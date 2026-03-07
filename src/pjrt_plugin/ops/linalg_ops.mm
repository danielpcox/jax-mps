// Linear algebra operations: cholesky, triangular_solve, QR, eigh, SVD
// Uses native MPS kernels (MPSMatrixDecompositionCholesky, MPSMatrixSolveTriangular)
// and Accelerate LAPACK (sgeqrf_, sorgqr_, ssyevd_, sgesdd_) for operations not
// available in MPS Graph.

#import <Accelerate/Accelerate.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#import "pjrt_plugin/mps_buffer.h"
#import "pjrt_plugin/ops/registry.h"

namespace jax_mps {

// ---------------------------------------------------------------------------
// Helpers for MPS row-byte alignment.
// MPS requires 16-byte-aligned row strides; rowBytesFromColumns returns the
// recommended value (e.g. 16 for a 2-column float32 matrix instead of 8).
// When the data stride differs we blit rows into an aligned staging buffer
// before calling the MPS kernel and blit back afterwards.
// ---------------------------------------------------------------------------

/// Blit rows between buffers with different row strides on the command buffer.
static void BlitRows(id<MTLCommandBuffer> cmdBuf, id<MTLBuffer> src, NSUInteger srcRowBytes,
                     id<MTLBuffer> dst, NSUInteger dstRowBytes, int64_t rows,
                     NSUInteger copyBytes) {
    id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
    for (int64_t r = 0; r < rows; r++) {
        [blit copyFromBuffer:src
                 sourceOffset:(NSUInteger)r * srcRowBytes
                     toBuffer:dst
            destinationOffset:(NSUInteger)r * dstRowBytes
                         size:copyBytes];
    }
    [blit endEncoding];
}

/// Copy contiguous rows from src to pre-allocated dst with different row strides.
/// If no padding is needed (strides match), just copies the data directly.
static void PadToBuffer(id<MTLCommandBuffer> cmdBuf, id<MTLBuffer> src, id<MTLBuffer> dst,
                        int64_t rows, NSUInteger dataRowBytes, NSUInteger mpsRowBytes) {
    if (mpsRowBytes == dataRowBytes) {
        id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
        [blit copyFromBuffer:src sourceOffset:0 toBuffer:dst destinationOffset:0 size:src.length];
        [blit endEncoding];
    } else {
        BlitRows(cmdBuf, src, dataRowBytes, dst, mpsRowBytes, rows, dataRowBytes);
    }
}

/// Copy padded rows from src to pre-allocated contiguous dst.
/// If no padding was used (strides match), just copies the data directly.
static void UnpadToBuffer(id<MTLCommandBuffer> cmdBuf, id<MTLBuffer> src, id<MTLBuffer> dst,
                          int64_t rows, NSUInteger dataRowBytes, NSUInteger mpsRowBytes) {
    if (mpsRowBytes == dataRowBytes) {
        id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
        [blit copyFromBuffer:src sourceOffset:0 toBuffer:dst destinationOffset:0 size:dst.length];
        [blit endEncoding];
    } else {
        BlitRows(cmdBuf, src, mpsRowBytes, dst, dataRowBytes, rows, dataRowBytes);
    }
}

// ---------------------------------------------------------------------------
// stablehlo.cholesky – native MPSMatrixDecompositionCholesky
// Supports batched inputs of shape [batch..., n, n] by looping over batch dims.
//
// NOTE: Unlike MPSMatrixMultiplication which has native batch support via
// batchStart/batchSize, MPSMatrixDecompositionCholesky only supports single
// matrix operations. The loop-based approach is necessary. This matches how
// other frameworks (e.g., mlx) handle batched Cholesky on MPS.
// ---------------------------------------------------------------------------

/// Fill a buffer with zeros using blit command encoder.
static void FillBufferWithZeros(id<MTLCommandBuffer> cmdBuf, id<MTLBuffer> buffer, size_t size) {
    id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
    [blit fillBuffer:buffer range:NSMakeRange(0, size) value:0];
    [blit endEncoding];
}

static NativeResult NativeHandle_cholesky(id<MTLDevice> device, id<MTLCommandBuffer> cmdBuf,
                                          mlir::Operation* op,
                                          const std::vector<id<MTLBuffer>>& inputs) {
    auto choleskyOp = mlir::dyn_cast<mlir::stablehlo::CholeskyOp>(op);
    if (!choleskyOp) {
        return NativeResult::Error("cholesky: expected CholeskyOp");
    }

    bool lower = true;
    if (choleskyOp.getLowerAttr()) {
        lower = choleskyOp.getLower();
    }

    auto resultType = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    auto shape = resultType.getShape();
    if (shape.size() < 2) {
        return NativeResult::Error("cholesky: expected at least rank 2 (got rank " +
                                   std::to_string(shape.size()) + ")");
    }
    int64_t n = shape[shape.size() - 1];
    int64_t m = shape[shape.size() - 2];
    if (n != m) {
        return NativeResult::Error("cholesky: expected square matrix (got " + std::to_string(m) +
                                   " x " + std::to_string(n) + ")");
    }

    // Compute batch size (product of all dimensions except last two).
    int64_t batchSize = 1;
    for (size_t i = 0; i < shape.size() - 2; i++) {
        batchSize *= shape[i];
    }

    if (!resultType.getElementType().isF32()) {
        return NativeResult::Error("cholesky: only float32 is supported");
    }

    MPSDataType mps_dtype = MlirTypeToMps(resultType.getElementType());
    int pjrt_dtype = MlirTypeToPjrtDtype(resultType.getElementType());
    size_t elem_size = DtypeByteSize(pjrt_dtype);
    NSUInteger dataRowBytes = (NSUInteger)(n * (int64_t)elem_size);
    NSUInteger mpsRowBytes = [MPSMatrixDescriptor rowBytesFromColumns:(NSUInteger)n
                                                             dataType:mps_dtype];
    size_t matrixDataSize = (size_t)(n * n) * elem_size;  // Size of one matrix in input.
    size_t matrixMpsSize = (size_t)n * mpsRowBytes;       // Size of one matrix with MPS alignment.

    // Allocate output buffer for all batches.
    size_t totalOutSize = (size_t)batchSize * matrixDataSize;
    id<MTLBuffer> outBuf = [device newBufferWithLength:totalOutSize
                                               options:MTLResourceStorageModeShared];

    // Compile the verification shader once.
    static id<MTLComputePipelineState> verifyPipeline = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
      NSString* source = @"#include <metal_stdlib>\n"
                          "using namespace metal;\n"
                          "kernel void cholesky_verify(\n"
                          "    device float *L [[buffer(0)]],\n"
                          "    constant uint &n [[buffer(1)]],\n"
                          "    constant uint &stride [[buffer(2)]],\n"
                          "    uint tid [[thread_position_in_grid]]\n"
                          ") {\n"
                          "    if (tid != 0) return;\n"
                          "    for (uint j = 0; j < n; j++) {\n"
                          "        if (L[j * stride + j] <= 0.0f) {\n"
                          "            for (uint r = 0; r < n; r++)\n"
                          "                for (uint c = 0; c < n; c++)\n"
                          "                    L[r * stride + c] = NAN;\n"
                          "            return;\n"
                          "        }\n"
                          "    }\n"
                          "}\n";
      NSError* error = nil;
      id<MTLLibrary> lib = [device newLibraryWithSource:source options:nil error:&error];
      if (lib) {
          id<MTLFunction> func = [lib newFunctionWithName:@"cholesky_verify"];
          verifyPipeline = [device newComputePipelineStateWithFunction:func error:&error];
      }
      if (!verifyPipeline) {
          MPS_LOG_ERROR("cholesky: failed to compile verify shader: %s\n",
                        error.localizedDescription.UTF8String);
      }
    });

    MPSMatrixDescriptor* desc = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)n
                                                                      columns:(NSUInteger)n
                                                                     rowBytes:mpsRowBytes
                                                                     dataType:mps_dtype];

    MPSMatrixDecompositionCholesky* cholesky =
        [[MPSMatrixDecompositionCholesky alloc] initWithDevice:device
                                                         lower:lower
                                                         order:(NSUInteger)n];

    // Pre-allocate reusable buffers for batch processing.
    // srcSlice: holds one matrix slice copied from input
    // srcBuf: padded source for MPS (same as srcSlice if no padding needed)
    // resultBuf: MPS output with alignment
    // unpaddedBuf: contiguous result (same as resultBuf if no padding needed)
    //
    // NOTE: Using MTLResourceStorageModeShared for simplicity. If profiling shows
    // memory bandwidth is a bottleneck, consider MTLResourceStorageModePrivate for
    // GPU-only intermediate buffers (srcBuf, resultBuf when padding is needed).
    bool needsPadding = (mpsRowBytes != dataRowBytes);
    id<MTLBuffer> srcSlice = [device newBufferWithLength:matrixDataSize
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> srcBuf = needsPadding ? [device newBufferWithLength:matrixMpsSize
                                                              options:MTLResourceStorageModeShared]
                                        : srcSlice;
    id<MTLBuffer> resultBuf = [device newBufferWithLength:matrixMpsSize
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> unpaddedBuf = needsPadding
                                    ? [device newBufferWithLength:matrixDataSize
                                                          options:MTLResourceStorageModeShared]
                                    : resultBuf;

    MPSMatrix* sourceMatrix = [[MPSMatrix alloc] initWithBuffer:srcBuf descriptor:desc];
    MPSMatrix* resultMatrix = [[MPSMatrix alloc] initWithBuffer:resultBuf descriptor:desc];

    // Verification kernel constants.
    uint32_t n32 = (uint32_t)n;
    uint32_t lStride = (uint32_t)(mpsRowBytes / elem_size);

    // Process each matrix in the batch.
    for (int64_t b = 0; b < batchSize; b++) {
        // Blit this matrix slice from input buffer.
        size_t srcOffset = (size_t)b * matrixDataSize;
        id<MTLBlitCommandEncoder> blitIn = [cmdBuf blitCommandEncoder];
        [blitIn copyFromBuffer:inputs[0]
                  sourceOffset:srcOffset
                      toBuffer:srcSlice
             destinationOffset:0
                          size:matrixDataSize];
        [blitIn endEncoding];

        // Pad source rows to MPS-recommended alignment if needed.
        if (needsPadding) {
            memset(srcBuf.contents, 0, matrixMpsSize);
            PadToBuffer(cmdBuf, srcSlice, srcBuf, n, dataRowBytes, mpsRowBytes);
        }

        // Zero-fill result buffer (unused triangle must be clean).
        FillBufferWithZeros(cmdBuf, resultBuf, matrixMpsSize);

        // The status buffer is unreliable on Apple Silicon — it always writes 0
        // (success) regardless of whether the input is positive definite.
        // See https://developer.apple.com/forums/thread/736787
        [cholesky encodeToCommandBuffer:cmdBuf
                           sourceMatrix:sourceMatrix
                           resultMatrix:resultMatrix
                                 status:nil];

        // Verification kernel to check diagonal and fill with NaN if non-positive.
        if (verifyPipeline) {
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:verifyPipeline];
            [enc setBuffer:resultBuf offset:0 atIndex:0];
            [enc setBytes:&n32 length:sizeof(n32) atIndex:1];
            [enc setBytes:&lStride length:sizeof(lStride) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [enc endEncoding];
        }

        // Unpad result to contiguous layout if needed.
        if (needsPadding) {
            UnpadToBuffer(cmdBuf, resultBuf, unpaddedBuf, n, dataRowBytes, mpsRowBytes);
        }

        // Blit the result to the output buffer at the correct offset.
        size_t dstOffset = (size_t)b * matrixDataSize;
        id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
        [blit copyFromBuffer:unpaddedBuf
                 sourceOffset:0
                     toBuffer:outBuf
            destinationOffset:dstOffset
                         size:matrixDataSize];
        [blit endEncoding];
    }

    return NativeResult::Buffer(outBuf);
}

REGISTER_NATIVE_MPS_OP("stablehlo.cholesky", NativeHandle_cholesky);

// ---------------------------------------------------------------------------
// stablehlo.triangular_solve – native MPSMatrixSolveTriangular
// Supports batched inputs of shape [batch..., n, n] by looping over batch dims.
//
// NOTE: MPSMatrixSolveTriangular only supports single matrix operations.
// The loop-based approach is necessary (same limitation as Cholesky above).
// ---------------------------------------------------------------------------

static NativeResult NativeHandle_triangular_solve(id<MTLDevice> device, id<MTLCommandBuffer> cmdBuf,
                                                  mlir::Operation* op,
                                                  const std::vector<id<MTLBuffer>>& inputs) {
    auto triSolveOp = mlir::dyn_cast<mlir::stablehlo::TriangularSolveOp>(op);
    if (!triSolveOp) {
        return NativeResult::Error("triangular_solve: expected TriangularSolveOp");
    }

    bool leftSide = triSolveOp.getLeftSide();
    bool lower = triSolveOp.getLower();
    bool unitDiagonal = triSolveOp.getUnitDiagonal();
    auto transposeA = triSolveOp.getTransposeA();
    bool transpose = (transposeA == mlir::stablehlo::Transpose::TRANSPOSE ||
                      transposeA == mlir::stablehlo::Transpose::ADJOINT);

    auto aType = mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    auto bType = mlir::cast<mlir::RankedTensorType>(op->getOperand(1).getType());
    auto aShape = aType.getShape();
    auto bShape = bType.getShape();

    if (aShape.size() < 2 || bShape.size() < 2) {
        return NativeResult::Error("triangular_solve: expected at least rank 2 (got ranks " +
                                   std::to_string(aShape.size()) + ", " +
                                   std::to_string(bShape.size()) + ")");
    }

    // A must be square.
    int64_t aRows = aShape[aShape.size() - 2];
    int64_t aCols = aShape[aShape.size() - 1];
    if (aRows != aCols) {
        return NativeResult::Error("triangular_solve: matrix A must be square (got " +
                                   std::to_string(aRows) + " x " + std::to_string(aCols) + ")");
    }

    // A and B must have matching rank and batch dimensions.
    if (aShape.size() != bShape.size()) {
        return NativeResult::Error("triangular_solve: A and B must have same rank (got " +
                                   std::to_string(aShape.size()) + " vs " +
                                   std::to_string(bShape.size()) + ")");
    }
    for (size_t i = 0; i < aShape.size() - 2; i++) {
        if (aShape[i] != bShape[i]) {
            return NativeResult::Error(
                "triangular_solve: A and B batch dimensions must match (dim " + std::to_string(i) +
                ": " + std::to_string(aShape[i]) + " vs " + std::to_string(bShape[i]) + ")");
        }
    }

    // Compute batch size (product of all dimensions except last two).
    // Both A and B must have the same batch dimensions.
    int64_t batchSize = 1;
    size_t batchRank = aShape.size() - 2;
    for (size_t i = 0; i < batchRank; i++) {
        batchSize *= aShape[i];
    }

    int64_t n = aShape[aShape.size() - 1];
    int64_t bRows = bShape[bShape.size() - 2];
    int64_t bCols = bShape[bShape.size() - 1];

    if (!bType.getElementType().isF32()) {
        return NativeResult::Error("triangular_solve: only float32 is supported");
    }

    MPSDataType mps_dtype = MlirTypeToMps(bType.getElementType());
    int pjrt_dtype = MlirTypeToPjrtDtype(bType.getElementType());
    size_t elem_size = DtypeByteSize(pjrt_dtype);

    // Number of right-hand sides depends on whether A is on the left or right.
    NSUInteger nrhs = leftSide ? (NSUInteger)bCols : (NSUInteger)bRows;

    // Row strides: data-contiguous vs MPS-recommended.
    NSUInteger aDataRowBytes = (NSUInteger)(n * (int64_t)elem_size);
    NSUInteger aMpsRowBytes = [MPSMatrixDescriptor rowBytesFromColumns:(NSUInteger)n
                                                              dataType:mps_dtype];
    NSUInteger bDataRowBytes = (NSUInteger)(bCols * (int64_t)elem_size);
    NSUInteger bMpsRowBytes = [MPSMatrixDescriptor rowBytesFromColumns:(NSUInteger)bCols
                                                              dataType:mps_dtype];

    size_t aMatrixDataSize = (size_t)(n * n) * elem_size;
    size_t bMatrixDataSize = (size_t)(bRows * bCols) * elem_size;

    // Allocate output buffer for all batches.
    size_t totalOutSize = (size_t)batchSize * bMatrixDataSize;
    id<MTLBuffer> outBuf = [device newBufferWithLength:totalOutSize
                                               options:MTLResourceStorageModeShared];

    MPSMatrixDescriptor* aDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)n
                                                                       columns:(NSUInteger)n
                                                                      rowBytes:aMpsRowBytes
                                                                      dataType:mps_dtype];

    MPSMatrixDescriptor* bDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)bRows
                                                                       columns:(NSUInteger)bCols
                                                                      rowBytes:bMpsRowBytes
                                                                      dataType:mps_dtype];

    MPSMatrixSolveTriangular* solver =
        [[MPSMatrixSolveTriangular alloc] initWithDevice:device
                                                   right:!leftSide
                                                   upper:!lower
                                               transpose:transpose
                                                    unit:unitDiagonal
                                                   order:(NSUInteger)n
                                  numberOfRightHandSides:nrhs
                                                   alpha:1.0];

    // Pre-allocate reusable buffers for batch processing.
    // NOTE: Using MTLResourceStorageModeShared for simplicity. If profiling shows
    // memory bandwidth is a bottleneck, consider MTLResourceStorageModePrivate for
    // GPU-only intermediate buffers (aBuf, bBuf, solBuf when padding is needed).
    bool aNeedsPadding = (aMpsRowBytes != aDataRowBytes);
    bool bNeedsPadding = (bMpsRowBytes != bDataRowBytes);
    size_t aMpsSize = (size_t)n * aMpsRowBytes;
    size_t bMpsSize = (size_t)bRows * bMpsRowBytes;

    id<MTLBuffer> aSlice = [device newBufferWithLength:aMatrixDataSize
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> aBuf = aNeedsPadding ? [device newBufferWithLength:aMpsSize
                                                             options:MTLResourceStorageModeShared]
                                       : aSlice;
    id<MTLBuffer> bSlice = [device newBufferWithLength:bMatrixDataSize
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> bBuf = bNeedsPadding ? [device newBufferWithLength:bMpsSize
                                                             options:MTLResourceStorageModeShared]
                                       : bSlice;
    id<MTLBuffer> solBuf = [device newBufferWithLength:bMpsSize
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> unpaddedBuf = bNeedsPadding
                                    ? [device newBufferWithLength:bMatrixDataSize
                                                          options:MTLResourceStorageModeShared]
                                    : solBuf;

    MPSMatrix* sourceMatrix = [[MPSMatrix alloc] initWithBuffer:aBuf descriptor:aDesc];
    MPSMatrix* rhsMatrix = [[MPSMatrix alloc] initWithBuffer:bBuf descriptor:bDesc];
    MPSMatrix* solutionMatrix = [[MPSMatrix alloc] initWithBuffer:solBuf descriptor:bDesc];

    // Process each matrix in the batch.
    for (int64_t b = 0; b < batchSize; b++) {
        // Blit matrix slices from input buffers.
        size_t aOffset = (size_t)b * aMatrixDataSize;
        size_t bOffset = (size_t)b * bMatrixDataSize;

        id<MTLBlitCommandEncoder> blitA = [cmdBuf blitCommandEncoder];
        [blitA copyFromBuffer:inputs[0]
                 sourceOffset:aOffset
                     toBuffer:aSlice
            destinationOffset:0
                         size:aMatrixDataSize];
        [blitA endEncoding];

        id<MTLBlitCommandEncoder> blitB = [cmdBuf blitCommandEncoder];
        [blitB copyFromBuffer:inputs[1]
                 sourceOffset:bOffset
                     toBuffer:bSlice
            destinationOffset:0
                         size:bMatrixDataSize];
        [blitB endEncoding];

        // Pad inputs to MPS alignment if needed.
        if (aNeedsPadding) {
            memset(aBuf.contents, 0, aMpsSize);
            PadToBuffer(cmdBuf, aSlice, aBuf, n, aDataRowBytes, aMpsRowBytes);
        }
        if (bNeedsPadding) {
            memset(bBuf.contents, 0, bMpsSize);
            PadToBuffer(cmdBuf, bSlice, bBuf, bRows, bDataRowBytes, bMpsRowBytes);
        }

        [solver encodeToCommandBuffer:cmdBuf
                         sourceMatrix:sourceMatrix
                  rightHandSideMatrix:rhsMatrix
                       solutionMatrix:solutionMatrix];

        // Unpad solution to contiguous layout if needed.
        if (bNeedsPadding) {
            UnpadToBuffer(cmdBuf, solBuf, unpaddedBuf, bRows, bDataRowBytes, bMpsRowBytes);
        }

        // Blit the result to the output buffer at the correct offset.
        size_t dstOffset = (size_t)b * bMatrixDataSize;
        id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
        [blit copyFromBuffer:unpaddedBuf
                 sourceOffset:0
                     toBuffer:outBuf
            destinationOffset:dstOffset
                         size:bMatrixDataSize];
        [blit endEncoding];
    }

    return NativeResult::Buffer(outBuf);
}

REGISTER_NATIVE_MPS_OP("stablehlo.triangular_solve", NativeHandle_triangular_solve);

// The native handler is used for triangular_solve. When it appears inside
// func.call (e.g., in jnp.linalg.inv → @_lu_solve), the execution engine's
// inline pass automatically inlines the callee so that segmented execution
// can handle the native op properly.

// ---------------------------------------------------------------------------
// QR decomposition via Accelerate LAPACK
// ---------------------------------------------------------------------------
// custom_call @"Qr": computes QR factorization using sgeqrf_/dgeqrf_
// Input: tensor<...xMxNxf32>
// Output 0: tensor<...xMxNxf32> (R in upper triangle, Householder vectors below)
// Output 1: tensor<...xmin(M,N)xf32> (tau values)

static NativeResult NativeHandle_Qr(id<MTLDevice> device, id<MTLCommandBuffer> cmdBuf,
                                     mlir::Operation* op,
                                     const std::vector<id<MTLBuffer>>& inputs) {
    if (inputs.empty())
        return NativeResult::Error("Qr: missing input");

    auto customCallOp = mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(op);
    if (!customCallOp || op->getNumResults() != 2)
        return NativeResult::Error("Qr: expected CustomCallOp with 2 results");

    auto inputType = mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    auto shape = inputType.getShape();
    if (shape.size() < 2)
        return NativeResult::Error("Qr: expected at least rank 2");

    if (!inputType.getElementType().isF32())
        return NativeResult::Error("Qr: only float32 supported");

    int64_t m = shape[shape.size() - 2];
    int64_t n = shape[shape.size() - 1];
    int64_t k = std::min(m, n);

    // Compute batch size
    int64_t batchSize = 1;
    for (size_t i = 0; i < shape.size() - 2; i++)
        batchSize *= shape[i];

    size_t matrixSize = (size_t)(m * n) * sizeof(float);
    size_t tauSize = (size_t)k * sizeof(float);

    // Allocate output buffers
    size_t totalMatrixSize = (size_t)batchSize * matrixSize;
    size_t totalTauSize = (size_t)batchSize * tauSize;

    id<MTLBuffer> outMatrix = [device newBufferWithLength:totalMatrixSize
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> outTau = [device newBufferWithLength:totalTauSize
                                               options:MTLResourceStorageModeShared];

    // The execution engine flushes pending GPU work before native steps,
    // so input data is available in shared memory.

    // Copy input to output (LAPACK works in-place)
    memcpy(outMatrix.contents, inputs[0].contents, totalMatrixSize);

    // Process each batch element
    for (int64_t b = 0; b < batchSize; b++) {
        float* a = (float*)outMatrix.contents + b * m * n;
        float* tau = (float*)outTau.contents + b * k;

        // LAPACK uses column-major, but JAX uses row-major.
        // sgeqrf on row-major A is equivalent to LQ on column-major A.
        // We transpose in place, call sgeqrf, and transpose back.
        // For simplicity, we transpose M×N → N×M, call sgeqrf(N, M), get taus of size k.

        // Transpose in-place: allocate temp buffer
        std::vector<float> temp(m * n);

        // Row-major A[i,j] = a[i*n + j] → column-major A[i,j] = temp[j*m + i]
        // Equivalently: transpose to get A^T in row-major = A in column-major
        for (int64_t i = 0; i < m; i++)
            for (int64_t j = 0; j < n; j++)
                temp[j * m + i] = a[i * n + j];

        __CLPK_integer lm = (__CLPK_integer)m;
        __CLPK_integer ln = (__CLPK_integer)n;
        __CLPK_integer lda = (__CLPK_integer)m;  // leading dim for column-major (m×n in col-major = m)
        __CLPK_integer info = 0;
        __CLPK_integer lwork = -1;

        // Query optimal workspace
        float work_query;
        sgeqrf_(&lm, &ln, temp.data(), &lda, tau, &work_query, &lwork, &info);
        lwork = (__CLPK_integer)work_query;
        std::vector<float> work(lwork);

        // Compute QR
        sgeqrf_(&lm, &ln, temp.data(), &lda, tau, work.data(), &lwork, &info);
        if (info != 0)
            return NativeResult::Error("Qr: sgeqrf failed with info=" + std::to_string(info));

        // Transpose back: column-major result → row-major
        for (int64_t i = 0; i < m; i++)
            for (int64_t j = 0; j < n; j++)
                a[i * n + j] = temp[j * m + i];
    }

    return NativeResult::Buffers({outMatrix, outTau});
}

// Register Qr as a NATIVE custom call target
static bool _cc_reg_Qr =
    CustomCallRegistry::Register("Qr", OpHandler::Native(NativeHandle_Qr));

// ---------------------------------------------------------------------------
// ProductOfElementaryHouseholderReflectors via Accelerate LAPACK (sorgqr_)
// ---------------------------------------------------------------------------
// custom_call @"ProductOfElementaryHouseholderReflectors"
// Input 0: tensor<...xMxNxf32> (packed QR result from Qr)
// Input 1: tensor<...xKxf32> (tau values)
// Output: tensor<...xMxNxf32> (Q matrix)

static NativeResult NativeHandle_HouseholderProduct(id<MTLDevice> device,
                                                     id<MTLCommandBuffer> cmdBuf,
                                                     mlir::Operation* op,
                                                     const std::vector<id<MTLBuffer>>& inputs) {
    if (inputs.size() < 2)
        return NativeResult::Error("HouseholderProduct: need 2 inputs");

    auto customCallOp = mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(op);
    if (!customCallOp || op->getNumResults() != 1)
        return NativeResult::Error("HouseholderProduct: expected CustomCallOp with 1 result");

    auto inputType = mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    auto tauType = mlir::cast<mlir::RankedTensorType>(op->getOperand(1).getType());
    auto shape = inputType.getShape();
    if (shape.size() < 2)
        return NativeResult::Error("HouseholderProduct: expected at least rank 2");

    if (!inputType.getElementType().isF32())
        return NativeResult::Error("HouseholderProduct: only float32 supported");

    int64_t m = shape[shape.size() - 2];
    int64_t n = shape[shape.size() - 1];
    int64_t k = tauType.getShape().back();

    int64_t batchSize = 1;
    for (size_t i = 0; i < shape.size() - 2; i++)
        batchSize *= shape[i];

    // Output shape matches input shape
    auto outType = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    auto outShape = outType.getShape();
    int64_t outM = outShape[outShape.size() - 2];
    int64_t outN = outShape[outShape.size() - 1];

    size_t outMatrixSize = (size_t)(outM * outN) * sizeof(float);
    size_t totalOutSize = (size_t)batchSize * outMatrixSize;

    id<MTLBuffer> outBuf = [device newBufferWithLength:totalOutSize
                                               options:MTLResourceStorageModeShared];

    // Copy packed QR input to output (sorgqr works in-place)
    size_t inputMatrixSize = (size_t)(m * n) * sizeof(float);
    size_t tauBatchSize = (size_t)k * sizeof(float);

    for (int64_t b = 0; b < batchSize; b++) {
        const float* aIn = (const float*)inputs[0].contents + b * m * n;
        const float* tauIn = (const float*)inputs[1].contents + b * k;
        float* out = (float*)outBuf.contents + b * outM * outN;

        // Transpose to column-major for LAPACK
        std::vector<float> temp(m * n);
        for (int64_t i = 0; i < m; i++)
            for (int64_t j = 0; j < n; j++)
                temp[j * m + i] = aIn[i * n + j];

        // Copy tau (1D, no transpose needed)
        std::vector<float> tau(k);
        memcpy(tau.data(), tauIn, k * sizeof(float));

        __CLPK_integer lm = (__CLPK_integer)m;
        __CLPK_integer ln = (__CLPK_integer)n;
        __CLPK_integer lk = (__CLPK_integer)k;
        __CLPK_integer lda = (__CLPK_integer)m;
        __CLPK_integer info = 0;
        __CLPK_integer lwork = -1;

        // Query workspace
        float work_query;
        sorgqr_(&lm, &ln, &lk, temp.data(), &lda, tau.data(), &work_query, &lwork, &info);
        lwork = (__CLPK_integer)work_query;
        std::vector<float> work(lwork);

        // Compute Q
        sorgqr_(&lm, &ln, &lk, temp.data(), &lda, tau.data(), work.data(), &lwork, &info);
        if (info != 0)
            return NativeResult::Error("HouseholderProduct: sorgqr failed with info=" +
                                       std::to_string(info));

        // Transpose back: column-major → row-major
        for (int64_t i = 0; i < outM; i++)
            for (int64_t j = 0; j < outN; j++)
                out[i * outN + j] = temp[j * m + i];
    }

    return NativeResult::Buffer(outBuf);
}

static bool _cc_reg_HouseholderProduct = CustomCallRegistry::Register(
    "ProductOfElementaryHouseholderReflectors",
    OpHandler::Native(NativeHandle_HouseholderProduct));

// ---------------------------------------------------------------------------
// Symmetric eigendecomposition via Accelerate LAPACK (ssyevd_)
// ---------------------------------------------------------------------------
// custom_call @"mps_syevd": computes eigenvalues and eigenvectors
// Input: tensor<...xNxNxf32> (symmetric matrix)
// Output 0: tensor<...xNxNxf32> (eigenvectors)
// Output 1: tensor<...xNxf32> (eigenvalues)

static NativeResult NativeHandle_Syevd(id<MTLDevice> device, id<MTLCommandBuffer> cmdBuf,
                                        mlir::Operation* op,
                                        const std::vector<id<MTLBuffer>>& inputs) {
    if (inputs.empty())
        return NativeResult::Error("Syevd: missing input");

    auto customCallOp = mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(op);
    if (!customCallOp || op->getNumResults() != 2)
        return NativeResult::Error("Syevd: expected CustomCallOp with 2 results");

    auto inputType = mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    auto shape = inputType.getShape();
    if (shape.size() < 2)
        return NativeResult::Error("Syevd: expected at least rank 2");
    if (!inputType.getElementType().isF32())
        return NativeResult::Error("Syevd: only float32 supported");

    int64_t n = shape[shape.size() - 1];

    int64_t batchSize = 1;
    for (size_t i = 0; i < shape.size() - 2; i++)
        batchSize *= shape[i];

    size_t matrixSize = (size_t)(n * n) * sizeof(float);
    size_t eigenvalSize = (size_t)n * sizeof(float);
    size_t totalMatrixSize = (size_t)batchSize * matrixSize;
    size_t totalEigenvalSize = (size_t)batchSize * eigenvalSize;

    id<MTLBuffer> outVectors = [device newBufferWithLength:totalMatrixSize
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> outValues = [device newBufferWithLength:totalEigenvalSize
                                                  options:MTLResourceStorageModeShared];

    // Copy input (ssyevd works in-place, overwrites with eigenvectors)
    memcpy(outVectors.contents, inputs[0].contents, totalMatrixSize);

    for (int64_t b = 0; b < batchSize; b++) {
        float* a = (float*)outVectors.contents + b * n * n;
        float* w = (float*)outValues.contents + b * n;

        // Transpose to column-major for LAPACK
        std::vector<float> temp(n * n);
        for (int64_t i = 0; i < n; i++)
            for (int64_t j = 0; j < n; j++)
                temp[j * n + i] = a[i * n + j];

        char jobz = 'V';  // Compute eigenvalues and eigenvectors
        char uplo = 'L';  // Lower triangle
        __CLPK_integer ln = (__CLPK_integer)n;
        __CLPK_integer lda = (__CLPK_integer)n;
        __CLPK_integer info = 0;
        __CLPK_integer lwork = -1;
        __CLPK_integer liwork = -1;

        // Query workspace
        float work_query;
        __CLPK_integer iwork_query;
        ssyevd_(&jobz, &uplo, &ln, temp.data(), &lda, w, &work_query, &lwork, &iwork_query, &liwork,
                &info);
        lwork = (__CLPK_integer)work_query;
        liwork = iwork_query;
        std::vector<float> work(lwork);
        std::vector<__CLPK_integer> iwork(liwork);

        // Compute eigendecomposition
        ssyevd_(&jobz, &uplo, &ln, temp.data(), &lda, w, work.data(), &lwork, iwork.data(), &liwork,
                &info);
        if (info != 0)
            return NativeResult::Error("Syevd: ssyevd failed with info=" + std::to_string(info));

        // Transpose eigenvectors back: column-major → row-major
        for (int64_t i = 0; i < n; i++)
            for (int64_t j = 0; j < n; j++)
                a[i * n + j] = temp[j * n + i];
    }

    return NativeResult::Buffers({outVectors, outValues});
}

static bool _cc_reg_Syevd =
    CustomCallRegistry::Register("mps_syevd", OpHandler::Native(NativeHandle_Syevd));

// ---------------------------------------------------------------------------
// SVD via Accelerate LAPACK (sgesdd_ – divide and conquer)
// ---------------------------------------------------------------------------
// custom_call @"mps_sgesdd": computes singular value decomposition
// Input: tensor<...xMxNxf32> (matrix)
// Output 0: tensor<...xKxf32> (singular values, K = min(M, N))
// Output 1: tensor<...xMxKxf32> or tensor<...xMxMxf32> (U matrix)
// Output 2: tensor<...xKxNxf32> or tensor<...xNxNxf32> (Vt matrix)

static NativeResult NativeHandle_Sgesdd(id<MTLDevice> device, id<MTLCommandBuffer> cmdBuf,
                                         mlir::Operation* op,
                                         const std::vector<id<MTLBuffer>>& inputs) {
    if (inputs.empty())
        return NativeResult::Error("Sgesdd: missing input");

    auto customCallOp = mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(op);
    if (!customCallOp || op->getNumResults() != 3)
        return NativeResult::Error("Sgesdd: expected CustomCallOp with 3 results");

    auto inputType = mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    auto shape = inputType.getShape();
    if (shape.size() < 2)
        return NativeResult::Error("Sgesdd: expected at least rank 2");
    if (!inputType.getElementType().isF32())
        return NativeResult::Error("Sgesdd: only float32 supported");

    int64_t m = shape[shape.size() - 2];
    int64_t n = shape[shape.size() - 1];
    int64_t k = std::min(m, n);

    // Get output shapes from result types
    auto sType = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    auto uType = mlir::cast<mlir::RankedTensorType>(op->getResult(1).getType());
    auto vtType = mlir::cast<mlir::RankedTensorType>(op->getResult(2).getType());
    auto uShape = uType.getShape();
    auto vtShape = vtType.getShape();

    int64_t uRows = uShape[uShape.size() - 2];
    int64_t uCols = uShape[uShape.size() - 1];
    int64_t vtRows = vtShape[vtShape.size() - 2];
    int64_t vtCols = vtShape[vtShape.size() - 1];

    // Determine if full_matrices from output shapes
    bool full_matrices = (uCols == m) || (vtRows == n);

    // Compute batch size
    int64_t batchSize = 1;
    for (size_t i = 0; i < shape.size() - 2; i++)
        batchSize *= shape[i];

    size_t inputMatrixSize = (size_t)(m * n) * sizeof(float);
    size_t sSize = (size_t)k * sizeof(float);
    size_t uSize = (size_t)(uRows * uCols) * sizeof(float);
    size_t vtSize = (size_t)(vtRows * vtCols) * sizeof(float);

    id<MTLBuffer> outS = [device newBufferWithLength:(size_t)batchSize * sSize
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> outU = [device newBufferWithLength:(size_t)batchSize * uSize
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> outVt = [device newBufferWithLength:(size_t)batchSize * vtSize
                                               options:MTLResourceStorageModeShared];

    for (int64_t b = 0; b < batchSize; b++) {
        const float* aIn = (const float*)inputs[0].contents + b * m * n;
        float* s = (float*)outS.contents + b * k;
        float* u = (float*)outU.contents + b * uRows * uCols;
        float* vt = (float*)outVt.contents + b * vtRows * vtCols;

        // Transpose input to column-major for LAPACK
        std::vector<float> aCm(m * n);
        for (int64_t i = 0; i < m; i++)
            for (int64_t j = 0; j < n; j++)
                aCm[j * m + i] = aIn[i * n + j];

        // LAPACK output dimensions (column-major)
        // U is m×ucols in column-major, Vt is vtrows×n in column-major
        int64_t lapack_ucols = full_matrices ? m : k;
        int64_t lapack_vtrows = full_matrices ? n : k;
        std::vector<float> uCm(m * lapack_ucols);
        std::vector<float> vtCm(lapack_vtrows * n);

        char jobz = full_matrices ? 'A' : 'S';
        __CLPK_integer lm = (__CLPK_integer)m;
        __CLPK_integer ln = (__CLPK_integer)n;
        __CLPK_integer lda = (__CLPK_integer)m;
        __CLPK_integer ldu = (__CLPK_integer)m;
        __CLPK_integer ldvt = (__CLPK_integer)lapack_vtrows;
        __CLPK_integer info = 0;
        __CLPK_integer lwork = -1;

        // Query optimal workspace
        float work_query;
        __CLPK_integer iwork_buf[8 * std::min(m, n)];
        sgesdd_(&jobz, &lm, &ln, aCm.data(), &lda, s, uCm.data(), &ldu,
                vtCm.data(), &ldvt, &work_query, &lwork, iwork_buf, &info);
        lwork = (__CLPK_integer)work_query;
        std::vector<float> work(lwork);
        std::vector<__CLPK_integer> iwork(8 * k);

        // Compute SVD
        sgesdd_(&jobz, &lm, &ln, aCm.data(), &lda, s, uCm.data(), &ldu,
                vtCm.data(), &ldvt, work.data(), &lwork, iwork.data(), &info);
        if (info != 0)
            return NativeResult::Error("Sgesdd: sgesdd failed with info=" + std::to_string(info));

        // Transpose U from column-major to row-major
        // LAPACK U (col-major): m × lapack_ucols, U[i,j] = uCm[j*m + i]
        // Output U (row-major): uRows × uCols
        for (int64_t i = 0; i < uRows; i++)
            for (int64_t j = 0; j < uCols; j++)
                u[i * uCols + j] = uCm[j * m + i];

        // Transpose Vt from column-major to row-major
        // LAPACK Vt (col-major): lapack_vtrows × n, Vt[i,j] = vtCm[j*lapack_vtrows + i]
        // Output Vt (row-major): vtRows × vtCols
        for (int64_t i = 0; i < vtRows; i++)
            for (int64_t j = 0; j < vtCols; j++)
                vt[i * vtCols + j] = vtCm[j * lapack_vtrows + i];
    }

    return NativeResult::Buffers({outS, outU, outVt});
}

static bool _cc_reg_Sgesdd =
    CustomCallRegistry::Register("mps_sgesdd", OpHandler::Native(NativeHandle_Sgesdd));

}  // namespace jax_mps
