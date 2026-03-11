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

    // Complex Cholesky via Accelerate LAPACK cpotrf_
    if (mlir::isa<mlir::ComplexType>(resultType.getElementType())) {
        size_t elem_size = sizeof(float) * 2;  // complex<float>
        size_t matrixDataSize = (size_t)(n * n) * elem_size;
        size_t totalOutSize = (size_t)batchSize * matrixDataSize;

        id<MTLBuffer> outBuf = [device newBufferWithLength:totalOutSize
                                                   options:MTLResourceStorageModeShared];

        // Copy input to output (LAPACK works in-place)
        memcpy(outBuf.contents, inputs[0].contents, totalOutSize);

        char uplo = lower ? 'L' : 'U';
        int ln = (int)n;

        for (int64_t b = 0; b < batchSize; b++) {
            auto* data = (std::complex<float>*)outBuf.contents + b * n * n;

            // Transpose to column-major for LAPACK
            std::vector<std::complex<float>> colMajor(n * n);
            for (int64_t i = 0; i < n; i++)
                for (int64_t j = 0; j < n; j++)
                    colMajor[j * n + i] = data[i * n + j];

            int info = 0;
            cpotrf_(&uplo, &ln, (__CLPK_complex*)colMajor.data(), &ln, &info);

            if (info != 0) {
                // Not positive definite — fill with NaN (matching CPU behavior)
                for (int64_t i = 0; i < n * n; i++)
                    data[i] = std::complex<float>(NAN, NAN);
                continue;
            }

            // Transpose back to row-major
            for (int64_t i = 0; i < n; i++)
                for (int64_t j = 0; j < n; j++)
                    data[i * n + j] = colMajor[j * n + i];

            // Zero out the upper (or lower) triangle
            for (int64_t i = 0; i < n; i++)
                for (int64_t j = 0; j < n; j++) {
                    if (lower && j > i)
                        data[i * n + j] = {0, 0};
                    if (!lower && j < i)
                        data[i * n + j] = {0, 0};
                }
        }

        return NativeResult::Buffer(outBuf);
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

    // Complex triangular solve via Accelerate BLAS ctrsm_
    if (mlir::isa<mlir::ComplexType>(bType.getElementType())) {
        bool adjoint = (transposeA == mlir::stablehlo::Transpose::ADJOINT);
        size_t elem_size = sizeof(float) * 2;  // complex<float>
        size_t bMatrixDataSize = (size_t)(bRows * bCols) * elem_size;
        size_t totalOutSize = (size_t)batchSize * bMatrixDataSize;

        id<MTLBuffer> outBuf = [device newBufferWithLength:totalOutSize
                                                   options:MTLResourceStorageModeShared];

        // The execution engine flushes pending GPU work before native steps,
        // so input data is available in shared memory.
        const auto* aData = (const std::complex<float>*)inputs[0].contents;
        const auto* bData = (const std::complex<float>*)inputs[1].contents;
        auto* outData = (std::complex<float>*)outBuf.contents;

        // ctrsm_ parameters (Fortran column-major conventions)
        char side = leftSide ? 'L' : 'R';
        char uplo = lower ? 'L' : 'U';
        char transa = adjoint ? 'C' : (transpose ? 'T' : 'N');
        char diag = unitDiagonal ? 'U' : 'N';
        std::complex<float> alpha(1.0f, 0.0f);

        for (int64_t b = 0; b < batchSize; b++) {
            const auto* aSrc = aData + b * n * n;
            const auto* bSrc = bData + b * bRows * bCols;
            auto* dst = outData + b * bRows * bCols;

            // Transpose A (row-major -> column-major)
            std::vector<std::complex<float>> aCm(n * n);
            for (int64_t i = 0; i < n; i++)
                for (int64_t j = 0; j < n; j++)
                    aCm[j * n + i] = aSrc[i * n + j];

            // Transpose B (row-major -> column-major), write into bCm
            std::vector<std::complex<float>> bCm(bRows * bCols);
            for (int64_t i = 0; i < bRows; i++)
                for (int64_t j = 0; j < bCols; j++)
                    bCm[j * bRows + i] = bSrc[i * bCols + j];

            // ctrsm_ solves op(A) * X = alpha * B (left) or X * op(A) = alpha * B (right)
            // B is overwritten with X.
            int lm = leftSide ? (int)n : (int)bRows;
            int ln = leftSide ? (int)bCols : (int)n;
            int lda = (int)n;
            int ldb = (int)bRows;
            ctrsm_(&side, &uplo, &transa, &diag, &lm, &ln, (__CLPK_complex*)&alpha,
                   (__CLPK_complex*)aCm.data(), &lda, (__CLPK_complex*)bCm.data(), &ldb);

            // Transpose result back (column-major -> row-major)
            for (int64_t i = 0; i < bRows; i++)
                for (int64_t j = 0; j < bCols; j++)
                    dst[i * bCols + j] = bCm[j * bRows + i];
        }

        return NativeResult::Buffer(outBuf);
    }

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
                                    mlir::Operation* op, const std::vector<id<MTLBuffer>>& inputs) {
    if (inputs.empty())
        return NativeResult::Error("Qr: missing input");

    auto customCallOp = mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(op);
    if (!customCallOp || op->getNumResults() != 2)
        return NativeResult::Error("Qr: expected CustomCallOp with 2 results");

    auto inputType = mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    auto shape = inputType.getShape();
    if (shape.size() < 2)
        return NativeResult::Error("Qr: expected at least rank 2");

    int64_t m = shape[shape.size() - 2];
    int64_t n = shape[shape.size() - 1];
    int64_t k = std::min(m, n);

    // Compute batch size
    int64_t batchSize = 1;
    for (size_t i = 0; i < shape.size() - 2; i++)
        batchSize *= shape[i];

    bool isComplex = mlir::isa<mlir::ComplexType>(inputType.getElementType());
    size_t elemSize = isComplex ? sizeof(float) * 2 : sizeof(float);
    size_t matrixSize = (size_t)(m * n) * elemSize;
    size_t tauSize = (size_t)k * elemSize;

    if (!isComplex && !inputType.getElementType().isF32())
        return NativeResult::Error("Qr: only float32 and complex64 supported");

    // Allocate output buffers
    size_t totalMatrixSize = (size_t)batchSize * matrixSize;
    size_t totalTauSize = (size_t)batchSize * tauSize;

    id<MTLBuffer> outMatrix = [device newBufferWithLength:totalMatrixSize
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> outTau = [device newBufferWithLength:totalTauSize
                                               options:MTLResourceStorageModeShared];

    // Copy input to output (LAPACK works in-place)
    memcpy(outMatrix.contents, inputs[0].contents, totalMatrixSize);

    if (isComplex) {
        for (int64_t b = 0; b < batchSize; b++) {
            auto* a = (std::complex<float>*)outMatrix.contents + b * m * n;
            auto* tau = (std::complex<float>*)outTau.contents + b * k;

            std::vector<std::complex<float>> temp(m * n);
            for (int64_t i = 0; i < m; i++)
                for (int64_t j = 0; j < n; j++)
                    temp[j * m + i] = a[i * n + j];

            __CLPK_integer lm = (__CLPK_integer)m;
            __CLPK_integer ln = (__CLPK_integer)n;
            __CLPK_integer lda = (__CLPK_integer)m;
            __CLPK_integer info = 0;
            __CLPK_integer lwork = -1;

            std::complex<float> work_query;
            cgeqrf_(&lm, &ln, (__CLPK_complex*)temp.data(), &lda, (__CLPK_complex*)tau,
                    (__CLPK_complex*)&work_query, &lwork, &info);
            lwork = (__CLPK_integer)work_query.real();
            std::vector<std::complex<float>> work(lwork);
            cgeqrf_(&lm, &ln, (__CLPK_complex*)temp.data(), &lda, (__CLPK_complex*)tau,
                    (__CLPK_complex*)work.data(), &lwork, &info);
            if (info != 0)
                return NativeResult::Error("Qr: cgeqrf failed with info=" + std::to_string(info));

            for (int64_t i = 0; i < m; i++)
                for (int64_t j = 0; j < n; j++)
                    a[i * n + j] = temp[j * m + i];
        }
    } else {
        // Process each batch element (float32 path)
        for (int64_t b = 0; b < batchSize; b++) {
            float* a = (float*)outMatrix.contents + b * m * n;
            float* tau = (float*)outTau.contents + b * k;

            std::vector<float> temp(m * n);
            for (int64_t i = 0; i < m; i++)
                for (int64_t j = 0; j < n; j++)
                    temp[j * m + i] = a[i * n + j];

            __CLPK_integer lm = (__CLPK_integer)m;
            __CLPK_integer ln = (__CLPK_integer)n;
            __CLPK_integer lda = (__CLPK_integer)m;
            __CLPK_integer info = 0;
            __CLPK_integer lwork = -1;

            float work_query;
            sgeqrf_(&lm, &ln, temp.data(), &lda, tau, &work_query, &lwork, &info);
            lwork = (__CLPK_integer)work_query;
            std::vector<float> work(lwork);
            sgeqrf_(&lm, &ln, temp.data(), &lda, tau, work.data(), &lwork, &info);
            if (info != 0)
                return NativeResult::Error("Qr: sgeqrf failed with info=" + std::to_string(info));

            for (int64_t i = 0; i < m; i++)
                for (int64_t j = 0; j < n; j++)
                    a[i * n + j] = temp[j * m + i];
        }
    }

    return NativeResult::Buffers({outMatrix, outTau});
}

// Register Qr as a NATIVE custom call target
static bool _cc_reg_Qr = CustomCallRegistry::Register("Qr", OpHandler::Native(NativeHandle_Qr));

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

    int64_t m = shape[shape.size() - 2];
    int64_t n = shape[shape.size() - 1];
    int64_t k = tauType.getShape().back();

    int64_t batchSize = 1;
    for (size_t i = 0; i < shape.size() - 2; i++)
        batchSize *= shape[i];

    bool isComplex = mlir::isa<mlir::ComplexType>(inputType.getElementType());
    if (!isComplex && !inputType.getElementType().isF32())
        return NativeResult::Error("HouseholderProduct: only float32 and complex64 supported");

    size_t elemSize = isComplex ? sizeof(float) * 2 : sizeof(float);

    auto outType = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    auto outShape = outType.getShape();
    int64_t outM = outShape[outShape.size() - 2];
    int64_t outN = outShape[outShape.size() - 1];

    size_t outMatrixSize = (size_t)(outM * outN) * elemSize;
    size_t totalOutSize = (size_t)batchSize * outMatrixSize;

    id<MTLBuffer> outBuf = [device newBufferWithLength:totalOutSize
                                               options:MTLResourceStorageModeShared];

    if (isComplex) {
        for (int64_t b = 0; b < batchSize; b++) {
            const auto* aIn = (const std::complex<float>*)inputs[0].contents + b * m * n;
            const auto* tauIn = (const std::complex<float>*)inputs[1].contents + b * k;
            auto* out = (std::complex<float>*)outBuf.contents + b * outM * outN;

            std::vector<std::complex<float>> temp(m * n);
            for (int64_t i = 0; i < m; i++)
                for (int64_t j = 0; j < n; j++)
                    temp[j * m + i] = aIn[i * n + j];

            std::vector<std::complex<float>> tau(k);
            memcpy(tau.data(), tauIn, k * sizeof(std::complex<float>));

            __CLPK_integer lm = (__CLPK_integer)m;
            __CLPK_integer ln = (__CLPK_integer)n;
            __CLPK_integer lk = (__CLPK_integer)k;
            __CLPK_integer lda = (__CLPK_integer)m;
            __CLPK_integer info = 0;
            __CLPK_integer lwork = -1;

            std::complex<float> work_query;
            cungqr_(&lm, &ln, &lk, (__CLPK_complex*)temp.data(), &lda, (__CLPK_complex*)tau.data(),
                    (__CLPK_complex*)&work_query, &lwork, &info);
            lwork = (__CLPK_integer)work_query.real();
            std::vector<std::complex<float>> work(lwork);
            cungqr_(&lm, &ln, &lk, (__CLPK_complex*)temp.data(), &lda, (__CLPK_complex*)tau.data(),
                    (__CLPK_complex*)work.data(), &lwork, &info);
            if (info != 0)
                return NativeResult::Error("HouseholderProduct: cungqr failed with info=" +
                                           std::to_string(info));

            for (int64_t i = 0; i < outM; i++)
                for (int64_t j = 0; j < outN; j++)
                    out[i * outN + j] = temp[j * m + i];
        }
    } else {
        for (int64_t b = 0; b < batchSize; b++) {
            const float* aIn = (const float*)inputs[0].contents + b * m * n;
            const float* tauIn = (const float*)inputs[1].contents + b * k;
            float* out = (float*)outBuf.contents + b * outM * outN;

            std::vector<float> temp(m * n);
            for (int64_t i = 0; i < m; i++)
                for (int64_t j = 0; j < n; j++)
                    temp[j * m + i] = aIn[i * n + j];

            std::vector<float> tau(k);
            memcpy(tau.data(), tauIn, k * sizeof(float));

            __CLPK_integer lm = (__CLPK_integer)m;
            __CLPK_integer ln = (__CLPK_integer)n;
            __CLPK_integer lk = (__CLPK_integer)k;
            __CLPK_integer lda = (__CLPK_integer)m;
            __CLPK_integer info = 0;
            __CLPK_integer lwork = -1;

            float work_query;
            sorgqr_(&lm, &ln, &lk, temp.data(), &lda, tau.data(), &work_query, &lwork, &info);
            lwork = (__CLPK_integer)work_query;
            std::vector<float> work(lwork);
            sorgqr_(&lm, &ln, &lk, temp.data(), &lda, tau.data(), work.data(), &lwork, &info);
            if (info != 0)
                return NativeResult::Error("HouseholderProduct: sorgqr failed with info=" +
                                           std::to_string(info));

            for (int64_t i = 0; i < outM; i++)
                for (int64_t j = 0; j < outN; j++)
                    out[i * outN + j] = temp[j * m + i];
        }
    }

    return NativeResult::Buffer(outBuf);
}

static bool _cc_reg_HouseholderProduct = CustomCallRegistry::Register(
    "ProductOfElementaryHouseholderReflectors", OpHandler::Native(NativeHandle_HouseholderProduct));

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
    int64_t n = shape[shape.size() - 1];

    int64_t batchSize = 1;
    for (size_t i = 0; i < shape.size() - 2; i++)
        batchSize *= shape[i];

    bool isComplex = mlir::isa<mlir::ComplexType>(inputType.getElementType());
    if (!isComplex && !inputType.getElementType().isF32())
        return NativeResult::Error("Syevd: only float32 and complex64 supported");

    size_t matrixElemSize = isComplex ? sizeof(float) * 2 : sizeof(float);
    size_t matrixSize = (size_t)(n * n) * matrixElemSize;
    size_t eigenvalSize = (size_t)n * sizeof(float);  // eigenvalues are always real
    size_t totalMatrixSize = (size_t)batchSize * matrixSize;
    size_t totalEigenvalSize = (size_t)batchSize * eigenvalSize;

    id<MTLBuffer> outVectors = [device newBufferWithLength:totalMatrixSize
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> outValues = [device newBufferWithLength:totalEigenvalSize
                                                  options:MTLResourceStorageModeShared];

    memcpy(outVectors.contents, inputs[0].contents, totalMatrixSize);

    if (isComplex) {
        for (int64_t b = 0; b < batchSize; b++) {
            auto* a = (std::complex<float>*)outVectors.contents + b * n * n;
            float* w = (float*)outValues.contents + b * n;

            std::vector<std::complex<float>> temp(n * n);
            for (int64_t i = 0; i < n; i++)
                for (int64_t j = 0; j < n; j++)
                    temp[j * n + i] = a[i * n + j];

            char jobz = 'V';
            char uplo = 'L';
            __CLPK_integer ln = (__CLPK_integer)n;
            __CLPK_integer lda = (__CLPK_integer)n;
            __CLPK_integer info = 0;
            __CLPK_integer lwork = -1;
            __CLPK_integer lrwork = -1;
            __CLPK_integer liwork = -1;

            std::complex<float> work_query;
            float rwork_query;
            __CLPK_integer iwork_query;
            cheevd_(&jobz, &uplo, &ln, (__CLPK_complex*)temp.data(), &lda, w,
                    (__CLPK_complex*)&work_query, &lwork, &rwork_query, &lrwork, &iwork_query,
                    &liwork, &info);
            lwork = (__CLPK_integer)work_query.real();
            lrwork = (__CLPK_integer)rwork_query;
            liwork = iwork_query;
            std::vector<std::complex<float>> work(lwork);
            std::vector<float> rwork(lrwork);
            std::vector<__CLPK_integer> iwork(liwork);

            cheevd_(&jobz, &uplo, &ln, (__CLPK_complex*)temp.data(), &lda, w,
                    (__CLPK_complex*)work.data(), &lwork, rwork.data(), &lrwork, iwork.data(),
                    &liwork, &info);
            if (info != 0)
                return NativeResult::Error("Syevd: cheevd failed with info=" +
                                           std::to_string(info));

            for (int64_t i = 0; i < n; i++)
                for (int64_t j = 0; j < n; j++)
                    a[i * n + j] = temp[j * n + i];
        }
    } else {
        for (int64_t b = 0; b < batchSize; b++) {
            float* a = (float*)outVectors.contents + b * n * n;
            float* w = (float*)outValues.contents + b * n;

            std::vector<float> temp(n * n);
            for (int64_t i = 0; i < n; i++)
                for (int64_t j = 0; j < n; j++)
                    temp[j * n + i] = a[i * n + j];

            char jobz = 'V';
            char uplo = 'L';
            __CLPK_integer ln = (__CLPK_integer)n;
            __CLPK_integer lda = (__CLPK_integer)n;
            __CLPK_integer info = 0;
            __CLPK_integer lwork = -1;
            __CLPK_integer liwork = -1;

            float work_query;
            __CLPK_integer iwork_query;
            ssyevd_(&jobz, &uplo, &ln, temp.data(), &lda, w, &work_query, &lwork, &iwork_query,
                    &liwork, &info);
            lwork = (__CLPK_integer)work_query;
            liwork = iwork_query;
            std::vector<float> work(lwork);
            std::vector<__CLPK_integer> iwork(liwork);

            ssyevd_(&jobz, &uplo, &ln, temp.data(), &lda, w, work.data(), &lwork, iwork.data(),
                    &liwork, &info);
            if (info != 0)
                return NativeResult::Error("Syevd: ssyevd failed with info=" +
                                           std::to_string(info));

            for (int64_t i = 0; i < n; i++)
                for (int64_t j = 0; j < n; j++)
                    a[i * n + j] = temp[j * n + i];
        }
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
    bool isComplex = mlir::isa<mlir::ComplexType>(inputType.getElementType());
    if (!inputType.getElementType().isF32() && !isComplex)
        return NativeResult::Error("Sgesdd: only float32 and complex64 supported");

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

    // Element size: 8 bytes for complex64 (two floats), 4 bytes for float32
    size_t elemSize = isComplex ? sizeof(float) * 2 : sizeof(float);
    size_t sSize = (size_t)k * sizeof(float);  // singular values are always real
    size_t uSize = (size_t)(uRows * uCols) * elemSize;
    size_t vtSize = (size_t)(vtRows * vtCols) * elemSize;

    id<MTLBuffer> outS = [device newBufferWithLength:(size_t)batchSize * sSize
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> outU = [device newBufferWithLength:(size_t)batchSize * uSize
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> outVt = [device newBufferWithLength:(size_t)batchSize * vtSize
                                              options:MTLResourceStorageModeShared];

    for (int64_t b = 0; b < batchSize; b++) {
        float* s = (float*)outS.contents + b * k;

        // LAPACK output dimensions (column-major)
        int64_t lapack_ucols = full_matrices ? m : k;
        int64_t lapack_vtrows = full_matrices ? n : k;

        char jobz = full_matrices ? 'A' : 'S';
        __CLPK_integer lm = (__CLPK_integer)m;
        __CLPK_integer ln = (__CLPK_integer)n;
        __CLPK_integer lda = (__CLPK_integer)m;
        __CLPK_integer ldu = (__CLPK_integer)m;
        __CLPK_integer ldvt = (__CLPK_integer)lapack_vtrows;
        __CLPK_integer info = 0;
        __CLPK_integer lwork = -1;
        std::vector<__CLPK_integer> iwork(8 * k);

        if (isComplex) {
            const float* aInRaw = (const float*)inputs[0].contents + b * m * n * 2;
            float* u = (float*)outU.contents + b * uRows * uCols * 2;
            float* vt = (float*)outVt.contents + b * vtRows * vtCols * 2;

            // Transpose input to column-major (each element is 2 floats)
            std::vector<float> aCm(m * n * 2);
            for (int64_t i = 0; i < m; i++)
                for (int64_t j = 0; j < n; j++) {
                    aCm[(j * m + i) * 2] = aInRaw[(i * n + j) * 2];
                    aCm[(j * m + i) * 2 + 1] = aInRaw[(i * n + j) * 2 + 1];
                }

            std::vector<float> uCm(m * lapack_ucols * 2);
            std::vector<float> vtCm(lapack_vtrows * n * 2);

            // Query optimal workspace
            __CLPK_complex work_query;
            float rwork_query;
            __CLPK_integer lrwork = -1;
            cgesdd_(&jobz, &lm, &ln, (__CLPK_complex*)aCm.data(), &lda, s,
                    (__CLPK_complex*)uCm.data(), &ldu, (__CLPK_complex*)vtCm.data(), &ldvt,
                    &work_query, &lwork, &rwork_query, iwork.data(), &info);
            lwork = (__CLPK_integer)work_query.r;
            std::vector<__CLPK_complex> work(lwork);

            // rwork size for cgesdd_
            int64_t mn_min = std::min(m, n);
            int64_t mn_max = std::max(m, n);
            __CLPK_integer lrwork_val;
            if (jobz == 'N') {
                lrwork_val = 7 * mn_min;
            } else {
                lrwork_val = std::max(5 * mn_min * mn_min + 5 * mn_min,
                                      2 * mn_max * mn_min + 2 * mn_min * mn_min + mn_min);
            }
            std::vector<float> rwork(lrwork_val);

            // Compute complex SVD
            cgesdd_(&jobz, &lm, &ln, (__CLPK_complex*)aCm.data(), &lda, s,
                    (__CLPK_complex*)uCm.data(), &ldu, (__CLPK_complex*)vtCm.data(), &ldvt,
                    work.data(), &lwork, rwork.data(), iwork.data(), &info);
            if (info != 0)
                return NativeResult::Error("Sgesdd: cgesdd failed with info=" +
                                           std::to_string(info));

            // Transpose U from column-major to row-major (complex)
            for (int64_t i = 0; i < uRows; i++)
                for (int64_t j = 0; j < uCols; j++) {
                    u[(i * uCols + j) * 2] = uCm[(j * m + i) * 2];
                    u[(i * uCols + j) * 2 + 1] = uCm[(j * m + i) * 2 + 1];
                }

            // Transpose Vt from column-major to row-major (complex)
            for (int64_t i = 0; i < vtRows; i++)
                for (int64_t j = 0; j < vtCols; j++) {
                    vt[(i * vtCols + j) * 2] = vtCm[(j * lapack_vtrows + i) * 2];
                    vt[(i * vtCols + j) * 2 + 1] = vtCm[(j * lapack_vtrows + i) * 2 + 1];
                }
        } else {
            const float* aIn = (const float*)inputs[0].contents + b * m * n;
            float* u = (float*)outU.contents + b * uRows * uCols;
            float* vt = (float*)outVt.contents + b * vtRows * vtCols;

            // Transpose input to column-major for LAPACK
            std::vector<float> aCm(m * n);
            for (int64_t i = 0; i < m; i++)
                for (int64_t j = 0; j < n; j++)
                    aCm[j * m + i] = aIn[i * n + j];

            std::vector<float> uCm(m * lapack_ucols);
            std::vector<float> vtCm(lapack_vtrows * n);

            // Query optimal workspace
            float work_query;
            __CLPK_integer iwork_buf[8 * std::min(m, n)];
            sgesdd_(&jobz, &lm, &ln, aCm.data(), &lda, s, uCm.data(), &ldu, vtCm.data(), &ldvt,
                    &work_query, &lwork, iwork_buf, &info);
            lwork = (__CLPK_integer)work_query;
            std::vector<float> work(lwork);

            // Compute SVD
            sgesdd_(&jobz, &lm, &ln, aCm.data(), &lda, s, uCm.data(), &ldu, vtCm.data(), &ldvt,
                    work.data(), &lwork, iwork.data(), &info);
            if (info != 0)
                return NativeResult::Error("Sgesdd: sgesdd failed with info=" +
                                           std::to_string(info));

            // Transpose U from column-major to row-major
            for (int64_t i = 0; i < uRows; i++)
                for (int64_t j = 0; j < uCols; j++)
                    u[i * uCols + j] = uCm[j * m + i];

            // Transpose Vt from column-major to row-major
            for (int64_t i = 0; i < vtRows; i++)
                for (int64_t j = 0; j < vtCols; j++)
                    vt[i * vtCols + j] = vtCm[j * lapack_vtrows + i];
        }
    }

    return NativeResult::Buffers({outS, outU, outVt});
}

static bool _cc_reg_Sgesdd =
    CustomCallRegistry::Register("mps_sgesdd", OpHandler::Native(NativeHandle_Sgesdd));

// ---------------------------------------------------------------------------
// General eigendecomposition via Accelerate LAPACK (sgeev_)
// ---------------------------------------------------------------------------
// custom_call @"mps_sgeev": computes eigenvalues and optionally eigenvectors
// of a general (non-symmetric) real matrix.
// Input: tensor<...xNxNxf32> (square matrix)
// Output 0: tensor<...xNxcomplex<f32>> (eigenvalues)
// Output 1: tensor<...xNxNxcomplex<f32>> (left eigenvectors, may be unused)
// Output 2: tensor<...xNxNxcomplex<f32>> (right eigenvectors, may be unused)
// Output 3: tensor<i32> (info)

static NativeResult NativeHandle_Sgeev(id<MTLDevice> device, id<MTLCommandBuffer> cmdBuf,
                                       mlir::Operation* op,
                                       const std::vector<id<MTLBuffer>>& inputs) {
    if (inputs.empty())
        return NativeResult::Error("Sgeev: missing input");

    auto customCallOp = mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(op);
    if (!customCallOp || op->getNumResults() != 4)
        return NativeResult::Error("Sgeev: expected CustomCallOp with 4 results");

    auto inputType = mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    auto shape = inputType.getShape();
    if (shape.size() < 2)
        return NativeResult::Error("Sgeev: expected at least rank 2");
    bool isComplex = mlir::isa<mlir::ComplexType>(inputType.getElementType());
    if (!inputType.getElementType().isF32() && !isComplex)
        return NativeResult::Error("Sgeev: only float32 and complex64 supported");

    int64_t n = shape[shape.size() - 1];
    int64_t batchSize = 1;
    for (size_t i = 0; i < shape.size() - 2; i++)
        batchSize *= shape[i];

    // Parse backend_config for compute_left, compute_right
    // Format: "compute_left,compute_right" (0 or 1)
    bool computeLeft = false;
    bool computeRight = true;
    auto configOpt = customCallOp.getBackendConfig();
    if (configOpt.has_value()) {
        if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(*configOpt)) {
            auto config = strAttr.getValue().str();
            if (config.size() >= 3) {
                computeLeft = (config[0] == '1');
                computeRight = (config[2] == '1');
            }
        }
    }

    // Complex eigenvalue buffer: N complex<float> = N*2 floats
    size_t complexEigenvalSize = (size_t)(n * 2) * sizeof(float);
    size_t totalComplexEigenvalSize = (size_t)batchSize * complexEigenvalSize;
    // Complex eigenvector buffers: N*N complex<float> = N*N*2 floats
    size_t complexMatrixSize = (size_t)(n * n * 2) * sizeof(float);
    size_t totalComplexMatrixSize = (size_t)batchSize * complexMatrixSize;

    id<MTLBuffer> outW = [device newBufferWithLength:totalComplexEigenvalSize
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> outVl = [device newBufferWithLength:totalComplexMatrixSize
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> outVr = [device newBufferWithLength:totalComplexMatrixSize
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> outInfo = [device newBufferWithLength:sizeof(int32_t)
                                                options:MTLResourceStorageModeShared];

    int32_t globalInfo = 0;

    for (int64_t b = 0; b < batchSize; b++) {
        char jobvl = computeLeft ? 'V' : 'N';
        char jobvr = computeRight ? 'V' : 'N';
        __CLPK_integer ln = (__CLPK_integer)n;
        __CLPK_integer lda = (__CLPK_integer)n;
        __CLPK_integer ldvl = (__CLPK_integer)n;
        __CLPK_integer ldvr = (__CLPK_integer)n;
        __CLPK_integer info = 0;
        __CLPK_integer lwork = -1;

        if (isComplex) {
            const float* inputRaw = (const float*)inputs[0].contents + b * n * n * 2;

            // Transpose to column-major (complex elements = 2 floats each)
            std::vector<float> aCm(n * n * 2);
            for (int64_t i = 0; i < n; i++)
                for (int64_t j = 0; j < n; j++) {
                    aCm[(j * n + i) * 2] = inputRaw[(i * n + j) * 2];
                    aCm[(j * n + i) * 2 + 1] = inputRaw[(i * n + j) * 2 + 1];
                }

            // cgeev_ outputs complex eigenvalues directly
            std::vector<__CLPK_complex> w(n);
            std::vector<__CLPK_complex> vl(computeLeft ? n * n : 1);
            std::vector<__CLPK_complex> vr(computeRight ? n * n : 1);
            std::vector<float> rwork(2 * n);

            // Query workspace
            __CLPK_complex work_query;
            cgeev_(&jobvl, &jobvr, &ln, (__CLPK_complex*)aCm.data(), &lda, w.data(), vl.data(),
                   &ldvl, vr.data(), &ldvr, &work_query, &lwork, rwork.data(), &info);
            lwork = (__CLPK_integer)work_query.r;
            std::vector<__CLPK_complex> work(lwork);

            // Compute
            cgeev_(&jobvl, &jobvr, &ln, (__CLPK_complex*)aCm.data(), &lda, w.data(), vl.data(),
                   &ldvl, vr.data(), &ldvr, work.data(), &lwork, rwork.data(), &info);

            if (info != 0) {
                globalInfo = (int32_t)info;
            }

            // Copy eigenvalues (already complex)
            float* wOut = (float*)outW.contents + b * n * 2;
            for (int64_t j = 0; j < n; j++) {
                wOut[j * 2] = w[j].r;
                wOut[j * 2 + 1] = w[j].i;
            }

            // Transpose eigenvectors from column-major to row-major
            if (computeLeft) {
                float* vlOut = (float*)outVl.contents + b * n * n * 2;
                for (int64_t i = 0; i < n; i++)
                    for (int64_t j = 0; j < n; j++) {
                        vlOut[(i * n + j) * 2] = vl[j * n + i].r;
                        vlOut[(i * n + j) * 2 + 1] = vl[j * n + i].i;
                    }
            }
            if (computeRight) {
                float* vrOut = (float*)outVr.contents + b * n * n * 2;
                for (int64_t i = 0; i < n; i++)
                    for (int64_t j = 0; j < n; j++) {
                        vrOut[(i * n + j) * 2] = vr[j * n + i].r;
                        vrOut[(i * n + j) * 2 + 1] = vr[j * n + i].i;
                    }
            }
        } else {
            const float* inputMatrix = (const float*)inputs[0].contents + b * n * n;

            // Temporary real/imag eigenvalue arrays for LAPACK
            std::vector<float> wr(n), wi(n);

            // Transpose to column-major for LAPACK
            std::vector<float> a(n * n);
            for (int64_t i = 0; i < n; i++)
                for (int64_t j = 0; j < n; j++)
                    a[j * n + i] = inputMatrix[i * n + j];

            // LAPACK sgeev returns real eigenvectors in a packed format
            std::vector<float> vl_real(computeLeft ? n * n : 1);
            std::vector<float> vr_real(computeRight ? n * n : 1);

            // Query workspace
            float work_query;
            sgeev_(&jobvl, &jobvr, &ln, a.data(), &lda, wr.data(), wi.data(), vl_real.data(), &ldvl,
                   vr_real.data(), &ldvr, &work_query, &lwork, &info);
            lwork = (__CLPK_integer)work_query;
            std::vector<float> work(lwork);

            // Compute eigendecomposition
            sgeev_(&jobvl, &jobvr, &ln, a.data(), &lda, wr.data(), wi.data(), vl_real.data(), &ldvl,
                   vr_real.data(), &ldvr, work.data(), &lwork, &info);

            if (info != 0) {
                globalInfo = (int32_t)info;
            }

            // Pack eigenvalues as complex: [re0, im0, re1, im1, ...]
            float* wOut = (float*)outW.contents + b * n * 2;
            for (int64_t j = 0; j < n; j++) {
                wOut[j * 2] = wr[j];
                wOut[j * 2 + 1] = wi[j];
            }

            // Convert LAPACK's packed real eigenvector format to complex.
            auto unpackEigenvectors = [&](const std::vector<float>& v_real, float* outComplex) {
                int64_t j = 0;
                while (j < n) {
                    if (wi[j] == 0.0f) {
                        for (int64_t i = 0; i < n; i++) {
                            outComplex[(i * n + j) * 2] = v_real[j * n + i];
                            outComplex[(i * n + j) * 2 + 1] = 0.0f;
                        }
                        j++;
                    } else {
                        for (int64_t i = 0; i < n; i++) {
                            float re = v_real[j * n + i];
                            float im = v_real[(j + 1) * n + i];
                            outComplex[(i * n + j) * 2] = re;
                            outComplex[(i * n + j) * 2 + 1] = im;
                            outComplex[(i * n + (j + 1)) * 2] = re;
                            outComplex[(i * n + (j + 1)) * 2 + 1] = -im;
                        }
                        j += 2;
                    }
                }
            };

            if (computeLeft) {
                float* vlOut = (float*)outVl.contents + b * n * n * 2;
                unpackEigenvectors(vl_real, vlOut);
            }
            if (computeRight) {
                float* vrOut = (float*)outVr.contents + b * n * n * 2;
                unpackEigenvectors(vr_real, vrOut);
            }
        }
    }

    *(int32_t*)outInfo.contents = globalInfo;
    return NativeResult::Buffers({outW, outVl, outVr, outInfo});
}

static bool _cc_reg_Sgeev =
    CustomCallRegistry::Register("mps_sgeev", OpHandler::Native(NativeHandle_Sgeev));

// ---------------------------------------------------------------------------
// Schur decomposition via Accelerate LAPACK (sgees_ / cgees_)
// ---------------------------------------------------------------------------
// custom_call @"mps_sgees": computes Schur decomposition A = Q*T*Q^H
// Input: tensor<...xNxNxf32> or tensor<...xNxNxcomplex<f32>> (square matrix)
// Output 0: tensor<...xNxNx(same)> (T - Schur form)
// Output 1: tensor<...xNxNx(same)> (Q - Schur vectors, if compute_schur_vectors)

static NativeResult NativeHandle_Sgees(id<MTLDevice> device, id<MTLCommandBuffer> cmdBuf,
                                       mlir::Operation* op,
                                       const std::vector<id<MTLBuffer>>& inputs) {
    if (inputs.empty())
        return NativeResult::Error("Sgees: missing input");

    auto inputType = mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    auto shape = inputType.getShape();
    if (shape.size() < 2)
        return NativeResult::Error("Sgees: expected at least rank 2");

    bool isComplex = mlir::isa<mlir::ComplexType>(inputType.getElementType());
    if (!inputType.getElementType().isF32() && !isComplex)
        return NativeResult::Error("Sgees: only float32 and complex64 supported");

    int64_t n = shape[shape.size() - 1];
    int64_t batchSize = 1;
    for (size_t i = 0; i < shape.size() - 2; i++)
        batchSize *= shape[i];

    // Parse backend_config for compute_schur_vectors
    bool computeVectors = true;
    auto customCallOp = mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(op);
    if (customCallOp) {
        auto configOpt = customCallOp.getBackendConfig();
        if (configOpt.has_value()) {
            if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(*configOpt)) {
                auto config = strAttr.getValue().str();
                computeVectors = (config[0] == '1');
            }
        }
    }

    size_t elemSize = isComplex ? sizeof(float) * 2 : sizeof(float);
    size_t matSize = (size_t)(n * n) * elemSize;

    // Output: T (Schur form)
    id<MTLBuffer> outT = [device newBufferWithLength:(size_t)batchSize * matSize
                                             options:MTLResourceStorageModeShared];
    // Output: Q (Schur vectors) - always allocated even if not computed
    id<MTLBuffer> outQ = [device newBufferWithLength:(size_t)batchSize * matSize
                                             options:MTLResourceStorageModeShared];

    for (int64_t b = 0; b < batchSize; b++) {
        char jobvs = computeVectors ? 'V' : 'N';
        char sort = 'N';  // No sorting
        __CLPK_integer ln = (__CLPK_integer)n;
        __CLPK_integer sdim = 0;
        __CLPK_integer info = 0;
        __CLPK_integer lwork = -1;
        __CLPK_integer ldvs = (__CLPK_integer)n;

        if (isComplex) {
            const float* inputRaw = (const float*)inputs[0].contents + b * n * n * 2;

            // Transpose to column-major
            std::vector<float> aCm(n * n * 2);
            for (int64_t i = 0; i < n; i++)
                for (int64_t j = 0; j < n; j++) {
                    aCm[(j * n + i) * 2] = inputRaw[(i * n + j) * 2];
                    aCm[(j * n + i) * 2 + 1] = inputRaw[(i * n + j) * 2 + 1];
                }

            std::vector<__CLPK_complex> w(n);
            std::vector<__CLPK_complex> vs(computeVectors ? n * n : 1);
            std::vector<float> rwork(n);

            // Query workspace
            __CLPK_complex work_query;
            cgees_(&jobvs, &sort, nullptr, &ln, (__CLPK_complex*)aCm.data(), &ln, &sdim, w.data(),
                   vs.data(), &ldvs, &work_query, &lwork, rwork.data(), nullptr, &info);
            lwork = (__CLPK_integer)work_query.r;
            std::vector<__CLPK_complex> work(lwork);

            // Compute Schur decomposition
            cgees_(&jobvs, &sort, nullptr, &ln, (__CLPK_complex*)aCm.data(), &ln, &sdim, w.data(),
                   vs.data(), &ldvs, work.data(), &lwork, rwork.data(), nullptr, &info);

            if (info != 0)
                return NativeResult::Error("Sgees: cgees failed with info=" + std::to_string(info));

            // Transpose T (in aCm after cgees_) from column-major to row-major
            float* tOut = (float*)outT.contents + b * n * n * 2;
            for (int64_t i = 0; i < n; i++)
                for (int64_t j = 0; j < n; j++) {
                    tOut[(i * n + j) * 2] = aCm[(j * n + i) * 2];
                    tOut[(i * n + j) * 2 + 1] = aCm[(j * n + i) * 2 + 1];
                }

            // Transpose Q (vs) from column-major to row-major
            if (computeVectors) {
                float* qOut = (float*)outQ.contents + b * n * n * 2;
                for (int64_t i = 0; i < n; i++)
                    for (int64_t j = 0; j < n; j++) {
                        qOut[(i * n + j) * 2] = vs[j * n + i].r;
                        qOut[(i * n + j) * 2 + 1] = vs[j * n + i].i;
                    }
            }
        } else {
            const float* inputRaw = (const float*)inputs[0].contents + b * n * n;

            // Transpose to column-major
            std::vector<float> aCm(n * n);
            for (int64_t i = 0; i < n; i++)
                for (int64_t j = 0; j < n; j++)
                    aCm[j * n + i] = inputRaw[i * n + j];

            std::vector<float> wr(n), wi(n);
            std::vector<float> vs(computeVectors ? n * n : 1);

            // Query workspace
            float work_query;
            sgees_(&jobvs, &sort, nullptr, &ln, aCm.data(), &ln, &sdim, wr.data(), wi.data(),
                   vs.data(), &ldvs, &work_query, &lwork, nullptr, &info);
            lwork = (__CLPK_integer)work_query;
            std::vector<float> work(lwork);

            // Compute Schur decomposition
            sgees_(&jobvs, &sort, nullptr, &ln, aCm.data(), &ln, &sdim, wr.data(), wi.data(),
                   vs.data(), &ldvs, work.data(), &lwork, nullptr, &info);

            if (info != 0)
                return NativeResult::Error("Sgees: sgees failed with info=" + std::to_string(info));

            // Transpose T from column-major to row-major
            float* tOut = (float*)outT.contents + b * n * n;
            for (int64_t i = 0; i < n; i++)
                for (int64_t j = 0; j < n; j++)
                    tOut[i * n + j] = aCm[j * n + i];

            // Transpose Q from column-major to row-major
            if (computeVectors) {
                float* qOut = (float*)outQ.contents + b * n * n;
                for (int64_t i = 0; i < n; i++)
                    for (int64_t j = 0; j < n; j++)
                        qOut[i * n + j] = vs[j * n + i];
            }
        }
    }

    if (computeVectors)
        return NativeResult::Buffers({outT, outQ});
    else
        return NativeResult::Buffers({outT});
}

static bool _cc_reg_Sgees =
    CustomCallRegistry::Register("mps_sgees", OpHandler::Native(NativeHandle_Sgees));

}  // namespace jax_mps
