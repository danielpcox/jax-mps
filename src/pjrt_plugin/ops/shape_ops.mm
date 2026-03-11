// Shape operations: broadcast, reshape, convert, slice, concatenate,
// custom_call, etc.

#import "pjrt_plugin/ops/gather_scatter_utils.h"
#import "pjrt_plugin/ops/registry.h"

namespace jax_mps {

static ProcessResult HandleBroadcast(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("broadcast: missing input tensor");
    NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
    MPSGraphTensor* result = [ctx.graph broadcastTensor:input toShape:outputShape name:nil];
    return Result(ctx, result, "broadcast");
}
REGISTER_MPS_OP("stablehlo.broadcast", HandleBroadcast);

// broadcast_in_dim needs special handling for dimension mapping
static ProcessResult HandleBroadcastInDim(HandlerContext& ctx) {
    auto broadcastOp = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(ctx.op);
    if (!broadcastOp) {
        return ProcessResult::Error("broadcast_in_dim: expected BroadcastInDimOp");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input) {
        return ProcessResult::Error("broadcast_in_dim: input tensor not found");
    }

    NSArray<NSNumber*>* inputShape = input.shape;
    NSUInteger inputRank = inputShape.count;

    NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
    NSUInteger outputRank = outputShape.count;

    auto broadcastDims = broadcastOp.getBroadcastDimensions();

    MPSGraphTensor* result = nil;

    // If broadcast_dims is empty or ranks match, just broadcast directly
    if (broadcastDims.empty() || inputRank == outputRank) {
        result = [ctx.graph broadcastTensor:input toShape:outputShape name:nil];
    } else {
        // Build intermediate shape: start with all 1s, then fill in from broadcast_dims
        NSMutableArray<NSNumber*>* intermediateShape =
            [NSMutableArray arrayWithCapacity:outputRank];
        for (NSUInteger i = 0; i < outputRank; i++) {
            [intermediateShape addObject:@1];
        }

        // Map input dimensions to output dimensions according to broadcast_dims
        for (size_t i = 0; i < broadcastDims.size() && i < inputRank; i++) {
            int64_t outDim = broadcastDims[i];
            if (outDim >= 0 && (NSUInteger)outDim < outputRank) {
                intermediateShape[outDim] = inputShape[i];
            }
        }

        // Reshape input to intermediate shape (same rank as output)
        MPSGraphTensor* reshaped = [ctx.graph reshapeTensor:input
                                                  withShape:intermediateShape
                                                       name:nil];

        // Now broadcast to final output shape
        result = [ctx.graph broadcastTensor:reshaped toShape:outputShape name:nil];
    }

    return Result(ctx, result, "broadcast_in_dim");
}
REGISTER_MPS_OP("stablehlo.broadcast_in_dim", HandleBroadcastInDim);

static ProcessResult HandleReshape(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("reshape: missing input tensor");
    NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
    MPSGraphTensor* result = [ctx.graph reshapeTensor:input withShape:outputShape name:nil];
    return Result(ctx, result, "reshape");
}
REGISTER_MPS_OP("stablehlo.reshape", HandleReshape);

static ProcessResult HandleTranspose(HandlerContext& ctx) {
    auto transposeOp = mlir::dyn_cast<mlir::stablehlo::TransposeOp>(ctx.op);
    if (!transposeOp) {
        return ProcessResult::Error("transpose: expected TransposeOp");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("transpose: missing input tensor");

    auto permutation = transposeOp.getPermutation();
    NSMutableArray<NSNumber*>* perm = [NSMutableArray array];
    for (int64_t d : permutation) {
        [perm addObject:@(d)];
    }

    MPSGraphTensor* result = [ctx.graph transposeTensor:input permutation:perm name:nil];
    return Result(ctx, result, "transpose");
}
REGISTER_MPS_OP("stablehlo.transpose", HandleTranspose);

static ProcessResult HandleConvert(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("convert: missing input tensor");

    MPSDataType dtype = GetResultMpsType(ctx.op);
    if (dtype == MPSDataTypeInvalid) {
        return ProcessResult::Error("convert: invalid dtype for convert operation");
    }
    MPSGraphTensor* result = [ctx.graph castTensor:input toType:dtype name:nil];
    return Result(ctx, result, "convert");
}
REGISTER_MPS_OP("stablehlo.convert", HandleConvert);

// Slice - extract a portion of a tensor (static indices)
static ProcessResult HandleSlice(HandlerContext& ctx) {
    auto sliceOp = mlir::dyn_cast<mlir::stablehlo::SliceOp>(ctx.op);
    if (!sliceOp) {
        return ProcessResult::Error("slice: expected SliceOp");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("slice: missing input tensor");

    NSMutableArray<NSNumber*>* starts = [NSMutableArray array];
    NSMutableArray<NSNumber*>* ends = [NSMutableArray array];
    NSMutableArray<NSNumber*>* strides = [NSMutableArray array];

    for (int64_t s : sliceOp.getStartIndices()) {
        [starts addObject:@(s)];
    }
    for (int64_t l : sliceOp.getLimitIndices()) {
        [ends addObject:@(l)];
    }
    for (int64_t s : sliceOp.getStrides()) {
        [strides addObject:@(s)];
    }

    MPSGraphTensor* result = [ctx.graph sliceTensor:input
                                             starts:starts
                                               ends:ends
                                            strides:strides
                                               name:nil];
    return Result(ctx, result, "slice");
}
REGISTER_MPS_OP("stablehlo.slice", HandleSlice);

// Dynamic slice - extract a portion using runtime indices
static ProcessResult HandleDynamicSlice(HandlerContext& ctx) {
    auto dynSliceOp = mlir::dyn_cast<mlir::stablehlo::DynamicSliceOp>(ctx.op);
    if (!dynSliceOp) {
        return ProcessResult::Error("dynamic_slice: expected DynamicSliceOp");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("dynamic_slice: missing input tensor");

    auto sliceSizes = dynSliceOp.getSliceSizes();
    NSUInteger rank = sliceSizes.size();

    // Build the output shape from slice sizes
    NSMutableArray<NSNumber*>* outputShape = [NSMutableArray array];
    for (int64_t s : sliceSizes) {
        [outputShape addObject:@(s)];
    }

    // Get input shape for clamping
    NSArray<NSNumber*>* inputShape = input.shape;

    // Get start indices as tensors (operands 1 through N)
    // Per StableHLO spec, start indices must be clamped:
    //   start[i] = clamp(start[i], 0, dim_size[i] - slice_size[i])
    NSMutableArray<MPSGraphTensor*>* indexTensors = [NSMutableArray array];
    for (NSUInteger dim = 0; dim < rank; dim++) {
        // Get the start index tensor for this dimension (scalar tensor)
        MPSGraphTensor* startIdx = GetInputTensor(ctx, dim + 1);
        if (!startIdx) {
            return ProcessResult::Error("dynamic_slice: missing start index for dimension");
        }

        // Clamp start index: max(0, min(start, dim_size - slice_size))
        int64_t dimSize = inputShape[dim].longLongValue;
        int64_t sliceSize = sliceSizes[dim];
        int64_t maxStart = std::max(dimSize - sliceSize, (int64_t)0);

        MPSGraphTensor* zero = [ctx.graph constantWithScalar:0 dataType:startIdx.dataType];
        MPSGraphTensor* maxVal = [ctx.graph constantWithScalar:static_cast<double>(maxStart) dataType:startIdx.dataType];
        startIdx = [ctx.graph clampWithTensor:startIdx
                               minValueTensor:zero
                               maxValueTensor:maxVal
                                         name:nil];

        // Create coordinate tensor for this dimension (0, 1, 2, ..., slice_size-1)
        MPSGraphTensor* coords = [ctx.graph coordinateAlongAxis:(NSInteger)dim
                                                      withShape:outputShape
                                                           name:nil];

        // Cast coordinates to match start index type for addition
        coords = [ctx.graph castTensor:coords toType:startIdx.dataType name:nil];

        // Add start index to coordinates (broadcasts the scalar start index)
        MPSGraphTensor* adjustedCoords = [ctx.graph additionWithPrimaryTensor:coords
                                                              secondaryTensor:startIdx
                                                                         name:nil];

        [indexTensors addObject:adjustedCoords];
    }

    // Stack the coordinate tensors along a new last axis to form indices tensor
    // Shape: [slice_size_0, slice_size_1, ..., rank]
    MPSGraphTensor* indices = [ctx.graph stackTensors:indexTensors axis:(NSInteger)rank name:nil];

    // Use SafeGatherND to handle integer precision issues
    MPSGraphTensor* result = SafeGatherND(ctx.graph, input, indices, 0);

    return Result(ctx, result, "dynamic_slice");
}
REGISTER_MPS_OP("stablehlo.dynamic_slice", HandleDynamicSlice);

// Bitcast convert - reinterpret bits as a different type
static ProcessResult HandleBitcastConvert(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("bitcast_convert: missing input tensor");

    MPSDataType dtype = GetResultMpsType(ctx.op);
    if (dtype == MPSDataTypeInvalid) {
        return ProcessResult::Error("bitcast_convert: invalid dtype");
    }

    // MPS reinterpretCastTensor doesn't support rank-0 (scalar) tensors.
    // Work around by reshaping to rank-1, casting, then reshaping back.
    NSArray<NSNumber*>* inputShape = input.shape;
    bool isScalar = (inputShape.count == 0);

    if (isScalar) {
        // Reshape scalar to [1]
        input = [ctx.graph reshapeTensor:input withShape:@[@1] name:nil];
    }

    // Use reinterpretCast which preserves bit patterns
    MPSGraphTensor* result = [ctx.graph reinterpretCastTensor:input toType:dtype name:nil];

    if (isScalar) {
        // Reshape back to scalar
        result = [ctx.graph reshapeTensor:result withShape:@[] name:nil];
    }

    return Result(ctx, result, "bitcast_convert");
}
REGISTER_MPS_OP("stablehlo.bitcast_convert", HandleBitcastConvert);

// Concatenate - joins tensors along a dimension
static ProcessResult HandleConcatenate(HandlerContext& ctx) {
    auto concatOp = mlir::dyn_cast<mlir::stablehlo::ConcatenateOp>(ctx.op);
    if (!concatOp) {
        return ProcessResult::Error("concatenate: expected ConcatenateOp");
    }

    // Gather all input tensors
    NSMutableArray<MPSGraphTensor*>* input_tensors = [NSMutableArray array];
    for (mlir::Value operand : ctx.op->getOperands()) {
        MPSGraphTensor* tensor = GetTensor(ctx.values, operand);
        if (tensor) {
            [input_tensors addObject:tensor];
        }
    }

    if (input_tensors.count == 0) {
        return ProcessResult::Error("concatenate: no valid inputs");
    }

    // Get the concatenate dimension from the op
    NSInteger dimension = static_cast<NSInteger>(concatOp.getDimension());

    MPSGraphTensor* result = [ctx.graph concatTensors:input_tensors dimension:dimension name:nil];
    return Result(ctx, result, "concatenate");
}
REGISTER_MPS_OP("stablehlo.concatenate", HandleConcatenate);

// Sharding is a marker used by JAX for partitioning - just pass through the input
static ProcessResult HandleSharding(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("sharding: missing input tensor");
    SetOutputTensor(ctx.values, ctx.op, input);
    return ProcessResult{};
}
REGISTER_CUSTOM_CALL("Sharding", HandleSharding, sharding);

// Helper: apply interior padding to a tensor along each dimension.
// For each dimension with interior padding p > 0:
//   Reshape to add axis, concat zeros, flatten, slice off trailing elements.
// All ops used (reshape, concat, slice) have working MPS backward passes.
static MPSGraphTensor* ApplyInteriorPadding(MPSGraph* graph, MPSGraphTensor* tensor,
                                            llvm::ArrayRef<int64_t> interiorPadding,
                                            NSUInteger rank) {
    MPSGraphTensor* current = tensor;
    for (NSUInteger dim = 0; dim < rank; dim++) {
        int64_t interior = interiorPadding[dim];
        if (interior <= 0)
            continue;

        NSArray<NSNumber*>* curShape = current.shape;
        NSUInteger curRank = curShape.count;
        int64_t dimSize = [curShape[dim] longLongValue];

        // [..., N, ...] -> [..., N, 1, ...]
        NSMutableArray<NSNumber*>* expandedShape = [NSMutableArray arrayWithCapacity:curRank + 1];
        for (NSUInteger d = 0; d < curRank; d++) {
            [expandedShape addObject:curShape[d]];
            if (d == dim)
                [expandedShape addObject:@1];
        }
        MPSGraphTensor* expanded = [graph reshapeTensor:current withShape:expandedShape name:nil];

        // Zeros: [..., N, p, ...]
        NSMutableArray<NSNumber*>* zerosShape = [NSMutableArray arrayWithArray:expandedShape];
        zerosShape[dim + 1] = @(interior);
        MPSGraphTensor* zeros = [graph constantWithScalar:0.0 dataType:current.dataType];
        zeros = [graph broadcastTensor:zeros toShape:zerosShape name:nil];

        // Concat -> [..., N, 1+p, ...], flatten -> [..., N*(1+p), ...]
        MPSGraphTensor* interleaved = [graph concatTensors:@[expanded, zeros]
                                                 dimension:(NSInteger)(dim + 1)
                                                      name:nil];
        NSMutableArray<NSNumber*>* flatShape = [NSMutableArray arrayWithCapacity:curRank];
        for (NSUInteger d = 0; d < curRank; d++) {
            [flatShape addObject:(d == dim) ? @(dimSize * (1 + interior)) : curShape[d]];
        }
        MPSGraphTensor* flat = [graph reshapeTensor:interleaved withShape:flatShape name:nil];

        // Slice to [..., N + (N-1)*p, ...]
        NSMutableArray<NSNumber*>* starts = [NSMutableArray arrayWithCapacity:curRank];
        NSMutableArray<NSNumber*>* ends = [NSMutableArray arrayWithCapacity:curRank];
        NSMutableArray<NSNumber*>* strides = [NSMutableArray arrayWithCapacity:curRank];
        for (NSUInteger d = 0; d < curRank; d++) {
            [starts addObject:@0];
            [ends addObject:(d == dim) ? @(dimSize + (dimSize - 1) * interior) : flatShape[d]];
            [strides addObject:@1];
        }
        current = [graph sliceTensor:flat starts:starts ends:ends strides:strides name:nil];
    }
    return current;
}

// Pad - add padding around tensor
// Avoids sliceUpdateDataTensor which crashes MPS's backward pass (issue #59).
// Uses padTensor for non-negative edge padding and slice for negative (crop).
static ProcessResult HandlePad(HandlerContext& ctx) {
    auto padOp = mlir::dyn_cast<mlir::stablehlo::PadOp>(ctx.op);
    if (!padOp)
        return ProcessResult::Error("pad: expected PadOp");

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    MPSGraphTensor* paddingValue = GetInputTensor(ctx, 1);
    if (!input || !paddingValue)
        return ProcessResult::Error("pad: missing input tensor");

    auto edgePaddingLow = padOp.getEdgePaddingLow();
    auto edgePaddingHigh = padOp.getEdgePaddingHigh();
    auto interiorPadding = padOp.getInteriorPadding();
    NSUInteger rank = edgePaddingLow.size();

    // Step 1: Apply interior padding (if any)
    MPSGraphTensor* current = ApplyInteriorPadding(ctx.graph, input, interiorPadding, rank);

    // Step 2: Handle negative edge padding (crop) via slice, then positive via padTensor.
    // Separate negative and non-negative parts.
    NSMutableArray<NSNumber*>* cropStarts = [NSMutableArray arrayWithCapacity:rank];
    NSMutableArray<NSNumber*>* cropEnds = [NSMutableArray arrayWithCapacity:rank];
    NSMutableArray<NSNumber*>* cropStrides = [NSMutableArray arrayWithCapacity:rank];
    NSMutableArray<NSNumber*>* leftPad = [NSMutableArray arrayWithCapacity:rank];
    NSMutableArray<NSNumber*>* rightPad = [NSMutableArray arrayWithCapacity:rank];
    bool needsCrop = false;
    bool needsPad = false;

    NSArray<NSNumber*>* curShape = current.shape;
    for (NSUInteger i = 0; i < rank; i++) {
        int64_t low = edgePaddingLow[i];
        int64_t high = edgePaddingHigh[i];
        int64_t dimSize = [curShape[i] longLongValue];

        // Crop: negative padding removes elements
        int64_t cropLow = (low < 0) ? -low : 0;
        int64_t cropHigh = (high < 0) ? -high : 0;
        [cropStarts addObject:@(cropLow)];
        [cropEnds addObject:@(dimSize - cropHigh)];
        [cropStrides addObject:@1];
        if (cropLow > 0 || cropHigh > 0)
            needsCrop = true;

        // Pad: non-negative padding adds elements
        int64_t padLow = (low > 0) ? low : 0;
        int64_t padHigh = (high > 0) ? high : 0;
        [leftPad addObject:@(padLow)];
        [rightPad addObject:@(padHigh)];
        if (padLow > 0 || padHigh > 0)
            needsPad = true;
    }

    if (needsCrop) {
        current = [ctx.graph sliceTensor:current
                                  starts:cropStarts
                                    ends:cropEnds
                                 strides:cropStrides
                                    name:nil];
    }

    if (needsPad) {
        current = [ctx.graph padTensor:current
                       withPaddingMode:MPSGraphPaddingModeConstant
                           leftPadding:leftPad
                          rightPadding:rightPad
                         constantValue:0.0
                                  name:nil];
    }

    // Step 3: Handle non-zero padding value.
    // padTensor pads with 0. To support arbitrary padding values, use a mask:
    // result = current + (1 - mask) * paddingValue
    // where mask is 1 at original data positions, 0 at padded positions.
    if (needsPad || llvm::any_of(interiorPadding, [](int64_t p) { return p > 0; })) {
        MPSGraphTensor* ones = [ctx.graph constantWithScalar:1.0 dataType:input.dataType];
        ones = [ctx.graph broadcastTensor:ones toShape:input.shape name:nil];
        MPSGraphTensor* mask = ApplyInteriorPadding(ctx.graph, ones, interiorPadding, rank);
        if (needsCrop) {
            mask = [ctx.graph sliceTensor:mask
                                   starts:cropStarts
                                     ends:cropEnds
                                  strides:cropStrides
                                     name:nil];
        }
        if (needsPad) {
            mask = [ctx.graph padTensor:mask
                        withPaddingMode:MPSGraphPaddingModeConstant
                            leftPadding:leftPad
                           rightPadding:rightPad
                          constantValue:0.0
                                   name:nil];
        }

        MPSGraphTensor* invMask = [ctx.graph
            subtractionWithPrimaryTensor:[ctx.graph constantWithScalar:1.0 dataType:mask.dataType]
                         secondaryTensor:mask
                                    name:nil];
        MPSGraphTensor* padFill = [ctx.graph multiplicationWithPrimaryTensor:invMask
                                                             secondaryTensor:paddingValue
                                                                        name:nil];
        current = [ctx.graph additionWithPrimaryTensor:current secondaryTensor:padFill name:nil];
    }

    return Result(ctx, current, "pad");
}
REGISTER_MPS_OP("stablehlo.pad", HandlePad);

// Dynamic update slice - update a portion of a tensor with new values
static ProcessResult HandleDynamicUpdateSlice(HandlerContext& ctx) {
    auto updateSliceOp = mlir::dyn_cast<mlir::stablehlo::DynamicUpdateSliceOp>(ctx.op);
    if (!updateSliceOp) {
        return ProcessResult::Error("dynamic_update_slice: expected DynamicUpdateSliceOp");
    }

    MPSGraphTensor* operand = GetInputTensor(ctx, 0);
    MPSGraphTensor* update = GetInputTensor(ctx, 1);
    if (!operand || !update)
        return ProcessResult::Error("dynamic_update_slice: missing input tensor");

    NSArray<NSNumber*>* updateShape = update.shape;
    NSUInteger rank = updateShape.count;

    // Get operand shape for clamping
    NSArray<NSNumber*>* operandShape = operand.shape;

    // Get start indices (operands 2 through N)
    // Per StableHLO spec, start indices must be clamped:
    //   start[i] = clamp(start[i], 0, dim_size[i] - update_size[i])
    NSMutableArray<MPSGraphTensor*>* startIndices = [NSMutableArray array];
    for (NSUInteger i = 0; i < rank; i++) {
        MPSGraphTensor* startIdx = GetInputTensor(ctx, i + 2);
        if (!startIdx) {
            return ProcessResult::Error("dynamic_update_slice: missing start index");
        }
        [startIndices addObject:startIdx];
    }

    // Create coordinate tensors for the update region
    NSMutableArray<MPSGraphTensor*>* indexTensors = [NSMutableArray array];
    for (NSUInteger dim = 0; dim < rank; dim++) {
        MPSGraphTensor* startIdx = startIndices[dim];

        // Clamp start index: max(0, min(start, dim_size - update_size))
        int64_t dimSize = operandShape[dim].longLongValue;
        int64_t updateSize = updateShape[dim].longLongValue;
        int64_t maxStart = std::max(dimSize - updateSize, (int64_t)0);

        MPSGraphTensor* zero = [ctx.graph constantWithScalar:0 dataType:startIdx.dataType];
        MPSGraphTensor* maxVal = [ctx.graph constantWithScalar:static_cast<double>(maxStart) dataType:startIdx.dataType];
        startIdx = [ctx.graph clampWithTensor:startIdx
                               minValueTensor:zero
                               maxValueTensor:maxVal
                                         name:nil];

        // Create coordinate tensor for this dimension (0, 1, 2, ..., update_size-1)
        MPSGraphTensor* coords = [ctx.graph coordinateAlongAxis:(NSInteger)dim
                                                      withShape:updateShape
                                                           name:nil];

        // Cast coordinates to match start index type
        coords = [ctx.graph castTensor:coords toType:startIdx.dataType name:nil];

        // Add start index to coordinates
        MPSGraphTensor* adjustedCoords = [ctx.graph additionWithPrimaryTensor:coords
                                                              secondaryTensor:startIdx
                                                                         name:nil];

        [indexTensors addObject:adjustedCoords];
    }

    // Stack the coordinate tensors along a new last axis to form indices tensor
    MPSGraphTensor* indices = [ctx.graph stackTensors:indexTensors axis:(NSInteger)rank name:nil];

    // Cast indices to int32 if needed
    indices = EnsureInt32(ctx.graph, indices);

    // Use SafeScatterND to handle integer precision issues
    MPSGraphTensor* result =
        SafeScatterND(ctx.graph, operand, update, indices, 0, MPSGraphScatterModeSet);
    return Result(ctx, result, "dynamic_update_slice");
}
REGISTER_MPS_OP("stablehlo.dynamic_update_slice", HandleDynamicUpdateSlice);

// Gather - generalized indexing operation
// Handles embedding lookups and other gather patterns
static ProcessResult HandleGather(HandlerContext& ctx) {
    auto gatherOp = mlir::dyn_cast<mlir::stablehlo::GatherOp>(ctx.op);
    if (!gatherOp) {
        return ProcessResult::Error("gather: expected GatherOp");
    }

    MPSGraphTensor* operand = GetInputTensor(ctx, 0);
    MPSGraphTensor* startIndices = GetInputTensor(ctx, 1);
    if (!operand || !startIndices)
        return ProcessResult::Error("gather: missing input tensor");

    auto dimNumbers = gatherOp.getDimensionNumbers();
    auto collapsedSliceDims = dimNumbers.getCollapsedSliceDims();
    auto offsetDims = dimNumbers.getOffsetDims();
    auto startIndexMap = dimNumbers.getStartIndexMap();
    int64_t indexVectorDim = dimNumbers.getIndexVectorDim();
    auto operandBatchingDims = dimNumbers.getOperandBatchingDims();
    auto startIndicesBatchingDims = dimNumbers.getStartIndicesBatchingDims();

    NSArray<NSNumber*>* indicesShape = startIndices.shape;
    NSUInteger indicesRank = indicesShape.count;

    // Handle batched gather pattern (e.g., from vmap over dynamic_index_in_dim or dynamic_slice):
    // Pattern: gather with batching dimensions where each batch element gathers independently
    // Example 1 (point gather): operand [3,10], indices [3,1], slice_sizes=[1,1]
    //   - For each batch i, gather operand[i, indices[i,0]]
    // Example 2 (batched dynamic_slice): operand [5,10], indices [5,1], slice_sizes=[1,4]
    //   - For each batch i, gather operand[i, indices[i,0]:indices[i,0]+4]
    // Also handles batched gather with collapsed dims and offset dims (e.g., from batched LU).
    if (!operandBatchingDims.empty() && !startIndicesBatchingDims.empty() &&
        operandBatchingDims.size() == startIndicesBatchingDims.size() &&
        operandBatchingDims == startIndicesBatchingDims && startIndexMap.size() == 1 &&
        indexVectorDim == (int64_t)indicesRank - 1 &&
        [indicesShape[indicesRank - 1] integerValue] == 1) {
        int64_t gatherAxis = startIndexMap[0];
        auto sliceSizes = gatherOp.getSliceSizes();
        int64_t sliceLen = sliceSizes[gatherAxis];

        // Compute total gather points N from indices dims between batch and index_vector_dim.
        // Indices shape: [batch..., N..., 1] where N dims are the gather dimensions.
        NSUInteger batchCount = operandBatchingDims.size();
        int64_t numGatherPoints = 1;
        for (NSUInteger i = batchCount; i < indicesRank - 1; ++i) {
            numGatherPoints *= [indicesShape[i] integerValue];
        }

        // Squeeze the trailing 1 from indices: [batch..., N..., 1] -> [batch..., N...]
        NSMutableArray<NSNumber*>* squeezedShape = [NSMutableArray array];
        for (NSUInteger i = 0; i < indicesRank - 1; ++i) {
            [squeezedShape addObject:indicesShape[i]];
        }
        MPSGraphTensor* squeezedIndices = [ctx.graph reshapeTensor:startIndices
                                                         withShape:squeezedShape
                                                              name:nil];
        squeezedIndices = EnsureInt32(ctx.graph, squeezedIndices);

        // If slice length > 1, build iota offsets: indices[i] + [0, 1, ..., sliceLen-1]
        // This converts a batched dynamic_slice into a batched gather with explicit indices.
        if (sliceLen > 1) {
            // Create iota [0, 1, ..., sliceLen-1]
            MPSGraphTensor* iota = [ctx.graph coordinateAlongAxis:0
                                                        withShape:@[@(sliceLen)]
                                                             name:nil];
            iota = [ctx.graph castTensor:iota toType:MPSDataTypeInt32 name:nil];

            // Reshape squeezedIndices to [..., 1] and iota to [1..., sliceLen] for broadcast
            NSMutableArray<NSNumber*>* idxBroadcastShape = [NSMutableArray array];
            NSMutableArray<NSNumber*>* iotaBroadcastShape = [NSMutableArray array];
            for (NSUInteger i = 0; i < squeezedShape.count; ++i) {
                [idxBroadcastShape addObject:squeezedShape[i]];
                [iotaBroadcastShape addObject:@1];
            }
            [idxBroadcastShape addObject:@1];
            [iotaBroadcastShape addObject:@(sliceLen)];

            squeezedIndices = [ctx.graph reshapeTensor:squeezedIndices
                                             withShape:idxBroadcastShape
                                                  name:nil];
            iota = [ctx.graph reshapeTensor:iota withShape:iotaBroadcastShape name:nil];

            // Broadcast and add: result shape [batch..., N..., sliceLen]
            NSMutableArray<NSNumber*>* combinedShape = [NSMutableArray array];
            for (NSUInteger i = 0; i < squeezedShape.count; ++i) {
                [combinedShape addObject:squeezedShape[i]];
            }
            [combinedShape addObject:@(sliceLen)];

            squeezedIndices = [ctx.graph broadcastTensor:squeezedIndices
                                                 toShape:combinedShape
                                                    name:nil];
            iota = [ctx.graph broadcastTensor:iota toShape:combinedShape name:nil];
            squeezedIndices = [ctx.graph additionWithPrimaryTensor:squeezedIndices
                                                   secondaryTensor:iota
                                                              name:nil];
            // Now squeezedIndices has shape [batch..., N..., sliceLen]
            // Update numGatherPoints to include slice length
            numGatherPoints *= sliceLen;
        }

        // Build broadcast shape matching operand rank:
        //   - batch dims get batch sizes
        //   - gather axis gets numGatherPoints (N * sliceLen or just N)
        //   - other dims get 1 (expanded) / operand size (broadcast)
        NSMutableArray<NSNumber*>* expandedShape = [NSMutableArray array];
        NSMutableArray<NSNumber*>* broadcastShape = [NSMutableArray array];
        NSUInteger operandRank = operand.shape.count;
        for (NSUInteger d = 0; d < operandRank; d++) {
            bool isBatchDim = false;
            for (auto bd : operandBatchingDims) {
                if ((NSUInteger)bd == d) {
                    isBatchDim = true;
                    break;
                }
            }
            if (isBatchDim) {
                [expandedShape addObject:operand.shape[d]];
                [broadcastShape addObject:operand.shape[d]];
            } else if ((int64_t)d == gatherAxis) {
                [expandedShape addObject:@(numGatherPoints)];
                [broadcastShape addObject:@(numGatherPoints)];
            } else {
                [expandedShape addObject:@1];
                [broadcastShape addObject:operand.shape[d]];
            }
        }

        MPSGraphTensor* reshapedIndices = [ctx.graph reshapeTensor:squeezedIndices
                                                         withShape:expandedShape
                                                              name:nil];
        reshapedIndices = [ctx.graph broadcastTensor:reshapedIndices
                                             toShape:broadcastShape
                                                name:nil];

        MPSGraphTensor* result =
            SafeGatherAlongAxis(ctx.graph, (NSInteger)gatherAxis, operand, reshapedIndices);

        // Reshape to expected output shape
        auto outputShape = GetOutputShape(ctx.op);
        result = [ctx.graph reshapeTensor:result withShape:outputShape name:nil];

        return Result(ctx, result, "gather");
    }

    // Handle common embedding lookup pattern:
    // operand: [num_embeddings, embedding_dim]
    // indices: [batch..., 1] where the last dim is the index vector
    // offset_dims: [last_dim] - the embedding dimension
    // collapsed_slice_dims: [0] - the looked-up dimension
    // start_index_map: [0] - indices point into dim 0

    // Check if index_vector_dim is the last dimension and has size 1
    // This is the common embedding pattern
    if (indexVectorDim == (int64_t)indicesRank - 1 &&
        [indicesShape[indicesRank - 1] integerValue] == 1 && startIndexMap.size() == 1 &&
        collapsedSliceDims.size() == 1 && collapsedSliceDims[0] == startIndexMap[0]) {
        int64_t gatherAxis = startIndexMap[0];

        // Squeeze the index vector dimension from indices
        // [batch..., 1] -> [batch...]
        NSMutableArray<NSNumber*>* squeezedShape = [NSMutableArray array];
        for (NSUInteger i = 0; i < indicesRank - 1; i++) {
            [squeezedShape addObject:indicesShape[i]];
        }

        MPSGraphTensor* squeezedIndices = [ctx.graph reshapeTensor:startIndices
                                                         withShape:squeezedShape
                                                              name:nil];

        // Cast indices to int32 if needed (MPS gather requires int32)
        squeezedIndices = EnsureInt32(ctx.graph, squeezedIndices);

        // Use SafeGather to handle integer precision issues
        MPSGraphTensor* result =
            SafeGather(ctx.graph, operand, squeezedIndices, (NSUInteger)gatherAxis, 0);

        return Result(ctx, result, "gather");
    }

    // Handle uncollapsed point gather pattern (e.g., searchsorted binary search):
    // Like the embedding pattern above but the gathered dimension is kept in the output
    // as a size-1 offset dim instead of being collapsed.
    // Example: operand [5], indices [6, 1], start_index_map=[0], collapsed=[],
    //   offset_dims=[1], slice_sizes=[1] -> output [6, 1]
    if (indexVectorDim == (int64_t)indicesRank - 1 &&
        [indicesShape[indicesRank - 1] integerValue] == 1 && startIndexMap.size() == 1 &&
        collapsedSliceDims.empty()) {
        int64_t gatherAxis = startIndexMap[0];
        auto sliceSizes = gatherOp.getSliceSizes();

        // Verify slice size for the mapped dim is 1
        if (sliceSizes[gatherAxis] == 1) {
            // Squeeze index vector dim: [batch..., 1] -> [batch...]
            NSMutableArray<NSNumber*>* squeezedShape = [NSMutableArray array];
            for (NSUInteger i = 0; i < indicesRank - 1; i++) {
                [squeezedShape addObject:indicesShape[i]];
            }

            MPSGraphTensor* squeezedIndices = [ctx.graph reshapeTensor:startIndices
                                                             withShape:squeezedShape
                                                                  name:nil];
            squeezedIndices = EnsureInt32(ctx.graph, squeezedIndices);

            MPSGraphTensor* result =
                SafeGather(ctx.graph, operand, squeezedIndices, (NSUInteger)gatherAxis, 0);

            // Reshape to expected output (adds back the size-1 offset dim)
            NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
            result = [ctx.graph reshapeTensor:result withShape:outputShape name:nil];

            return Result(ctx, result, "gather");
        }
    }

    // Handle full-index gather pattern (e.g., x[0, 0, 0] on a rank-3 tensor):
    // - indices is a 1D vector of length input_rank
    // - index_vector_dim is 0, meaning the entire indices tensor is one index vector
    // - start_index_map covers all operand dimensions in order [0, 1, ..., rank-1]
    // - offset_dims is empty (no trailing slice dimensions)
    // - collapsed_slice_dims is either empty (slice sizes are all 1) or fully collapsed
    //   ([0, 1, ..., rank-1]); both represent point gathers yielding a scalar
    if (indexVectorDim == 0 && indicesRank == 1 && startIndexMap.size() == operand.shape.count &&
        offsetDims.empty() && [indicesShape[0] integerValue] == (NSInteger)operand.shape.count) {
        bool hasFullCollapsedSliceDims = collapsedSliceDims.size() == operand.shape.count;
        if (hasFullCollapsedSliceDims) {
            for (NSUInteger dim = 0; dim < operand.shape.count; ++dim) {
                if (collapsedSliceDims[dim] != (int64_t)dim) {
                    hasFullCollapsedSliceDims = false;
                    break;
                }
            }
        }

        bool fullRange = true;
        for (NSUInteger dim = 0; dim < operand.shape.count; ++dim) {
            if (startIndexMap[dim] != (int64_t)dim) {
                fullRange = false;
                break;
            }
        }

        if (!fullRange || !(collapsedSliceDims.empty() || hasFullCollapsedSliceDims)) {
            return ProcessResult::Error("gather: unsupported full-index gather pattern");
        }

        // MPS gatherND expects indices as [N, rank]. Reshape [rank] -> [1, rank].
        MPSGraphTensor* ndIndices = [ctx.graph reshapeTensor:startIndices
                                                   withShape:@[@1, @(operand.shape.count)]
                                                        name:nil];
        ndIndices = EnsureInt32(ctx.graph, ndIndices);

        // Use SafeGatherND to handle integer precision issues
        MPSGraphTensor* gathered = SafeGatherND(ctx.graph, operand, ndIndices, 0);

        // Result is [1] for a scalar point gather; reshape to scalar.
        if (gathered.shape.count == 1 && [gathered.shape[0] integerValue] == 1) {
            gathered = [ctx.graph reshapeTensor:gathered withShape:@[] name:nil];
        }
        return Result(ctx, gathered, "gather");
    }

    // Handle dynamic_slice-like gather pattern:
    // Pattern: indices are a coordinate vector (index_vector_dim=0), all dims are offset,
    // no collapsed dims. This is a multi-dimensional dynamic slice.
    // Single-index example: operand [4,16], indices [1], start_index_map=[1], slice_sizes=[4,15]
    // Multi-index example: operand [2,3,3], indices [2], start_index_map=[1,2], slice_sizes=[2,1,1]
    if (operandBatchingDims.empty() && startIndicesBatchingDims.empty() &&
        collapsedSliceDims.empty() && indexVectorDim == 0 && indicesRank == 1 &&
        !startIndexMap.empty() &&
        (int64_t)[indicesShape[0] integerValue] == (int64_t)startIndexMap.size()) {
        auto sliceSizes = gatherOp.getSliceSizes();
        // Apply dynamic slices sequentially along each indexed dimension
        MPSGraphTensor* gathered = operand;
        for (size_t i = 0; i < startIndexMap.size(); ++i) {
            int64_t sliceAxis = startIndexMap[i];
            int64_t sliceLen = sliceSizes[sliceAxis];

            // Extract the i-th index from the indices vector
            MPSGraphTensor* startIdx = [ctx.graph sliceTensor:startIndices
                                                       starts:@[@(i)]
                                                         ends:@[@(i + 1)]
                                                      strides:@[@1]
                                                         name:nil];
            startIdx = EnsureInt32(ctx.graph, startIdx);
            startIdx = [ctx.graph reshapeTensor:startIdx withShape:@[] name:nil];

            // Create iota range [0, 1, ..., sliceLen-1] + startIdx
            MPSGraphTensor* iota = [ctx.graph coordinateAlongAxis:0
                                                        withShape:@[@(sliceLen)]
                                                             name:nil];
            iota = [ctx.graph castTensor:iota toType:MPSDataTypeInt32 name:nil];
            MPSGraphTensor* offsetIota = [ctx.graph additionWithPrimaryTensor:iota
                                                              secondaryTensor:startIdx
                                                                         name:nil];

            // Build broadcast shape for gatherAlongAxis
            NSUInteger curRank = gathered.shape.count;
            NSMutableArray<NSNumber*>* iotaShape = [NSMutableArray array];
            NSMutableArray<NSNumber*>* broadcastShape = [NSMutableArray array];
            for (NSUInteger d = 0; d < curRank; ++d) {
                if ((int64_t)d == sliceAxis) {
                    [iotaShape addObject:@(sliceLen)];
                    [broadcastShape addObject:@(sliceLen)];
                } else {
                    [iotaShape addObject:@1];
                    [broadcastShape addObject:gathered.shape[d]];
                }
            }
            offsetIota = [ctx.graph reshapeTensor:offsetIota withShape:iotaShape name:nil];
            offsetIota = [ctx.graph broadcastTensor:offsetIota toShape:broadcastShape name:nil];

            gathered = SafeGatherAlongAxis(ctx.graph, (NSInteger)sliceAxis, gathered, offsetIota);
        }

        // Reshape to expected output
        NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
        gathered = [ctx.graph reshapeTensor:gathered withShape:outputShape name:nil];
        return Result(ctx, gathered, "gather");
    }

    // Handle slice-gather pattern (e.g., embedding lookup via vmap):
    // Like multi-index gather but some mapped dims have full-size slices (not collapsed).
    // Pattern: operand [10,4], indices [N,2], start_index_map=[0,1], slice_sizes=[1,4]
    //   collapsed=[], offset_dims=[1,2]. The second index is always 0 (full dim slice).
    // Strategy: identify the single "real" gather dim (slice_size=1), use simple gather.
    if (operandBatchingDims.empty() && startIndicesBatchingDims.empty() &&
        indexVectorDim == (int64_t)indicesRank - 1 && startIndexMap.size() > 1) {
        auto sliceSizes = gatherOp.getSliceSizes();
        auto operandType = mlir::cast<mlir::RankedTensorType>(ctx.op->getOperand(0).getType());
        auto operandShape = operandType.getShape();

        // Find dims where slice_size = 1 (real point-indexed dims)
        // and dims where slice_size = full operand dim (passthrough dims)
        llvm::SmallVector<int64_t> realIndexDims;
        llvm::SmallVector<size_t> realIndexColumns;  // column index in startIndexMap
        bool allPassthroughsAreFull = true;
        for (size_t i = 0; i < startIndexMap.size(); ++i) {
            int64_t dim = startIndexMap[i];
            if (sliceSizes[dim] == 1) {
                realIndexDims.push_back(dim);
                realIndexColumns.push_back(i);
            } else if (sliceSizes[dim] == operandShape[dim]) {
                // Full-size slice — passthrough, index must be 0
            } else {
                allPassthroughsAreFull = false;
            }
        }

        // This handler applies when: exactly 1 real index dim, rest are full passthrough
        if (allPassthroughsAreFull && realIndexDims.size() == 1 && realIndexColumns.size() == 1) {
            int64_t gatherAxis = realIndexDims[0];
            size_t indexCol = realIndexColumns[0];

            // Extract the relevant index column: indices[..., indexCol]
            NSMutableArray<NSNumber*>* starts = [NSMutableArray array];
            NSMutableArray<NSNumber*>* ends = [NSMutableArray array];
            NSMutableArray<NSNumber*>* strides_arr = [NSMutableArray array];
            for (NSUInteger d = 0; d < indicesRank; ++d) {
                if (d == (NSUInteger)indexVectorDim) {
                    [starts addObject:@(indexCol)];
                    [ends addObject:@(indexCol + 1)];
                } else {
                    [starts addObject:@0];
                    [ends addObject:indicesShape[d]];
                }
                [strides_arr addObject:@1];
            }
            MPSGraphTensor* indexCol_t = [ctx.graph sliceTensor:startIndices
                                                         starts:starts
                                                           ends:ends
                                                        strides:strides_arr
                                                           name:nil];

            // Squeeze the index vector dim: [N,1] -> [N]
            NSMutableArray<NSNumber*>* squeezedShape = [NSMutableArray array];
            for (NSUInteger d = 0; d < indicesRank; ++d) {
                if (d != (NSUInteger)indexVectorDim) {
                    [squeezedShape addObject:indexCol_t.shape[d]];
                }
            }
            indexCol_t = [ctx.graph reshapeTensor:indexCol_t withShape:squeezedShape name:nil];
            indexCol_t = EnsureInt32(ctx.graph, indexCol_t);

            // Gather along the real index axis
            MPSGraphTensor* gathered =
                SafeGather(ctx.graph, operand, indexCol_t, (NSUInteger)gatherAxis, 0);

            // Reshape to expected output shape (handles the offset dims)
            NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
            gathered = [ctx.graph reshapeTensor:gathered withShape:outputShape name:nil];

            return Result(ctx, gathered, "gather");
        }
    }

    // Handle multi-index gather pattern:
    // - indices: [N, k] where each row is a k-dimensional coordinate
    // - index_vector_dim is the last dimension
    // - collapsed_slice_dims ⊆ start_index_map (same set or subset)
    // - all mapped slice_sizes are 1 (for collapsed dims) or pass through (for non-collapsed)
    // - offset_dims may be empty (pure point gather) or non-empty (with offset/batch dims)
    // - no batching dimensions
    //
    // Examples:
    //   Pure point gather: x[arange(n), arange(n)] for diagonal
    //     offset_dims=[], collapsed=[0,1], start_index_map=[0,1], slice_sizes=[1,1]
    //   Gather with offset: jnp.diagonal(a, axis1=-1, axis2=-2) on [3,2,2] tensor
    //     offset_dims=[0], collapsed=[1,2], start_index_map=[1,2], slice_sizes=[3,1,1]
    //   Embedding via vmap (slice gather): operand [10,4], indices [3,2]
    //     offset_dims=[1,2], collapsed=[], start_index_map=[0,1], slice_sizes=[1,4]
    if (operandBatchingDims.empty() && startIndicesBatchingDims.empty() &&
        indexVectorDim == (int64_t)indicesRank - 1 &&
        collapsedSliceDims.size() <= startIndexMap.size() && startIndexMap.size() > 1) {
        // Verify collapsed_slice_dims and start_index_map contain the same dims
        llvm::SmallVector<int64_t> sortedCollapsed(collapsedSliceDims.begin(),
                                                   collapsedSliceDims.end());
        llvm::SmallVector<int64_t> sortedMap(startIndexMap.begin(), startIndexMap.end());
        llvm::sort(sortedCollapsed);
        llvm::sort(sortedMap);

        bool dimsMatch = (sortedCollapsed == sortedMap);

        // Verify all mapped slice_sizes are 1
        auto sliceSizes = gatherOp.getSliceSizes();
        bool allOnes = true;
        for (int64_t dim : startIndexMap) {
            if (sliceSizes[dim] != 1) {
                allOnes = false;
                break;
            }
        }

        if (dimsMatch && allOnes) {
            NSUInteger operandRank = operand.shape.count;

            // Identify which operand dims are indexed vs offset
            llvm::SmallDenseSet<int64_t, 8> indexedDimSet(sortedMap.begin(), sortedMap.end());

            if (offsetDims.empty()) {
                // Pure point gather: all dims are indexed
                bool coversAllDims = (startIndexMap.size() == operandRank);
                if (coversAllDims) {
                    for (NSUInteger d = 0; d < operandRank; ++d) {
                        if (sortedMap[d] != (int64_t)d) {
                            coversAllDims = false;
                            break;
                        }
                    }
                }

                if (coversAllDims) {
                    // Indices are [N, rank] — use gatherND directly
                    MPSGraphTensor* ndIndices = EnsureInt32(ctx.graph, startIndices);
                    MPSGraphTensor* gathered = SafeGatherND(ctx.graph, operand, ndIndices, 0);
                    return Result(ctx, gathered, "gather");
                }
            }

            // Gather with offset dimensions: some dims pass through, others are point-gathered.
            // Strategy: transpose operand so indexed dims come first, then flatten indexed dims,
            // compute flat indices from multi-dim index coordinates, and gather along the flat
            // axis. This produces [N, offset_dims...] directly, which we then transpose if needed.

            // Build permutation: indexed dims first, then offset dims
            llvm::SmallVector<int64_t> perm;
            llvm::SmallVector<int64_t> offsetOperandDims;
            for (int64_t d : sortedMap) {
                perm.push_back(d);
            }
            for (NSUInteger d = 0; d < operandRank; ++d) {
                if (!indexedDimSet.count(d)) {
                    perm.push_back(d);
                    offsetOperandDims.push_back(d);
                }
            }

            // Check if permutation is identity (no transpose needed)
            bool needsTranspose = false;
            for (size_t i = 0; i < perm.size(); ++i) {
                if (perm[i] != (int64_t)i) {
                    needsTranspose = true;
                    break;
                }
            }

            MPSGraphTensor* transposed = operand;
            if (needsTranspose) {
                NSMutableArray<NSNumber*>* permArr = [NSMutableArray array];
                for (int64_t p : perm) {
                    [permArr addObject:@(p)];
                }
                transposed = [ctx.graph transposeTensor:operand permutation:permArr name:nil];
            }

            // Flatten the indexed dimensions into one.
            // After transpose, shape is [idx_dim0, ..., idx_dimM, offset_dim0, ..., offset_dimK]
            NSUInteger numIndexedDims = sortedMap.size();
            NSMutableArray<NSNumber*>* flatShape = [NSMutableArray array];
            int64_t flatIndexedSize = 1;
            for (NSUInteger d = 0; d < numIndexedDims; ++d) {
                flatIndexedSize *= [transposed.shape[d] integerValue];
            }
            [flatShape addObject:@(flatIndexedSize)];
            for (NSUInteger d = numIndexedDims; d < operandRank; ++d) {
                [flatShape addObject:transposed.shape[d]];
            }

            MPSGraphTensor* flatOperand = [ctx.graph reshapeTensor:transposed
                                                         withShape:flatShape
                                                              name:nil];

            // Convert multi-dim indices to flat indices using strides.
            // indices shape: [N, k] where k = number of indexed dims
            // After transpose, indexed dims are in sorted (ascending) order.
            // Compute strides in sorted order, then map back to startIndexMap order.
            llvm::SmallVector<int64_t> sortedDimSizes;
            for (int64_t d : sortedMap) {
                sortedDimSizes.push_back([operand.shape[(NSUInteger)d] integerValue]);
            }

            // Compute row-major strides for the sorted indexed dims
            llvm::SmallVector<int64_t> sortedStrides(sortedMap.size(), 1);
            for (int i = (int)sortedMap.size() - 2; i >= 0; --i) {
                sortedStrides[i] = sortedStrides[i + 1] * sortedDimSizes[i + 1];
            }

            // Map strides back to startIndexMap order: for each index column i,
            // find where startIndexMap[i] sits in sortedMap to get the correct stride.
            llvm::SmallVector<int64_t> strides(startIndexMap.size());
            for (size_t i = 0; i < startIndexMap.size(); ++i) {
                for (size_t j = 0; j < sortedMap.size(); ++j) {
                    if (sortedMap[j] == startIndexMap[i]) {
                        strides[i] = sortedStrides[j];
                        break;
                    }
                }
            }

            MPSGraphTensor* indicesInt = EnsureInt32(ctx.graph, startIndices);

            // Compute flat_index = sum(indices[:, i] * stride[i]) for each index point
            MPSGraphTensor* flatIndices = nil;
            for (size_t i = 0; i < strides.size(); ++i) {
                // Extract the i-th column of indices: indices[:, i]
                // indices shape is [N, k], so slice along dim 1
                NSMutableArray<NSNumber*>* starts = [NSMutableArray array];
                NSMutableArray<NSNumber*>* ends = [NSMutableArray array];
                NSMutableArray<NSNumber*>* stridesArr = [NSMutableArray array];
                for (NSUInteger d = 0; d < indicesRank; ++d) {
                    if (d == (NSUInteger)indexVectorDim) {
                        [starts addObject:@(i)];
                        [ends addObject:@(i + 1)];
                    } else {
                        [starts addObject:@0];
                        [ends addObject:indicesInt.shape[d]];
                    }
                    [stridesArr addObject:@1];
                }
                MPSGraphTensor* col = [ctx.graph sliceTensor:indicesInt
                                                      starts:starts
                                                        ends:ends
                                                     strides:stridesArr
                                                        name:nil];
                // Squeeze the index vector dimension: [N,1] -> [N]
                NSMutableArray<NSNumber*>* squeezedShape = [NSMutableArray array];
                for (NSUInteger d = 0; d < indicesRank; ++d) {
                    if (d != (NSUInteger)indexVectorDim) {
                        [squeezedShape addObject:col.shape[d]];
                    }
                }
                col = [ctx.graph reshapeTensor:col withShape:squeezedShape name:nil];

                // Clamp each index to [0, dim_size-1] to match CPU per-dim clamping
                // (otherwise flat index clamping gives different OOB behavior).
                int64_t dimSize = [operand.shape[(NSUInteger)startIndexMap[i]] integerValue];
                MPSGraphTensor* zero = [ctx.graph constantWithScalar:0 dataType:MPSDataTypeInt32];
                MPSGraphTensor* maxIdx = [ctx.graph constantWithScalar:(double)(dimSize - 1)
                                                              dataType:MPSDataTypeInt32];
                col = [ctx.graph clampWithTensor:col
                                  minValueTensor:zero
                                  maxValueTensor:maxIdx
                                            name:nil];

                MPSGraphTensor* strideT = [ctx.graph constantWithScalar:(double)strides[i]
                                                               dataType:MPSDataTypeInt32];
                MPSGraphTensor* term = [ctx.graph multiplicationWithPrimaryTensor:col
                                                                  secondaryTensor:strideT
                                                                             name:nil];
                if (flatIndices == nil) {
                    flatIndices = term;
                } else {
                    flatIndices = [ctx.graph additionWithPrimaryTensor:flatIndices
                                                       secondaryTensor:term
                                                                  name:nil];
                }
            }

            // Gather along the flattened indexed axis (axis 0, since indexed dims are first)
            NSUInteger gatherAxis = 0;

            // When indices represent a single point (scalar flat index), add a
            // leading batch dim so SafeGatherAlongAxis gets matching ranks.
            bool singlePoint = (flatIndices.shape.count == 0);
            if (singlePoint) {
                flatIndices = [ctx.graph reshapeTensor:flatIndices withShape:@[@1] name:nil];
            }

            // MPS gatherAlongAxis requires updates and indices to have the same rank.
            // When flatIndices has rank > 1 (from multi-dimensional index arrays),
            // flatten it to 1D before gathering, then reshape back afterwards.
            NSArray<NSNumber*>* origBatchShape = nil;
            bool flattenedBatch = false;
            if (!singlePoint && flatIndices.shape.count > 1) {
                origBatchShape = [flatIndices.shape copy];
                int64_t totalBatch = 1;
                for (NSUInteger d = 0; d < flatIndices.shape.count; ++d) {
                    totalBatch *= [flatIndices.shape[d] integerValue];
                }
                flatIndices = [ctx.graph reshapeTensor:flatIndices
                                             withShape:@[@(totalBatch)]
                                                  name:nil];
                flattenedBatch = true;
            }

            // Reshape flatIndices for broadcasting: add size-1 dims for offset dims
            // then broadcast to match operand shape in all non-gather dimensions
            NSMutableArray<NSNumber*>* gatherIndicesShape = [NSMutableArray array];
            for (NSUInteger d = 0; d < flatIndices.shape.count; ++d) {
                [gatherIndicesShape addObject:flatIndices.shape[d]];
            }
            for (NSUInteger d = 1; d < flatOperand.shape.count; ++d) {
                [gatherIndicesShape addObject:flatOperand.shape[d]];
            }

            // First reshape to add dims, then broadcast
            NSMutableArray<NSNumber*>* unsqueezedShape = [NSMutableArray array];
            for (NSUInteger d = 0; d < flatIndices.shape.count; ++d) {
                [unsqueezedShape addObject:flatIndices.shape[d]];
            }
            for (NSUInteger d = 1; d < flatOperand.shape.count; ++d) {
                [unsqueezedShape addObject:@1];
            }
            MPSGraphTensor* reshapedIndices = [ctx.graph reshapeTensor:flatIndices
                                                             withShape:unsqueezedShape
                                                                  name:nil];
            reshapedIndices = [ctx.graph broadcastTensor:reshapedIndices
                                                 toShape:gatherIndicesShape
                                                    name:nil];

            MPSGraphTensor* gathered =
                SafeGatherAlongAxis(ctx.graph, (NSInteger)gatherAxis, flatOperand, reshapedIndices);

            // If we flattened multi-dimensional batch indices, restore the batch dims.
            // gathered shape is [totalBatch, offset_dims...], reshape to
            // [batch_dim0, batch_dim1, ..., offset_dims...]
            if (flattenedBatch) {
                NSMutableArray<NSNumber*>* unflattenedShape = [NSMutableArray array];
                for (NSUInteger d = 0; d < origBatchShape.count; ++d) {
                    [unflattenedShape addObject:origBatchShape[d]];
                }
                for (NSUInteger d = 1; d < gathered.shape.count; ++d) {
                    [unflattenedShape addObject:gathered.shape[d]];
                }
                gathered = [ctx.graph reshapeTensor:gathered withShape:unflattenedShape name:nil];
            }

            // If we added a batch dim for single-point gather, squeeze it now.
            if (singlePoint) {
                NSMutableArray<NSNumber*>* squeezed = [NSMutableArray array];
                for (NSUInteger d = 1; d < gathered.shape.count; ++d) {
                    [squeezed addObject:gathered.shape[d]];
                }
                gathered = [ctx.graph reshapeTensor:gathered withShape:squeezed name:nil];
            }

            // After gathering, result shape is [N0, N1, ..., offset_0, offset_1, ...].
            // The output needs offset dims at positions specified by offsetDims
            // and index dims at the remaining positions.
            // Build a permutation to reorder: for each output position,
            // determine which after-gather dimension goes there.
            NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
            NSUInteger outputRank = outputShape.count;
            // After squeezing single-point batch dim or unflattening,
            // numIndexDims reflects the actual index dims remaining in the gathered result.
            NSUInteger numIndexDims =
                singlePoint ? 0 : (flattenedBatch ? origBatchShape.count : flatIndices.shape.count);
            NSUInteger numOffsetDims = offsetDims.size();

            // Map output positions: offset dims go to specified positions,
            // index dims fill the rest in order.
            llvm::SmallDenseSet<int64_t, 4> offsetDimSet(offsetDims.begin(), offsetDims.end());
            NSMutableArray<NSNumber*>* postPerm = [NSMutableArray array];
            NSUInteger idxPos = 0;
            NSUInteger offPos = 0;
            for (NSUInteger d = 0; d < outputRank; ++d) {
                if (offsetDimSet.count(d)) {
                    // This output position is an offset dim
                    [postPerm addObject:@(numIndexDims + offPos)];
                    offPos++;
                } else {
                    // This output position is an index dim
                    [postPerm addObject:@(idxPos)];
                    idxPos++;
                }
            }

            // Check if permutation is identity
            bool needsPostTranspose = false;
            for (NSUInteger d = 0; d < outputRank; ++d) {
                if ([postPerm[d] unsignedIntegerValue] != d) {
                    needsPostTranspose = true;
                    break;
                }
            }

            if (needsPostTranspose) {
                gathered = [ctx.graph transposeTensor:gathered permutation:postPerm name:nil];
            }

            // Reshape to expected output shape
            gathered = [ctx.graph reshapeTensor:gathered withShape:outputShape name:nil];

            return Result(ctx, gathered, "gather");
        }
    }

    // For now, log unsupported patterns
    return ProcessResult::Error("gather: unsupported gather pattern");
}
REGISTER_MPS_OP("stablehlo.gather", HandleGather);

// Helper to determine scatter mode from the update computation region
static MPSGraphScatterMode GetScatterMode(mlir::stablehlo::ScatterOp scatterOp) {
    MPSGraphScatterMode mode = MPSGraphScatterModeSet;
    auto& updateRegion = scatterOp.getUpdateComputation();
    if (!updateRegion.empty()) {
        auto& block = updateRegion.front();
        for (auto& innerOp : block) {
            if (mlir::isa<mlir::stablehlo::AddOp>(innerOp)) {
                return MPSGraphScatterModeAdd;
            } else if (mlir::isa<mlir::stablehlo::SubtractOp>(innerOp)) {
                return MPSGraphScatterModeSub;
            } else if (mlir::isa<mlir::stablehlo::MulOp>(innerOp)) {
                return MPSGraphScatterModeMul;
            } else if (mlir::isa<mlir::stablehlo::DivOp>(innerOp)) {
                return MPSGraphScatterModeDiv;
            } else if (mlir::isa<mlir::stablehlo::MaxOp>(innerOp)) {
                return MPSGraphScatterModeMax;
            } else if (mlir::isa<mlir::stablehlo::MinOp>(innerOp)) {
                return MPSGraphScatterModeMin;
            }
        }
    }
    return mode;
}

// Scatter - update tensor at specified indices
// This handles the common pattern used by gather gradients
static ProcessResult HandleScatter(HandlerContext& ctx) {
    auto scatterOp = mlir::dyn_cast<mlir::stablehlo::ScatterOp>(ctx.op);
    if (!scatterOp) {
        return ProcessResult::Error("scatter: expected ScatterOp");
    }

    // Get inputs (may be variadic, but we handle single input case)
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    MPSGraphTensor* scatterIndices = GetInputTensor(ctx, 1);
    MPSGraphTensor* updates = GetInputTensor(ctx, 2);
    if (!input || !scatterIndices || !updates)
        return ProcessResult::Error("scatter: missing input tensor");

    auto dimNumbers = scatterOp.getScatterDimensionNumbers();
    auto insertedWindowDims = dimNumbers.getInsertedWindowDims();
    auto scatterDimsToOperandDims = dimNumbers.getScatterDimsToOperandDims();
    auto inputBatchingDims = dimNumbers.getInputBatchingDims();
    auto scatterIndicesBatchingDims = dimNumbers.getScatterIndicesBatchingDims();

    // Handle identity scatter: empty indices with full-window updates.
    // This pattern occurs in LU decomposition where scatter effectively copies
    // the entire updates tensor to the output (no actual scattering).
    // Pattern: scatter_dims_to_operand_dims empty, inserted_window_dims empty,
    // indices has 0 elements, update_window_dims covers all update dims.
    if (scatterDimsToOperandDims.empty() && insertedWindowDims.empty() &&
        inputBatchingDims.empty()) {
        // With no scatter dims, 0 scatter points → output = data unchanged (Set mode)
        // or output = data (with identity update computation).
        // In practice, JAX uses this for full-tensor assignment: output = updates.
        MPSGraphScatterMode mode = GetScatterMode(scatterOp);
        if (mode == MPSGraphScatterModeSet) {
            // Full-window set: output = updates (reshaped to match input if needed)
            MPSGraphTensor* result = updates;
            if (updates.shape.count != input.shape.count) {
                result = [ctx.graph reshapeTensor:updates withShape:input.shape name:nil];
            }
            return Result(ctx, result, "scatter");
        }
        // For other modes (add, etc), identity scatter means output = data
        return Result(ctx, input, "scatter");
    }
    int64_t indexVectorDim = dimNumbers.getIndexVectorDim();

    NSArray<NSNumber*>* indicesShape = scatterIndices.shape;
    NSUInteger indicesRank = indicesShape.count;

    // Handle batched scatter pattern:
    // Pattern: scatter with batching dimensions where each batch element scatters independently
    // Variant 1 (sort gradients): insertedWindowDims has the scatter dim
    //   Example: input [5,7], indices [5,7,1], updates [5,7]
    // Variant 2 (gather gradients): insertedWindowDims empty, update_window_dims has trailing dim
    //   Example: input [3,10], indices [3,1], updates [3,1]
    auto updateWindowDimsForBatched = dimNumbers.getUpdateWindowDims();
    if (!inputBatchingDims.empty() && !scatterIndicesBatchingDims.empty() &&
        inputBatchingDims.size() == scatterIndicesBatchingDims.size() &&
        scatterDimsToOperandDims.size() == 1 && indexVectorDim == (int64_t)indicesRank - 1 &&
        [indicesShape[indicesRank - 1] integerValue] == 1 &&
        (insertedWindowDims.size() == 1 ||
         (insertedWindowDims.empty() && updateWindowDimsForBatched.size() == 1))) {
        int64_t scatterAxis = scatterDimsToOperandDims[0];

        MPSGraphTensor* scatterUpdates = updates;
        MPSGraphTensor* scatterIdx = scatterIndices;

        if (insertedWindowDims.size() == 1) {
            // Variant 1: indices [batch..., N, 1] -> squeeze index vector dim -> [batch..., N]
            NSMutableArray<NSNumber*>* squeezedShape = [NSMutableArray array];
            for (NSUInteger i = 0; i < indicesRank - 1; i++) {
                [squeezedShape addObject:indicesShape[i]];
            }
            scatterIdx = [ctx.graph reshapeTensor:scatterIndices withShape:squeezedShape name:nil];
        } else {
            // Variant 2: indices [batch, 1], updates [batch, 1]
            // scatterAlongAxis needs indices to have same rank as input.
            // Indices already have the right shape [batch, 1] for axis=scatterAxis.
            // Just use them directly (index_vector_dim semantics: the last dim
            // is the index coordinate, which has size 1 for single-dim scatter).
        }

        // scatterAlongAxis requires indices and updates to have matching shapes.
        // Build the shapes by analyzing which dims in the squeezed indices are
        // batch vs scatter.
        NSUInteger inputRank = input.shape.count;

        // After squeezing the index_vector_dim, identify which dims of scatterIdx
        // correspond to batch dims vs scatter dims.
        // scatterIdx shape is indices shape with ivd removed.
        // scatterIndicesBatchingDims refers to the original indices tensor dims.
        llvm::SmallDenseSet<int64_t, 4> idxBatchDimSet;
        for (auto bd : scatterIndicesBatchingDims) {
            // Adjust for squeeze: if bd > indexVectorDim, subtract 1
            int64_t adjustedBd = bd;
            if (bd > indexVectorDim)
                adjustedBd--;
            idxBatchDimSet.insert(adjustedBd);
        }

        // Compute scatter count N from non-batch dims of squeezed indices
        int64_t scatterN = 1;
        for (NSUInteger i = 0; i < scatterIdx.shape.count; ++i) {
            if (!idxBatchDimSet.count(i)) {
                scatterN *= [scatterIdx.shape[i] integerValue];
            }
        }

        // For scatterAlongAxis, indices must match input in all dims except scatter axis.
        // Reshape squeezed indices [scatter..., batch...] -> [input_shape] with
        // scatter axis = N, batch dims = input batch sizes, other dims = 1 for broadcast.
        NSMutableArray<NSNumber*>* idxExpandShape = [NSMutableArray array];
        NSMutableArray<NSNumber*>* targetShape = [NSMutableArray array];
        for (NSUInteger d = 0; d < inputRank; d++) {
            bool isBatchDim = false;
            for (auto bd : inputBatchingDims) {
                if ((NSUInteger)bd == d) {
                    isBatchDim = true;
                    break;
                }
            }
            if (isBatchDim) {
                [idxExpandShape addObject:input.shape[d]];
                [targetShape addObject:input.shape[d]];
            } else if ((int64_t)d == scatterAxis) {
                [idxExpandShape addObject:@(scatterN)];
                [targetShape addObject:@(scatterN)];
            } else {
                [idxExpandShape addObject:@1];
                [targetShape addObject:input.shape[d]];
            }
        }
        scatterIdx = [ctx.graph reshapeTensor:scatterIdx withShape:idxExpandShape name:nil];
        scatterIdx = [ctx.graph broadcastTensor:scatterIdx toShape:targetShape name:nil];
        scatterIdx = EnsureInt32(ctx.graph, scatterIdx);

        // Reshape updates to target shape
        if (scatterUpdates.shape.count < inputRank) {
            NSMutableArray<NSNumber*>* expandedShape =
                [NSMutableArray arrayWithArray:scatterUpdates.shape];
            [expandedShape insertObject:@1 atIndex:static_cast<NSUInteger>(scatterAxis)];
            scatterUpdates = [ctx.graph reshapeTensor:scatterUpdates
                                            withShape:expandedShape
                                                 name:nil];
        }
        scatterUpdates = [ctx.graph broadcastTensor:scatterUpdates toShape:targetShape name:nil];

        MPSGraphScatterMode mode = GetScatterMode(scatterOp);

        MPSGraphTensor* result =
            SafeScatterAlongAxis(ctx.graph, static_cast<NSInteger>(scatterAxis), input,
                                 scatterUpdates, scatterIdx, mode);
        return Result(ctx, result, "scatter");
    }

    // Handle common embedding gradient pattern (reverse of gather):
    // input: [num_embeddings, embedding_dim] - zeros initially
    // indices: [batch..., 1] where last dim is index vector
    // updates: [batch..., embedding_dim] - gradients to scatter
    // Result: accumulate updates into input at specified indices

    // Check for the common pattern where:
    // - index_vector_dim is the last dimension of indices
    // - indices has size 1 in that dimension
    // - we're scattering along a single dimension
    // - no batching dimensions
    if (inputBatchingDims.empty() && indexVectorDim == (int64_t)indicesRank - 1 &&
        [indicesShape[indicesRank - 1] integerValue] == 1 && scatterDimsToOperandDims.size() == 1 &&
        insertedWindowDims.size() == 1 && insertedWindowDims[0] == scatterDimsToOperandDims[0]) {
        int64_t scatterAxis = scatterDimsToOperandDims[0];

        // Squeeze the index vector dimension from indices
        NSMutableArray<NSNumber*>* squeezedShape = [NSMutableArray array];
        for (NSUInteger i = 0; i < indicesRank - 1; i++) {
            [squeezedShape addObject:indicesShape[i]];
        }

        // If squeezing produces a scalar, keep as [1] so MPS has a valid rank for the axis
        if (squeezedShape.count == 0)
            [squeezedShape addObject:@1];

        MPSGraphTensor* squeezedIndices = [ctx.graph reshapeTensor:scatterIndices
                                                         withShape:squeezedShape
                                                              name:nil];
        squeezedIndices = EnsureInt32(ctx.graph, squeezedIndices);

        MPSGraphScatterMode mode = GetScatterMode(scatterOp);

        // Ensure updates is at least rank 1 (MPS doesn't support scalar updates)
        if (updates.shape.count == 0)
            updates = [ctx.graph reshapeTensor:updates withShape:@[@1] name:nil];

        // MPS scatter requires updates rank to match operand rank. When updates has higher
        // rank (e.g. batch of indices into embedding table: indices [B, S, 1] -> squeezed [B, S],
        // updates [B, S, D] into operand [V, D]), flatten the batch dimensions so MPS can handle
        // it.
        if (updates.shape.count > input.shape.count) {
            // The extra dimensions in updates/indices are batch dims that need flattening.
            // updates: [batch..., window_dims...] -> [flat_batch, window_dims...]
            // indices: [batch...] -> [flat_batch]
            NSUInteger numBatchDims =
                updates.shape.count - (input.shape.count - insertedWindowDims.size());
            if (numBatchDims > 0 && squeezedIndices.shape.count == numBatchDims) {
                // Flatten all batch dims into one
                NSInteger flatBatch = 1;
                for (NSUInteger i = 0; i < numBatchDims; i++) {
                    flatBatch *= [squeezedIndices.shape[i] integerValue];
                }

                // Flatten indices: [B1, B2, ...] -> [flat]
                squeezedIndices = [ctx.graph reshapeTensor:squeezedIndices
                                                 withShape:@[@(flatBatch)]
                                                      name:nil];

                // Flatten updates: [B1, B2, ..., D1, D2, ...] -> [flat, D1, D2, ...]
                NSMutableArray<NSNumber*>* flatUpdatesShape = [NSMutableArray array];
                [flatUpdatesShape addObject:@(flatBatch)];
                for (NSUInteger i = numBatchDims; i < updates.shape.count; i++) {
                    [flatUpdatesShape addObject:updates.shape[i]];
                }
                updates = [ctx.graph reshapeTensor:updates withShape:flatUpdatesShape name:nil];
            } else {
                return ProcessResult::Error(
                    "scatter: cannot flatten batch dimensions - updates rank (" +
                    std::to_string(updates.shape.count) + ") > operand rank (" +
                    std::to_string(input.shape.count) + ").");
            }
        }

        // Scalar index updates in StableHLO can drop the scattered axis from the update
        // shape (e.g. input [10,1,4], updates [1,4] for axis 0). MPS scatter expects the
        // update tensor rank to match the operand rank, so reinsert singleton axes as needed.
        if (updates.shape.count < input.shape.count) {
            NSArray<NSNumber*>* updatesShape = updates.shape;
            NSMutableArray<NSNumber*>* alignedUpdatesShape =
                [NSMutableArray arrayWithCapacity:input.shape.count];
            NSUInteger updateDimIndex = 0;

            for (NSUInteger dim = 0; dim < input.shape.count; ++dim) {
                if ((int64_t)dim == scatterAxis) {
                    [alignedUpdatesShape addObject:@1];
                    continue;
                }

                if (updateDimIndex >= updatesShape.count) {
                    [alignedUpdatesShape addObject:@1];
                    continue;
                }

                [alignedUpdatesShape addObject:updatesShape[updateDimIndex++]];
            }

            updates = [ctx.graph reshapeTensor:updates withShape:alignedUpdatesShape name:nil];
        }

        // Use SafeScatter to handle integer precision issues
        MPSGraphTensor* result = SafeScatter(ctx.graph, input, updates, squeezedIndices,
                                             static_cast<NSInteger>(scatterAxis), mode);
        return Result(ctx, result, "scatter");
    }

    // Handle full-rank point updates (e.g. x.at[0,0,...].set(value) on MPS):
    // the update index tensor stores a full index vector and update_window_dims is empty.
    auto updateWindowDims = dimNumbers.getUpdateWindowDims();
    if (updateWindowDims.empty() && inputBatchingDims.empty() &&
        scatterDimsToOperandDims.size() == input.shape.count &&
        insertedWindowDims.size() == input.shape.count && indicesRank == 1 &&
        [indicesShape[0] integerValue] == (NSInteger)input.shape.count && indexVectorDim == 0) {
        bool fullRange = true;
        NSUInteger inputRank = input.shape.count;
        for (NSUInteger dim = 0; dim < inputRank; ++dim) {
            if (scatterDimsToOperandDims[dim] != (int64_t)dim ||
                insertedWindowDims[dim] != (int64_t)dim) {
                fullRange = false;
                break;
            }
        }
        if (!fullRange) {
            return ProcessResult::Error("scatter: unsupported full-rank scatter pattern");
        }

        MPSGraphScatterMode mode = GetScatterMode(scatterOp);

        // MPS scatterND expects indices as [N, rank]. For full point updates with a scalar
        // index vector (e.g. [0,0,0]), reshape to one update row.
        MPSGraphTensor* ndIndices = [ctx.graph reshapeTensor:scatterIndices
                                                   withShape:@[@1, indicesShape[0]]
                                                        name:nil];
        ndIndices = EnsureInt32(ctx.graph, ndIndices);

        if (updates.shape.count == 0) {
            updates = [ctx.graph reshapeTensor:updates withShape:@[@1] name:nil];
        }

        // Use SafeScatterND to handle integer precision issues
        MPSGraphTensor* result = SafeScatterND(ctx.graph, input, updates, ndIndices, 0, mode);
        return Result(ctx, result, "scatter");
    }

    // Handle slice-scatter pattern (reverse of slice-gather, e.g., embedding grad via vmap):
    // Pattern: multi-index scatter where some mapped dims are full passthrough.
    // Example: input [10,4], indices [3,2], updates [3,1,4]
    //   scatter_dims_to_operand_dims=[0,1], inserted_window_dims=[], update_window_dims=[1,2]
    //   The second index column is always 0 (full dim passthrough).
    // Strategy: identify the single "real" scatter dim, reduce to simple scatter.
    if (inputBatchingDims.empty() && insertedWindowDims.empty() &&
        indexVectorDim == (int64_t)indicesRank - 1 && scatterDimsToOperandDims.size() > 1) {
        auto updateWindowDimsSlice = dimNumbers.getUpdateWindowDims();
        auto inputType = mlir::cast<mlir::RankedTensorType>(ctx.op->getOperand(0).getType());
        auto inputShape = inputType.getShape();

        // Find dims where the update window covers the full operand dim (passthrough)
        // vs dims where the update window is size 1 (real scatter point)
        // update_window_dims tells us which update dims are window dims.
        // The window shape comes from the update's window dims.
        llvm::SmallVector<int64_t> realScatterDims;  // operand dims with point scatter
        llvm::SmallVector<size_t> realIndexColumns;  // index columns for real dims
        bool allPassthroughsAreFull = true;

        for (size_t i = 0; i < scatterDimsToOperandDims.size(); ++i) {
            int64_t operandDim = scatterDimsToOperandDims[i];
            // Find the corresponding window dim size from the update shape
            // The update shape is [scatter_dims..., window_dims...]
            // window_dims are listed in updateWindowDimsSlice
            // The window shape for this operand dim: find the window dim index
            // that corresponds to this operand dim
            int64_t windowSize = -1;
            // operandDim maps to a window dim: it's the position of operandDim
            // among all operand dims that appear in update_window_dims
            // Since inserted_window_dims is empty, all operand dims appear as window dims
            // in the same order. So the window size for operandDim is the update size
            // at the corresponding window dim position.
            if ((size_t)operandDim <
                updateWindowDimsSlice.size() + scatterDimsToOperandDims.size()) {
                // Find which window dim corresponds to this operand dim
                for (size_t w = 0; w < updateWindowDimsSlice.size(); ++w) {
                    // The w-th window dim corresponds to the w-th operand dim
                    // (since no dims are inserted/collapsed)
                    if ((int64_t)w == operandDim) {
                        int64_t updateDim = updateWindowDimsSlice[w];
                        auto updatesType =
                            mlir::cast<mlir::RankedTensorType>(ctx.op->getOperand(2).getType());
                        windowSize = updatesType.getShape()[updateDim];
                        break;
                    }
                }
            }

            if (windowSize == 1) {
                realScatterDims.push_back(operandDim);
                realIndexColumns.push_back(i);
            } else if (windowSize == inputShape[operandDim]) {
                // Full passthrough - index is always 0
            } else {
                allPassthroughsAreFull = false;
            }
        }

        if (allPassthroughsAreFull && realScatterDims.size() == 1) {
            int64_t scatterAxis = realScatterDims[0];
            size_t indexCol = realIndexColumns[0];

            // Extract the relevant index column
            NSMutableArray<NSNumber*>* starts = [NSMutableArray array];
            NSMutableArray<NSNumber*>* ends = [NSMutableArray array];
            NSMutableArray<NSNumber*>* strides_arr = [NSMutableArray array];
            for (NSUInteger d = 0; d < indicesRank; ++d) {
                if (d == (NSUInteger)indexVectorDim) {
                    [starts addObject:@(indexCol)];
                    [ends addObject:@(indexCol + 1)];
                } else {
                    [starts addObject:@0];
                    [ends addObject:indicesShape[d]];
                }
                [strides_arr addObject:@1];
            }
            MPSGraphTensor* indexCol_t = [ctx.graph sliceTensor:scatterIndices
                                                         starts:starts
                                                           ends:ends
                                                        strides:strides_arr
                                                           name:nil];

            // Squeeze the index vector dim
            NSMutableArray<NSNumber*>* squeezedShape = [NSMutableArray array];
            for (NSUInteger d = 0; d < indicesRank; ++d) {
                if (d != (NSUInteger)indexVectorDim) {
                    [squeezedShape addObject:indexCol_t.shape[d]];
                }
            }
            if (squeezedShape.count == 0)
                [squeezedShape addObject:@1];
            indexCol_t = [ctx.graph reshapeTensor:indexCol_t withShape:squeezedShape name:nil];
            indexCol_t = EnsureInt32(ctx.graph, indexCol_t);

            // Reshape updates to match input rank for scatter
            // updates shape: [N, 1, 4] -> need [N, 4] for scatter along axis 0
            // Remove the size-1 window dim corresponding to the scatter axis
            NSMutableArray<NSNumber*>* reshapedUpdatesShape = [NSMutableArray array];
            auto updatesType = mlir::cast<mlir::RankedTensorType>(ctx.op->getOperand(2).getType());
            auto updatesShape = updatesType.getShape();

            // Build new shape: keep scatter dims (non-window) and window dims
            // except drop the window dim for the scatter axis (it's size 1)
            for (size_t d = 0; d < (size_t)updatesShape.size(); ++d) {
                bool isScatterAxisWindow = false;
                // Check if this update dim is the window dim for the scatter axis
                for (size_t w = 0; w < updateWindowDimsSlice.size(); ++w) {
                    if (updateWindowDimsSlice[w] == (int64_t)d && (int64_t)w == scatterAxis) {
                        isScatterAxisWindow = true;
                        break;
                    }
                }
                if (isScatterAxisWindow && updatesShape[d] == 1) {
                    continue;  // Drop the size-1 scatter axis window dim
                }
                [reshapedUpdatesShape addObject:@(updatesShape[d])];
            }

            MPSGraphTensor* reshapedUpdates = [ctx.graph reshapeTensor:updates
                                                             withShape:reshapedUpdatesShape
                                                                  name:nil];

            // scatterAlongAxis requires input, updates, and indices to have the same rank.
            // When dropping the scatter axis window dim made updates rank < input rank,
            // reinsert a size-1 dim at the scatter axis position.
            NSUInteger inputRank = input.shape.count;
            if (reshapedUpdates.shape.count < inputRank &&
                reshapedUpdates.shape.count == inputRank - 1) {
                NSMutableArray<NSNumber*>* fixedShape =
                    [NSMutableArray arrayWithArray:reshapedUpdates.shape];
                [fixedShape insertObject:@1 atIndex:static_cast<NSUInteger>(scatterAxis)];
                reshapedUpdates = [ctx.graph reshapeTensor:reshapedUpdates
                                                 withShape:fixedShape
                                                      name:nil];
            }

            // When updates rank > input rank (batch dims from scatter points), flatten
            // the scatter batch dims so updates matches input rank.
            if (reshapedUpdates.shape.count > inputRank) {
                // Flatten scatter batch dims: updates [B1, B2, ..., W1, W2, ...] -> [B1*B2*..., W1,
                // W2, ...] The first (updatesRank - inputRank + 1) dims become one flat dim
                NSUInteger numBatchDims = reshapedUpdates.shape.count - inputRank + 1;
                int64_t flatBatch = 1;
                for (NSUInteger d = 0; d < numBatchDims; ++d) {
                    flatBatch *= [reshapedUpdates.shape[d] integerValue];
                }
                NSMutableArray<NSNumber*>* flatUpdatesShape = [NSMutableArray array];
                [flatUpdatesShape addObject:@(flatBatch)];
                for (NSUInteger d = numBatchDims; d < reshapedUpdates.shape.count; ++d) {
                    [flatUpdatesShape addObject:reshapedUpdates.shape[d]];
                }
                reshapedUpdates = [ctx.graph reshapeTensor:reshapedUpdates
                                                 withShape:flatUpdatesShape
                                                      name:nil];

                // Flatten indices similarly
                int64_t flatIdx = 1;
                for (NSUInteger d = 0; d < indexCol_t.shape.count; ++d) {
                    flatIdx *= [indexCol_t.shape[d] integerValue];
                }
                indexCol_t = [ctx.graph reshapeTensor:indexCol_t withShape:@[@(flatIdx)] name:nil];
            }

            // scatterAlongAxis requires indices to have the same shape as updates.
            // Reshape index to [N, 1] and broadcast to match updates shape.
            if (indexCol_t.shape.count < reshapedUpdates.shape.count) {
                NSMutableArray<NSNumber*>* expandedIdxShape =
                    [NSMutableArray arrayWithArray:indexCol_t.shape];
                while (expandedIdxShape.count < reshapedUpdates.shape.count) {
                    [expandedIdxShape addObject:@1];
                }
                indexCol_t = [ctx.graph reshapeTensor:indexCol_t
                                            withShape:expandedIdxShape
                                                 name:nil];
            }
            // Broadcast indices to match updates shape
            indexCol_t = [ctx.graph broadcastTensor:indexCol_t
                                            toShape:reshapedUpdates.shape
                                               name:nil];

            MPSGraphScatterMode mode = GetScatterMode(scatterOp);
            MPSGraphTensor* result =
                SafeScatterAlongAxis(ctx.graph, static_cast<NSInteger>(scatterAxis), input,
                                     reshapedUpdates, indexCol_t, mode);
            return Result(ctx, result, "scatter");
        }
    }

    // Dynamic-update-slice scatter: N=1 scatter point, all update dims are window dims,
    // insertedWindowDims empty. Finds the single real scatter dim (where window < operand)
    // and uses scatter_along_axis.
    // Example: input [4,2,97], indices [1,2], updates [4,1,97],
    //   scatter_dims_to_operand_dims=[1,2], update_window_dims=[0,1,2], inserted_window_dims=[]
    if (insertedWindowDims.empty() && !scatterDimsToOperandDims.empty() &&
        indexVectorDim == (int64_t)indicesRank - 1) {
        auto updateWindowDimsDUS = dimNumbers.getUpdateWindowDims();
        NSUInteger updatesRank = updates.shape.count;
        bool allWindowDims = (updateWindowDimsDUS.size() == updatesRank);

        if (allWindowDims) {
            // Check N=1 (indices has shape [1, K])
            int64_t N = 1;
            for (NSUInteger i = 0; i < indicesRank - 1; i++) {
                N *= [indicesShape[i] integerValue];
            }

            if (N == 1) {
                NSUInteger operandRank = input.shape.count;
                NSUInteger K = scatterDimsToOperandDims.size();

                // Find the real scatter dim: where updates window < operand dim
                int64_t realScatterDim = -1;
                int64_t realScatterIdx = -1;
                int64_t numRealScatterDims = 0;
                for (NSUInteger i = 0; i < K; i++) {
                    int64_t opDim = scatterDimsToOperandDims[i];
                    int64_t opSize = [input.shape[(NSUInteger)opDim] integerValue];
                    int64_t winSize = [updates.shape[(NSUInteger)opDim] integerValue];
                    if (winSize < opSize) {
                        realScatterDim = opDim;
                        realScatterIdx = i;
                        numRealScatterDims++;
                    }
                }

                if (numRealScatterDims <= 1) {
                    MPSGraphScatterMode mode = GetScatterMode(scatterOp);

                    if (numRealScatterDims == 0) {
                        // All window dims match operand dims: scatter is an identity/add
                        if (mode == MPSGraphScatterModeAdd) {
                            MPSGraphTensor* result = [ctx.graph additionWithPrimaryTensor:input
                                                                          secondaryTensor:updates
                                                                                     name:nil];
                            return Result(ctx, result, "scatter");
                        }
                        // For set mode, just return updates reshaped to input
                        NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
                        MPSGraphTensor* result = [ctx.graph reshapeTensor:updates
                                                                withShape:outputShape
                                                                     name:nil];
                        return Result(ctx, result, "scatter");
                    }

                    // Extract the index for the real scatter dim
                    // indices is [1, K], extract column realScatterIdx
                    MPSGraphTensor* idx = scatterIndices;
                    idx = [ctx.graph reshapeTensor:idx withShape:@[@(K)] name:nil];
                    // Slice out the single index value
                    idx = [ctx.graph sliceTensor:idx
                                       dimension:0
                                           start:(NSInteger)realScatterIdx
                                          length:1
                                            name:nil];
                    idx = EnsureInt32(ctx.graph, idx);

                    // Build sequential indices: start_idx + 0, start_idx + 1, ...
                    // scatter_along_axis places each update at its index, so we need
                    // [start, start+1, ..., start+winSize-1] along the scatter dim.
                    int64_t winSize = [updates.shape[(NSUInteger)realScatterDim] integerValue];
                    std::vector<int32_t> offsetBuf(winSize);
                    for (int64_t i = 0; i < winSize; i++)
                        offsetBuf[i] = (int32_t)i;
                    NSData* offsetData = [NSData dataWithBytes:offsetBuf.data()
                                                        length:winSize * sizeof(int32_t)];
                    MPSGraphTensor* offsets = [ctx.graph constantWithData:offsetData
                                                                    shape:@[@(winSize)]
                                                                 dataType:MPSDataTypeInt32];
                    // idx is [1], offsets is [winSize] → add gives [winSize]
                    idx = [ctx.graph additionWithPrimaryTensor:idx
                                               secondaryTensor:offsets
                                                          name:nil];

                    // Reshape index to match updates shape for scatter_along_axis:
                    // updates is [d0, d1, ..., dk] and we scatter along realScatterDim
                    NSMutableArray<NSNumber*>* idxShape = [NSMutableArray array];
                    for (NSUInteger d = 0; d < updatesRank; d++) {
                        [idxShape addObject:((int64_t)d == realScatterDim) ? updates.shape[d] : @1];
                    }
                    idx = [ctx.graph reshapeTensor:idx withShape:idxShape name:nil];
                    idx = [ctx.graph broadcastTensor:idx toShape:updates.shape name:nil];

                    MPSGraphTensor* result = SafeScatterAlongAxis(
                        ctx.graph, (NSInteger)realScatterDim, input, updates, idx, mode);
                    return Result(ctx, result, "scatter");
                }
            }
        }
    }

    // General ScatterND fallback: handles arbitrary scatter dimension numbers
    // by reshaping indices and updates into the [batch..., N, K] / [batch..., N, window...]
    // layout expected by MPS scatterNDWithDataTensor.
    //
    // Handles two sources of "batch" dimensions:
    //   1. StableHLO batching dims (inputBatchingDims / scatterIndicesBatchingDims)
    //   2. Leading update_window_dims that correspond to leading operand dims before
    //      the scattered dims — these become MPS batch dims, and the indices tensor
    //      is broadcast to include them.
    //
    // Requirements:
    //   - indexVectorDim is the last dim of the indices tensor
    //   - scatterDimsToOperandDims maps to contiguous operand dims
    //   - insertedWindowDims matches the scattered operand dims
    {
        // updateWindowDims was already extracted above for the full-rank pattern
        NSUInteger numStableHLOBatch = inputBatchingDims.size();
        NSUInteger K = scatterDimsToOperandDims.size();  // index vector size

        // When index_vector_dim == indicesRank, the entire indices tensor IS
        // the index vector (one scatter point, N=1). Reshape to [1, K] so the
        // rest of the fallback can treat index_vector_dim as the last dim.
        if (indexVectorDim == (int64_t)indicesRank && (int64_t)indicesRank == (int64_t)K) {
            NSMutableArray<NSNumber*>* newShape = [NSMutableArray array];
            [newShape addObject:@1];  // N=1
            for (NSUInteger i = 0; i < indicesRank; i++) {
                [newShape addObject:indicesShape[i]];
            }
            scatterIndices = [ctx.graph reshapeTensor:scatterIndices withShape:newShape name:nil];
            indicesShape = scatterIndices.shape;
            indicesRank = indicesShape.count;
            indexVectorDim = (int64_t)indicesRank - 1;
        }

        // Verify index_vector_dim is the last dimension of the indices tensor
        if (indexVectorDim != (int64_t)indicesRank - 1) {
            return ProcessResult::Error(
                "scatter: general fallback requires index_vector_dim == last dim of indices");
        }

        // Verify the index vector size matches K
        if ([indicesShape[indicesRank - 1] integerValue] != (NSInteger)K) {
            return ProcessResult::Error("scatter: index vector size mismatch in general fallback");
        }

        // Find the first scattered operand dim to determine MPS batch dim count.
        // All operand dims before min(scatterDimsToOperandDims) that are NOT in
        // insertedWindowDims are leading window dims that become MPS batch dims.
        int64_t minScatterDim = scatterDimsToOperandDims[0];
        for (NSUInteger i = 1; i < K; ++i) {
            minScatterDim = std::min(minScatterDim, scatterDimsToOperandDims[i]);
        }
        NSUInteger mpsBatchDims = (NSUInteger)minScatterDim;

        // StableHLO batching dims must precede the scatter dims in the operand.
        if (mpsBatchDims < numStableHLOBatch) {
            return ProcessResult::Error(
                "scatter: general fallback requires scatter dims after batching dims");
        }

        // Verify scatterDimsToOperandDims maps to contiguous dims
        // [mpsBatchDims, mpsBatchDims+1, ..., mpsBatchDims+K-1]
        // If not contiguous, transpose to make them so, then transpose back.
        bool contiguousDims = true;
        for (NSUInteger i = 0; i < K; ++i) {
            if (scatterDimsToOperandDims[i] != static_cast<int64_t>(mpsBatchDims) + i) {
                contiguousDims = false;
                break;
            }
        }

        NSMutableArray<NSNumber*>* scatterTransposePerm = nil;
        NSMutableArray<NSNumber*>* scatterInversePerm = nil;
        if (!contiguousDims) {
            // Build permutation: batch dims first, then scatter dims sorted,
            // then remaining (window) dims.
            NSUInteger operandRank = input.shape.count;
            llvm::SmallVector<int64_t> sortedScatter(scatterDimsToOperandDims.begin(),
                                                     scatterDimsToOperandDims.end());
            llvm::sort(sortedScatter);
            llvm::SmallDenseSet<int64_t, 8> scatterSet(sortedScatter.begin(), sortedScatter.end());

            llvm::SmallVector<int64_t> perm;
            // Batch dims first (dims before min scatter dim)
            for (NSUInteger d = 0; d < mpsBatchDims; ++d) {
                perm.push_back(d);
            }
            // Then scatter dims in sorted order
            for (int64_t d : sortedScatter) {
                perm.push_back(d);
            }
            // Then remaining dims
            for (NSUInteger d = 0; d < operandRank; ++d) {
                if (d >= mpsBatchDims && !scatterSet.count(d)) {
                    perm.push_back(d);
                }
            }

            // Build forward and inverse permutations
            scatterTransposePerm = [NSMutableArray array];
            scatterInversePerm = [NSMutableArray arrayWithCapacity:operandRank];
            for (NSUInteger d = 0; d < operandRank; ++d) {
                [scatterTransposePerm addObject:@(perm[d])];
                [scatterInversePerm addObject:@0];  // placeholder
            }
            for (NSUInteger d = 0; d < operandRank; ++d) {
                scatterInversePerm[perm[d]] = @(d);
            }

            // Transpose input
            input = [ctx.graph transposeTensor:input permutation:scatterTransposePerm name:nil];

            // Remap scatterDimsToOperandDims and insertedWindowDims.
            // Build old→new dim mapping from perm.
            llvm::SmallDenseMap<int64_t, int64_t, 8> dimRemap;
            for (NSUInteger d = 0; d < operandRank; ++d) {
                dimRemap[perm[d]] = d;
            }

            // Create mutable copies with remapped dims
            llvm::SmallVector<int64_t> remappedScatterDims(K);
            for (NSUInteger i = 0; i < K; ++i) {
                remappedScatterDims[i] = dimRemap[scatterDimsToOperandDims[i]];
            }
            scatterDimsToOperandDims = remappedScatterDims;

            llvm::SmallVector<int64_t> remappedInserted(insertedWindowDims.begin(),
                                                        insertedWindowDims.end());
            for (size_t i = 0; i < remappedInserted.size(); ++i) {
                remappedInserted[i] = dimRemap[remappedInserted[i]];
            }
            llvm::sort(remappedInserted);
            insertedWindowDims = remappedInserted;

            // Now dims should be contiguous
            contiguousDims = true;
        }

        // Special case: dynamic_update_slice-style scatter.
        // When insertedWindowDims is empty, the update tensor has the same rank as
        // the operand and the indices specify a start position for placing the
        // update window (like dynamic_update_slice but expressed as scatter).
        // Convert to coordinate-based scatter using the same approach as
        // HandleDynamicUpdateSlice.
        if (insertedWindowDims.empty() && updateWindowDims.size() == updates.shape.count) {
            NSUInteger updRank = updates.shape.count;
            NSArray<NSNumber*>* updShape = updates.shape;
            NSMutableArray<MPSGraphTensor*>* indexTensors = [NSMutableArray array];

            for (NSUInteger dim = 0; dim < updRank; dim++) {
                // Create coordinate tensor (0, 1, ..., size-1) along this dim
                MPSGraphTensor* coords = [ctx.graph coordinateAlongAxis:(NSInteger)dim
                                                              withShape:updShape
                                                                   name:nil];
                coords = EnsureInt32(ctx.graph, coords);

                // If this operand dim is addressed by the index, add the start offset
                for (NSUInteger k = 0; k < K; k++) {
                    if (scatterDimsToOperandDims[k] == (int64_t)dim) {
                        // Extract start index for this dim from indices tensor
                        // (indices shape is [..., K] after earlier reshaping)
                        MPSGraphTensor* startIdx =
                            [ctx.graph sliceTensor:scatterIndices
                                         dimension:(NSInteger)(indicesRank - 1)
                                             start:(NSInteger)k
                                            length:1
                                              name:nil];

                        // Reshape to [1, 1, ..., 1] for broadcasting with coords
                        NSMutableArray<NSNumber*>* scalarShape = [NSMutableArray array];
                        for (NSUInteger d = 0; d < updRank; d++) {
                            [scalarShape addObject:@1];
                        }
                        startIdx = [ctx.graph reshapeTensor:startIdx
                                                  withShape:scalarShape
                                                       name:nil];
                        startIdx = EnsureInt32(ctx.graph, startIdx);

                        // Clamp: max(0, min(start, dim_size - update_size))
                        int64_t dimSize = [input.shape[dim] integerValue];
                        int64_t updateSize = [updShape[dim] integerValue];
                        int64_t maxStart = dimSize - updateSize;
                        if (maxStart < 0)
                            maxStart = 0;

                        MPSGraphTensor* zero = [ctx.graph constantWithScalar:0
                                                                    dataType:MPSDataTypeInt32];
                        MPSGraphTensor* maxVal = [ctx.graph constantWithScalar:maxStart
                                                                      dataType:MPSDataTypeInt32];
                        startIdx = [ctx.graph clampWithTensor:startIdx
                                               minValueTensor:zero
                                               maxValueTensor:maxVal
                                                         name:nil];

                        coords = [ctx.graph additionWithPrimaryTensor:coords
                                                      secondaryTensor:startIdx
                                                                 name:nil];
                        break;
                    }
                }

                [indexTensors addObject:coords];
            }

            // Stack to form [update_shape..., rank] indices
            MPSGraphTensor* fullIndices = [ctx.graph stackTensors:indexTensors
                                                             axis:(NSInteger)updRank
                                                             name:nil];

            MPSGraphScatterMode mode = GetScatterMode(scatterOp);
            MPSGraphTensor* result = SafeScatterND(ctx.graph, input, updates, fullIndices, 0, mode);
            if (scatterInversePerm) {
                result = [ctx.graph transposeTensor:result permutation:scatterInversePerm name:nil];
            }
            return Result(ctx, result, "scatter");
        }

        // Slice-scatter: some indexed dims are "inserted" (point-indexed) while
        // the update window covers the remaining dims.  Reshape the updates to
        // full operand rank by inserting size-1 dims for each insertedWindowDim,
        // then fall through to the DUS-style coordinate scatter.
        if (!insertedWindowDims.empty() && insertedWindowDims.size() < (size_t)K &&
            updateWindowDims.size() + insertedWindowDims.size() ==
                updates.shape.count + insertedWindowDims.size()) {
            // Build the full-rank update shape: for each operand dim,
            // size 1 if it's an inserted window dim, otherwise take the next
            // window dim size from the update tensor.
            NSUInteger operandRank = input.shape.count;
            llvm::SmallDenseSet<int64_t, 8> insertedSet(insertedWindowDims.begin(),
                                                        insertedWindowDims.end());

            NSMutableArray<NSNumber*>* fullShape = [NSMutableArray array];
            NSUInteger windowIdx = 0;
            for (NSUInteger d = 0; d < operandRank; ++d) {
                if (insertedSet.count(d)) {
                    [fullShape addObject:@1];
                } else if (windowIdx < updateWindowDims.size()) {
                    [fullShape addObject:updates.shape[updateWindowDims[windowIdx]]];
                    windowIdx++;
                } else {
                    [fullShape addObject:@1];
                }
            }

            MPSGraphTensor* reshapedUpdates = [ctx.graph reshapeTensor:updates
                                                             withShape:fullShape
                                                                  name:nil];

            // Now use the DUS-style coordinate scatter approach.
            NSUInteger updRank = reshapedUpdates.shape.count;
            NSArray<NSNumber*>* updShape = reshapedUpdates.shape;
            NSMutableArray<MPSGraphTensor*>* indexTensors = [NSMutableArray array];

            for (NSUInteger dim = 0; dim < updRank; dim++) {
                MPSGraphTensor* coords = [ctx.graph coordinateAlongAxis:(NSInteger)dim
                                                              withShape:updShape
                                                                   name:nil];
                coords = EnsureInt32(ctx.graph, coords);

                for (NSUInteger k = 0; k < K; k++) {
                    if (scatterDimsToOperandDims[k] == (int64_t)dim) {
                        MPSGraphTensor* startIdx =
                            [ctx.graph sliceTensor:scatterIndices
                                         dimension:(NSInteger)(indicesRank - 1)
                                             start:(NSInteger)k
                                            length:1
                                              name:nil];

                        NSMutableArray<NSNumber*>* scalarShape = [NSMutableArray array];
                        for (NSUInteger d = 0; d < updRank; d++) {
                            [scalarShape addObject:@1];
                        }
                        startIdx = [ctx.graph reshapeTensor:startIdx
                                                  withShape:scalarShape
                                                       name:nil];
                        startIdx = EnsureInt32(ctx.graph, startIdx);

                        int64_t dimSize = [input.shape[dim] integerValue];
                        int64_t updateSize = [updShape[dim] integerValue];
                        int64_t maxStart = dimSize - updateSize;
                        if (maxStart < 0)
                            maxStart = 0;

                        MPSGraphTensor* zero = [ctx.graph constantWithScalar:0
                                                                    dataType:MPSDataTypeInt32];
                        MPSGraphTensor* maxVal = [ctx.graph constantWithScalar:maxStart
                                                                      dataType:MPSDataTypeInt32];
                        startIdx = [ctx.graph clampWithTensor:startIdx
                                               minValueTensor:zero
                                               maxValueTensor:maxVal
                                                         name:nil];

                        coords = [ctx.graph additionWithPrimaryTensor:coords
                                                      secondaryTensor:startIdx
                                                                 name:nil];
                        break;
                    }
                }

                [indexTensors addObject:coords];
            }

            MPSGraphTensor* fullIndices = [ctx.graph stackTensors:indexTensors
                                                             axis:(NSInteger)updRank
                                                             name:nil];

            MPSGraphScatterMode mode = GetScatterMode(scatterOp);
            MPSGraphTensor* result =
                SafeScatterND(ctx.graph, input, reshapedUpdates, fullIndices, 0, mode);
            if (scatterInversePerm) {
                result = [ctx.graph transposeTensor:result permutation:scatterInversePerm name:nil];
            }
            return Result(ctx, result, "scatter");
        }

        // Verify insertedWindowDims matches the scattered operand dims
        bool insertedMatch = insertedWindowDims.size() == K;
        if (insertedMatch) {
            for (NSUInteger i = 0; i < K; ++i) {
                if (insertedWindowDims[i] != static_cast<int64_t>(mpsBatchDims) + i) {
                    insertedMatch = false;
                    break;
                }
            }
        }
        if (!insertedMatch) {
            return ProcessResult::Error(
                "scatter: general fallback requires insertedWindowDims to match indexed dims");
        }

        // Identify scatter dims vs window dims in the updates tensor.
        // update_window_dims lists which update dims are window dims; the rest are scatter dims.
        NSUInteger updatesRank = updates.shape.count;
        std::vector<NSUInteger> updateScatterDims;
        for (NSUInteger d = 0; d < updatesRank; ++d) {
            bool isWindow = false;
            for (auto wd : updateWindowDims) {
                if (wd == (int64_t)d) {
                    isWindow = true;
                    break;
                }
            }
            if (!isWindow) {
                updateScatterDims.push_back(d);
            }
        }

        // Count leading window dims (window dims before the first scatter dim in updates).
        // These correspond to leading operand dims and become MPS batch dims.
        // When updateScatterDims is empty (N=1, single scatter point), the leading
        // window dims are the first mpsBatchDims dims, not all of them.
        NSUInteger leadingWindowDims =
            updateScatterDims.empty() ? mpsBatchDims : updateScatterDims[0];
        if (leadingWindowDims != mpsBatchDims) {
            return ProcessResult::Error(
                "scatter: general fallback requires leading update window dims "
                "to match operand batch dims");
        }

        // Compute total scatter points N from the updates scatter dims
        int64_t N = 1;
        for (auto sd : updateScatterDims) {
            N *= [updates.shape[sd] integerValue];
        }

        // Compute N from indices scatter dims for verification
        int64_t indicesN = 1;
        NSUInteger idxScatterStart = numStableHLOBatch;
        NSUInteger idxScatterEnd = indicesRank - 1;
        for (NSUInteger i = idxScatterStart; i < idxScatterEnd; ++i) {
            indicesN *= [indicesShape[i] integerValue];
        }
        if (N != indicesN) {
            return ProcessResult::Error(
                "scatter: general fallback N mismatch between updates and indices");
        }

        // Build the MPS batch shape from the operand's leading dims
        NSMutableArray<NSNumber*>* batchShape = [NSMutableArray array];
        for (NSUInteger d = 0; d < mpsBatchDims; ++d) {
            [batchShape addObject:input.shape[d]];
        }

        // Reshape indices: [stablehlo_batch..., scatter_dims..., K] -> [N, K]
        // Then broadcast to [mps_batch..., N, K] if there are window-based batch dims.
        MPSGraphTensor* ndIndices = [ctx.graph reshapeTensor:scatterIndices
                                                   withShape:@[@(indicesN), @(K)]
                                                        name:nil];
        ndIndices = EnsureInt32(ctx.graph, ndIndices);

        if (mpsBatchDims > numStableHLOBatch) {
            // Leading window dims in the operand don't appear in the indices tensor.
            // Broadcast indices to include these batch dims: [N, K] -> [batch..., N, K]
            NSMutableArray<NSNumber*>* expandedShape = [NSMutableArray array];
            for (NSUInteger d = 0; d < mpsBatchDims; ++d) {
                [expandedShape addObject:@1];
            }
            [expandedShape addObject:@(N)];
            [expandedShape addObject:@(K)];
            ndIndices = [ctx.graph reshapeTensor:ndIndices withShape:expandedShape name:nil];

            NSMutableArray<NSNumber*>* broadcastShape = [NSMutableArray arrayWithArray:batchShape];
            [broadcastShape addObject:@(N)];
            [broadcastShape addObject:@(K)];
            ndIndices = [ctx.graph broadcastTensor:ndIndices toShape:broadcastShape name:nil];
        }

        // Reshape updates to [mps_batch..., N, trailing_window...].
        // The updates tensor has layout:
        //   [leading_window_dims..., scatter_dims..., trailing_window_dims...]
        // Leading window dims match the MPS batch shape. Scatter dims flatten to N.
        // Trailing window dims stay as-is.
        NSMutableArray<NSNumber*>* ndUpdatesShape = [NSMutableArray arrayWithArray:batchShape];
        [ndUpdatesShape addObject:@(N)];
        // When updateScatterDims is empty (N=1), trailing window dims start
        // after the leading batch dims (mpsBatchDims), not at the end.
        NSUInteger trailingStart =
            updateScatterDims.empty() ? mpsBatchDims : (NSUInteger)(updateScatterDims.back() + 1);
        NSArray<NSNumber*>* updatesShape = updates.shape;
        for (NSUInteger d = trailingStart; d < updatesRank; ++d) {
            [ndUpdatesShape addObject:updatesShape[d]];
        }

        MPSGraphTensor* ndUpdates = [ctx.graph reshapeTensor:updates
                                                   withShape:ndUpdatesShape
                                                        name:nil];

        MPSGraphScatterMode mode = GetScatterMode(scatterOp);
        MPSGraphTensor* result =
            SafeScatterND(ctx.graph, input, ndUpdates, ndIndices, mpsBatchDims, mode);
        if (scatterInversePerm) {
            result = [ctx.graph transposeTensor:result permutation:scatterInversePerm name:nil];
        }
        return Result(ctx, result, "scatter");
    }
}
REGISTER_MPS_OP("stablehlo.scatter", HandleScatter);

// Reverse - reverse elements along specified dimensions
static ProcessResult HandleReverse(HandlerContext& ctx) {
    auto reverseOp = mlir::dyn_cast<mlir::stablehlo::ReverseOp>(ctx.op);
    if (!reverseOp) {
        return ProcessResult::Error("reverse: expected ReverseOp");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("reverse: missing input tensor");

    auto dimensions = reverseOp.getDimensions();
    NSMutableArray<NSNumber*>* axes = [NSMutableArray array];
    for (int64_t dim : dimensions) {
        [axes addObject:@(dim)];
    }

    MPSGraphTensor* result = [ctx.graph reverseTensor:input axes:axes name:nil];
    return Result(ctx, result, "reverse");
}
REGISTER_MPS_OP("stablehlo.reverse", HandleReverse);

}  // namespace jax_mps
