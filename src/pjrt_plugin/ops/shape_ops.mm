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

    // Get start indices as tensors (operands 1 through N)
    // and create coordinate tensors offset by the start indices
    NSMutableArray<MPSGraphTensor*>* indexTensors = [NSMutableArray array];
    for (NSUInteger dim = 0; dim < rank; dim++) {
        // Get the start index tensor for this dimension (scalar tensor)
        MPSGraphTensor* startIdx = GetInputTensor(ctx, dim + 1);
        if (!startIdx) {
            return ProcessResult::Error("dynamic_slice: missing start index for dimension");
        }

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

        MPSGraphTensor* invMask = [ctx.graph subtractionWithPrimaryTensor:
                                    [ctx.graph constantWithScalar:1.0 dataType:mask.dataType]
                                                        secondaryTensor:mask
                                                                   name:nil];
        MPSGraphTensor* padFill = [ctx.graph multiplicationWithPrimaryTensor:invMask
                                                            secondaryTensor:paddingValue
                                                                       name:nil];
        current = [ctx.graph additionWithPrimaryTensor:current
                                      secondaryTensor:padFill
                                                 name:nil];
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

    // Get start indices (operands 2 through N)
    NSMutableArray<MPSGraphTensor*>* startIndices = [NSMutableArray array];
    for (NSUInteger i = 0; i < rank; i++) {
        MPSGraphTensor* startIdx = GetInputTensor(ctx, i + 2);
        if (!startIdx) {
            return ProcessResult::Error("dynamic_update_slice: missing start index");
        }
        [startIndices addObject:startIdx];
    }

    // Build starts array by reading the scalar start indices
    // For sliceUpdateDataTensor, we need static starts/ends/strides
    // But the start indices are dynamic tensors, so we need to use scatter instead

    // Create coordinate tensors for the update region
    NSMutableArray<MPSGraphTensor*>* indexTensors = [NSMutableArray array];
    for (NSUInteger dim = 0; dim < rank; dim++) {
        MPSGraphTensor* startIdx = startIndices[dim];

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

    // Handle batched gather pattern (e.g., from vmap over dynamic_index_in_dim):
    // Pattern: gather with batching dimensions where each batch element gathers independently
    // Example: operand [3,10], indices [3,1]
    //   - operand_batching_dims = [0], start_indices_batching_dims = [0]
    //   - For each batch i, gather operand[i,:] at indices[i,0]
    if (!operandBatchingDims.empty() && !startIndicesBatchingDims.empty() &&
        operandBatchingDims.size() == startIndicesBatchingDims.size() &&
        operandBatchingDims == startIndicesBatchingDims && startIndexMap.size() == 1 &&
        indexVectorDim == (int64_t)indicesRank - 1 &&
        [indicesShape[indicesRank - 1] integerValue] == 1) {
        int64_t gatherAxis = startIndexMap[0];

        // Build indices shape matching operand rank: insert size-1 dims for non-batch,
        // non-axis dims. For operand [batch, D] with axis=1, indices should be [batch, 1].
        NSMutableArray<NSNumber*>* expandedShape = [NSMutableArray array];
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
            } else {
                [expandedShape addObject:@1];
            }
        }

        MPSGraphTensor* reshapedIndices = [ctx.graph reshapeTensor:startIndices
                                                         withShape:expandedShape
                                                              name:nil];
        reshapedIndices = EnsureInt32(ctx.graph, reshapedIndices);

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

    // Handle multi-index point gather pattern:
    // - indices: [N, k] where each row is a k-dimensional coordinate
    // - index_vector_dim is the last dimension
    // - collapsed_slice_dims matches start_index_map (same set of dims)
    // - all mapped slice_sizes are 1
    // - offset_dims may be empty (pure point gather) or non-empty (with offset/batch dims)
    // - no batching dimensions
    //
    // Examples:
    //   Pure point gather: x[arange(n), arange(n)] for diagonal
    //     offset_dims=[], collapsed=[0,1], start_index_map=[0,1], slice_sizes=[1,1]
    //   Gather with offset: jnp.diagonal(a, axis1=-1, axis2=-2) on [3,2,2] tensor
    //     offset_dims=[0], collapsed=[1,2], start_index_map=[1,2], slice_sizes=[3,1,1]
    if (operandBatchingDims.empty() && startIndicesBatchingDims.empty() &&
        indexVectorDim == (int64_t)indicesRank - 1 &&
        collapsedSliceDims.size() == startIndexMap.size() && startIndexMap.size() > 1) {
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

            // Check if indexed dims are contiguous (required for flatten approach)
            bool indexedDimsContiguous = true;
            for (size_t i = 1; i < sortedMap.size(); ++i) {
                if (sortedMap[i] != sortedMap[i - 1] + 1) {
                    indexedDimsContiguous = false;
                    break;
                }
            }

            if (!indexedDimsContiguous) {
                return ProcessResult::Error(
                    "gather: multi-index pattern requires contiguous indexed dims");
            }

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
            // Strategy: transpose operand so offset dims come first, then flatten indexed dims,
            // compute flat indices from multi-dim index coordinates, and gather along the flat axis.

            // Build permutation: offset dims first, then indexed dims
            llvm::SmallVector<int64_t> perm;
            llvm::SmallVector<int64_t> offsetOperandDims;
            for (NSUInteger d = 0; d < operandRank; ++d) {
                if (!indexedDimSet.count(d)) {
                    perm.push_back(d);
                    offsetOperandDims.push_back(d);
                }
            }
            for (int64_t d : sortedMap) {
                perm.push_back(d);
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
                transposed = [ctx.graph transposeTensor:operand
                                            permutation:permArr
                                                   name:nil];
            }

            // Flatten the indexed dimensions into one.
            // After transpose, shape is [offset_dim0, ..., offset_dimK, idx_dim0, ..., idx_dimM]
            NSMutableArray<NSNumber*>* flatShape = [NSMutableArray array];
            int64_t flatIndexedSize = 1;
            for (NSUInteger d = 0; d < operandRank; ++d) {
                if (d < offsetOperandDims.size()) {
                    [flatShape addObject:transposed.shape[d]];
                } else {
                    flatIndexedSize *= [transposed.shape[d] integerValue];
                }
            }
            [flatShape addObject:@(flatIndexedSize)];

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

                MPSGraphTensor* strideT =
                    [ctx.graph constantWithScalar:(double)strides[i]
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

            // Gather along the flattened indexed axis
            NSUInteger gatherAxis = offsetOperandDims.size();

            // Reshape flatIndices for broadcasting: add size-1 dims for offset dims
            // then broadcast to match operand shape in all non-gather dimensions
            NSMutableArray<NSNumber*>* gatherIndicesShape = [NSMutableArray array];
            for (NSUInteger d = 0; d < gatherAxis; ++d) {
                [gatherIndicesShape addObject:flatOperand.shape[d]];
            }
            for (NSUInteger d = 0; d < flatIndices.shape.count; ++d) {
                [gatherIndicesShape addObject:flatIndices.shape[d]];
            }

            // First reshape to add dims, then broadcast
            NSMutableArray<NSNumber*>* unsqueezedShape = [NSMutableArray array];
            for (NSUInteger d = 0; d < gatherAxis; ++d) {
                [unsqueezedShape addObject:@1];
            }
            for (NSUInteger d = 0; d < flatIndices.shape.count; ++d) {
                [unsqueezedShape addObject:flatIndices.shape[d]];
            }
            MPSGraphTensor* reshapedIndices = [ctx.graph reshapeTensor:flatIndices
                                                             withShape:unsqueezedShape
                                                                  name:nil];
            reshapedIndices = [ctx.graph broadcastTensor:reshapedIndices
                                                toShape:gatherIndicesShape
                                                   name:nil];

            MPSGraphTensor* gathered = SafeGatherAlongAxis(ctx.graph, (NSInteger)gatherAxis,
                                                           flatOperand, reshapedIndices);

            // Reshape to expected output shape
            NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
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
        scatterDimsToOperandDims.size() == 1 &&
        indexVectorDim == (int64_t)indicesRank - 1 &&
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
            scatterIdx = [ctx.graph reshapeTensor:scatterIndices
                                         withShape:squeezedShape
                                              name:nil];
        } else {
            // Variant 2: indices [batch, 1], updates [batch, 1]
            // scatterAlongAxis needs indices to have same rank as input.
            // Indices already have the right shape [batch, 1] for axis=scatterAxis.
            // Just use them directly (index_vector_dim semantics: the last dim
            // is the index coordinate, which has size 1 for single-dim scatter).
        }
        scatterIdx = EnsureInt32(ctx.graph, scatterIdx);

        MPSGraphScatterMode mode = GetScatterMode(scatterOp);

        MPSGraphTensor* result = SafeScatterAlongAxis(
            ctx.graph, static_cast<NSInteger>(scatterAxis), input, scatterUpdates, scatterIdx, mode);
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
        // rank (e.g. 2D batch of indices into 2D embedding table), MPS cannot handle it.
        if (updates.shape.count > input.shape.count) {
            return ProcessResult::Error(
                "scatter: MPS does not support scatter where updates rank (" +
                std::to_string(updates.shape.count) + ") > operand rank (" +
                std::to_string(input.shape.count) +
                "). This can occur with multi-dimensional indices into embeddings.");
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
        bool contiguousDims = true;
        for (NSUInteger i = 0; i < K; ++i) {
            if (scatterDimsToOperandDims[i] != static_cast<int64_t>(mpsBatchDims) + i) {
                contiguousDims = false;
                break;
            }
        }
        if (!contiguousDims) {
            return ProcessResult::Error(
                "scatter: general fallback requires contiguous scatterDimsToOperandDims");
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
        NSUInteger leadingWindowDims =
            updateScatterDims.empty() ? updatesRank : updateScatterDims[0];
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
        NSUInteger trailingStart =
            updateScatterDims.empty() ? updatesRank : (NSUInteger)(updateScatterDims.back() + 1);
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
