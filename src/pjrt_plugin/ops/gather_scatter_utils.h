// Centralized safe wrappers for MPS gather/scatter operations.
//
// MPSGraph's gather operations internally convert integer data to float32,
// causing precision loss for 32-bit and 64-bit integers with values > 2^24.
// This affects random key operations like jax.random.split().
//
// These wrappers apply the bitcast workaround:
// 1. Bitcast 32-bit integers to float32 (preserving bit patterns, not values)
// 2. Perform the gather/scatter on the "float32" data
// 3. Bitcast back to the original integer type
// For 64-bit integers, we reshape to pairs of 32-bit values first.

#pragma once

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

namespace jax_mps {

// Helper to check if a type needs the bitcast workaround
inline bool NeedsBitcastWorkaround(MPSDataType type) {
    return type == MPSDataTypeInt32 || type == MPSDataTypeUInt32 || type == MPSDataTypeInt64 ||
           type == MPSDataTypeUInt64;
}

inline bool IsComplexType(MPSDataType type) {
    return type == MPSDataTypeComplexFloat32 || type == MPSDataTypeComplexFloat16;
}

inline bool IsBoolType(MPSDataType type) {
    return type == MPSDataTypeBool;
}

inline bool Is64BitInteger(MPSDataType type) {
    return type == MPSDataTypeInt64 || type == MPSDataTypeUInt64;
}

// Prepare a tensor for gather/scatter by bitcasting integers to float32.
// Also handles booleans, which MPS scatter silently ignores.
// Returns the prepared tensor, and sets the output parameters for reversal.
inline MPSGraphTensor* PrepareIntegerTensor(MPSGraph* graph, MPSGraphTensor* input,
                                            MPSDataType& originalType, bool& needsReverse,
                                            bool& is64Bit) {
    originalType = input.dataType;
    needsReverse = (originalType == MPSDataTypeInt32 || originalType == MPSDataTypeUInt32);
    is64Bit = Is64BitInteger(originalType);

    // MPS scatter operations silently fail on boolean tensors.
    // Cast to int8 so scatter works, then cast back in FinalizeIntegerTensor.
    if (IsBoolType(originalType)) {
        input = [graph castTensor:input toType:MPSDataTypeInt8 name:@"bool_to_int8"];
        return input;
    }

    if (is64Bit) {
        // Reshape input: [shape...] -> [shape..., 2] treating each 64-bit as two 32-bits
        NSMutableArray<NSNumber*>* expandedShape = [NSMutableArray arrayWithArray:input.shape];
        [expandedShape addObject:@2];

        // Reinterpret as uint32 pairs
        input = [graph reinterpretCastTensor:input toType:MPSDataTypeUInt32 name:nil];
        input = [graph reshapeTensor:input withShape:expandedShape name:nil];
        needsReverse = true;
    }

    if (needsReverse) {
        // Bitcast to float32 (reinterpret bits, no conversion)
        input = [graph reinterpretCastTensor:input toType:MPSDataTypeFloat32 name:nil];
    }

    return input;
}

// Reverse the PrepareIntegerTensor operation - bitcasts back to original type.
inline MPSGraphTensor* FinalizeIntegerTensor(MPSGraph* graph, MPSGraphTensor* result,
                                             MPSDataType originalType, bool needsReverse,
                                             bool is64Bit) {
    // Reverse the bool->int8 cast from PrepareIntegerTensor.
    if (IsBoolType(originalType)) {
        return [graph castTensor:result toType:MPSDataTypeBool name:@"int8_to_bool"];
    }

    if (needsReverse) {
        // MPS reinterpret_cast doesn't work on scalar tensors (rank 0).
        // If the result is a scalar, reshape to [1], cast, then reshape back.
        bool isScalar = result.shape.count == 0;
        if (isScalar) {
            result = [graph reshapeTensor:result withShape:@[@1] name:nil];
        }

        // Bitcast back to original type (or uint32 for 64-bit case)
        MPSDataType targetType = is64Bit ? MPSDataTypeUInt32 : originalType;
        result = [graph reinterpretCastTensor:result toType:targetType name:nil];

        if (isScalar) {
            result = [graph reshapeTensor:result withShape:@[] name:nil];
        }
    }

    if (is64Bit) {
        // Reinterpret as original 64-bit type
        bool isScalar = result.shape.count == 0;
        if (isScalar) {
            result = [graph reshapeTensor:result withShape:@[@1] name:nil];
        }
        result = [graph reinterpretCastTensor:result toType:originalType name:nil];
        if (isScalar) {
            result = [graph reshapeTensor:result withShape:@[] name:nil];
        }
    }

    return result;
}

// Safe wrapper for gatherNDWithUpdatesTensor
// Handles int32/uint32/int64/uint64 precision fix and complex type decomposition
inline MPSGraphTensor* SafeGatherND(MPSGraph* graph, MPSGraphTensor* updatesTensor,
                                    MPSGraphTensor* indicesTensor, NSUInteger batchDimensions) {
    // MPS gatherND doesn't support complex types - decompose into real/imag parts
    if (IsComplexType(updatesTensor.dataType)) {
        MPSGraphTensor* real = [graph realPartOfTensor:updatesTensor name:nil];
        MPSGraphTensor* imag = [graph imaginaryPartOfTensor:updatesTensor name:nil];
        MPSGraphTensor* realResult = SafeGatherND(graph, real, indicesTensor, batchDimensions);
        MPSGraphTensor* imagResult = SafeGatherND(graph, imag, indicesTensor, batchDimensions);
        return [graph complexTensorWithRealTensor:realResult imaginaryTensor:imagResult name:nil];
    }

    MPSDataType originalType;
    bool needsReverse = false;
    bool is64Bit = false;
    updatesTensor = PrepareIntegerTensor(graph, updatesTensor, originalType, needsReverse, is64Bit);

    MPSGraphTensor* result = [graph gatherNDWithUpdatesTensor:updatesTensor
                                                indicesTensor:indicesTensor
                                              batchDimensions:batchDimensions
                                                         name:nil];

    return FinalizeIntegerTensor(graph, result, originalType, needsReverse, is64Bit);
}

// Clamp gather indices to valid range [0, dimSize-1] along the given axis.
// This matches XLA/CPU behavior where out-of-bounds gather indices are clamped.
inline MPSGraphTensor* ClampGatherIndices(MPSGraph* graph, MPSGraphTensor* indicesTensor,
                                          MPSGraphTensor* updatesTensor, NSUInteger axis) {
    NSNumber* dimSize = updatesTensor.shape[axis];
    int64_t rawMax = [dimSize longLongValue] - 1;
    int64_t maxIdx = rawMax > 0 ? rawMax : 0;
    MPSGraphTensor* zero = [graph constantWithScalar:0 dataType:indicesTensor.dataType];
    MPSGraphTensor* maxVal = [graph constantWithScalar:static_cast<double>(maxIdx)
                                              dataType:indicesTensor.dataType];
    return [graph clampWithTensor:indicesTensor minValueTensor:zero maxValueTensor:maxVal name:nil];
}

// Safe wrapper for gatherWithUpdatesTensor (axis-based gather)
inline MPSGraphTensor* SafeGather(MPSGraph* graph, MPSGraphTensor* updatesTensor,
                                  MPSGraphTensor* indicesTensor, NSUInteger axis,
                                  NSUInteger batchDimensions) {
    if (IsComplexType(updatesTensor.dataType)) {
        MPSGraphTensor* real = [graph realPartOfTensor:updatesTensor name:nil];
        MPSGraphTensor* imag = [graph imaginaryPartOfTensor:updatesTensor name:nil];
        MPSGraphTensor* realResult = SafeGather(graph, real, indicesTensor, axis, batchDimensions);
        MPSGraphTensor* imagResult = SafeGather(graph, imag, indicesTensor, axis, batchDimensions);
        return [graph complexTensorWithRealTensor:realResult imaginaryTensor:imagResult name:nil];
    }

    // Clamp indices to valid range (matches XLA/CPU behavior)
    indicesTensor = ClampGatherIndices(graph, indicesTensor, updatesTensor, axis);

    MPSDataType originalType;
    bool needsReverse = false;
    bool is64Bit = false;
    updatesTensor = PrepareIntegerTensor(graph, updatesTensor, originalType, needsReverse, is64Bit);

    MPSGraphTensor* result = [graph gatherWithUpdatesTensor:updatesTensor
                                              indicesTensor:indicesTensor
                                                       axis:axis
                                            batchDimensions:batchDimensions
                                                       name:nil];

    return FinalizeIntegerTensor(graph, result, originalType, needsReverse, is64Bit);
}

// Safe wrapper for gatherAlongAxis
inline MPSGraphTensor* SafeGatherAlongAxis(MPSGraph* graph, NSInteger axis,
                                           MPSGraphTensor* updatesTensor,
                                           MPSGraphTensor* indicesTensor) {
    if (IsComplexType(updatesTensor.dataType)) {
        MPSGraphTensor* real = [graph realPartOfTensor:updatesTensor name:nil];
        MPSGraphTensor* imag = [graph imaginaryPartOfTensor:updatesTensor name:nil];
        MPSGraphTensor* realResult = SafeGatherAlongAxis(graph, axis, real, indicesTensor);
        MPSGraphTensor* imagResult = SafeGatherAlongAxis(graph, axis, imag, indicesTensor);
        return [graph complexTensorWithRealTensor:realResult imaginaryTensor:imagResult name:nil];
    }

    // Clamp indices to valid range (matches XLA/CPU behavior)
    indicesTensor = ClampGatherIndices(graph, indicesTensor, updatesTensor, (NSUInteger)axis);

    MPSDataType originalType;
    bool needsReverse = false;
    bool is64Bit = false;
    updatesTensor = PrepareIntegerTensor(graph, updatesTensor, originalType, needsReverse, is64Bit);

    MPSGraphTensor* result = [graph gatherAlongAxis:axis
                                  withUpdatesTensor:updatesTensor
                                      indicesTensor:indicesTensor
                                               name:nil];

    return FinalizeIntegerTensor(graph, result, originalType, needsReverse, is64Bit);
}

// Safe wrapper for scatterNDWithDataTensor
// Scatter operations have the same float32 conversion bug as gather operations.
// Testing confirmed that scatter corrupts uint32 values > 2^24 identically to gather.
// IMPORTANT: The bitcast workaround only works for Set mode. For arithmetic modes
// (Add, Sub, Mul, etc.), bitcasting int to float makes the arithmetic meaningless
// (e.g., int 1 becomes ~1.4e-45 as float, so add gives 0). Skip the workaround
// for non-Set modes — arithmetic scatter on integers works correctly without it.
inline MPSGraphTensor* SafeScatterND(MPSGraph* graph, MPSGraphTensor* dataTensor,
                                     MPSGraphTensor* updatesTensor, MPSGraphTensor* indicesTensor,
                                     NSUInteger batchDimensions, MPSGraphScatterMode mode) {
    if (IsComplexType(dataTensor.dataType)) {
        MPSGraphTensor* dataReal = [graph realPartOfTensor:dataTensor name:nil];
        MPSGraphTensor* dataImag = [graph imaginaryPartOfTensor:dataTensor name:nil];
        MPSGraphTensor* updReal = [graph realPartOfTensor:updatesTensor name:nil];
        MPSGraphTensor* updImag = [graph imaginaryPartOfTensor:updatesTensor name:nil];
        MPSGraphTensor* realResult =
            SafeScatterND(graph, dataReal, updReal, indicesTensor, batchDimensions, mode);
        MPSGraphTensor* imagResult =
            SafeScatterND(graph, dataImag, updImag, indicesTensor, batchDimensions, mode);
        return [graph complexTensorWithRealTensor:realResult imaginaryTensor:imagResult name:nil];
    }

    if (mode != MPSGraphScatterModeSet) {
        return [graph scatterNDWithDataTensor:dataTensor
                                updatesTensor:updatesTensor
                                indicesTensor:indicesTensor
                              batchDimensions:batchDimensions
                                         mode:mode
                                         name:nil];
    }

    // For scatter, both data and updates need the workaround if they're integers
    MPSDataType originalDataType;
    bool dataReverse = false;
    bool dataIs64Bit = false;
    dataTensor =
        PrepareIntegerTensor(graph, dataTensor, originalDataType, dataReverse, dataIs64Bit);

    MPSDataType originalUpdatesType;
    bool updatesReverse = false;
    bool updatesIs64Bit = false;
    updatesTensor = PrepareIntegerTensor(graph, updatesTensor, originalUpdatesType, updatesReverse,
                                         updatesIs64Bit);

    MPSGraphTensor* result = [graph scatterNDWithDataTensor:dataTensor
                                              updatesTensor:updatesTensor
                                              indicesTensor:indicesTensor
                                            batchDimensions:batchDimensions
                                                       mode:mode
                                                       name:nil];

    // The result should match the data tensor type
    return FinalizeIntegerTensor(graph, result, originalDataType, dataReverse, dataIs64Bit);
}

// Safe wrapper for scatterWithDataTensor (axis-based scatter)
inline MPSGraphTensor* SafeScatter(MPSGraph* graph, MPSGraphTensor* dataTensor,
                                   MPSGraphTensor* updatesTensor, MPSGraphTensor* indicesTensor,
                                   NSInteger axis, MPSGraphScatterMode mode) {
    if (IsComplexType(dataTensor.dataType)) {
        MPSGraphTensor* dataReal = [graph realPartOfTensor:dataTensor name:nil];
        MPSGraphTensor* dataImag = [graph imaginaryPartOfTensor:dataTensor name:nil];
        MPSGraphTensor* updReal = [graph realPartOfTensor:updatesTensor name:nil];
        MPSGraphTensor* updImag = [graph imaginaryPartOfTensor:updatesTensor name:nil];
        MPSGraphTensor* realResult =
            SafeScatter(graph, dataReal, updReal, indicesTensor, axis, mode);
        MPSGraphTensor* imagResult =
            SafeScatter(graph, dataImag, updImag, indicesTensor, axis, mode);
        return [graph complexTensorWithRealTensor:realResult imaginaryTensor:imagResult name:nil];
    }

    if (mode != MPSGraphScatterModeSet) {
        return [graph scatterWithDataTensor:dataTensor
                              updatesTensor:updatesTensor
                              indicesTensor:indicesTensor
                                       axis:axis
                                       mode:mode
                                       name:nil];
    }

    MPSDataType originalDataType;
    bool dataReverse = false;
    bool dataIs64Bit = false;
    dataTensor =
        PrepareIntegerTensor(graph, dataTensor, originalDataType, dataReverse, dataIs64Bit);

    MPSDataType originalUpdatesType;
    bool updatesReverse = false;
    bool updatesIs64Bit = false;
    updatesTensor = PrepareIntegerTensor(graph, updatesTensor, originalUpdatesType, updatesReverse,
                                         updatesIs64Bit);

    MPSGraphTensor* result = [graph scatterWithDataTensor:dataTensor
                                            updatesTensor:updatesTensor
                                            indicesTensor:indicesTensor
                                                     axis:axis
                                                     mode:mode
                                                     name:nil];

    return FinalizeIntegerTensor(graph, result, originalDataType, dataReverse, dataIs64Bit);
}

// Safe wrapper for scatterAlongAxis
inline MPSGraphTensor* SafeScatterAlongAxis(MPSGraph* graph, NSInteger axis,
                                            MPSGraphTensor* dataTensor,
                                            MPSGraphTensor* updatesTensor,
                                            MPSGraphTensor* indicesTensor,
                                            MPSGraphScatterMode mode) {
    if (IsComplexType(dataTensor.dataType)) {
        MPSGraphTensor* dataReal = [graph realPartOfTensor:dataTensor name:nil];
        MPSGraphTensor* dataImag = [graph imaginaryPartOfTensor:dataTensor name:nil];
        MPSGraphTensor* updReal = [graph realPartOfTensor:updatesTensor name:nil];
        MPSGraphTensor* updImag = [graph imaginaryPartOfTensor:updatesTensor name:nil];
        MPSGraphTensor* realResult =
            SafeScatterAlongAxis(graph, axis, dataReal, updReal, indicesTensor, mode);
        MPSGraphTensor* imagResult =
            SafeScatterAlongAxis(graph, axis, dataImag, updImag, indicesTensor, mode);
        return [graph complexTensorWithRealTensor:realResult imaginaryTensor:imagResult name:nil];
    }

    if (mode != MPSGraphScatterModeSet) {
        return [graph scatterAlongAxis:axis
                        withDataTensor:dataTensor
                         updatesTensor:updatesTensor
                         indicesTensor:indicesTensor
                                  mode:mode
                                  name:nil];
    }

    MPSDataType originalDataType;
    bool dataReverse = false;
    bool dataIs64Bit = false;
    dataTensor =
        PrepareIntegerTensor(graph, dataTensor, originalDataType, dataReverse, dataIs64Bit);

    MPSDataType originalUpdatesType;
    bool updatesReverse = false;
    bool updatesIs64Bit = false;
    updatesTensor = PrepareIntegerTensor(graph, updatesTensor, originalUpdatesType, updatesReverse,
                                         updatesIs64Bit);

    MPSGraphTensor* result = [graph scatterAlongAxis:axis
                                      withDataTensor:dataTensor
                                       updatesTensor:updatesTensor
                                       indicesTensor:indicesTensor
                                                mode:mode
                                                name:nil];

    return FinalizeIntegerTensor(graph, result, originalDataType, dataReverse, dataIs64Bit);
}

}  // namespace jax_mps
