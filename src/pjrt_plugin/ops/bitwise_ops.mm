// Bitwise operations: and, or, xor, not, shift_left, shift_right_logical,
// shift_right_arithmetic, popcnt, count_leading_zeros

#import "pjrt_plugin/ops/registry.h"

namespace jax_mps {

// Helper to check if operation result is boolean type
static bool isBooleanResult(mlir::Operation* op) {
    if (op->getNumResults() == 0)
        return false;
    auto resultType = op->getResult(0).getType();
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(resultType);
    if (!tensorType)
        return false;
    auto elemType = tensorType.getElementType();
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elemType)) {
        return intType.getWidth() == 1;
    }
    return false;
}

// Macro for logical/bitwise operations (AND, OR, XOR) that dispatch based on boolean type
#define REGISTER_LOGICAL_BITWISE_OP(mlir_op_name, logical_method, bitwise_method, reg_suffix) \
    static ProcessResult Handle##reg_suffix(HandlerContext& ctx) {                            \
        MPSGraphTensor* lhs = GetInputTensor(ctx, 0);                                         \
        MPSGraphTensor* rhs = GetInputTensor(ctx, 1);                                         \
        if (!lhs || !rhs)                                                                     \
            return ProcessResult::Error(#reg_suffix ": missing input tensor");                \
        MPSGraphTensor* result = nil;                                                         \
        if (isBooleanResult(ctx.op)) {                                                        \
            result = [ctx.graph logical_method##WithPrimaryTensor:lhs                         \
                                                  secondaryTensor:rhs                         \
                                                             name:nil];                       \
        } else {                                                                              \
            result = [ctx.graph bitwise_method##WithPrimaryTensor:lhs                         \
                                                  secondaryTensor:rhs                         \
                                                             name:nil];                       \
        }                                                                                     \
        return Result(ctx, result, #reg_suffix);                                              \
    }                                                                                         \
    REGISTER_MPS_OP(mlir_op_name, Handle##reg_suffix)

REGISTER_LOGICAL_BITWISE_OP("stablehlo.and", logicalAND, bitwiseAND, And);
REGISTER_LOGICAL_BITWISE_OP("stablehlo.or", logicalOR, bitwiseOR, Or);
REGISTER_LOGICAL_BITWISE_OP("stablehlo.xor", logicalXOR, bitwiseXOR, Xor);

static ProcessResult HandleNot(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("not: missing input tensor");

    MPSGraphTensor* result = nil;
    if (isBooleanResult(ctx.op)) {
        MPSGraphTensor* falseTensor = [ctx.graph constantWithScalar:0 dataType:input.dataType];
        result = [ctx.graph equalWithPrimaryTensor:input secondaryTensor:falseTensor name:nil];
    } else {
        result = [ctx.graph bitwiseNOTWithTensor:input name:nil];
    }

    return Result(ctx, result, "not");
}
REGISTER_MPS_OP("stablehlo.not", HandleNot);

// Helper to get bit width from tensor's element type
static int getBitWidth(mlir::Operation* op) {
    auto resultType = op->getResult(0).getType();
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(resultType);
    if (!tensorType)
        return 0;
    auto elemType = tensorType.getElementType();
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elemType)) {
        return static_cast<int>(intType.getWidth());
    }
    return 0;
}

static MPSDataType toUnsignedIntegerDataType(MPSDataType dataType) {
    switch (dataType) {
        case MPSDataTypeInt8:
            return MPSDataTypeUInt8;
        case MPSDataTypeInt16:
            return MPSDataTypeUInt16;
        case MPSDataTypeInt32:
            return MPSDataTypeUInt32;
        case MPSDataTypeInt64:
            return MPSDataTypeUInt64;
        case MPSDataTypeUInt8:
        case MPSDataTypeUInt16:
        case MPSDataTypeUInt32:
        case MPSDataTypeUInt64:
            return dataType;
        default:
            return MPSDataTypeInvalid;
    }
}

// StableHLO defines shift overflow in terms of bit-pattern shift counts.
// MPSGraph masks shift counts modulo bit-width, so we materialize overflow masks
// explicitly and use unsigned comparisons for shift amounts.
static MPSGraphTensor* BuildShiftOverflowMask(MPSGraph* g, MPSGraphTensor* shiftAmount,
                                              int bitWidth) {
    MPSDataType unsignedShiftType = toUnsignedIntegerDataType(shiftAmount.dataType);
    MPSGraphTensor* shiftForCompare = shiftAmount;
    if (unsignedShiftType != MPSDataTypeInvalid && unsignedShiftType != shiftAmount.dataType) {
        shiftForCompare = [g castTensor:shiftAmount toType:unsignedShiftType name:nil];
    }

    MPSGraphTensor* bitWidthTensor = [g constantWithScalar:bitWidth
                                                     shape:@[@1]
                                                  dataType:shiftForCompare.dataType];
    return [g greaterThanOrEqualToWithPrimaryTensor:shiftForCompare
                                    secondaryTensor:bitWidthTensor
                                               name:nil];
}

enum class ShiftMode {
    kLeft,
    kRightLogical,
    kRightArithmetic,
};

static MPSGraphTensor* BuildShiftOverflowValue(MPSGraph* g, MPSGraphTensor* input, ShiftMode mode) {
    MPSGraphTensor* zeroTensor = [g constantWithScalar:0 shape:@[@1] dataType:input.dataType];
    if (mode != ShiftMode::kRightArithmetic) {
        return zeroTensor;
    }

    MPSGraphTensor* minusOneTensor = [g constantWithScalar:-1 shape:@[@1] dataType:input.dataType];
    MPSGraphTensor* isNegative = [g lessThanWithPrimaryTensor:input
                                              secondaryTensor:zeroTensor
                                                         name:nil];
    return [g selectWithPredicateTensor:isNegative
                    truePredicateTensor:minusOneTensor
                   falsePredicateTensor:zeroTensor
                                   name:nil];
}

// Shared helper for shift operations with StableHLO overflow handling.
static MPSGraphTensor* HandleShiftOp(HandlerContext& ctx, ShiftMode mode) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    MPSGraphTensor* shiftAmount = GetInputTensor(ctx, 1);
    if (!input || !shiftAmount)
        return nullptr;

    int bitWidth = getBitWidth(ctx.op);
    if (bitWidth == 0)
        return nullptr;

    MPSGraphTensor* shiftedResult = mode == ShiftMode::kLeft
                                        ? [ctx.graph bitwiseLeftShiftWithPrimaryTensor:input
                                                                       secondaryTensor:shiftAmount
                                                                                  name:nil]
                                        : [ctx.graph bitwiseRightShiftWithPrimaryTensor:input
                                                                        secondaryTensor:shiftAmount
                                                                                   name:nil];

    MPSGraphTensor* overflowMask = BuildShiftOverflowMask(ctx.graph, shiftAmount, bitWidth);
    MPSGraphTensor* overflowValue = BuildShiftOverflowValue(ctx.graph, input, mode);

    return [ctx.graph selectWithPredicateTensor:overflowMask
                            truePredicateTensor:overflowValue
                           falsePredicateTensor:shiftedResult
                                           name:nil];
}

static ProcessResult HandleShiftLeft(HandlerContext& ctx) {
    MPSGraphTensor* result = HandleShiftOp(ctx, ShiftMode::kLeft);
    return Result(ctx, result, "shift_left");
}
REGISTER_MPS_OP("stablehlo.shift_left", HandleShiftLeft);

static ProcessResult HandleShiftRightLogical(HandlerContext& ctx) {
    MPSGraphTensor* result = HandleShiftOp(ctx, ShiftMode::kRightLogical);
    return Result(ctx, result, "shift_right_logical");
}
REGISTER_MPS_OP("stablehlo.shift_right_logical", HandleShiftRightLogical);

static ProcessResult HandleShiftRightArithmetic(HandlerContext& ctx) {
    MPSGraphTensor* result = HandleShiftOp(ctx, ShiftMode::kRightArithmetic);
    return Result(ctx, result, "shift_right_arithmetic");
}
REGISTER_MPS_OP("stablehlo.shift_right_arithmetic", HandleShiftRightArithmetic);

// count_leading_zeros: counts the number of leading zero bits.
// MPS doesn't have a native CLZ op, so we implement it using a binary search
// approach with bitwise shifts and selects.
static ProcessResult HandleCountLeadingZeros(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("clz: missing input tensor");

    int bitWidth = getBitWidth(ctx.op);
    if (bitWidth == 0)
        return ProcessResult::Error("clz: could not determine bit width");

    // Cast to unsigned for logical shift behavior
    MPSDataType unsignedType = toUnsignedIntegerDataType(input.dataType);
    if (unsignedType == MPSDataTypeInvalid)
        return ProcessResult::Error("clz: unsupported data type");

    MPSGraphTensor* x = input;
    if (unsignedType != input.dataType)
        x = [ctx.graph castTensor:input toType:unsignedType name:nil];

    // Compute CLZ via: bitWidth - 1 - floor(log2(x)) for x > 0, bitWidth for x == 0.
    // We use the float conversion trick: cast to float, extract the exponent.
    // For int32, cast to float32 gives exact exponent for the highest set bit.
    // floor(log2(x)) = exponent of float(x) - bias = exponent - 127 for float32.
    //
    // Note: float32 has 23-bit mantissa, so values up to 2^24 are exact.
    // For larger values, the cast may round, but the exponent (and thus CLZ) is still correct
    // because rounding only affects mantissa bits below the MSB.

    MPSGraphTensor* zero = [ctx.graph constantWithScalar:0 shape:@[@1] dataType:unsignedType];
    MPSGraphTensor* isZero = [ctx.graph equalWithPrimaryTensor:x secondaryTensor:zero name:nil];

    // Compute CLZ using float conversion: floor(log2(x)) gives MSB position.
    // Precision issue: float32 has 24-bit mantissa, so for 32-bit integers near
    // powers of 2, float(x) may round up to the next power of 2, giving log2
    // one too high. Fix: right-shift x by 1 and use (bitWidth - 2) - floor(log2(x>>1))
    // for the top-bit case. Or simpler: compute via float, then verify by
    // checking if (1 << (bitWidth-1-n)) > x (n was too small).

    MPSGraphTensor* xFloat = [ctx.graph castTensor:x toType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* log2x = [ctx.graph logarithmBase2WithTensor:xFloat name:nil];
    MPSGraphTensor* floorLog2 = [ctx.graph floorWithTensor:log2x name:nil];

    MPSGraphTensor* bwMinus1F = [ctx.graph constantWithScalar:(bitWidth - 1)
                                                        shape:@[@1]
                                                     dataType:MPSDataTypeFloat32];
    MPSGraphTensor* clzFloat = [ctx.graph subtractionWithPrimaryTensor:bwMinus1F
                                                       secondaryTensor:floorLog2
                                                                  name:nil];
    MPSGraphTensor* n = [ctx.graph castTensor:clzFloat toType:unsignedType name:nil];

    // Correction for float32 rounding: when float(x) rounds up to 2^(k+1),
    // log2 gives k+1 instead of k, so n is one too small.
    // Check: if x >> (bitWidth - 1 - n) == 0, then n should be n + 1.
    // (The MSB isn't actually at position bitWidth-1-n.)
    MPSGraphTensor* bwMinus1 = [ctx.graph constantWithScalar:(bitWidth - 1)
                                                       shape:@[@1]
                                                    dataType:unsignedType];
    MPSGraphTensor* msbPos = [ctx.graph subtractionWithPrimaryTensor:bwMinus1
                                                     secondaryTensor:n
                                                                name:nil];
    MPSGraphTensor* checkBit = [ctx.graph bitwiseRightShiftWithPrimaryTensor:x
                                                             secondaryTensor:msbPos
                                                                        name:nil];
    MPSGraphTensor* needsCorrection = [ctx.graph equalWithPrimaryTensor:checkBit
                                                        secondaryTensor:zero
                                                                   name:nil];
    MPSGraphTensor* one = [ctx.graph constantWithScalar:1 shape:@[@1] dataType:unsignedType];
    MPSGraphTensor* nPlusOne = [ctx.graph additionWithPrimaryTensor:n secondaryTensor:one name:nil];
    n = [ctx.graph selectWithPredicateTensor:needsCorrection
                         truePredicateTensor:nPlusOne
                        falsePredicateTensor:n
                                        name:nil];

    // For x == 0, CLZ = bitWidth
    MPSGraphTensor* bitWidthTensor = [ctx.graph constantWithScalar:bitWidth
                                                             shape:@[@1]
                                                          dataType:unsignedType];
    n = [ctx.graph selectWithPredicateTensor:isZero
                         truePredicateTensor:bitWidthTensor
                        falsePredicateTensor:n
                                        name:nil];

    // Cast back to original type
    MPSDataType outType = GetResultMpsType(ctx.op);
    if (outType != MPSDataTypeInvalid && n.dataType != outType)
        n = [ctx.graph castTensor:n toType:outType name:nil];

    return Result(ctx, n, "clz");
}
REGISTER_MPS_OP("stablehlo.count_leading_zeros", HandleCountLeadingZeros);

static ProcessResult HandlePopcnt(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("popcnt: missing input tensor");

    if (ctx.op->getNumOperands() == 0) {
        return ProcessResult::Error("popcnt: requires one integer operand");
    }
    auto operandType = ctx.op->getOperand(0).getType();
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(operandType);
    if (!tensorType || !mlir::isa<mlir::IntegerType>(tensorType.getElementType())) {
        return ProcessResult::Error("popcnt: requires an integer operand");
    }

    MPSGraphTensor* count = [ctx.graph bitwisePopulationCountWithTensor:input name:nil];
    if (!count)
        return ProcessResult::Error("popcnt: bitwisePopulationCount returned null");

    MPSDataType outType = GetResultMpsType(ctx.op);
    MPSGraphTensor* result = count;
    if (outType != MPSDataTypeInvalid && count.dataType != outType) {
        result = [ctx.graph castTensor:count toType:outType name:nil];
    }

    return Result(ctx, result, "popcnt");
}
REGISTER_MPS_OP("stablehlo.popcnt", HandlePopcnt);

}  // namespace jax_mps
