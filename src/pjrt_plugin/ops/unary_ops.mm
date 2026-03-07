// Unary operations: math functions, complex part extraction/construction

#import "pjrt_plugin/ops/registry.h"

namespace jax_mps {

// Unary ops using macros
REGISTER_MLIR_UNARY_OP("stablehlo.tanh", tanh, tanh);
REGISTER_MLIR_UNARY_OP("stablehlo.exponential", exponent, exp);
REGISTER_MLIR_UNARY_OP("stablehlo.log", logarithm, log);
REGISTER_MLIR_UNARY_OP("stablehlo.negate", negative, negate);
// abs: MPS absoluteWithTensor: on complex input returns complex (magnitude in
// real part, zero imaginary). StableHLO expects a real-valued result, so we
// extract the real part for complex inputs.
static ProcessResult HandleAbs(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("abs: missing input tensor");
    MPSGraphTensor* result = [ctx.graph absoluteWithTensor:input name:nil];
    if (input.dataType == MPSDataTypeComplexFloat32 || input.dataType == MPSDataTypeComplexFloat16)
        result = [ctx.graph realPartOfTensor:result name:nil];
    return Result(ctx, result, "abs");
}
REGISTER_MPS_OP("stablehlo.abs", HandleAbs);
REGISTER_MLIR_UNARY_OP("stablehlo.sqrt", squareRoot, sqrt);
REGISTER_MLIR_UNARY_OP("stablehlo.rsqrt", reciprocalSquareRoot, rsqrt);
REGISTER_MLIR_UNARY_OP("stablehlo.erf", erf, erf);
REGISTER_MLIR_UNARY_OP("chlo.erf", erf, chlo_erf);
REGISTER_MLIR_UNARY_OP("stablehlo.floor", floor, floor);
// sign: for complex inputs, stablehlo.sign returns x / |x| (or 0 for x == 0).
// MPS signWithTensor: applies component-wise sign which is wrong for complex.
static ProcessResult HandleSign(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("sign: missing input tensor");

    MPSGraphTensor* result = nil;
    if (input.dataType != MPSDataTypeComplexFloat32 &&
        input.dataType != MPSDataTypeComplexFloat16) {
        result = [ctx.graph signWithTensor:input name:nil];
    } else {
        MPSGraphTensor* re = [ctx.graph realPartOfTensor:input name:nil];
        MPSGraphTensor* im = [ctx.graph imaginaryPartOfTensor:input name:nil];
        MPSDataType floatType = re.dataType;

        // magnitude = |x| (as real)
        MPSGraphTensor* magnitude =
            [ctx.graph realPartOfTensor:[ctx.graph absoluteWithTensor:input name:nil] name:nil];

        // Avoid division by zero: use 1 where magnitude is 0, then mask result to 0.
        MPSGraphTensor* zero = [ctx.graph constantWithScalar:0.0 dataType:floatType];
        MPSGraphTensor* one = [ctx.graph constantWithScalar:1.0 dataType:floatType];
        MPSGraphTensor* is_zero = [ctx.graph equalWithPrimaryTensor:magnitude
                                                    secondaryTensor:zero
                                                               name:nil];
        MPSGraphTensor* safe_mag = [ctx.graph selectWithPredicateTensor:is_zero
                                                    truePredicateTensor:one
                                                   falsePredicateTensor:magnitude
                                                                   name:nil];

        // x / |x|, zeroed where |x| == 0
        MPSGraphTensor* norm_re = [ctx.graph divisionWithPrimaryTensor:re
                                                       secondaryTensor:safe_mag
                                                                  name:nil];
        MPSGraphTensor* norm_im = [ctx.graph divisionWithPrimaryTensor:im
                                                       secondaryTensor:safe_mag
                                                                  name:nil];
        norm_re = [ctx.graph selectWithPredicateTensor:is_zero
                                   truePredicateTensor:zero
                                  falsePredicateTensor:norm_re
                                                  name:nil];
        norm_im = [ctx.graph selectWithPredicateTensor:is_zero
                                   truePredicateTensor:zero
                                  falsePredicateTensor:norm_im
                                                  name:nil];

        result = [ctx.graph complexTensorWithRealTensor:norm_re imaginaryTensor:norm_im name:nil];
    }

    return Result(ctx, result, "sign");
}
REGISTER_MPS_OP("stablehlo.sign", HandleSign);
REGISTER_MLIR_UNARY_OP("stablehlo.is_finite", isFinite, is_finite);
REGISTER_MLIR_UNARY_OP("chlo.square", square, chlo_square);
REGISTER_MLIR_UNARY_OP("stablehlo.ceil", ceil, ceil);
REGISTER_MLIR_UNARY_OP("stablehlo.round_nearest_even", rint, round_nearest_even);
REGISTER_MLIR_UNARY_OP("stablehlo.cosine", cos, cosine);
REGISTER_MLIR_UNARY_OP("stablehlo.sine", sin, sine);
REGISTER_MLIR_UNARY_OP("stablehlo.tan", tan, tan);
REGISTER_MLIR_UNARY_OP("chlo.asin", asin, asin);
REGISTER_MLIR_UNARY_OP("chlo.acos", acos, acos);
REGISTER_MLIR_UNARY_OP("chlo.sinh", sinh, sinh);
REGISTER_MLIR_UNARY_OP("chlo.cosh", cosh, cosh);
REGISTER_MLIR_UNARY_OP("chlo.asinh", asinh, asinh);
REGISTER_MLIR_UNARY_OP("chlo.acosh", acosh, acosh);
REGISTER_MLIR_UNARY_OP("chlo.atanh", atanh, atanh);

// Custom call targets for mhlo.* variants of unary ops
REGISTER_CUSTOM_CALL_UNARY_OP("mhlo.erf", erf, mhlo_erf);
REGISTER_CUSTOM_CALL_UNARY_OP("mhlo.asin", asin, mhlo_asin);
REGISTER_CUSTOM_CALL_UNARY_OP("mhlo.acos", acos, mhlo_acos);
REGISTER_CUSTOM_CALL_UNARY_OP("mhlo.sinh", sinh, mhlo_sinh);
REGISTER_CUSTOM_CALL_UNARY_OP("mhlo.cosh", cosh, mhlo_cosh);
REGISTER_CUSTOM_CALL_UNARY_OP("mhlo.asinh", asinh, mhlo_asinh);
REGISTER_CUSTOM_CALL_UNARY_OP("mhlo.acosh", acosh, mhlo_acosh);
REGISTER_CUSTOM_CALL_UNARY_OP("mhlo.atanh", atanh, mhlo_atanh);

// Complex part extraction (methods use OfTensor, not WithTensor, so can't use the macro)
static ProcessResult HandleReal(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("real: missing input tensor");
    MPSGraphTensor* result = [ctx.graph realPartOfTensor:input name:nil];
    return Result(ctx, result, "real");
}
REGISTER_MPS_OP("stablehlo.real", HandleReal);

static ProcessResult HandleImag(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("imag: missing input tensor");
    MPSGraphTensor* result = [ctx.graph imaginaryPartOfTensor:input name:nil];
    return Result(ctx, result, "imag");
}
REGISTER_MPS_OP("stablehlo.imag", HandleImag);

// Complex construction from real and imaginary parts
static ProcessResult HandleComplex(HandlerContext& ctx) {
    MPSGraphTensor* real = GetInputTensor(ctx, 0);
    MPSGraphTensor* imag = GetInputTensor(ctx, 1);
    if (!real || !imag)
        return ProcessResult::Error("complex: missing input tensor");
    MPSGraphTensor* result = [ctx.graph complexTensorWithRealTensor:real
                                                    imaginaryTensor:imag
                                                               name:nil];
    return Result(ctx, result, "complex");
}
REGISTER_MPS_OP("stablehlo.complex", HandleComplex);

// exponential_minus_one: exp(x) - 1
static ProcessResult HandleExpm1(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("exponential_minus_one: missing input tensor");
    MPSGraphTensor* exp_x = [ctx.graph exponentWithTensor:input name:nil];
    MPSGraphTensor* one = [ctx.graph constantWithScalar:1.0 dataType:input.dataType];
    MPSGraphTensor* result = [ctx.graph subtractionWithPrimaryTensor:exp_x
                                                     secondaryTensor:one
                                                                name:nil];
    return Result(ctx, result, "exponential_minus_one");
}
REGISTER_MPS_OP("stablehlo.exponential_minus_one", HandleExpm1);

// log_plus_one: log(1+x) using compensated formula for numerical stability.
//
// The naive log(1+x) loses precision for small |x| because 1.0 + x rounds away
// the low-order bits of x. The compensated formula (Kahan / Cephes / musl libc):
//
//   u = 1 + x
//   if u == 1:  result = x            (x so small that 1+x rounds to 1)
//   else:       result = log(u) * x / (u - 1)
//
// The ratio x/(u-1) corrects for the rounding error introduced in computing u.
static ProcessResult HandleLogPlusOne(HandlerContext& ctx) {
    MPSGraphTensor* x = GetInputTensor(ctx, 0);
    if (!x)
        return ProcessResult::Error("log_plus_one: missing input tensor");

    MPSGraphTensor* one = [ctx.graph constantWithScalar:1.0 dataType:x.dataType];
    MPSGraphTensor* u = [ctx.graph additionWithPrimaryTensor:one secondaryTensor:x name:nil];

    // Guard: when u == 1, (u - 1) is zero so we return x directly.
    MPSGraphTensor* uEqualsOne = [ctx.graph equalWithPrimaryTensor:u secondaryTensor:one name:nil];

    MPSGraphTensor* logU = [ctx.graph logarithmWithTensor:u name:nil];
    MPSGraphTensor* uMinusOne = [ctx.graph subtractionWithPrimaryTensor:u
                                                        secondaryTensor:one
                                                                   name:nil];
    // correction = x / (u - 1) compensates for rounding in u = 1 + x
    MPSGraphTensor* correction = [ctx.graph divisionWithPrimaryTensor:x
                                                      secondaryTensor:uMinusOne
                                                                 name:nil];
    MPSGraphTensor* compensated = [ctx.graph multiplicationWithPrimaryTensor:logU
                                                             secondaryTensor:correction
                                                                        name:nil];

    MPSGraphTensor* result = [ctx.graph selectWithPredicateTensor:uEqualsOne
                                              truePredicateTensor:x
                                             falsePredicateTensor:compensated
                                                             name:nil];
    return Result(ctx, result, "log_plus_one");
}
REGISTER_MPS_OP("stablehlo.log_plus_one", HandleLogPlusOne);

// Inverse error function (erfinv) using Winitzki approximation
// erfinv(x) ≈ sign(x) * sqrt(sqrt(t² - log(1-x²)/a) - t)
// where t = 2/(π*a) + log(1-x²)/2, a ≈ 0.147
static ProcessResult HandleErfInv(HandlerContext& ctx) {
    MPSGraphTensor* x = GetInputTensor(ctx, 0);
    if (!x)
        return ProcessResult::Error("erf_inv: missing input tensor");

    MPSDataType dtype = x.dataType;

    // Constants for Winitzki approximation
    // a = 8*(π-3)/(3*π*(4-π)) ≈ 0.140012
    double a = 0.140012;
    double two_over_pi_a = 2.0 / (M_PI * a);  // ≈ 4.546884

    MPSGraphTensor* one = [ctx.graph constantWithScalar:1.0 dataType:dtype];
    MPSGraphTensor* half = [ctx.graph constantWithScalar:0.5 dataType:dtype];
    MPSGraphTensor* const_a = [ctx.graph constantWithScalar:a dataType:dtype];
    MPSGraphTensor* const_two_pi_a = [ctx.graph constantWithScalar:two_over_pi_a dataType:dtype];

    // x² = x * x
    MPSGraphTensor* x_sq = [ctx.graph multiplicationWithPrimaryTensor:x secondaryTensor:x name:nil];

    // 1 - x²
    MPSGraphTensor* one_minus_x_sq = [ctx.graph subtractionWithPrimaryTensor:one
                                                             secondaryTensor:x_sq
                                                                        name:nil];

    // Clamp to avoid log(0) - use small epsilon
    MPSGraphTensor* epsilon = [ctx.graph constantWithScalar:1e-7 dataType:dtype];
    MPSGraphTensor* clamped = [ctx.graph maximumWithPrimaryTensor:one_minus_x_sq
                                                  secondaryTensor:epsilon
                                                             name:nil];

    // log(1 - x²)
    MPSGraphTensor* log_term = [ctx.graph logarithmWithTensor:clamped name:nil];

    // t = 2/(π*a) + log(1-x²)/2
    MPSGraphTensor* half_log = [ctx.graph multiplicationWithPrimaryTensor:log_term
                                                          secondaryTensor:half
                                                                     name:nil];
    MPSGraphTensor* t = [ctx.graph additionWithPrimaryTensor:const_two_pi_a
                                             secondaryTensor:half_log
                                                        name:nil];

    // t²
    MPSGraphTensor* t_sq = [ctx.graph multiplicationWithPrimaryTensor:t secondaryTensor:t name:nil];

    // log(1-x²) / a
    MPSGraphTensor* log_over_a = [ctx.graph divisionWithPrimaryTensor:log_term
                                                      secondaryTensor:const_a
                                                                 name:nil];

    // t² - log(1-x²)/a
    MPSGraphTensor* inner = [ctx.graph subtractionWithPrimaryTensor:t_sq
                                                    secondaryTensor:log_over_a
                                                               name:nil];

    // sqrt(t² - log(1-x²)/a)
    MPSGraphTensor* sqrt_inner = [ctx.graph squareRootWithTensor:inner name:nil];

    // sqrt(...) - t
    MPSGraphTensor* diff = [ctx.graph subtractionWithPrimaryTensor:sqrt_inner
                                                   secondaryTensor:t
                                                              name:nil];

    // sqrt(sqrt(...) - t) = |erfinv(x)|
    MPSGraphTensor* abs_result = [ctx.graph squareRootWithTensor:diff name:nil];

    // sign(x) * |result|
    MPSGraphTensor* sign_x = [ctx.graph signWithTensor:x name:nil];
    MPSGraphTensor* result = [ctx.graph multiplicationWithPrimaryTensor:sign_x
                                                        secondaryTensor:abs_result
                                                                   name:nil];
    return Result(ctx, result, "erf_inv");
}
REGISTER_MPS_OP("chlo.erf_inv", HandleErfInv);

// cbrt: cube root via sign(x) * pow(|x|, 1/3)
static ProcessResult HandleCbrt(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("cbrt: missing input tensor");
    MPSGraphTensor* abs_input = [ctx.graph absoluteWithTensor:input name:nil];
    MPSGraphTensor* third = [ctx.graph constantWithScalar:(1.0 / 3.0) dataType:input.dataType];
    MPSGraphTensor* pow_result = [ctx.graph powerWithPrimaryTensor:abs_input
                                                   secondaryTensor:third
                                                              name:nil];
    MPSGraphTensor* sign = [ctx.graph signWithTensor:input name:nil];
    MPSGraphTensor* result = [ctx.graph multiplicationWithPrimaryTensor:sign
                                                        secondaryTensor:pow_result
                                                                   name:nil];
    return Result(ctx, result, "cbrt");
}
REGISTER_MPS_OP("stablehlo.cbrt", HandleCbrt);

// reduce_precision: truncate mantissa and clamp exponent of floating-point values.
// Implements the StableHLO reduce_precision op which reduces the precision of
// float32 values to the specified number of exponent and mantissa bits.
//
// Algorithm (for float32: sign=1, exp=8, mantissa=23):
//   1. Bitcast float → uint32
//   2. If mantissa_bits < 23: round-to-nearest-even and truncate low mantissa bits
//   3. If exponent_bits < 8: clamp exponent; overflow → ±inf, underflow → ±0
//   4. Bitcast uint32 → float
//   5. NaN passthrough (NaN stays NaN)
static ProcessResult HandleReducePrecision(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("reduce_precision: missing input tensor");

    auto rpOp = mlir::dyn_cast<mlir::stablehlo::ReducePrecisionOp>(ctx.op);
    if (!rpOp)
        return ProcessResult::Error("reduce_precision: failed to cast to ReducePrecisionOp");

    int exponent_bits = static_cast<int>(rpOp.getExponentBits());
    int mantissa_bits = static_cast<int>(rpOp.getMantissaBits());

    // For float32: 8 exponent bits, 23 mantissa bits
    // If both match, this is a no-op
    if (exponent_bits >= 8 && mantissa_bits >= 23) {
        return Result(ctx, input, "reduce_precision");
    }

    // For float16: 5 exponent bits, 10 mantissa bits
    const int FLOAT32_EXP_BITS = 8;
    const int FLOAT32_MANTISSA_BITS = 23;
    const int FLOAT32_EXP_BIAS = 127;

    MPSGraph* g = ctx.graph;

    // Step 1: Bitcast float32 → int32 (reinterpret bits)
    MPSGraphTensor* bits = [g reinterpretCastTensor:input toType:MPSDataTypeInt32 name:nil];

    // Step 2: Round and truncate mantissa
    if (mantissa_bits < FLOAT32_MANTISSA_BITS) {
        int shift = FLOAT32_MANTISSA_BITS - mantissa_bits;

        // Round to nearest even: add rounding bias, then truncate
        // rounding_bias = (1 << (shift - 1)) - 1 + ((bits >> shift) & 1)
        // The ((bits >> shift) & 1) term implements round-to-even
        MPSGraphTensor* shiftConst = [g constantWithScalar:shift dataType:MPSDataTypeInt32];
        MPSGraphTensor* shifted = [g bitwiseRightShiftWithPrimaryTensor:bits
                                                        secondaryTensor:shiftConst
                                                                   name:nil];
        MPSGraphTensor* lsb = [g bitwiseANDWithPrimaryTensor:shifted
                                              secondaryTensor:[g constantWithScalar:1
                                                                           dataType:MPSDataTypeInt32]
                                                         name:nil];
        int base_bias = (1 << (shift - 1)) - 1;
        MPSGraphTensor* baseBias = [g constantWithScalar:base_bias dataType:MPSDataTypeInt32];
        MPSGraphTensor* roundingBias = [g additionWithPrimaryTensor:baseBias
                                                    secondaryTensor:lsb
                                                               name:nil];

        // Add rounding bias (may overflow mantissa into exponent — that's correct behavior)
        bits = [g additionWithPrimaryTensor:bits secondaryTensor:roundingBias name:nil];

        // Truncate: zero out the low mantissa bits
        int32_t mask = ~((1 << shift) - 1);
        MPSGraphTensor* maskTensor = [g constantWithScalar:mask dataType:MPSDataTypeInt32];
        bits = [g bitwiseANDWithPrimaryTensor:bits secondaryTensor:maskTensor name:nil];
    }

    // Step 3: Clamp exponent range
    if (exponent_bits < FLOAT32_EXP_BITS) {
        // Extract sign and magnitude
        // Note: 0x80000000 overflows int32, use INT32_MIN (-2147483648)
        MPSGraphTensor* signMask = [g constantWithScalar:INT32_MIN dataType:MPSDataTypeInt32];
        MPSGraphTensor* signBit = [g bitwiseANDWithPrimaryTensor:bits
                                                 secondaryTensor:signMask
                                                            name:nil];
        MPSGraphTensor* magMask = [g constantWithScalar:0x7FFFFFFF dataType:MPSDataTypeInt32];
        MPSGraphTensor* magnitude = [g bitwiseANDWithPrimaryTensor:bits
                                                   secondaryTensor:magMask
                                                              name:nil];

        // The reduced format has exponent range [-2^(e-1)+2, 2^(e-1)-1] (biased: [1, 2^e - 2])
        // In float32 biased exponent, the max representable biased exponent is:
        //   FLOAT32_EXP_BIAS + (2^(exponent_bits-1) - 1)
        // The min representable biased exponent is:
        //   FLOAT32_EXP_BIAS - (2^(exponent_bits-1) - 2)
        int max_exp = (1 << (exponent_bits - 1)) - 1;   // e.g., 15 for 5-bit exponent
        int min_exp = -(1 << (exponent_bits - 1)) + 2;  // e.g., -14 for 5-bit exponent

        // In float32 biased representation:
        int max_biased = FLOAT32_EXP_BIAS + max_exp;  // e.g., 142 for 5-bit
        int min_biased = FLOAT32_EXP_BIAS + min_exp;  // e.g., 113 for 5-bit

        // Max representable magnitude (max exponent, all mantissa bits 1):
        // biased_exp << 23 | mantissa_mask
        int32_t max_magnitude = (max_biased << FLOAT32_MANTISSA_BITS) | 0x7FFFFF;
        // Min representable magnitude (min normal exponent, mantissa 0):
        int32_t min_magnitude = (min_biased << FLOAT32_MANTISSA_BITS);

        MPSGraphTensor* maxMag = [g constantWithScalar:max_magnitude dataType:MPSDataTypeInt32];
        MPSGraphTensor* minMag = [g constantWithScalar:min_magnitude dataType:MPSDataTypeInt32];

        // Detect NaN: magnitude > inf magnitude (0x7F800000) — must preserve NaN
        MPSGraphTensor* inf = [g constantWithScalar:0x7F800000 dataType:MPSDataTypeInt32];
        MPSGraphTensor* isNaN = [g greaterThanWithPrimaryTensor:magnitude
                                                secondaryTensor:inf
                                                           name:nil];

        // Overflow: magnitude > max → set to inf (NaN excluded below)
        MPSGraphTensor* isOverflow = [g greaterThanWithPrimaryTensor:magnitude
                                                     secondaryTensor:maxMag
                                                                name:nil];

        // Underflow: 0 < magnitude < min → set to 0
        MPSGraphTensor* zero = [g constantWithScalar:0 dataType:MPSDataTypeInt32];
        MPSGraphTensor* isNonZeroUnderflow = [g logicalANDWithPrimaryTensor:
                                                [g greaterThanWithPrimaryTensor:magnitude
                                                                secondaryTensor:zero
                                                                           name:nil]
                                                            secondaryTensor:
                                                [g lessThanWithPrimaryTensor:magnitude
                                                                secondaryTensor:minMag
                                                                           name:nil]
                                                                       name:nil];

        // Apply: overflow → inf, underflow → 0, else magnitude
        magnitude = [g selectWithPredicateTensor:isOverflow
                              truePredicateTensor:inf
                             falsePredicateTensor:magnitude
                                             name:nil];
        magnitude = [g selectWithPredicateTensor:isNonZeroUnderflow
                              truePredicateTensor:zero
                             falsePredicateTensor:magnitude
                                             name:nil];

        // Recombine sign and magnitude
        magnitude = [g selectWithPredicateTensor:isNaN
                              truePredicateTensor:[g bitwiseANDWithPrimaryTensor:bits
                                                                 secondaryTensor:magMask
                                                                            name:nil]
                             falsePredicateTensor:magnitude
                                             name:nil];
        bits = [g bitwiseORWithPrimaryTensor:signBit secondaryTensor:magnitude name:nil];
    }

    // Step 4: Bitcast int32 → float32
    MPSGraphTensor* result = [g reinterpretCastTensor:bits toType:MPSDataTypeFloat32 name:nil];

    return Result(ctx, result, "reduce_precision");
}
REGISTER_MPS_OP("stablehlo.reduce_precision", HandleReducePrecision);

}  // namespace jax_mps
