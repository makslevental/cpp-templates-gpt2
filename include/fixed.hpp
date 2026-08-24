// fixed.hpp -- Q16.16 fixed-point scalar arithmetic as template metafunctions.
//
// CMake/C++ have the same 64-bit two's-complement integer semantics this relies
// on: `>>` is an arithmetic shift (floors toward -inf), `/` truncates toward
// zero. Every value is a `long long` holding a real number scaled by 2^16.
//
// Each op is a metafunction: a struct template exposing `static constexpr long
// long value`. No constexpr *functions* -- the arithmetic lives in the type
// system, evaluated as the compiler instantiates the templates. Rounding must
// match tools/gen_model.py exactly (mul: half-up; div: half-away-from-zero;
// dot defers its single shift -- see tensor.hpp) or the golden ids won't match.
#pragma once

namespace gpt2 {

inline constexpr long long FX_FRAC = 16;
inline constexpr long long FX_ONE  = 65536;      // 1.0
inline constexpr long long FX_HALF = 32768;      // 0.5
inline constexpr long long FX_MAX  = 2147483647; //  2^31 - 1
inline constexpr long long FX_MIN  = -2147483648;// -2^31

// saturate to [FX_MIN, FX_MAX]
template<long long V>
struct FxSat { static constexpr long long value = V > FX_MAX ? FX_MAX
                                                : V < FX_MIN ? FX_MIN : V; };

// fx_mul: round-half-up, saturating.  |a|,|b| <= 2^31  =>  a*b fits in 64 bits.
template<long long A, long long B>
struct FxMul { static constexpr long long value =
    FxSat<((A * B + FX_HALF) >> FX_FRAC)>::value; };

// fx_div: Q/Q -> Q, round half away from zero (sign-aware bias compensates for
// truncating division). b != 0 required.
template<long long A, long long B>
struct FxDiv {
    static_assert(B != 0, "fx_div by zero");
    static constexpr long long num = A << FX_FRAC;
    static constexpr long long value =
        num >= 0 ? (num + B / 2) / B : -((-num + B / 2) / B);
};

// fx_div_int: divide a Q value by a plain integer, same half-away rounding.
template<long long A, long long N>
struct FxDivInt {
    static_assert(N != 0, "fx_div_int by zero");
    static constexpr long long value =
        A >= 0 ? (A + N / 2) / N : -((-A + N / 2) / N);
};

// _fx_isqrt: floor integer sqrt via Newton/Heron iteration, expressed as tail
// recursion. Invariant mirrors the Python `while x > y: x=(x+y)/2; y=n/x`.
template<long long N, long long X, long long Y, bool Done = (X <= Y)>
struct ISqrtStep { static constexpr long long value =
    ISqrtStep<N, (X + Y) / 2, N / ((X + Y) / 2)>::value; };
template<long long N, long long X, long long Y>
struct ISqrtStep<N, X, Y, true> { static constexpr long long value = X; };

template<long long N>
struct ISqrt { static constexpr long long value =
    N < 2 ? N : ISqrtStep<N, N, 1>::value; };

// fx_sqrt: sqrt of a Q16.16 value, returned in Q16.16.  isqrt(v << 16) works
// because sqrt(v * 2^16) = sqrt(v/2^16) * 2^16. v >= 0 required.
template<long long V>
struct FxSqrt {
    static_assert(V >= 0, "fx_sqrt of negative");
    static constexpr long long value = ISqrt<(V << FX_FRAC)>::value;
};

// --- static-assert unit checks against known values ---
static_assert(FxMul<FX_ONE, FX_ONE>::value == FX_ONE);       // 1*1 == 1
static_assert(FxMul<FX_HALF, FX_HALF>::value == 16384);      // 0.5*0.5 == 0.25
static_assert(FxDiv<FX_ONE, (2 << 16)>::value == FX_HALF);   // 1/2 == 0.5
static_assert(FxDivInt<FX_ONE, 2>::value == FX_HALF);        // 1/2 == 0.5
static_assert(FxSqrt<(4 << 16)>::value == (2 << 16));        // sqrt(4) == 2
static_assert(FxSat<(3LL << 40)>::value == FX_MAX);          // clamps high
static_assert(FxSat<(-(3LL << 40))>::value == FX_MIN);       // clamps low

}  // namespace gpt2
