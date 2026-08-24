// tensor.hpp -- vectors/matrices as types, ops as metafunctions.
//
// A vector is a type carrying its elements as NTTPs: `Vec<long long...>`. A
// matrix is a list of row types: `Mat<Vec<...>, ...>`. Every operation is a
// struct template yielding either `::type` (a new Vec) or `::value` (a scalar).
//
// Reductions over small packs (<=64 here) use fold expressions or shallow
// recursion; the only large table (the 801-entry EXP LUT) is a constexpr data
// array in tables.hpp indexed directly, so instantiation depth stays tiny.
//
// Semantics are ported from tools/gen_model.py and must stay bit-exact:
//  - Dot defers its single round+shift (raw accumulate).
//  - Softmax sum and LayerNorm mean/var sums are raw (Python builtin sum).
//  - Attention value-aggregation floor-shifts each term (done in gpt2.hpp).
#pragma once
#include <cstddef>
#include <utility>
#include <type_traits>
#include "fixed.hpp"
#include "tables.hpp"

namespace gpt2 {

template<long long... Xs> struct Vec { static constexpr int size = sizeof...(Xs); };
template<class... Rows>   struct Mat { static constexpr int rows = sizeof...(Rows); };

// ---- indexing: Get<I, Vec> in O(1) instantiation depth via a masked fold ----
template<int Idx, class V, class Seq> struct GetImpl;
template<int Idx, long long... Xs, std::size_t... Is>
struct GetImpl<Idx, Vec<Xs...>, std::index_sequence<Is...>> {
    static constexpr long long value = ((static_cast<int>(Is) == Idx ? Xs : 0LL) + ... + 0LL);
};
template<int Idx, class V> struct Get;
template<int Idx, long long... Xs>
struct Get<Idx, Vec<Xs...>> {
    static constexpr long long value =
        GetImpl<Idx, Vec<Xs...>, std::make_index_sequence<sizeof...(Xs)>>::value;
};

// ---- Slice<Start, Len, Vec> -> Vec of Len elements starting at Start ----
template<int Start, class V, class Seq> struct SliceImpl;
template<int Start, class V, std::size_t... Is>
struct SliceImpl<Start, V, std::index_sequence<Is...>> {
    using type = Vec<Get<Start + static_cast<int>(Is), V>::value...>;
};
template<int Start, int Len, class V>
using Slice = typename SliceImpl<Start, V, std::make_index_sequence<Len>>::type;

// ---- Concat ----
template<class A, class B> struct Concat;
template<long long... As, long long... Bs>
struct Concat<Vec<As...>, Vec<Bs...>> { using type = Vec<As..., Bs...>; };

// ---- RowOf<I, Mat> -> the I-th row type ----
template<int I, class M> struct RowOf;
template<int I, class R0, class... Rs>
struct RowOf<I, Mat<R0, Rs...>> { using type = typename RowOf<I - 1, Mat<Rs...>>::type; };
template<class R0, class... Rs>
struct RowOf<0, Mat<R0, Rs...>> { using type = R0; };

// ---- elementwise vector ops (saturating add/sub, scale) ----
template<class A, class B> struct VecAdd;
template<long long... As, long long... Bs>
struct VecAdd<Vec<As...>, Vec<Bs...>> { using type = Vec<FxSat<As + Bs>::value...>; };

template<class A, class B> struct VecSub;
template<long long... As, long long... Bs>
struct VecSub<Vec<As...>, Vec<Bs...>> { using type = Vec<FxSat<As - Bs>::value...>; };

template<long long Sc, class V> struct VecScale;
template<long long Sc, long long... Xs>
struct VecScale<Sc, Vec<Xs...>> { using type = Vec<FxMul<Xs, Sc>::value...>; };

// ---- Dot: raw accumulate, one round+shift+sat at the end (fx_dot) ----
template<class A, class B> struct Dot;
template<long long... As, long long... Bs>
struct Dot<Vec<As...>, Vec<Bs...>> {
    static constexpr long long acc   = ((As * Bs) + ... + 0LL);
    static constexpr long long value = FxSat<((acc + FX_HALF) >> FX_FRAC)>::value;
};

// ---- MatVec: y = W @ x, one Dot per row ----
template<class X, class M> struct MatVec;
template<class X, class... Rows>
struct MatVec<X, Mat<Rows...>> { using type = Vec<Dot<X, Rows>::value...>; };

// ---- map a scalar metafunction F<long long>::value over a Vec ----
template<template<long long> class F, class V> struct MapElem;
template<template<long long> class F, long long... Xs>
struct MapElem<F, Vec<Xs...>> { using type = Vec<F<Xs>::value...>; };

// ---- reductions: max, argmax (first-max wins, matching Python) ----
template<class V> struct VecMax;
template<long long X> struct VecMax<Vec<X>> { static constexpr long long value = X; };
template<long long X0, long long... Xs>
struct VecMax<Vec<X0, Xs...>> {
    static constexpr long long rest  = VecMax<Vec<Xs...>>::value;
    static constexpr long long value = X0 > rest ? X0 : rest;
};

template<int I, int BestI, long long BestV, class V> struct ArgMaxImpl;
template<int I, int BestI, long long BestV>
struct ArgMaxImpl<I, BestI, BestV, Vec<>> { static constexpr int value = BestI; };
template<int I, int BestI, long long BestV, long long X0, long long... Xs>
struct ArgMaxImpl<I, BestI, BestV, Vec<X0, Xs...>> {
    static constexpr bool take = X0 > BestV;                 // strict -> first max wins
    static constexpr int  value =
        ArgMaxImpl<I + 1, take ? I : BestI, take ? X0 : BestV, Vec<Xs...>>::value;
};
template<class V> struct ArgMax;
template<long long X0, long long... Xs>
struct ArgMax<Vec<X0, Xs...>> { static constexpr int value = ArgMaxImpl<1, 0, X0, Vec<Xs...>>::value; };

// ---- exp(z) for z <= 0, via LUT + linear interpolation (fx_exp_neg) ----
template<long long Z>
struct ExpNeg {
    static_assert(Z <= 0, "exp_neg requires z <= 0");
    static constexpr long long t    = -Z;
    static constexpr long long idx  = t >> 12;               // LUT step = 2^12
    static constexpr long long frac = t & 4095;
    static constexpr long long lo   = idx >= 800 ? 0 : EXP_LUT[idx];
    static constexpr long long hi   = idx >= 800 ? 0 : EXP_LUT[idx + 1];
    static constexpr long long value = idx >= 800 ? 0 : lo - ((lo - hi) * frac) / 4096;
};

// ---- softmax: subtract max (=> exp args <= 0), raw-sum, divide (fx_softmax) ----
template<class V> struct Softmax;
template<long long... Xs>
struct Softmax<Vec<Xs...>> {
    static constexpr long long m = VecMax<Vec<Xs...>>::value;
    static constexpr long long s = (ExpNeg<Xs - m>::value + ... + 0LL);
    using type = Vec<FxDiv<ExpNeg<Xs - m>::value, s>::value...>;
};

// ---- tanh (odd symmetry; only the needed branch is instantiated) ----
template<long long U> struct TanhPos {                       // U >= 0
    static constexpr long long e     = ExpNeg<-(U << 1)>::value;
    static constexpr long long value = FxDiv<(FX_ONE - e), (FX_ONE + e)>::value;
};
template<long long U, bool Neg = (U < 0)> struct Tanh;
template<long long U> struct Tanh<U, false> { static constexpr long long value =  TanhPos<U>::value; };
template<long long U> struct Tanh<U, true>  { static constexpr long long value = -TanhPos<-U>::value; };

// ---- GELU (tanh approximation), matching fx_gelu's exact sat/shift points ----
template<long long X> struct Gelu {
    static constexpr long long x2    = FxMul<X, X>::value;
    static constexpr long long x3    = FxMul<x2, X>::value;
    static constexpr long long inner = (GELU_A * x3 + FX_HALF) >> FX_FRAC;      // no sat (Python)
    static constexpr long long u     = X + inner;
    static constexpr long long u2    = (SQRT_2_PI * u + FX_HALF) >> FX_FRAC;    // no sat (Python)
    static constexpr long long t     = Tanh<u2>::value;
    static constexpr long long value = FxSat<((X * (FX_ONE + t) + FX_ONE) >> 17)>::value;
};

// ---- LayerNorm (mean/var raw sums, floor isqrt, per-element scale+shift) ----
template<class V, class G, class B, long long Eps> struct LayerNorm;
template<long long... Xs, long long... Gs, long long... Bs, long long Eps>
struct LayerNorm<Vec<Xs...>, Vec<Gs...>, Vec<Bs...>, Eps> {
    static constexpr int  n      = sizeof...(Xs);
    static constexpr long long sum    = (Xs + ... + 0LL);
    static constexpr long long mean   = FxDivInt<sum, n>::value;
    static constexpr long long varsum = ((((Xs - mean) * (Xs - mean) + FX_HALF) >> FX_FRAC) + ... + 0LL);
    static constexpr long long var    = FxDivInt<varsum, n>::value;
    static constexpr long long std0   = ISqrt<((var + Eps) << FX_FRAC)>::value;
    static constexpr long long stdv   = std0 == 0 ? 1 : std0;
    using type = Vec<
        FxSat<(((FxDiv<(Xs - mean), stdv>::value * Gs + FX_HALF) >> FX_FRAC) + Bs)>::value ...>;
};

// --- static-assert unit checks ---
static_assert(Get<2, Vec<10, 20, 30, 40>>::value == 30);
static_assert(std::is_same_v<Slice<1, 2, Vec<5, 6, 7, 8>>, Vec<6, 7>>);
static_assert(std::is_same_v<Concat<Vec<1, 2>, Vec<3>>::type, Vec<1, 2, 3>>);
static_assert(std::is_same_v<typename RowOf<1, Mat<Vec<1>, Vec<2>, Vec<3>>>::type, Vec<2>>);
static_assert(std::is_same_v<VecAdd<Vec<1, 2>, Vec<3, 4>>::type, Vec<4, 6>>);
static_assert(Dot<Vec<FX_ONE, FX_ONE>, Vec<FX_ONE, FX_ONE>>::value == 2 * FX_ONE);
static_assert(ArgMax<Vec<3, 9, 9, 2>>::value == 1);          // first max
static_assert(ExpNeg<0>::value == FX_ONE);                   // exp(0) == 1
static_assert(Gelu<0>::value == 0);                          // gelu(0) == 0

}  // namespace gpt2
