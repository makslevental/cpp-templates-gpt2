// gpt2.hpp -- the GPT-2 forward pass as metafunctions over Mat/Vec state.
//
// Activations are a Mat whose row p is the [N_EMBD] vector at position p.
// Structure mirrors tools/gen_model.py's forward() exactly:
//   embed -> N_LAYER x (ln1, c_attn, causal MHA, c_proj+res, ln2, mlp+res)
//         -> final ln -> tied LM head -> argmax (greedy).
// Attention gathers K/V from earlier positions with RowOf; the value
// aggregation floor-shifts each term (not Dot's deferred shift) to stay exact.
#pragma once
#include <utility>
#include "model.hpp"

namespace gpt2 {

// map a unary metafunction over the rows of a Mat (F<Row>::type)
template<template<class> class F, class M> struct MapRows;
template<template<class> class F, class... Rows>
struct MapRows<F, Mat<Rows...>> { using type = Mat<typename F<Rows>::type...>; };

// map a binary metafunction over two Mats' rows in lockstep (F<A,B>::type)
template<template<class, class> class F, class MA, class MB> struct MapRows2;
template<template<class, class> class F, class... As, class... Bs>
struct MapRows2<F, Mat<As...>, Mat<Bs...>> { using type = Mat<typename F<As, Bs>::type...>; };

// concatenate a pack of Vecs
template<class... Vs> struct ConcatAll { using type = Vec<>; };
template<class V0, class... Vs> struct ConcatAll<V0, Vs...> {
    using type = typename Concat<V0, typename ConcatAll<Vs...>::type>::type;
};

// ---- embeddings: x[p] = WTE[id_p] + WPE[p] ----
template<class Seq, long long... Ids> struct EmbedImpl;
template<std::size_t... Ps, long long... Ids>
struct EmbedImpl<std::index_sequence<Ps...>, Ids...> {
    using type = Mat<typename VecAdd<
        typename RowOf<static_cast<int>(Ids), WTE>::type,
        typename RowOf<static_cast<int>(Ps),  WPE>::type>::type...>;
};
template<long long... Ids>
using Embed = typename EmbedImpl<std::make_index_sequence<sizeof...(Ids)>, Ids...>::type;

// ---- causal multi-head self-attention for one query position P ----
// scores[pp] = mul(dot(q_head, k_head_pp), INV_SQRT_HD), for pp in 0..P
template<int P, int HH, class QKV, class Seq> struct Scores;
template<int P, int HH, class QKV, std::size_t... PPs>
struct Scores<P, HH, QKV, std::index_sequence<PPs...>> {
    using qh = Slice<HH * HD, HD, typename RowOf<P, QKV>::type>;
    template<int PP> using kh = Slice<N_EMBD + HH * HD, HD, typename RowOf<PP, QKV>::type>;
    using type = Vec<FxMul<Dot<qh, kh<static_cast<int>(PPs)>>::value, INV_SQRT_HD>::value...>;
};

// head output element j: sat( sum_pp (probs[pp] * V[pp][j]) >> FX ), floor per term
template<int HH, int J, class QKV, class Probs, class SeqPP> struct AccJ;
template<int HH, int J, class QKV, class Probs, std::size_t... PPs>
struct AccJ<HH, J, QKV, Probs, std::index_sequence<PPs...>> {
    static constexpr long long acc =
        (((Get<static_cast<int>(PPs), Probs>::value *
           Get<2 * N_EMBD + HH * HD + J, typename RowOf<static_cast<int>(PPs), QKV>::type>::value)
          >> FX_FRAC) + ... + 0LL);
    static constexpr long long value = FxSat<acc>::value;
};

// one head -> Vec<HD>
template<int P, int HH, class QKV, class SeqJ> struct HeadOut;
template<int P, int HH, class QKV, std::size_t... Js>
struct HeadOut<P, HH, QKV, std::index_sequence<Js...>> {
    using scores = typename Scores<P, HH, QKV, std::make_index_sequence<P + 1>>::type;
    using probs  = typename Softmax<scores>::type;
    using type   = Vec<AccJ<HH, static_cast<int>(Js), QKV, probs, std::make_index_sequence<P + 1>>::value...>;
};

// all heads for position P concatenated -> Vec<N_EMBD>
template<int P, class QKV, class SeqH> struct AttnPos;
template<int P, class QKV, std::size_t... HHs>
struct AttnPos<P, QKV, std::index_sequence<HHs...>> {
    using type = typename ConcatAll<
        typename HeadOut<P, static_cast<int>(HHs), QKV, std::make_index_sequence<HD>>::type...>::type;
};

// attention for every position -> Mat
template<class QKV, class SeqP> struct AttnAll;
template<class QKV, std::size_t... Ps>
struct AttnAll<QKV, std::index_sequence<Ps...>> {
    using type = Mat<typename AttnPos<static_cast<int>(Ps), QKV, std::make_index_sequence<N_HEAD>>::type...>;
};

// ---- per-layer operations, bound to layer L's weights ----
template<int L> struct Ops {
    using W = LayerW<L>;
    // ln1 -> c_attn (+bias) -> qkv row [3*N_EMBD]
    template<class Row> struct Ln1Qkv {
        using h    = typename LayerNorm<Row, typename W::LN1G, typename W::LN1B, EPS>::type;
        using type = typename VecAdd<typename MatVec<h, typename W::CATTN>::type,
                                     typename W::CATTNB>::type;
    };
    // c_proj(attn) + bias + residual
    template<class Xrow, class Arow> struct CProjResid {
        using proj = typename VecAdd<typename MatVec<Arow, typename W::CPROJ>::type,
                                     typename W::CPROJB>::type;
        using type = typename VecAdd<Xrow, proj>::type;
    };
    // ln2 -> c_fc(+bias) -> gelu -> c_proj2(+bias) + residual
    template<class Row> struct MlpResid {
        using h2   = typename LayerNorm<Row, typename W::LN2G, typename W::LN2B, EPS>::type;
        using up   = typename VecAdd<typename MatVec<h2, typename W::FC>::type, typename W::FCB>::type;
        using ff   = typename MapElem<Gelu, up>::type;
        using down = typename VecAdd<typename MatVec<ff, typename W::MPROJ>::type, typename W::MPROJB>::type;
        using type = typename VecAdd<Row, down>::type;
    };
};

// ---- one transformer block: Mat -> Mat ----
template<int L, class X> struct Block {
    static constexpr int P = X::rows;
    using QKV  = typename MapRows<Ops<L>::template Ln1Qkv, X>::type;
    using Attn = typename AttnAll<QKV, std::make_index_sequence<P>>::type;
    using X1   = typename MapRows2<Ops<L>::template CProjResid, X, Attn>::type;
    using type = typename MapRows<Ops<L>::template MlpResid, X1>::type;
};

// fold blocks over the layers
template<int L, class X, bool Done = (L >= N_LAYER)> struct RunBlocks {
    using type = typename RunBlocks<L + 1, typename Block<L, X>::type>::type;
};
template<int L, class X> struct RunBlocks<L, X, true> { using type = X; };

// ---- logits for the next token: embed -> blocks -> final ln -> tied LM head ----
template<long long... Ids> struct Logits {
    using X0      = Embed<Ids...>;
    using XL      = typename RunBlocks<0, X0>::type;
    static constexpr int last = static_cast<int>(sizeof...(Ids)) - 1;
    using lastRow = typename RowOf<last, XL>::type;
    using ln      = typename LayerNorm<lastRow, LNF_G, LNF_B, EPS>::type;
    using type    = typename MatVec<ln, WTE>::type;          // Vec<N_VOCAB>
};

// ---- greedy autoregressive generation ----
template<int N, long long... Ids> struct Generate {
    static constexpr int nxt = ArgMax<typename Logits<Ids...>::type>::value;
    using type = typename Generate<N - 1, Ids..., nxt>::type;
};
template<long long... Ids> struct Generate<0, Ids...> { using type = Vec<Ids...>; };

// unpack a prompt Vec and generate N tokens
template<int N, class Ids> struct Run;
template<int N, long long... Ids> struct Run<N, Vec<Ids...>> {
    using type = typename Generate<N, Ids...>::type;
};

}  // namespace gpt2
