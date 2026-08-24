# cpp-template-gpt2

A toy GPT-2 forward pass that runs **entirely at compile time** in C++ via
classic template metaprogramming. The prompt is fixed at compile time and the
generated tokens are produced by the compiler — **you never run the binary**.

It is the C++ analogue of [`gpt2.cmake`](../gpt2.cmake): where that project uses
CMake's integer `math()` with Q16.16 fixed-point (CMake has no floats), this one
carries every value as a `long long` non-type template parameter and does every
operation as a template metafunction. No floating point, no `constexpr`
functions for the math — the arithmetic lives in the type system and is
evaluated as the compiler instantiates templates.

## Run it

```sh
python3 tools/gen_model.py          # (re)generate include/model.hpp + tables.hpp
cmake -S . -B build
cmake --build build                 # a clean build IS the result — see below
```

The forward pass is `using Generated = Run<N_GEN, PromptIds>::type;` in
`main.cpp`, gated by:

```cpp
static_assert(std::is_same_v<Generated, Golden>, "...");
```

`Golden` is the bit-exact reference output emitted by `tools/gen_model.py`. **If
the build succeeds, the compiler ran the model and reproduced the reference** —
no execution required.

### See the generated tokens at compile time

```sh
cmake -S . -B build_show -DSHOW=ON
cmake --build build_show            # fails on purpose, printing the answer
```

Produces (in the compiler diagnostic):

```
incomplete type 'ShowIds<8, 9, 13, 13, 13, 13, 13, 13>'
incomplete type 'ShowText<'h', 'i', 'm', 'm', 'm', 'm', 'm', 'm'>'
```

i.e. prompt `"hi"` greedily continues to `himmmmmm`, read straight from the
error message. Change the prompt in `tools/gen_model.py`, regenerate, and the
ids change — all at compile time.

## How it works

| Layer | File | Idea |
|-------|------|------|
| scalars | `include/fixed.hpp` | Q16.16 arithmetic as metafunctions (`FxMul`, `FxDiv`, `ISqrt`/`FxSqrt`, saturation). Rounding matches the reference exactly. |
| tensors | `include/tensor.hpp` | `Vec<long long...>` / `Mat<Vec...>`; `Dot`, `MatVec`, `Softmax`, `Gelu`, `LayerNorm`, `ExpNeg` (LUT + interpolation). O(1)-depth indexing via masked folds. |
| model | `include/gpt2.hpp` | Embeddings, causal multi-head attention (K/V gathered across positions with `RowOf`), residual MLP, tied LM head, greedy `Generate`. |
| data | `include/model.hpp`, `include/tables.hpp` | Generated: config, weights as `Vec`/`Mat` aliases, the EXP LUT, and `Golden`. |
| driver | `main.cpp` | Instantiates the forward pass; `static_assert` gate + `-DSHOW` printer. |

The single source of truth for numerics is `tools/gen_model.py`, ported
bit-for-bit from the CMake project's generator (same Q16.16 kernels, EXP LUT,
truncating division, floor sqrt, saturation). The toy config is tiny
(`N_EMBD=16, N_HEAD=4, N_LAYER=2, N_VOCAB=32`), so the whole thing compiles in
well under a second.

## License

BSD 3-Clause (matching the upstream `gpt2.cmake`).
