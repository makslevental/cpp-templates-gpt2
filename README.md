# cpp-template-gpt2

A toy GPT-2 forward pass that runs **entirely at compile time** in C++ via
classic template metaprogramming. The prompt is fixed at compile time and the
generated tokens are produced by the compiler — **you never run the binary**.

Inspired by [`gpt2.cmake`](https://github.com/AlpinDale/gpt2.cmake): where that
project runs GPT-2 in CMake's integer `math()` with Q16.16 fixed-point (CMake
has no floats), this one carries every value as a `long long` non-type template
parameter and does every operation as a template metafunction. No floating
point, no `constexpr` functions for the math — the arithmetic lives in the type
system and is evaluated as the compiler instantiates templates.

> **Disclaimer:** this is fully for the lulz. It is not fast, not useful, and
> not a sensible way to run a neural network. The point is that you *can*.

## Build it

```sh
python3 tools/gen_model.py          # (re)generate include/model.hpp + tables.hpp
cmake -S . -B build
cmake --build build                 # a clean build IS the result — see below
```

The forward pass is `using Generated = Run<GEN_N, PromptIds>::type;` in
`main.cpp`, gated by a `static_assert` against `Golden` (the bit-exact
reference emitted by `tools/gen_model.py`). **If the build succeeds, the
compiler ran the model and reproduced the reference** — no execution required.

### Configure the prompt at build time

The prompt is tokenized **at compile time** (char-level lookup into the vocab),
so you set it with a CMake variable — no regeneration, nothing to run:

```sh
cmake -S . -B build -DPROMPT="hello" -DGEN_N=6
cmake --build build
```

The vocab is 32 chars: space, `a`–`z`, and `. , ' ! ?`. Out-of-vocab characters,
an empty prompt, or exceeding the context window (`N_CTX = 16`) each produce a
clear `static_assert` error. The `Golden` gate stays active only for the default
prompt; for a custom prompt use `-DSHOW=ON` to reveal the output.

The weights are random (fixed seed), so outputs are deterministic but not
linguistic — the model echoes the prompt and settles on a repeated character:

| Prompt | `GEN_N` | Output |
|--------|--------:|--------|
| `hi` | 6 | `himmmmmm` |
| `hello` | 6 | `hellooooooo` |
| `no` | 6 | `nooooooo` |
| `yes` | 6 | `yesssssss` |
| `the cat` | 6 | `the cattttttt` |

Compute any prompt's output with the reference: `python3 tools/gen_model.py
--predict "hello" --n 6`.

### See the generated tokens at compile time

```sh
cmake -S . -B build_show -DPROMPT="hello" -DSHOW=ON
cmake --build build_show            # fails on purpose, printing the answer
```

Produces (in the compiler diagnostic):

```
incomplete type 'ShowIds<8, 5, 12, 12, 15, 15, 15, 15, 15, 15, 15>'
incomplete type 'ShowText<'h', 'e', 'l', 'l', 'o', 'o', 'o', 'o', 'o', 'o', 'o'>'
```

i.e. prompt `"hello"` continues to `hellooooooo`, read straight from the error
message.

## CI

`.github/workflows/ci.yml` builds on **ubuntu-latest** and **macos-latest**:
it regenerates the headers (and checks they match the committed output),
builds the default (golden gate), then builds several prompts through a
compiler-independent `-DEXPECT="<text>"` gate (which `static_assert`s the
generated text) — so a green build *is* a passing test — and cross-checks each
against `gen_model.py --predict`.

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
