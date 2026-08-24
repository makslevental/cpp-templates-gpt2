#!/usr/bin/env python3
"""Generate a toy GPT-2's weights + config as C++ headers for the compile-time
template-metaprogramming forward pass, plus an integer-exact reference forward
pass that produces the golden token ids.

The reference kernels below are ported verbatim from
~/dev_projects/gpt2.cmake/tools/gen_model.py so the C++ template evaluation must
match the golden ids bit-for-bit (Q16.16, truncating division, floor sqrt,
saturating ops, EXP LUT). Only the *emission* differs: we write C++ `Vec<...>` /
`Mat<...>` type aliases and constexpr tables instead of CMake `set()`.

Outputs:
  include/tables.hpp  -- EXP_LUT + GELU constants (constexpr data)
  include/model.hpp   -- config, weights (Vec/Mat aliases), Golden ids, vocab
"""
import math
import os

S = 65536
FX = 16
MAX = 2147483647
MIN = -2147483648

# --- toy config (identical to the CMake reference) ---
N_EMBD = 16
N_HEAD = 4
N_LAYER = 2
N_CTX = 16
HD = N_EMBD // N_HEAD
VOCAB = " abcdefghijklmnopqrstuvwxyz.,'!?"  # 32 chars
assert len(VOCAB) == 32
EPS = 1  # 1e-5 in Q16.16
INV_SQRT_HD = round(S / math.sqrt(HD))  # 1/sqrt(head_dim) in Q16.16

PROMPT = "hi"
N_GEN = 6

# --- exact Q16.16 replicas of the CMake kernels ---
def sat(v):
    if v > MAX: return MAX
    if v < MIN: return MIN
    return v

def trunc_div(a, b):  # CMake / truncates toward zero
    q = abs(a) // abs(b)
    return -q if (a < 0) ^ (b < 0) else q

def rdiv(a, b):  # round half away from zero, matches fx_div
    return (a + b // 2) // b if a >= 0 else -((-a + b // 2) // b)

def mul(a, b):  # fx_mul (round-half-up, saturating)
    return sat((a * b + 32768) >> FX)

# EXP LUT identical to tables.cmake
LUT = [round(math.exp(-i / 16) * S) for i in range(801)]

def exp_neg(z):  # fx_exp_neg, z <= 0
    t = -z
    idx = t >> 12
    if idx >= 800:
        return 0
    frac = t & 4095
    lo = LUT[idx]; hi = LUT[idx + 1]
    return lo - ((lo - hi) * frac) // 4096

def dot(a, b):  # fx_dot: raw-sum accumulate, round once, sat at end
    acc = 0
    for x, y in zip(a, b):
        acc += x * y
    return sat((acc + 32768) >> FX)

def vec_add(a, b):  # fx_vec_add (saturating elementwise)
    return [sat(x + y) for x, y in zip(a, b)]

def softmax(xs):  # fx_softmax
    m = max(xs)
    es = [exp_neg(x - m) for x in xs]
    s = sum(es)
    return [rdiv(e << FX, s) for e in es]

def tanh_pos(u):  # fx_tanh, u >= 0
    e = exp_neg(-(u << 1))
    return rdiv((S - e) << FX, S + e)

def gelu(x):  # fx_gelu
    a = 52290; b = 2930
    x2 = sat((x * x + 32768) >> FX)
    x3 = sat((x2 * x + 32768) >> FX)
    u = x + ((b * x3 + 32768) >> FX)
    u2 = (a * u + 32768) >> FX
    t = tanh_pos(u2) if u2 >= 0 else -tanh_pos(-u2)
    return sat((x * (S + t) + 65536) >> 17)

def isqrt(n):
    if n < 2:
        return n
    x, y = n, 1
    while x > y:
        x = (x + y) // 2
        y = n // x
    return x

def layernorm(xs, gamma, beta, eps):  # fx_layernorm
    n = len(xs)
    mean = rdiv(sum(xs), n)
    d = [x - mean for x in xs]
    var = rdiv(sum((v * v + 32768) >> FX for v in d), n)
    std = isqrt((var + eps) << FX)
    if std == 0:
        std = 1
    y = [rdiv(v << FX, std) for v in d]
    return [sat(((y[i] * gamma[i] + 32768) >> FX) + beta[i]) for i in range(n)]

# --- weights (random, fixed seed) ---
import random
rng = random.Random(42)
def W(rows, cols, scale):
    return [[sat(round(rng.gauss(0, scale) * S)) for _ in range(cols)] for _ in range(rows)]

d = N_EMBD
wte = W(len(VOCAB), d, 0.1)          # [vocab, d]
wpe = W(N_CTX, d, 0.1)               # [ctx, d]
layers = []
resid_scale = 0.02 / math.sqrt(2 * N_LAYER)
for _ in range(N_LAYER):
    layers.append({
        "ln1_g": [sat(round(S))] * d,          # gamma init 1.0
        "ln1_b": [0] * d,
        "ln2_g": [sat(round(S))] * d,
        "ln2_b": [0] * d,
        "c_attn_w": W(3 * d, d, 0.02),
        "c_attn_b": [0] * (3 * d),
        "c_proj_w": W(d, d, resid_scale),
        "c_proj_b": [0] * d,
        "c_fc_w": W(4 * d, d, 0.02),
        "c_fc_b": [0] * (4 * d),
        "c_proj2_w": W(d, 4 * d, resid_scale),
        "c_proj2_b": [0] * d,
    })
lnf_g = [sat(round(S))] * d
lnf_b = [0] * d

# --- reference forward pass ---
def forward(tokens):
    pos = len(tokens) - 1
    # embeddings: x[p] = wte[t_p] + wpe[p]
    x = [vec_add(wte[tokens[p]], wpe[p]) for p in range(pos + 1)]
    for L in layers:
        # ln1 + qkv per position
        qkv = []
        for p in range(pos + 1):
            h = layernorm(x[p], L["ln1_g"], L["ln1_b"], EPS)
            qkv.append(vec_add([dot(L["c_attn_w"][r], h) for r in range(3 * d)], L["c_attn_b"]))
        # attention over all positions
        attn = []
        for p in range(pos + 1):
            heads = []
            for hh in range(N_HEAD):
                qh = [qkv[p][hh * HD + j] for j in range(HD)]
                scores = []
                for pp in range(p + 1):
                    kh = [qkv[pp][d + hh * HD + j] for j in range(HD)]
                    scores.append(mul(dot(qh, kh), INV_SQRT_HD))
                probs = softmax(scores)
                for j in range(HD):
                    acc = 0
                    for pp in range(p + 1):
                        acc += (probs[pp] * qkv[pp][2 * d + hh * HD + j]) >> FX
                    heads.append(sat(acc))
            attn.append(heads)  # [d]
        # c_proj + residual (all positions)
        x = [vec_add(x[p], vec_add([dot(L["c_proj_w"][r], attn[p]) for r in range(d)], L["c_proj_b"]))
             for p in range(pos + 1)]
        # ln2 + mlp + residual (all positions)
        x = [vec_add(x[p], vec_add([dot(L["c_proj2_w"][r], [gelu(v) for v in vec_add(
            [dot(L["c_fc_w"][rr], layernorm(x[p], L["ln2_g"], L["ln2_b"], EPS)) for rr in range(4 * d)],
            L["c_fc_b"])]) for r in range(d)], L["c_proj2_b"]))
             for p in range(pos + 1)]
    # final ln + lm head (tied wte)
    last = layernorm(x[pos], lnf_g, lnf_b, EPS)
    return [dot(wte[r], last) for r in range(len(VOCAB))]

def argmax(xs):
    return max(range(len(xs)), key=lambda i: xs[i])

def run_prompt(prompt, n):
    ids = [VOCAB.index(c) for c in prompt]
    for _ in range(n):
        ids.append(argmax(forward(ids)))
    return ids, "".join(VOCAB[i] for i in ids)

# --predict lets CI (and humans) compute the deterministic output for an
# arbitrary in-vocab prompt without rewriting the generated headers.
import argparse, sys
_ap = argparse.ArgumentParser(description="toy GPT-2 weight generator / predictor")
_ap.add_argument("--predict", metavar="PROMPT",
                 help="print the greedy continuation of PROMPT and exit (no file writes)")
_ap.add_argument("--n", type=int, default=N_GEN, help="tokens to generate")
_args = _ap.parse_args()
if _args.predict is not None:
    _ids, _text = run_prompt(_args.predict, _args.n)
    print(f"prompt: {_args.predict!r}")
    print(f"ids: {_ids}")
    print(f"text: {_text!r}")
    sys.exit(0)

# --- C++ emission helpers ---
def vec_cpp(vals):
    return "Vec<" + ", ".join(str(v) for v in vals) + ">"

def mat_cpp(rows):
    return "Mat<\n    " + ",\n    ".join(vec_cpp(r) for r in rows) + ">"

out_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "include"))
os.makedirs(out_dir, exist_ok=True)

# --- tables.hpp ---
tbl = [
    "// tables.hpp -- generated by tools/gen_model.py. Do not edit.",
    "// Q16.16 lookup data: exp(-t) LUT (step 1/16) + GELU constants.",
    "#pragma once",
    "namespace gpt2 {",
    f"inline constexpr int  EXP_LUT_N = {len(LUT)};",
    "inline constexpr long long EXP_LUT[EXP_LUT_N] = {",
]
# 12 entries per line for readability
for i in range(0, len(LUT), 12):
    tbl.append("    " + ", ".join(str(v) for v in LUT[i:i + 12]) + ",")
tbl += [
    "};",
    "inline constexpr long long SQRT_2_PI = 52290;  // sqrt(2/pi) in Q16.16",
    "inline constexpr long long GELU_A    = 2930;   // 0.044715 in Q16.16",
    "}  // namespace gpt2",
    "",
]
with open(os.path.join(out_dir, "tables.hpp"), "w") as f:
    f.write("\n".join(tbl))

# --- model.hpp ---
def esc(c):
    if c == "\\": return "\\\\"
    if c == "'": return "\\'"
    return c

m = [
    "// model.hpp -- generated by tools/gen_model.py. Do not edit.",
    "// Toy GPT-2 weights (Q16.16) + config as template-parameter data.",
    "#pragma once",
    '#include "tensor.hpp"',
    "namespace gpt2 {",
    "",
    f"inline constexpr int N_EMBD  = {N_EMBD};",
    f"inline constexpr int N_HEAD  = {N_HEAD};",
    f"inline constexpr int N_LAYER = {N_LAYER};",
    f"inline constexpr int N_CTX   = {N_CTX};",
    f"inline constexpr int N_VOCAB = {len(VOCAB)};",
    f"inline constexpr int HD      = {HD};",
    f"inline constexpr long long EPS         = {EPS};",
    f"inline constexpr long long INV_SQRT_HD = {INV_SQRT_HD};",
    f'inline constexpr char VOCAB[] = "{"".join(esc(c) for c in VOCAB)}";',
    "",
    "// token + positional embeddings (rows indexed via RowOf)",
    f"using WTE = {mat_cpp(wte)};",
    f"using WPE = {mat_cpp(wpe)};",
    "",
]
# per-layer weight bundles
for l, L in enumerate(layers):
    m.append(f"// ---- layer {l} ----")
    m.append(f"using L{l}_CATTN  = {mat_cpp(L['c_attn_w'])};")
    m.append(f"using L{l}_CATTNB = {vec_cpp(L['c_attn_b'])};")
    m.append(f"using L{l}_CPROJ  = {mat_cpp(L['c_proj_w'])};")
    m.append(f"using L{l}_CPROJB = {vec_cpp(L['c_proj_b'])};")
    m.append(f"using L{l}_FC     = {mat_cpp(L['c_fc_w'])};")
    m.append(f"using L{l}_FCB    = {vec_cpp(L['c_fc_b'])};")
    m.append(f"using L{l}_MPROJ  = {mat_cpp(L['c_proj2_w'])};")
    m.append(f"using L{l}_MPROJB = {vec_cpp(L['c_proj2_b'])};")
    m.append(f"using L{l}_LN1G   = {vec_cpp(L['ln1_g'])};")
    m.append(f"using L{l}_LN1B   = {vec_cpp(L['ln1_b'])};")
    m.append(f"using L{l}_LN2G   = {vec_cpp(L['ln2_g'])};")
    m.append(f"using L{l}_LN2B   = {vec_cpp(L['ln2_b'])};")
    m.append("")
m.append(f"using LNF_G = {vec_cpp(lnf_g)};")
m.append(f"using LNF_B = {vec_cpp(lnf_b)};")
m.append("")

# per-layer accessor: LayerW<l>::CATTN, etc.
m.append("template<int L> struct LayerW;")
for l in range(N_LAYER):
    m.append(f"template<> struct LayerW<{l}> {{")
    for name in ("CATTN", "CATTNB", "CPROJ", "CPROJB", "FC", "FCB",
                 "MPROJ", "MPROJB", "LN1G", "LN1B", "LN2G", "LN2B"):
        m.append(f"  using {name} = L{l}_{name};")
    m.append("};")
m.append("")

# golden ids + text (default prompt)
ids, text = run_prompt(PROMPT, N_GEN)
prompt_ids = [VOCAB.index(c) for c in PROMPT]

m.append(f"// default prompt {PROMPT!r} -> greedy-generate {N_GEN} tokens")
m.append(f'inline constexpr char DEFAULT_PROMPT[] = "{"".join(esc(c) for c in PROMPT)}";')
m.append(f"using DefaultPromptIds = Vec<{', '.join(str(i) for i in prompt_ids)}>;")
m.append(f"inline constexpr int N_GEN = {N_GEN};")
m.append(f"using Golden = Vec<{', '.join(str(i) for i in ids)}>;  // includes prompt")
m.append(f'inline constexpr char GOLDEN_TEXT[] = "{"".join(esc(c) for c in text)}";')
m.append("}  // namespace gpt2")
m.append("")

with open(os.path.join(out_dir, "model.hpp"), "w") as f:
    f.write("\n".join(m))

print("wrote include/tables.hpp and include/model.hpp")
print(f"golden ids:  {ids}")
print(f"golden text: {text!r}")
