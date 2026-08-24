// main.cpp -- run the toy GPT-2 forward pass entirely at compile time.
//
// The prompt is fixed at compile time and the generated token ids are computed
// by the compiler as it instantiates Run<>. You never need to run the binary.
//
// Configure the prompt/length at BUILD time:
//   cmake -S . -B build -DPROMPT="hello" -DGEN_N=6
//   cmake --build build            # a clean build == the model ran (see below)
//
// See the generated ids/text (prints them in a compiler diagnostic):
//   cmake -S . -B build_show -DPROMPT="hello" -DSHOW=ON
//   cmake --build build_show
#include "model.hpp"
#include "gpt2.hpp"

using namespace gpt2;

// Build-time knobs (defaults match tools/gen_model.py so the golden gate below
// stays meaningful when nothing is overridden).
#ifndef PROMPT
#define PROMPT "hi"        // keep in sync with gen_model.py's PROMPT default
#endif
#ifndef GEN_N
#define GEN_N N_GEN        // default: the generator's token count
#endif

// ---- compile-time char-level tokenizer: prompt string -> Vec of vocab ids ----
namespace {
constexpr long long char_to_id(char c) {
    for (int i = 0; i < N_VOCAB; ++i)
        if (VOCAB[i] == c) return i;
    return -1;  // not in vocab
}
constexpr int PROMPT_LEN = sizeof(PROMPT) - 1;   // string-literal length

template<class Seq> struct PromptTok;
template<std::size_t... Is>
struct PromptTok<std::index_sequence<Is...>> {
    static_assert(((char_to_id(PROMPT[Is]) >= 0) && ...),
                  "PROMPT has a character outside the toy vocab (space a-z . , ' ! ?)");
    using type = Vec<char_to_id(PROMPT[Is])...>;
};
}  // namespace

static_assert(PROMPT_LEN >= 1, "PROMPT must be non-empty");
static_assert(PROMPT_LEN + (GEN_N) <= N_CTX,
              "PROMPT length + GEN_N exceeds the context window (N_CTX = 16)");

using PromptIds = PromptTok<std::make_index_sequence<PROMPT_LEN>>::type;

// The entire forward pass + greedy decode, evaluated by the compiler:
using Generated = Run<GEN_N, PromptIds>::type;

// Proof of correctness at compile time: for the DEFAULT prompt/length we have a
// bit-exact golden reference from tools/gen_model.py. For a custom prompt there
// is no reference, so the gate is vacuously true and -DSHOW reveals the output.
static_assert(!(std::is_same_v<PromptIds, DefaultPromptIds> && (GEN_N) == N_GEN)
                  || std::is_same_v<Generated, Golden>,
              "default-prompt output does not match the golden reference");

#ifdef SHOW
// Surface the result in a compiler diagnostic without running anything:
// sizeof on an incomplete type prints its full name, so you read the answer
// straight from the error, e.g.
//   incomplete type 'ShowIds<8, 9, 13, ...>'
//   incomplete type 'ShowText<'h', 'i', 'm', 'm', 'm', 'm', 'm', 'm'>'
template<long long...> struct ShowIds;   // deliberately undefined
template<char...>      struct ShowText;  // deliberately undefined
template<class V> struct Show;
template<long long... Xs> struct Show<Vec<Xs...>> {
    using ids  = ShowIds<Xs...>;
    using text = ShowText<VOCAB[Xs]...>;   // decode id -> vocab char at compile time
};
constexpr int _dump_ids  = sizeof(Show<Generated>::ids);
constexpr int _dump_text = sizeof(Show<Generated>::text);
#endif

int main() { return 0; }
