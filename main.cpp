// main.cpp -- run the toy GPT-2 forward pass entirely at compile time.
//
// The prompt is fixed at compile time (PromptIds in model.hpp, from tools/
// gen_model.py). The generated token ids are computed by the compiler as it
// instantiates Run<>. The static_assert below is the proof: if it passes, the
// compile-time evaluation reproduced the bit-exact golden reference. You never
// need to run the resulting binary.
//
// Build normally to verify:        cmake --build build
// See the generated ids/text with: cmake -S . -B build -DSHOW=ON && cmake --build build
#include "model.hpp"
#include "gpt2.hpp"

using namespace gpt2;

// The entire forward pass + greedy decode, evaluated by the compiler:
using Generated = Run<N_GEN, PromptIds>::type;

// Proof of correctness at compile time (bit-exact vs the Python reference).
static_assert(std::is_same_v<Generated, Golden>,
              "compile-time GPT-2 output does not match the golden reference");

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
