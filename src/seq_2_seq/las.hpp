#include "attention.hpp"
#include "char_mlp.hpp"
#include "listener.hpp"
#include "speller.hpp"
#include "vocabulary.hpp"
#include <concepts>
#include <cstddef>
#include <string>

// LAS
constexpr size_t MAX_LEN = 256;

// Listener parameters
constexpr size_t INPUT_SAMPLE = 96;
constexpr size_t SAMPLE_SIZE = 40;
constexpr size_t LISTENER_LAYERS = 4;
constexpr size_t LISTENER_HIDDEN = 256;

// Speller parameters
constexpr size_t SPELLER_LAYERS = 2;
constexpr size_t SPELLER_HIDDEN = 512;

// MLP parameters
constexpr size_t MLP_HIDDEN = 256;

template <std::floating_point T, bool TRAIN_ENABLED = true> class LAS {

  Listener<T, INPUT_SAMPLE, SAMPLE_SIZE, LISTENER_LAYERS, LISTENER_HIDDEN,
           TRAIN_ENABLED>
      listener{};
  Speller<T, SPELLER_LAYERS, SPELLER_HIDDEN, VOCABULARY.size(), TRAIN_ENABLED>
      speller{};
  MLP<T, MLP_HIDDEN, 2 * SPELLER_HIDDEN, VOCABULARY.size()> mlp{};

  char prev_token = 42;
  std::array<T, SPELLER_HIDDEN> prev_context{};

  std::array<T, VOCABULARY.size()>
  char_mlp(const std::array<T, SPELLER_HIDDEN> &state,
           const std::array<T, SPELLER_HIDDEN> &context) {
    typename decltype(mlp)::in_t input;
    auto context_begin = std::copy(state.begin(), state.end(), input.begin());
    std::copy(context.begin(), context.end(), context_begin);
    return mlp.forward_pass(input);
  }

public:
  LAS() = default;
  LAS(const LAS &) = delete;
  LAS(LAS &&) = delete;
  LAS &operator=(const LAS &) = delete;
  LAS &operator=(LAS &&) = delete;

  std::string
  inference(const std::array<std::array<T, SAMPLE_SIZE>, INPUT_SAMPLE> &input) {
    prev_token = 42;
    prev_context = {};
    // Result initialization
    std::string res{};
    res.reserve(SPELLER_HIDDEN);

    // Reset the recurrent state for this utterance.
    listener.reset_state();
    speller.reset_state();
    auto enc = listener.forward_pass(input);

    size_t counter = 0;
    while (prev_token != 41 && ++counter < MAX_LEN) {
      std::array<T, SPELLER_HIDDEN> new_state =
          speller.forward_pass(prev_token, prev_context);
      std::array<T, SPELLER_HIDDEN> new_context = attend(new_state, enc);
      std::array<T, VOCABULARY.size()> scores =
          char_mlp(new_state, new_context);
      size_t best = std::distance(
          scores.begin(), std::max_element(scores.begin(), scores.end()));

      if (best == token_to_index("<EOS>"))
        break;

      res += VOCABULARY[best];
      prev_token = best;
      prev_context = new_context;
    }

    return res;
  }
};
