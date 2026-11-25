#pragma once
#include "lstm.hpp"
#include <array>
#include <cstddef>

template <std::floating_point T, size_t LAYERS, size_t HIDDEN_STATES,
          size_t VOCABULARY_SIZE, bool TRAIN_ENABLED = true>
class Speller {
  using context_in_t = std::array<T, HIDDEN_STATES + VOCABULARY_SIZE>;
  using context_out_t = std::array<T, HIDDEN_STATES>;

  LSTM<T, VOCABULARY_SIZE + HIDDEN_STATES, HIDDEN_STATES, TRAIN_ENABLED> lstm_1;
  LSTM<T, HIDDEN_STATES, HIDDEN_STATES, TRAIN_ENABLED> lstm_2;

public:
  Speller() = default;
  Speller(const Speller &) = delete;
  Speller(Speller &&) = delete;
  Speller &operator=(const Speller &) = delete;
  Speller &operator=(Speller &&) = delete;

  void reset_state() {
    lstm_1.reset_state();
    lstm_2.reset_state();
  }

  context_out_t forward_pass(char prev_token,
                             const context_out_t &prev_context) {
    context_in_t input{}; // Already zero, so no need to zero on the one_hot
    input[prev_token] = 1;
    for (size_t i = 0; i < HIDDEN_STATES; ++i) {
      input[i + VOCABULARY_SIZE] = prev_context[i];
    }
    context_out_t hidden = lstm_1.forward_pass(input);
    return lstm_2.forward_pass(hidden);
  }
};
