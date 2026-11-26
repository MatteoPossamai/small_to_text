#pragma once
#include "constant.hpp"
#include "lstm.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <filesystem>
#include <stdexcept>

template <std::floating_point T, size_t LAYERS, size_t HIDDEN_STATES,
          size_t VOCABULARY_SIZE, bool TRAIN_ENABLED = true>
class Speller {
  using context_in_t = std::array<T, HIDDEN_STATES + VOCABULARY_SIZE>;
  using context_out_t = std::array<T, HIDDEN_STATES>;

  LSTM<T, VOCABULARY_SIZE + HIDDEN_STATES, HIDDEN_STATES, TRAIN_ENABLED> lstm_1;
  LSTM<T, HIDDEN_STATES, HIDDEN_STATES, TRAIN_ENABLED> lstm_2;

  static context_in_t make_input(char prev_token,
                                 const context_out_t &prev_context) {
    const auto token_index = static_cast<size_t>(prev_token);
    if (token_index >= VOCABULARY_SIZE)
      throw std::out_of_range("Speller token is outside the vocabulary");

    context_in_t input{};
    input[token_index] = T{1};
    std::copy(prev_context.begin(), prev_context.end(),
              input.begin() + VOCABULARY_SIZE);
    return input;
  }

public:
  struct BackwardState {
    context_out_t first_hidden{}, first_cell{};
    context_out_t second_hidden{}, second_cell{};
  };

  Speller() = default;
  explicit Speller(const std::filesystem::path &directory) : Speller() {
    las_weights::load_network(lstm_1,
                              directory / las_weights::speller_filenames[0]);
    las_weights::load_network(lstm_2,
                              directory / las_weights::speller_filenames[1]);
  }

  Speller(const Speller &) = delete;
  Speller(Speller &&) = delete;
  Speller &operator=(const Speller &) = delete;
  Speller &operator=(Speller &&) = delete;

  void save_weights(const std::filesystem::path &directory) const {
    las_weights::save_network(lstm_1,
                              directory / las_weights::speller_filenames[0]);
    las_weights::save_network(lstm_2,
                              directory / las_weights::speller_filenames[1]);
  }

  void reset_state() {
    lstm_1.reset_state();
    lstm_2.reset_state();
  }

  context_out_t forward_pass(char prev_token,
                             const context_out_t &prev_context) {
    const auto hidden =
        lstm_1.forward_pass(make_input(prev_token, prev_context));
    return lstm_2.forward_pass(hidden);
  }

  context_out_t forward_pass_train(char prev_token,
                                   const context_out_t &prev_context)
    requires(TRAIN_ENABLED)
  {
    const auto hidden =
        lstm_1.forward_pass_train(make_input(prev_token, prev_context));
    return lstm_2.forward_pass_train(hidden);
  }

  // Start with a zero BackwardState and call once per step in reverse order.
  context_out_t backward_pass(const context_out_t &state_gradient,
                              BackwardState &future)
    requires(TRAIN_ENABLED)
  {
    // Each layer receives a gradient from this step and its next time step.
    for (size_t i = 0; i < HIDDEN_STATES; ++i)
      future.second_hidden[i] += state_gradient[i];
    const auto second =
        lstm_2.backward_pass(future.second_hidden, future.second_cell);
    future.second_hidden = second.previous_hidden_gradient;
    future.second_cell = second.previous_cell_gradient;

    for (size_t i = 0; i < HIDDEN_STATES; ++i)
      future.first_hidden[i] += second.input_gradient[i];
    const auto first =
        lstm_1.backward_pass(future.first_hidden, future.first_cell);
    future.first_hidden = first.previous_hidden_gradient;
    future.first_cell = first.previous_cell_gradient;

    // The input is the fixed one-hot token followed by the previous context.
    context_out_t previous_context_gradient;
    std::copy_n(first.input_gradient.begin() + VOCABULARY_SIZE, HIDDEN_STATES,
                previous_context_gradient.begin());
    return previous_context_gradient;
  }

  void apply_gradient(T learning_rate)
    requires(TRAIN_ENABLED)
  {
    lstm_1.apply_gradient(learning_rate);
    lstm_2.apply_gradient(learning_rate);
  }
};
