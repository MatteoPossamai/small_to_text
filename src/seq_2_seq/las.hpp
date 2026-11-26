#pragma once

#include "attention.hpp"
#include "char_mlp.hpp"
#include "constant.hpp"
#include "listener.hpp"
#include "speller.hpp"
#include "vocabulary.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

// LAS
constexpr size_t MAX_LEN = 256;

// Listener parameters
constexpr size_t SAMPLE_SIZE = 40;
constexpr size_t LISTENER_LAYERS = 4;
constexpr size_t LISTENER_HIDDEN = 256;

// Speller parameters
constexpr size_t SPELLER_LAYERS = 2;
constexpr size_t SPELLER_HIDDEN = 512;

// MLP parameters
constexpr size_t MLP_HIDDEN = 256;

template <std::floating_point T, bool TRAIN_ENABLED = true> class LAS {
  using listener_t =
      Listener<T, SAMPLE_SIZE, LISTENER_LAYERS, LISTENER_HIDDEN, TRAIN_ENABLED>;
  using speller_t = Speller<T, SPELLER_LAYERS, SPELLER_HIDDEN,
                            VOCABULARY.size(), TRAIN_ENABLED>;
  using mlp_t = MLP<T, MLP_HIDDEN, 2 * SPELLER_HIDDEN, VOCABULARY.size()>;
  using state_t = std::array<T, SPELLER_HIDDEN>;
  using audio_t = std::span<const std::array<T, SAMPLE_SIZE>>;

  listener_t listener;
  speller_t speller;
  mlp_t mlp;

  static std::filesystem::path
  create_checkpoint(const std::filesystem::path &directory) {
    std::filesystem::create_directories(directory);
    auto id = static_cast<std::uint64_t>(
        std::chrono::system_clock::now().time_since_epoch().count());
    for (;;) {
      const auto checkpoint =
          directory /
          (std::string(las_weights::checkpoint_prefix) + std::to_string(id++));
      if (std::filesystem::create_directory(checkpoint))
        return checkpoint;
    }
  }

  static std::filesystem::path
  resolve_checkpoint(const std::filesystem::path &directory) {
    const auto current = directory / las_weights::current_filename;
    if (!std::filesystem::exists(current))
      return directory;

    std::ifstream file(current);
    std::string name;
    const auto prefix = las_weights::checkpoint_prefix;
    if (!(file >> name) || !name.starts_with(prefix) ||
        name.size() == prefix.size() ||
        name.find_first_not_of("0123456789", prefix.size()) !=
            std::string::npos)
      throw std::runtime_error("Invalid LAS checkpoint reference in " +
                               current.string());
    file >> std::ws;
    if (file.bad() || !file.eof())
      throw std::runtime_error("Invalid LAS checkpoint reference in " +
                               current.string());
    return directory / name;
  }

  explicit LAS(const std::filesystem::path &checkpoint)
      : listener(checkpoint), speller(checkpoint) {
    las_weights::load_network(mlp, checkpoint / las_weights::mlp_filename);
  }

  static typename mlp_t::in_t character_input(const state_t &state,
                                              const state_t &context) {
    typename mlp_t::in_t input;
    auto context_begin = std::copy(state.begin(), state.end(), input.begin());
    std::copy(context.begin(), context.end(), context_begin);
    return input;
  }

public:
  LAS() = default;
  LAS(const LAS &) = delete;
  LAS(LAS &&) = delete;
  LAS &operator=(const LAS &) = delete;
  LAS &operator=(LAS &&) = delete;

  static std::unique_ptr<LAS> load_weights(
      const std::filesystem::path &directory = las_weights::directory) {
    return std::unique_ptr<LAS>(new LAS(resolve_checkpoint(directory)));
  }

  void save_weights(
      const std::filesystem::path &directory = las_weights::directory) const {
    const auto checkpoint = create_checkpoint(directory);
    try {
      listener.save_weights(checkpoint);
      speller.save_weights(checkpoint);
      las_weights::save_network(mlp, checkpoint / las_weights::mlp_filename);

      // Publish the checkpoint only after every network file has been closed.
      const auto pending = checkpoint / las_weights::current_filename;
      std::ofstream current(pending);
      current << checkpoint.filename().string() << '\n';
      current.close();
      if (!current)
        throw std::runtime_error("Could not write LAS checkpoint reference " +
                                 pending.string());
      std::filesystem::rename(pending,
                              directory / las_weights::current_filename);
    } catch (...) {
      std::error_code ignored;
      std::filesystem::remove_all(checkpoint, ignored);
      throw;
    }
  }

  std::string inference(audio_t input, size_t max_tokens = MAX_LEN - 1) {
    if (input.empty())
      throw std::invalid_argument("LAS requires at least one audio frame");
    listener.reset_state();
    speller.reset_state();
    const auto rows = listener.forward_pass(input);

    char prev_token = token_to_index("<SOS>");
    state_t context{};
    std::string result;
    for (size_t step = 0; step < max_tokens; ++step) {
      const auto state = speller.forward_pass(prev_token, context);
      context = attend(state, rows);
      const auto scores = mlp.forward_pass(character_input(state, context));
      const auto best = static_cast<size_t>(
          std::max_element(scores.begin(), scores.end()) - scores.begin());
      if (best == token_to_index("<EOS>"))
        break;
      result += VOCABULARY[best];
      prev_token = static_cast<char>(best);
    }
    return result;
  }

  // Train on one utterance, feeding the correct previous character each step.
  T train_utterance(audio_t input, std::string_view transcript, T learning_rate)
    requires(TRAIN_ENABLED)
  {
    if (input.empty())
      throw std::invalid_argument("LAS requires at least one audio frame");
    if (!std::isfinite(learning_rate) || learning_rate <= T{0})
      throw std::invalid_argument("Learning rate must be finite and positive");
    listener.reset_state();
    speller.reset_state();
    mlp.reset_state();
    const auto rows = listener.forward_pass_train(input);

    struct CharacterStep {
      size_t target;
      state_t state;
      std::vector<T> attention_weights;
    };

    char prev_token = token_to_index("<SOS>");
    state_t context{};
    std::vector<CharacterStep> steps;
    steps.reserve(transcript.size() + 1);
    for (size_t i = 0; i <= transcript.size(); ++i) {
      const size_t target = i == transcript.size()
                                ? token_to_index("<EOS>")
                                : token_to_index(transcript.substr(i, 1));
      steps.push_back(
          {target, speller.forward_pass_train(prev_token, context), {}});
      auto &step = steps.back();
      context = attend(step.state, rows, step.attention_weights);
      mlp.forward_pass_train(character_input(step.state, context));
      prev_token = static_cast<char>(target);
    }

    // Each context is used by the character MLP and by the following LSTM step.
    state_t context_gradient{};
    typename speller_t::BackwardState speller_gradient{};
    std::vector<typename listener_t::output_row> row_gradients(rows.size());
    T loss = 0;
    for (auto step = steps.rbegin(); step != steps.rend(); ++step) {
      const auto character = mlp.backward_pass(step->target);
      loss += character.loss;
      for (size_t i = 0; i < SPELLER_HIDDEN; ++i)
        context_gradient[i] += character.input_gradient[SPELLER_HIDDEN + i];

      auto state_gradient =
          attend_backward(step->state, rows, step->attention_weights,
                          context_gradient, row_gradients);
      for (size_t i = 0; i < SPELLER_HIDDEN; ++i)
        state_gradient[i] += character.input_gradient[i];
      context_gradient =
          speller.backward_pass(state_gradient, speller_gradient);
    }
    if (!std::isfinite(loss))
      throw std::runtime_error("LAS training produced a non-finite loss");

    listener.backward_pass(row_gradients);
    const T token_count = static_cast<T>(steps.size());
    const T step_size = learning_rate / token_count;
    listener.apply_gradient(step_size);
    speller.apply_gradient(step_size);
    mlp.apply_gradient(step_size);
    return loss / token_count;
  }
};
