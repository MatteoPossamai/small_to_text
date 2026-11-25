#pragma once

#include "attention.hpp"
#include "char_mlp.hpp"
#include "constant.hpp"
#include "listener.hpp"
#include "speller.hpp"
#include "vocabulary.hpp"
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>

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
