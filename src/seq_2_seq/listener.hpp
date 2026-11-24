#include "lstm.hpp"
#include <array>
#include <cstddef>
#include <cstring>
#include <utility>

template <std::floating_point T, size_t INPUT_SAMPLE, size_t SAMPLE_SIZE,
          size_t LAYERS, size_t HIDDEN_VALS, bool TRAIN_ENABLED = true>
class Listener {

  // Compile time computations
  static constexpr size_t OUTPUT_NETS =
      INPUT_SAMPLE / (static_cast<size_t>(1 << (LAYERS - 1)));

  // Verifications
  static_assert(INPUT_SAMPLE % static_cast<size_t>(std::pow(2, LAYERS)) == 0);

  // Type aliasing
  using lstm_f_t = LSTM<T, SAMPLE_SIZE, HIDDEN_VALS, TRAIN_ENABLED>;
  using lstm_h_t = LSTM<T, HIDDEN_VALS * 4, HIDDEN_VALS, TRAIN_ENABLED>;

  using l_int_t = std::array<T, HIDDEN_VALS * 4>;
  using listen_in_t = std::array<std::array<T, SAMPLE_SIZE>, INPUT_SAMPLE>;
  using listen_out_t = std::array<std::array<T, HIDDEN_VALS * 2>, OUTPUT_NETS>;

  // Elements
  std::pair<lstm_f_t, lstm_f_t> first_layer;
  std::array<std::pair<lstm_h_t, lstm_h_t>, LAYERS - 1> layers;

public:
  void reset_state() {
    first_layer.first.reset_state();
    first_layer.second.reset_state();
    for (size_t i = 0; i < LAYERS - 1; ++i) {
      layers[i].first.reset_state();
      layers[i].second.reset_state();
    }
  }

  listen_out_t forward_pass_train(const listen_in_t &input) {
    std::array<l_int_t, INPUT_SAMPLE / 2> cache{};
    for (size_t i = 0; i < INPUT_SAMPLE; i += 2) {
      const size_t j = INPUT_SAMPLE - 2 - i;
      auto &forward_row = cache[i / 2];
      auto &backward_row = cache[j / 2];

      const auto forward_1 = first_layer.first.forward_pass_train(input[i]);
      const auto forward_2 = first_layer.first.forward_pass_train(input[i + 1]);
      const auto backward_2 =
          first_layer.second.forward_pass_train(input[j + 1]);
      const auto backward_1 = first_layer.second.forward_pass_train(input[j]);

      std::copy(forward_1.begin(), forward_1.end(), forward_row.begin());
      std::copy(forward_2.begin(), forward_2.end(),
                forward_row.begin() + 2 * HIDDEN_VALS);
      std::copy(backward_1.begin(), backward_1.end(),
                backward_row.begin() + HIDDEN_VALS);
      std::copy(backward_2.begin(), backward_2.end(),
                backward_row.begin() + 3 * HIDDEN_VALS);
    }

    decltype(cache) scratch;
    auto *read_cache = cache.data();
    auto *write_cache = scratch.data();

    size_t num_inputs = INPUT_SAMPLE / 2;
    for (size_t x = 0; x + 1 < layers.size(); ++x) {
      auto &networks = layers[x];

      for (size_t i = 0; i < num_inputs; i += 2) {
        const size_t j = num_inputs - 2 - i;
        auto &forward_row = write_cache[i / 2];
        auto &backward_row = write_cache[j / 2];
        const auto forward_1 = networks.first.forward_pass_train(read_cache[i]);
        const auto forward_2 =
            networks.first.forward_pass_train(read_cache[i + 1]);
        const auto backward_2 =
            networks.second.forward_pass_train(read_cache[j + 1]);
        const auto backward_1 =
            networks.second.forward_pass_train(read_cache[j]);
        std::copy(forward_1.begin(), forward_1.end(), forward_row.begin());
        std::copy(forward_2.begin(), forward_2.end(),
                  forward_row.begin() + 2 * HIDDEN_VALS);
        std::copy(backward_1.begin(), backward_1.end(),
                  backward_row.begin() + HIDDEN_VALS);
        std::copy(backward_2.begin(), backward_2.end(),
                  backward_row.begin() + 3 * HIDDEN_VALS);
      }

      std::swap(read_cache, write_cache);
      num_inputs /= 2;
    }

    listen_out_t result{};
    auto &last = layers.back();
    for (size_t i = 0; i < num_inputs; ++i) {
      const size_t j = num_inputs - 1 - i;
      const auto forward = last.first.forward_pass(read_cache[i]);
      const auto backward = last.second.forward_pass(read_cache[j]);
      std::copy(forward.begin(), forward.end(), result[i].begin());
      std::copy(backward.begin(), backward.end(),
                result[j].begin() + HIDDEN_VALS);
    }
    return result;
  }

  listen_out_t forward_pass(const listen_in_t &input) {
    std::array<l_int_t, INPUT_SAMPLE / 2> cache{};
    for (size_t i = 0; i < INPUT_SAMPLE; i += 2) {
      const size_t j = INPUT_SAMPLE - 2 - i;
      auto &forward_row = cache[i / 2];
      auto &backward_row = cache[j / 2];

      const auto forward_1 = first_layer.first.forward_pass(input[i]);
      const auto forward_2 = first_layer.first.forward_pass(input[i + 1]);
      const auto backward_2 = first_layer.second.forward_pass(input[j + 1]);
      const auto backward_1 = first_layer.second.forward_pass(input[j]);

      std::copy(forward_1.begin(), forward_1.end(), forward_row.begin());
      std::copy(forward_2.begin(), forward_2.end(),
                forward_row.begin() + 2 * HIDDEN_VALS);
      std::copy(backward_1.begin(), backward_1.end(),
                backward_row.begin() + HIDDEN_VALS);
      std::copy(backward_2.begin(), backward_2.end(),
                backward_row.begin() + 3 * HIDDEN_VALS);
    }

    decltype(cache) scratch;
    auto *read_cache = cache.data();
    auto *write_cache = scratch.data();

    size_t num_inputs = INPUT_SAMPLE / 2;
    for (size_t x = 0; x + 1 < layers.size(); ++x) {
      auto &networks = layers[x];

      for (size_t i = 0; i < num_inputs; i += 2) {
        const size_t j = num_inputs - 2 - i;
        auto &forward_row = write_cache[i / 2];
        auto &backward_row = write_cache[j / 2];
        const auto forward_1 = networks.first.forward_pass(read_cache[i]);
        const auto forward_2 = networks.first.forward_pass(read_cache[i + 1]);
        const auto backward_2 = networks.second.forward_pass(read_cache[j + 1]);
        const auto backward_1 = networks.second.forward_pass(read_cache[j]);
        std::copy(forward_1.begin(), forward_1.end(), forward_row.begin());
        std::copy(forward_2.begin(), forward_2.end(),
                  forward_row.begin() + 2 * HIDDEN_VALS);
        std::copy(backward_1.begin(), backward_1.end(),
                  backward_row.begin() + HIDDEN_VALS);
        std::copy(backward_2.begin(), backward_2.end(),
                  backward_row.begin() + 3 * HIDDEN_VALS);
      }

      std::swap(read_cache, write_cache);
      num_inputs /= 2;
    }

    listen_out_t result{};
    auto &last = layers.back();
    for (size_t i = 0; i < num_inputs; ++i) {
      const size_t j = num_inputs - 1 - i;
      const auto forward = last.first.forward_pass(read_cache[i]);
      const auto backward = last.second.forward_pass(read_cache[j]);
      std::copy(forward.begin(), forward.end(), result[i].begin());
      std::copy(backward.begin(), backward.end(),
                result[j].begin() + HIDDEN_VALS);
    }
    return result;
  }
};

// TODO: learning with the full end to end rest of the stuff
