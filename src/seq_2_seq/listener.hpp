#include "lstm.hpp"
#include <array>
#include <cstddef>
#include <utility>

template <std::floating_point T, size_t INPUT_SAMPLE, size_t SAMPLE_SIZE,
          size_t HIDDEN_VALS, size_t LAYERS, Platform platform,
          bool TRAIN_ENABLED = true>
class Listener {
  // Verifications
  static_assert(LAYERS > 1);
  static_assert(INPUT_SAMPLE % static_cast<size_t>(std::pow(2, LAYERS)) == 0);

  // Compile time computations
  static constexpr size_t OUTPUT_NETS =
      INPUT_SAMPLE / (static_cast<size_t>(std::pow(2, LAYERS - 1)));

  // Type aliasing
  using lstm_f_t = LSTM<T, SAMPLE_SIZE, HIDDEN_VALS, platform, TRAIN_ENABLED>;
  using lstm_h_t =
      LSTM<T, HIDDEN_VALS * 4, HIDDEN_VALS, platform, TRAIN_ENABLED>;

  using l_int_t = std::array<T, HIDDEN_VALS * 4>;
  using listen_in_t = std::array<std::array<T, SAMPLE_SIZE>, INPUT_SAMPLE>;
  using listen_out_t = std::array<std::array<T, HIDDEN_VALS * 2>, OUTPUT_NETS>;

  // Elements
  std::pair<lstm_f_t, lstm_f_t> first_layer;
  std::array<std::pair<lstm_h_t, lstm_h_t>, LAYERS - 1> layers;

public:
  listen_out_t forward_pass(const listen_in_t &input) {
    listen_out_t res{};
    // TODO: complete forward pass
    return res;
  }
};
