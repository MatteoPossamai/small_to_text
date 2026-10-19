#include "utils.hpp"
#include <algorithm>

template <size_t IN_N, size_t L1, size_t L2, size_t OUT_N> class Network {
  // Input layer
  std::array<output_t, IN_N> input_layer{};
  // Hidden layer 1
  std::array<output_t, L1> layer1{};
  // Hidden layer 2
  std::array<output_t, L2> layer2{};
  // Output layer
  std::array<output_t, OUT_N> output_layer{};

  // After soft-max
  std::array<output_t, OUT_N> norm_output_layer{};

  // Weights
  std::array<std::array<output_t, IN_N>, IN_N> w0;
  std::array<std::array<output_t, L1>, L1> w1;
  std::array<std::array<output_t, L2>, L2> w2;

  [[gnu::always_inline]] inline output_t ReLU(output_t value) {
    return std::max(static_cast<output_t>(0), value);
  }

  void normalize_output_layer(); // TODO: impl

public:
  Network() = default;
  void load(std::array<output_t, IN_N> input); // TODO: impl
  void forward_pass();                         // TODO: impl

  const std::array<output_t, OUT_N> &output() const {
    return &norm_output_layer;
  }
};
