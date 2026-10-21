#pragma once

#include "network.hpp"
#include "utils.hpp"
#include "wav_weights.hpp"
#include <array>
#include <cstddef>
#include <random>
#include <string_view>

constexpr size_t N_LABELS = 36;

using training_set =
    std::vector<std::tuple<WavWeights, std::array<output_t, N_LABELS>>>;

constexpr std::array<std::string_view, N_LABELS> LABELS_MAPPINGS = {
    "_background_noise_",
    "backward",
    "bed",
    "bird",
    "cat",
    "dog",
    "down",
    "eight",
    "five",
    "follow",
    "forward",
    "four",
    "go",
    "happy",
    "house",
    "learn",
    "left",
    "marvin",
    "nine",
    "no",
    "off",
    "on",
    "one",
    "right",
    "seven",
    "sheila",
    "six",
    "stop",
    "three",
    "tree",
    "two",
    "up",
    "visual",
    "wow",
    "yes",
    "zero"};

// Move complete time windows, keeping their frequency bands together.
// Positive shifts delay the sound; newly exposed windows contain silence.
inline std::array<output_t, NETWORK_IN_VALUES>
time_shift(const std::array<output_t, NETWORK_IN_VALUES> &input, int shift) {
  constexpr int WINDOWS = NETWORK_IN_VALUES / NUM_BANDS;
  std::array<output_t, NETWORK_IN_VALUES> shifted{};
  for (int window = 0; window < WINDOWS; ++window) {
    const int source = window - shift;
    if (source < 0 || source >= WINDOWS)
      continue;
    for (size_t band = 0; band < NUM_BANDS; ++band)
      shifted[window * NUM_BANDS + band] = input[source * NUM_BANDS + band];
  }
  return shifted;
}

template <size_t IN_N, size_t L1, size_t L2, size_t OUT_N>
output_t train_network(Network<IN_N, L1, L2, OUT_N> &network,
                       output_t learning_rate,
                       const training_set &training_dataset,
                       const std::array<std::string_view, N_LABELS> mappings) {
  static std::mt19937 shift_generator(std::random_device{}());
  // Four 12.5 ms windows in either direction: at most 50 ms.
  std::uniform_int_distribution<int> shift_distribution(-4, 4);
  output_t loss = 0;
  for (const auto &training_point : training_dataset) {
    const auto &[weights, target] = training_point;

    // Perform forward pass
    const auto shifted =
        time_shift(weights.data(), shift_distribution(shift_generator));
    network.load(shifted);
    network.forward_pass();

    loss += network.backprop(target, learning_rate);
  }
  return loss;
}
