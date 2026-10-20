#pragma once

#include "network.hpp"
#include "wav_weights.hpp"
#include <array>
#include <cstddef>
#include <string_view>

constexpr size_t N_LABELS = 36;

using training_set =
    std::vector<std::tuple<WavWeights, std::array<output_t, N_LABELS>>>;

constexpr std::array<std::string_view, N_LABELS> LABELS_MAPPINGS = {
    "_backgroud_noise_",
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

template <size_t IN_N, size_t L1, size_t L2, size_t OUT_N>
void train_network(Network<IN_N, L1, L2, OUT_N> &network, size_t epochs,
                   size_t learning_rate, const training_set &training_dataset) {
}
