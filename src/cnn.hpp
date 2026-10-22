#pragma once

#include "utils.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <span>
#include <string>

template <size_t NETWORK_IN, size_t BANDS, size_t CONV_NEURONS,
          size_t WINDOW_SIZE, size_t COMPRESS_RATIO, size_t L1, size_t L2,
          size_t OUT_N>
class CNN {
  static_assert(BANDS > 0 && WINDOW_SIZE > 0 && COMPRESS_RATIO > 0);
  static_assert(CONV_NEURONS > 0 && L1 > 0 && L2 > 0 && OUT_N > 0);
  static_assert(NETWORK_IN % BANDS == 0,
                "The input must contain complete rows of bands");
  static_assert(NETWORK_IN / BANDS >= WINDOW_SIZE,
                "The convolution window must fit in the input");
  static_assert((NETWORK_IN / BANDS - WINDOW_SIZE + 1) % COMPRESS_RATIO == 0,
                "The averaging groups must cover all convolution positions");

  // Static constexpr
  static constexpr size_t POSITIONS = NETWORK_IN / BANDS - WINDOW_SIZE + 1;
  static constexpr size_t INPUT_N =
      POSITIONS / COMPRESS_RATIO *
      CONV_NEURONS; // ((1580 ÷ 20) - (5 - 1)) / 3 * 16 = 400 in the example

  // Inputs
  std::array<output_t, NETWORK_IN> input_data{};

  // Convolutional part
  std::array<std::array<output_t, CONV_NEURONS>, BANDS * WINDOW_SIZE>
      conv_weights{};
  std::array<output_t, CONV_NEURONS> conv_bias{};
  // Keep each position's output after ReLU for backpropagation.
  std::array<std::array<output_t, CONV_NEURONS>, POSITIONS> conv_output{};

  // Vanilla NN side
  std::array<output_t, INPUT_N> input_layer{};
  std::array<output_t, L1> layer1{};
  std::array<output_t, L2> layer2{};
  std::array<output_t, OUT_N> output_layer{};
  std::array<output_t, OUT_N> norm_output_layer{};

  std::array<std::array<output_t, L1>, INPUT_N> w0{};
  std::array<std::array<output_t, L2>, L1> w1{};
  std::array<std::array<output_t, OUT_N>, L2> w2{};

  std::array<output_t, L1> b0{};
  std::array<output_t, L2> b1{};
  std::array<output_t, OUT_N> b2{};

  [[gnu::always_inline]] inline output_t ReLU(output_t value) {
    return std::max(static_cast<output_t>(0), value);
  }

  [[gnu::always_inline]] inline output_t loss(output_t output) {
    return -std::log(output);
  }

  void normalize_output_layer();

  template <size_t X, size_t Y>
  void matrix_mul(std::span<const output_t> inputs,
                  const std::array<std::array<output_t, Y>, X> &weights,
                  std::array<output_t, Y> &res);

public:
  CNN() = default;
  CNN(std::array<output_t, NETWORK_IN> inputs) : input_data(inputs) {}
  CNN(const CNN &) = delete;
  CNN &operator=(const CNN &) = delete;

  void load_inputs(const std::array<output_t, NETWORK_IN> &inputs) {
    input_data = inputs;
  }

  void initialize_random_weights() {
    std::mt19937 generator(std::random_device{}());
    auto initialize = [&generator](auto &weights, output_t bound) {
      std::uniform_real_distribution<output_t> distribution(-bound, bound);
      for (auto &row : weights)
        for (auto &weight : row)
          weight = distribution(generator);
    };

    initialize(conv_weights, std::sqrt(output_t{6} / (BANDS * WINDOW_SIZE)));
    initialize(w0, std::sqrt(output_t{6} / INPUT_N));
    initialize(w1, std::sqrt(output_t{6} / L1));
    initialize(w2, std::sqrt(output_t{6} / (L2 + OUT_N)));

    conv_bias.fill(0);
    b0.fill(0);
    b1.fill(0);
    b2.fill(0);
  }

  void load_weights(const std::array<std::array<output_t, CONV_NEURONS>,
                                     BANDS * WINDOW_SIZE> &conv_weights_in,
                    const std::array<std::array<output_t, L1>, INPUT_N> &w0_in,
                    const std::array<std::array<output_t, L2>, L1> &w1_in,
                    const std::array<std::array<output_t, OUT_N>, L2> &w2_in,
                    const std::array<output_t, CONV_NEURONS> &conv_bias_in = {},
                    const std::array<output_t, L1> &b0_in = {},
                    const std::array<output_t, L2> &b1_in = {},
                    const std::array<output_t, OUT_N> &b2_in = {}) {
    conv_weights = conv_weights_in;
    conv_bias = conv_bias_in;
    w0 = w0_in;
    w1 = w1_in;
    w2 = w2_in;
    b0 = b0_in;
    b1 = b1_in;
    b2 = b2_in;
  }

  const std::array<output_t, OUT_N> &output() const {
    return norm_output_layer;
  }

  void forward_pass();
  bool from_file(const std::string &path);
  bool to_file(const std::string &path);
  output_t backprop(const std::array<output_t, OUT_N> &target,
                    output_t learning_rate);
};

template <size_t NETWORK_IN, size_t BANDS, size_t CONV_NEURONS,
          size_t WINDOW_SIZE, size_t COMPRESS_RATIO, size_t L1, size_t L2,
          size_t OUT_N>
template <size_t X, size_t Y>
void CNN<NETWORK_IN, BANDS, CONV_NEURONS, WINDOW_SIZE, COMPRESS_RATIO, L1, L2,
         OUT_N>::matrix_mul(std::span<const output_t> inputs,
                            const std::array<std::array<output_t, Y>, X>
                                &weights,
                            std::array<output_t, Y> &res) {
  res.fill(0);
  for (size_t i = 0; i < X; ++i) {
    for (size_t j = 0; j < Y; ++j) {
      res[j] += inputs[i] * weights[i][j];
    }
  }
}

template <size_t NETWORK_IN, size_t BANDS, size_t CONV_NEURONS,
          size_t WINDOW_SIZE, size_t COMPRESS_RATIO, size_t L1, size_t L2,
          size_t OUT_N>
void CNN<NETWORK_IN, BANDS, CONV_NEURONS, WINDOW_SIZE, COMPRESS_RATIO, L1, L2,
         OUT_N>::normalize_output_layer() {
  // Get the max and subtract it, to avoid overflow of the std::exp(x) call
  const auto max = *std::max_element(output_layer.begin(), output_layer.end());
  output_t softmax_sum = 0;
  for (size_t i = 0; i < OUT_N; ++i) {
    norm_output_layer[i] = std::exp(output_layer[i] - max);
    softmax_sum += norm_output_layer[i];
  }

  for (auto &out : norm_output_layer) {
    out /= softmax_sum;
  }
}

template <size_t NETWORK_IN, size_t BANDS, size_t CONV_NEURONS,
          size_t WINDOW_SIZE, size_t COMPRESS_RATIO, size_t L1, size_t L2,
          size_t OUT_N>
bool CNN<NETWORK_IN, BANDS, CONV_NEURONS, WINDOW_SIZE, COMPRESS_RATIO, L1, L2,
         OUT_N>::from_file(const std::string &file_path) {
  std::ifstream file(file_path);
  if (!file)
    return false;

  // CNN_V1, then all eight template dimensions. This is a different
  // format from the ordinary Network model files.
  std::string format;
  size_t network_in, bands, conv_neurons, window_size, compress_ratio;
  size_t l1, l2, out_n;
  if (!(file >> format >> network_in >> bands >> conv_neurons >> window_size >>
        compress_ratio >> l1 >> l2 >> out_n) ||
      format != "CNN_V1" || network_in != NETWORK_IN || bands != BANDS ||
      conv_neurons != CONV_NEURONS || window_size != WINDOW_SIZE ||
      compress_ratio != COMPRESS_RATIO || l1 != L1 || l2 != L2 ||
      out_n != OUT_N)
    return false;

  // Only replace the current parameters after the entire file is valid.
  CNN loaded;
  const auto read_values = [&file](auto &values) {
    for (auto &value : values) {
      if (!(file >> value) || !std::isfinite(value))
        return false;
    }
    return true;
  };
  const auto read_matrix = [&read_values](auto &matrix) {
    for (auto &row : matrix) {
      if (!read_values(row))
        return false;
    }
    return true;
  };

  // Biases first, then weight matrices; convolution precedes ordinary layers.
  if (!read_values(loaded.conv_bias) || !read_values(loaded.b0) ||
      !read_values(loaded.b1) || !read_values(loaded.b2) ||
      !read_matrix(loaded.conv_weights) || !read_matrix(loaded.w0) ||
      !read_matrix(loaded.w1) || !read_matrix(loaded.w2))
    return false;

  file >> std::ws;
  if (file.bad() || !file.eof())
    return false;

  conv_weights = loaded.conv_weights;
  conv_bias = loaded.conv_bias;
  w0 = loaded.w0;
  w1 = loaded.w1;
  w2 = loaded.w2;
  b0 = loaded.b0;
  b1 = loaded.b1;
  b2 = loaded.b2;

  return true;
}

template <size_t NETWORK_IN, size_t BANDS, size_t CONV_NEURONS,
          size_t WINDOW_SIZE, size_t COMPRESS_RATIO, size_t L1, size_t L2,
          size_t OUT_N>
bool CNN<NETWORK_IN, BANDS, CONV_NEURONS, WINDOW_SIZE, COMPRESS_RATIO, L1, L2,
         OUT_N>::to_file(const std::string &file_path) {
  std::ofstream file(file_path);
  if (!file)
    return false;

  file << "CNN_V1 " << NETWORK_IN << " " << BANDS << " " << CONV_NEURONS << " "
       << WINDOW_SIZE << " " << COMPRESS_RATIO << " " << L1 << " " << L2 << " "
       << OUT_N << "\n\n";
  file << std::setprecision(std::numeric_limits<output_t>::max_digits10);

  // Biases
  for (const auto &bias : conv_bias) {
    file << bias << " ";
  }
  file << "\n\n";

  for (const auto &bias : b0) {
    file << bias << " ";
  }
  file << "\n\n";

  for (const auto &bias : b1) {
    file << bias << " ";
  }
  file << "\n\n";

  for (const auto &bias : b2) {
    file << bias << " ";
  }
  file << "\n\n";

  // Weights
  for (const auto &weight_row : conv_weights) {
    for (const auto &weight : weight_row) {
      file << weight << " ";
    }
    file << "\n";
  }
  file << "\n";

  for (const auto &weight_row : w0) {
    for (const auto &weight : weight_row) {
      file << weight << " ";
    }
    file << "\n";
  }
  file << "\n";

  for (const auto &weight_row : w1) {
    for (const auto &weight : weight_row) {
      file << weight << " ";
    }
    file << "\n";
  }
  file << "\n";

  for (const auto &weight_row : w2) {
    for (const auto &weight : weight_row) {
      file << weight << " ";
    }
    file << "\n";
  }
  file << "\n";

  file.close();
  return static_cast<bool>(file);
}

template <size_t NETWORK_IN, size_t BANDS, size_t CONV_NEURONS,
          size_t WINDOW_SIZE, size_t COMPRESS_RATIO, size_t L1, size_t L2,
          size_t OUT_N>
void CNN<NETWORK_IN, BANDS, CONV_NEURONS, WINDOW_SIZE, COMPRESS_RATIO, L1, L2,
         OUT_N>::forward_pass() {
  input_layer.fill(0);
  std::span<const output_t> input_view(input_data);
  for (size_t position = 0; position < POSITIONS; ++position) {
    auto window = input_view.subspan(position * BANDS, WINDOW_SIZE * BANDS);
    auto &result = conv_output[position];

    matrix_mul(window, conv_weights, result);
    for (size_t neuron = 0; neuron < CONV_NEURONS; ++neuron) {
      result[neuron] = ReLU(result[neuron] + conv_bias[neuron]);
    }

    // Consecutive positions share an averaging group, preserving each neuron.
    const size_t group = position / COMPRESS_RATIO;
    for (size_t neuron = 0; neuron < CONV_NEURONS; ++neuron) {
      input_layer[group * CONV_NEURONS + neuron] += result[neuron];
    }
  }

  for (auto &value : input_layer) {
    value /= COMPRESS_RATIO;
  }

  matrix_mul(input_layer, w0, layer1);
  for (size_t i = 0; i < L1; ++i) {
    layer1[i] = ReLU(layer1[i] + b0[i]);
  }

  matrix_mul(layer1, w1, layer2);
  for (size_t i = 0; i < L2; ++i) {
    layer2[i] = ReLU(layer2[i] + b1[i]);
  }

  matrix_mul(layer2, w2, output_layer);
  for (size_t i = 0; i < OUT_N; ++i) {
    output_layer[i] += b2[i];
  }
  normalize_output_layer();
}

template <size_t NETWORK_IN, size_t BANDS, size_t CONV_NEURONS,
          size_t WINDOW_SIZE, size_t COMPRESS_RATIO, size_t L1, size_t L2,
          size_t OUT_N>
output_t CNN<NETWORK_IN, BANDS, CONV_NEURONS, WINDOW_SIZE, COMPRESS_RATIO, L1,
             L2, OUT_N>::backprop(const std::array<output_t, OUT_N> &target,
                                  output_t learning_rate) {
  std::array<output_t, OUT_N> losses = {};
  std::array<output_t, OUT_N> gradients = {};
  output_t scalar_loss = 0;
  for (size_t i = 0; i < losses.size(); i++) {
    losses[i] = loss(norm_output_layer[i]);
    scalar_loss += target[i] == 0 ? 0 : losses[i];
    gradients[i] = norm_output_layer[i] - target[i];
  }

  // Biases
  std::array<output_t, L1> db0{};
  std::array<output_t, L2> db1{};
  std::array<output_t, OUT_N> db2 = gradients;

  for (size_t i = 0; i < db2.size(); i++) {
    b2[i] -= learning_rate * db2[i];
  }

  for (size_t i = 0; i < db1.size(); i++) {
    output_t temp = 0;
    for (size_t j = 0; j < db2.size(); j++) {
      temp += w2[i][j] * db2[j];
    }
    db1[i] = layer2[i] > 0 ? temp : 0;
    b1[i] -= learning_rate * db1[i];
  }

  for (size_t i = 0; i < db0.size(); i++) {
    output_t temp = 0;
    for (size_t j = 0; j < db1.size(); j++) {
      temp += w1[i][j] * db1[j];
    }
    db0[i] = layer1[i] > 0 ? temp : 0;
    b0[i] -= learning_rate * db0[i];
  }

  // Weights
  for (size_t i = 0; i < layer2.size(); ++i) {
    for (size_t j = 0; j < db2.size(); ++j) {
      w2[i][j] -= learning_rate * layer2[i] * db2[j];
    }
  }

  for (size_t i = 0; i < layer1.size(); ++i) {
    for (size_t j = 0; j < db1.size(); ++j) {
      w1[i][j] -= learning_rate * layer1[i] * db1[j];
    }
  }

  for (size_t i = 0; i < input_layer.size(); ++i) {
    for (size_t j = 0; j < db0.size(); ++j) {
      w0[i][j] -= learning_rate * input_layer[i] * db0[j];
    }
  }

  // TODO: impl convolution

  return scalar_loss;
}
