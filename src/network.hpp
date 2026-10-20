#pragma once

#include "utils.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <random>

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
  std::array<std::array<output_t, L1>, IN_N> w0;
  std::array<std::array<output_t, L2>, L1> w1;
  std::array<std::array<output_t, OUT_N>, L2> w2;

  // One bias per hidden and output neuron.
  std::array<output_t, L1> b0{};
  std::array<output_t, L2> b1{};
  std::array<output_t, OUT_N> b2{};

  [[gnu::always_inline]] inline output_t ReLU(output_t value) {
    return std::max(static_cast<output_t>(0), value);
  }

  [[gnu::always_inline]] inline output_t loss(output_t output,
                                              output_t expected) {
    return -std::log(output);
  }

  void normalize_output_layer();

  template <size_t X, size_t Y>
  void matrix_mul(std::array<output_t, X> &inputs,
                  std::array<std::array<output_t, Y>, X> &weights,
                  std::array<output_t, Y> &res);

public:
  Network() = default;
  void load(const std::array<output_t, IN_N> &input) { input_layer = input; }

  void forward_pass();

  void initialize_random_weights() {
    std::random_device r;
    std::default_random_engine generator(r());
    std::uniform_int_distribution<size_t> distribution(0, 100);

    for (auto &row : w0)
      for (auto &el : row)
        el = static_cast<output_t>(distribution(r));

    for (auto &row : w1)
      for (auto &el : row)
        el = static_cast<output_t>(distribution(r));

    for (auto &row : w2)
      for (auto &el : row)
        el = static_cast<output_t>(distribution(r));

    b0.fill(0);
    b1.fill(0);
    b2.fill(0);
  }

  void load_weights(const std::array<std::array<output_t, L1>, IN_N> &w0_in,
                    const std::array<std::array<output_t, L2>, L1> &w1_in,
                    const std::array<std::array<output_t, OUT_N>, L2> &w2_in,
                    const std::array<output_t, L1> &b0_in = {},
                    const std::array<output_t, L2> &b1_in = {},
                    const std::array<output_t, OUT_N> &b2_in = {}) {
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

  bool from_file(const std::string &path);
  bool to_file(const std::string &path);
  void backprop(const std::array<output_t, OUT_N> &target,
                output_t learning_rate);
};

template <size_t IN_N, size_t L1, size_t L2, size_t OUT_N>
void Network<IN_N, L1, L2, OUT_N>::normalize_output_layer() {
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

template <size_t IN_N, size_t L1, size_t L2, size_t OUT_N>
template <size_t X, size_t Y>
void Network<IN_N, L1, L2, OUT_N>::matrix_mul(
    std::array<output_t, X> &inputs,
    std::array<std::array<output_t, Y>, X> &weights,
    std::array<output_t, Y> &res) {
  for (size_t i = 0; i < Y; ++i) {
    output_t total = 0;
    for (size_t j = 0; j < X; ++j) {
      total += inputs[j] * weights[j][i];
    }
    res[i] = total;
  }
}

template <size_t IN_N, size_t L1, size_t L2, size_t OUT_N>
void Network<IN_N, L1, L2, OUT_N>::forward_pass() {
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

template <size_t IN_N, size_t L1, size_t L2, size_t OUT_N>
bool Network<IN_N, L1, L2, OUT_N>::from_file(const std::string &file_path) {
  std::ifstream file(file_path);
  if (!file)
    return false;

  size_t in_n, l1, l2, out_n;
  if (!(file >> in_n >> l1 >> l2 >> out_n) || in_n != IN_N || l1 != L1 ||
      l2 != L2 || out_n != OUT_N)
    return false;

  // Biases
  for (auto &bias : b0) {
    if (!(file >> bias))
      return false;
  }
  for (auto &bias : b1) {
    if (!(file >> bias))
      return false;
  }
  for (auto &bias : b2) {
    if (!(file >> bias))
      return false;
  }

  // Weights
  for (auto &weight_row : w0) {
    for (auto &weight : weight_row) {
      if (!(file >> weight))
        return false;
    }
  }
  for (auto &weight_row : w1) {
    for (auto &weight : weight_row) {
      if (!(file >> weight))
        return false;
    }
  }
  for (auto &weight_row : w2) {
    for (auto &weight : weight_row) {
      if (!(file >> weight))
        return false;
    }
  }

  return true;
}

template <size_t IN_N, size_t L1, size_t L2, size_t OUT_N>
bool Network<IN_N, L1, L2, OUT_N>::to_file(const std::string &file_path) {
  std::ofstream file(file_path);
  if (!file)
    return false;

  file << IN_N << " " << L1 << " " << L2 << " " << OUT_N << "\n\n";
  file << std::setprecision(std::numeric_limits<double>::max_digits10);

  // Biases
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

  return static_cast<bool>(file);
}

template <size_t IN_N, size_t L1, size_t L2, size_t OUT_N>
void Network<IN_N, L1, L2, OUT_N>::backprop(
    const std::array<output_t, OUT_N> &target, output_t learning_rate) {

  std::array<output_t, OUT_N> losses = {};
  std::array<output_t, OUT_N> gradients = {};
  output_t scalar_loss = 0;
  for (size_t i = 0; i < losses.size(); i++) {
    losses[i] = loss(norm_output_layer[i], target[i]);
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
}
