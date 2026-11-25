#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

template <std::floating_point T, size_t H_SIZE, size_t IN_SIZE, size_t OUT_SIZE>
class MLP {
  static_assert(H_SIZE > 0 && IN_SIZE > 0 && OUT_SIZE > 0);

public:
  using in_t = std::array<T, IN_SIZE>;
  using out_t = std::array<T, OUT_SIZE>;

  struct BackwardResult {
    in_t input_gradient;
    T loss;
  };

private:
  using hidden_t = std::array<T, H_SIZE>;

  template <size_t In, size_t Out> struct DenseLayer {
    std::array<std::array<T, In>, Out> weights;
    std::array<T, Out> bias{};
    std::array<std::array<T, In>, Out> weight_gradient{};
    std::array<T, Out> bias_gradient{};

    void initialize(std::mt19937 &generator,
                    std::uniform_real_distribution<T> &distribution) {
      for (auto &row : weights)
        for (T &weight : row)
          weight = distribution(generator);
    }

    std::array<T, Out> forward(const std::array<T, In> &input) const {
      std::array<T, Out> output;
      for (size_t i = 0; i < Out; ++i) {
        T sum = 0;
        for (size_t j = 0; j < In; ++j)
          sum += weights[i][j] * input[j];
        output[i] = sum + bias[i];
      }
      return output;
    }

    std::array<T, In> backward(const std::array<T, In> &input,
                               const std::array<T, Out> &output_gradient) {
      std::array<T, In> input_gradient{};
      for (size_t i = 0; i < Out; ++i) {
        const T gradient = output_gradient[i];
        bias_gradient[i] += gradient;
        for (size_t j = 0; j < In; ++j) {
          weight_gradient[i][j] += gradient * input[j];
          input_gradient[j] += weights[i][j] * gradient;
        }
      }
      return input_gradient;
    }

    void clear_gradient() {
      for (auto &row : weight_gradient)
        row.fill(T{0});
      bias_gradient.fill(T{0});
    }

    void apply_gradient(T learning_rate) {
      for (size_t i = 0; i < Out; ++i) {
        for (size_t j = 0; j < In; ++j)
          weights[i][j] -= learning_rate * weight_gradient[i][j];
        bias[i] -= learning_rate * bias_gradient[i];
      }
      clear_gradient();
    }
  };

  struct Distribution {
    out_t probabilities;
    T max_logit;
    T exp_sum;
  };

  struct StepCache {
    in_t input;
    hidden_t hidden;
    out_t logits;
    Distribution distribution;
  };

  DenseLayer<IN_SIZE, H_SIZE> hidden_layer;
  DenseLayer<H_SIZE, OUT_SIZE> output_layer;
  std::vector<StepCache> cache;

  hidden_t hidden_activations(const in_t &input) const {
    auto hidden = hidden_layer.forward(input);
    for (T &value : hidden)
      value = std::max(T{0}, value);
    return hidden;
  }

  static Distribution softmax(const out_t &logits) {
    const T max_logit = *std::max_element(logits.begin(), logits.end());
    T exp_sum = 0;
    out_t probabilities;
    for (size_t i = 0; i < OUT_SIZE; ++i) {
      probabilities[i] = std::exp(logits[i] - max_logit);
      exp_sum += probabilities[i];
    }
    for (T &probability : probabilities)
      probability /= exp_sum;
    return {probabilities, max_logit, exp_sum};
  }

public:
  MLP() : MLP(std::random_device{}()) {}

  explicit MLP(std::uint32_t seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<T> distribution(-T{0.1}, T{0.1});
    hidden_layer.initialize(generator, distribution);
    output_layer.initialize(generator, distribution);
  }

  MLP(const MLP &) = delete;
  MLP &operator=(const MLP &) = delete;

  out_t forward_pass(const in_t &input) const {
    return softmax(output_layer.forward(hidden_activations(input)))
        .probabilities;
  }

  out_t forward_pass_train(const in_t &input) {
    const auto hidden = hidden_activations(input);
    const auto logits = output_layer.forward(hidden);
    const auto distribution = softmax(logits);
    cache.push_back({input, hidden, logits, distribution});
    return distribution.probabilities;
  }

  // Call backward in reverse order after saving the sequence's forward steps.
  BackwardResult backward_pass(size_t target_token) {
    if (target_token >= OUT_SIZE)
      throw std::out_of_range("MLP target token is out of range");
    if (cache.empty())
      throw std::logic_error("MLP backward pass requires a saved forward step");

    const auto &step = cache.back();
    auto logit_gradient = step.distribution.probabilities;
    logit_gradient[target_token] -= T{1};
    auto hidden_gradient = output_layer.backward(step.hidden, logit_gradient);
    for (size_t i = 0; i < H_SIZE; ++i)
      if (step.hidden[i] <= T{0})
        hidden_gradient[i] = T{0};

    const T loss = std::log(step.distribution.exp_sum) -
                   (step.logits[target_token] - step.distribution.max_logit);
    BackwardResult result{hidden_layer.backward(step.input, hidden_gradient),
                          loss};
    cache.pop_back();
    return result;
  }

  void apply_gradient(T learning_rate) {
    if (!cache.empty())
      throw std::logic_error("MLP has forward steps awaiting backward passes");
    hidden_layer.apply_gradient(learning_rate);
    output_layer.apply_gradient(learning_rate);
  }

  void reset_state() {
    cache.clear();
    hidden_layer.clear_gradient();
    output_layer.clear_gradient();
  }
};
