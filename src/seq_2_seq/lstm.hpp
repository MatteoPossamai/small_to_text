#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

template <std::floating_point T, size_t INPUT_SIZE, size_t OUTPUT_SIZE,
          bool TRAIN_ENABLED = true>
class LSTM {

  static_assert(INPUT_SIZE > 0);
  static_assert(OUTPUT_SIZE % 8 == 0 && OUTPUT_SIZE > 0);

public:
  using mem_t = std::array<T, OUTPUT_SIZE>;
  using in_t = std::array<T, INPUT_SIZE>;
  using out_t = std::array<T, OUTPUT_SIZE>;
  using weight_t =
      std::array<std::array<T, OUTPUT_SIZE + INPUT_SIZE>, OUTPUT_SIZE>;
  using concat_t = std::array<T, OUTPUT_SIZE + INPUT_SIZE>;

  struct StepCache {
    in_t input;
    out_t prev_l_t_mem;
    out_t prev_s_t_mem;
    out_t forget_gate;
    out_t input_gate;
    out_t candidate_gate;
    out_t output_gate;
    out_t new_cell;
  };

  struct BackwardResult {
    in_t input_gradient;
    out_t previous_hidden_gradient;
    out_t previous_cell_gradient;
  };

private:
  struct TrainingData {
    std::vector<StepCache> cache;
    weight_t W_f_grad{}, W_i_grad{}, W_c_grad{}, W_o_grad{};
    out_t B_f_grad{}, B_i_grad{}, B_c_grad{}, B_o_grad{};
  };

  struct NoTraining {};

  // Inference instances contain neither the cache nor the gradient arrays.
  [[no_unique_address]]
  std::conditional_t<TRAIN_ENABLED, TrainingData, NoTraining> training{};

  // Memory
  mem_t l_t_mem{}; // Long term memory, or c
  mem_t s_t_mem{}; // Short term memory, or h

  // Weights
  weight_t W_f{}; // forget
  weight_t W_i{}; // input
  weight_t W_c{}; // candidate
  weight_t W_o{}; // output

  // Biases
  out_t B_f{}; // forget
  out_t B_i{}; // input
  out_t B_c{}; // candidate
  out_t B_o{}; // output

  __attribute__((always_inline)) inline T sigmoid(T x) {
    return T{1} / (T{1} + std::exp(-x));
  }

  void sigmoid_inplace(out_t &array) {
    for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
      array[i] = sigmoid(array[i]);
    }
  }

  void tanh_inplace(out_t &array) {
    for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
      array[i] = std::tanh(array[i]);
    }
  }

  out_t matrix_mul_scalar(const weight_t &m1, const concat_t &m2,
                          const out_t &bias) {
    out_t res{};

    for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
#pragma clang loop vectorize(enable)
#pragma clang loop interleave(enable)
      for (size_t j = 0; j < OUTPUT_SIZE + INPUT_SIZE; ++j) {
        res[i] += m2[j] * m1[i][j];
      }
    }

#pragma clang loop vectorize(enable)
#pragma clang loop interleave(enable)
    for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
      res[i] += bias[i];
    }

    return res;
  }

  void set_l_t_mem_scalar(mem_t &__restrict__ mem,
                          const out_t &__restrict__ forget_gate,
                          const out_t &__restrict__ input_gate,
                          const out_t &__restrict__ candidate_gate) {
#pragma clang loop vectorize(enable)
#pragma clang loop interleave(enable)
    for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
      mem[i] = mem[i] * forget_gate[i] + input_gate[i] * candidate_gate[i];
    }
  }

  out_t forward_scalar_train(const in_t &x)
    requires(TRAIN_ENABLED)
  {
    StepCache cache_i{};
    cache_i.input = x;
    cache_i.prev_l_t_mem = l_t_mem;
    cache_i.prev_s_t_mem = s_t_mem;

    concat_t z_t;
    std::copy(x.begin(), x.end(), z_t.begin());
    std::copy(s_t_mem.begin(), s_t_mem.end(), z_t.begin() + x.size());

    // Forget gate
    out_t forget_gate = matrix_mul_scalar(W_f, z_t, B_f);
    sigmoid_inplace(forget_gate);
    cache_i.forget_gate = forget_gate;

    // Input gate
    out_t input_gate = matrix_mul_scalar(W_i, z_t, B_i);
    sigmoid_inplace(input_gate);
    cache_i.input_gate = input_gate;
    out_t candidate_gate = matrix_mul_scalar(W_c, z_t, B_c);
    tanh_inplace(candidate_gate);
    cache_i.candidate_gate = candidate_gate;

    // Update Long Term Memory
    set_l_t_mem_scalar(l_t_mem, forget_gate, input_gate, candidate_gate);
    cache_i.new_cell = l_t_mem;

    // Output Gate
    out_t output_gate = matrix_mul_scalar(W_o, z_t, B_o);
    sigmoid_inplace(output_gate);
    cache_i.output_gate = output_gate;

    for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
      s_t_mem[i] = std::tanh(l_t_mem[i]) * output_gate[i];
    }

    training.cache.push_back(cache_i);
    return s_t_mem;
  }

  out_t forward_scalar(const in_t &x) {
    concat_t z_t;
    std::copy(x.begin(), x.end(), z_t.begin());
    std::copy(s_t_mem.begin(), s_t_mem.end(), z_t.begin() + x.size());

    // Forget gate
    out_t forget_gate = matrix_mul_scalar(W_f, z_t, B_f);
    sigmoid_inplace(forget_gate);

    // Input gate
    out_t input_gate = matrix_mul_scalar(W_i, z_t, B_i);
    sigmoid_inplace(input_gate);
    out_t candidate_gate = matrix_mul_scalar(W_c, z_t, B_c);
    tanh_inplace(candidate_gate);

    // Update Long Term Memory
    set_l_t_mem_scalar(l_t_mem, forget_gate, input_gate, candidate_gate);

    // Output Gate
    out_t output_gate = matrix_mul_scalar(W_o, z_t, B_o);
    sigmoid_inplace(output_gate);

    for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
      s_t_mem[i] = std::tanh(l_t_mem[i]) * output_gate[i];
    }
    return s_t_mem;
  }

public:
  LSTM() : LSTM(std::random_device{}()) {}

  // A fixed seed makes initialization repeatable. Biases and memories start at
  // 0.
  explicit LSTM(std::uint32_t seed) {
    std::mt19937 generator(seed);
    // Xavier uniform: fan-in = INPUT_SIZE + OUTPUT_SIZE, fan-out = OUTPUT_SIZE.
    const T limit = std::sqrt(T{6} / (T(INPUT_SIZE) + T{2} * T(OUTPUT_SIZE)));
    std::uniform_real_distribution<T> distribution(-limit, limit);
    for (weight_t *weights : {&W_f, &W_i, &W_c, &W_o}) {
      for (auto &row : *weights) {
        for (auto &weight : row)
          weight = distribution(generator);
      }
    }
  }

  // Clear sequence memory while keeping the same weights and biases.
  void reset_state() {
    l_t_mem.fill(T{0});
    s_t_mem.fill(T{0});
    if constexpr (TRAIN_ENABLED) {
      training.cache.clear();

      for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
        training.W_f_grad[i].fill(0);
        training.W_i_grad[i].fill(0);
        training.W_c_grad[i].fill(0);
        training.W_o_grad[i].fill(0);
      }

      training.B_f_grad.fill(0);
      training.B_i_grad.fill(0);
      training.B_c_grad.fill(0);
      training.B_o_grad.fill(0);
    }
  }

  LSTM(weight_t &W_f_in, weight_t &W_i_in, weight_t &W_c_in, weight_t &W_o_in,
       out_t &B_f_in, out_t &B_i_in, out_t &B_c_in, out_t &B_o_in) {
    W_f = W_f_in;
    W_i = W_i_in;
    W_c = W_c_in;
    W_o = W_o_in;
    B_f = B_f_in;
    B_i = B_i_in;
    B_c = B_c_in;
    B_o = B_o_in;
  }

  LSTM(const LSTM &src) = default;
  LSTM(LSTM &&src) = default;
  LSTM &operator=(const LSTM &rhs) = default;
  LSTM &operator=(LSTM &&rhs) = default;

  // Save parameters only; recurrent memory, caches, and gradients are
  // temporary.
  bool to_file(const std::string &file_path) const {
    std::ofstream file(file_path);
    if (!file)
      return false;

    file << "LSTM_V1 " << INPUT_SIZE << ' ' << OUTPUT_SIZE << '\n';
    file << std::setprecision(std::numeric_limits<T>::max_digits10);

    const auto write_values = [&file](const auto &values) {
      for (const auto &value : values)
        file << value << ' ';
      file << '\n';
    };
    // Gate order is forget, input, candidate, output; biases precede matrices.
    for (const auto *bias : {&B_f, &B_i, &B_c, &B_o})
      write_values(*bias);
    for (const auto *weights : {&W_f, &W_i, &W_c, &W_o}) {
      for (const auto &row : *weights)
        write_values(row);
    }

    file.close();
    return static_cast<bool>(file);
  }

  bool from_file(const std::string &file_path) {
    std::ifstream file(file_path);
    if (!file)
      return false;

    std::string format;
    size_t input_size, output_size;
    if (!(file >> format >> input_size >> output_size) || format != "LSTM_V1" ||
        input_size != INPUT_SIZE || output_size != OUTPUT_SIZE)
      return false;

    // Read into temporary storage so a bad file cannot partially replace
    // weights. Keep large matrices off the stack.
    struct Parameters {
      weight_t W_f{}, W_i{}, W_c{}, W_o{};
      out_t B_f{}, B_i{}, B_c{}, B_o{};
    };
    auto loaded = std::make_unique<Parameters>();
    const auto read_values = [&file](auto &values) {
      for (auto &value : values) {
        if (!(file >> value) || !std::isfinite(value))
          return false;
      }
      return true;
    };
    for (auto *bias :
         {&loaded->B_f, &loaded->B_i, &loaded->B_c, &loaded->B_o}) {
      if (!read_values(*bias))
        return false;
    }
    for (auto *weights :
         {&loaded->W_f, &loaded->W_i, &loaded->W_c, &loaded->W_o}) {
      for (auto &row : *weights) {
        if (!read_values(row))
          return false;
      }
    }
    file >> std::ws;
    if (file.bad() || !file.eof())
      return false;

    W_f = loaded->W_f;
    W_i = loaded->W_i;
    W_c = loaded->W_c;
    W_o = loaded->W_o;
    B_f = loaded->B_f;
    B_i = loaded->B_i;
    B_c = loaded->B_c;
    B_o = loaded->B_o;
    reset_state();
    return true;
  }

  out_t forward_pass(const in_t &x) { return forward_scalar(x); }

  // Training API

  out_t forward_pass_train(const in_t &x)
    requires(TRAIN_ENABLED)
  {
    return forward_scalar_train(x);
  }

  BackwardResult backward_pass(const out_t &hidden_gradient,
                               const out_t &cell_gradient)
    requires(TRAIN_ENABLED)
  {
    // Consume one saved forward step, starting with the most recent one.
    const StepCache &last_cached = training.cache.back();
    const auto &forget = last_cached.forget_gate;
    const auto &input = last_cached.input_gate;
    const auto &cand = last_cached.candidate_gate;
    const auto &output = last_cached.output_gate;

    auto tanh_cell = last_cached.new_cell;
    tanh_inplace(tanh_cell);

    BackwardResult result{};
    out_t forget_score_gradient{}, input_score_gradient{};
    out_t candidate_score_gradient{}, output_score_gradient{};

#pragma clang loop vectorize(enable)
#pragma clang loop interleave(enable)
    for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
      // Forward: hidden = output * tanh(cell).
      // Derivative through tanh: d_cell = d_hidden * output * (1 -
      // tanh(cell)^2). Add the cell gradient arriving directly from the
      // following time step.
      const T total_cell_gradient =
          cell_gradient[i] +
          hidden_gradient[i] * output[i] * (T{1} - tanh_cell[i] * tanh_cell[i]);

      // Forward: cell = forget * previous_cell + input * candidate.
      // Derivative with respect to previous_cell: d_previous_cell = d_cell *
      // forget.
      result.previous_cell_gradient[i] = total_cell_gradient * forget[i];

      // Forward: forget = sigmoid(forget_score).
      // d_forget = d_cell * previous_cell; sigmoid'(score) = forget * (1 -
      // forget).
      forget_score_gradient[i] = total_cell_gradient *
                                 last_cached.prev_l_t_mem[i] * forget[i] *
                                 (T{1} - forget[i]);

      // Forward: input = sigmoid(input_score).
      // d_input = d_cell * candidate; sigmoid'(score) = input * (1 - input).
      input_score_gradient[i] =
          total_cell_gradient * cand[i] * input[i] * (T{1} - input[i]);

      // Forward: candidate = tanh(candidate_score).
      // d_candidate = d_cell * input; tanh'(score) = 1 - candidate^2.
      candidate_score_gradient[i] =
          total_cell_gradient * input[i] * (T{1} - cand[i] * cand[i]);

      // Forward: hidden = output * tanh(cell); output = sigmoid(output_score).
      // d_output = d_hidden * tanh(cell); sigmoid'(score) = output * (1 -
      // output).
      output_score_gradient[i] =
          hidden_gradient[i] * tanh_cell[i] * output[i] * (T{1} - output[i]);
    }

    concat_t combined_input{};
    std::copy(last_cached.input.begin(), last_cached.input.end(),
              combined_input.begin());
    std::copy(last_cached.prev_s_t_mem.begin(), last_cached.prev_s_t_mem.end(),
              combined_input.begin() + INPUT_SIZE);
    concat_t combined_input_gradient{};

    // Forward, for each gate: score[row] = sum(W[row][col] * z[col]) +
    // bias[row]. Derivatives: d_W[row][col] = d_score[row] * z[col];
    // d_bias[row] = d_score[row]. Also d_z[col] = sum(W[row][col] *
    // d_score[row]), summed over all four gates.
    const auto accumulate_gate =
        [&](const weight_t &weights, weight_t &weight_gradients,
            out_t &bias_gradients, const out_t &score_gradient) {
          for (size_t row = 0; row < OUTPUT_SIZE; ++row) {
            const T gradient = score_gradient[row];
            bias_gradients[row] += gradient;
#pragma clang loop vectorize(enable)
#pragma clang loop interleave(enable)
            for (size_t col = 0; col < INPUT_SIZE + OUTPUT_SIZE; ++col) {
              weight_gradients[row][col] += gradient * combined_input[col];
              combined_input_gradient[col] += weights[row][col] * gradient;
            }
          }
        };
    accumulate_gate(W_f, training.W_f_grad, training.B_f_grad,
                    forget_score_gradient);
    accumulate_gate(W_i, training.W_i_grad, training.B_i_grad,
                    input_score_gradient);
    accumulate_gate(W_c, training.W_c_grad, training.B_c_grad,
                    candidate_score_gradient);
    accumulate_gate(W_o, training.W_o_grad, training.B_o_grad,
                    output_score_gradient);

    // Forward: z = [input, previous_hidden]. Split its gradient in the same
    // order.
    std::copy_n(combined_input_gradient.begin(), INPUT_SIZE,
                result.input_gradient.begin());
    std::copy_n(combined_input_gradient.begin() + INPUT_SIZE, OUTPUT_SIZE,
                result.previous_hidden_gradient.begin());

    // Clean up and return
    training.cache.pop_back();
    return result;
  }

  void apply_gradient(T learning_rate)
    requires(TRAIN_ENABLED)
  {
    for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
      for (size_t j = 0; j < OUTPUT_SIZE + INPUT_SIZE; ++j) {
        W_f[i][j] -= learning_rate * training.W_f_grad[i][j];
        W_i[i][j] -= learning_rate * training.W_i_grad[i][j];
        W_c[i][j] -= learning_rate * training.W_c_grad[i][j];
        W_o[i][j] -= learning_rate * training.W_o_grad[i][j];
      }

      B_f[i] -= learning_rate * training.B_f_grad[i];
      B_i[i] -= learning_rate * training.B_i_grad[i];
      B_c[i] -= learning_rate * training.B_c_grad[i];
      B_o[i] -= learning_rate * training.B_o_grad[i];
    }
  }
};
