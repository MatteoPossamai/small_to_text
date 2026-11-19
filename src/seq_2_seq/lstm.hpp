#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

enum class Platform { X86_64, VULKAN_GPU, ANY };

template <std::floating_point T, size_t INPUT_SIZE, size_t OUTPUT_SIZE,
          Platform platform>
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

  // Training cache
  std::vector<StepCache> cache;

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

  out_t forward_scalar_train(const in_t &x) {
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

    cache.push_back(cache_i);
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
    cache.clear();

    for (size_t i = 0; i < OUTPUT_SIZE; ++i) {
      W_f_grad[i].fill(0);
      W_i_grad[i].fill(0);
      W_c_grad[i].fill(0);
      W_o_grad[i].fill(0);
    }

    B_f_grad.fill(0);
    B_i_grad.fill(0);
    B_c_grad.fill(0);
    B_o_grad.fill(0);
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

  out_t forward_pass(const in_t &x) {
    // if constexpr (platform == Platform::X86_64) {
    //   return forward_x86(x); // TODO: implement
    // } else {
    return forward_scalar(x);
    // }
  }

  // Training API
  // TODO: can be wrapped at compile time to avoid memory loss at inference?

  out_t forward_pass_train(const in_t &x) {
    // TODO: x86 one
    return forward_scalar_train(x);
  }

  BackwardResult backward_pass(const out_t &hidden_gradient,
                               const out_t &cell_gradient) {
    // Consume one saved forward step, starting with the most recent one.
    const StepCache &last_cached = cache.back();
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
    accumulate_gate(W_f, W_f_grad, B_f_grad, forget_score_gradient);
    accumulate_gate(W_i, W_i_grad, B_i_grad, input_score_gradient);
    accumulate_gate(W_c, W_c_grad, B_c_grad, candidate_score_gradient);
    accumulate_gate(W_o, W_o_grad, B_o_grad, output_score_gradient);

    // Forward: z = [input, previous_hidden]. Split its gradient in the same
    // order.
    std::copy_n(combined_input_gradient.begin(), INPUT_SIZE,
                result.input_gradient.begin());
    std::copy_n(combined_input_gradient.begin() + INPUT_SIZE, OUTPUT_SIZE,
                result.previous_hidden_gradient.begin());

    // Clean up and return
    cache.pop_back();
    return result;
  }

  void apply_gradient(T learning_rate) {
    for (size_t i=0;i<OUTPUT_SIZE; ++i) {
      for (size_t j=0;j<OUTPUT_SIZE + INPUT_SIZE; ++j) {
        W_f[i][j] -= learning_rate * W_f_grad[i][j];
        W_i[i][j] -= learning_rate * W_i_grad[i][j];
        W_c[i][j] -= learning_rate * W_c_grad[i][j];
        W_o[i][j] -= learning_rate * W_o_grad[i][j];
      }

      B_f[i] -= learning_rate * B_f_grad[i];
      B_i[i] -= learning_rate * B_i_grad[i];
      B_c[i] -= learning_rate * B_c_grad[i];
      B_o[i] -= learning_rate * B_o_grad[i];
    }
  }

private:
  // Weight's Gradient
  weight_t W_f_grad{}; // forget
  weight_t W_i_grad{}; // input
  weight_t W_c_grad{}; // candidate
  weight_t W_o_grad{}; // output

  // Biases
  out_t B_f_grad{}; // forget
  out_t B_i_grad{}; // input
  out_t B_c_grad{}; // candidate
  out_t B_o_grad{}; // output
};
