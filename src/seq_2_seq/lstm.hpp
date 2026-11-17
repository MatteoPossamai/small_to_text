#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>

enum class Platform { X86_64, VULKAN_GPU, ANY };

template <std::floating_point T, size_t INPUT_SIZE, size_t OUTPUT_SIZE,
          Platform platform>
class LSTM {

  static_assert(INPUT_SIZE > 0);
  static_assert(OUTPUT_SIZE % 8 == 0 && OUTPUT_SIZE > 0);

  using mem_t = std::array<T, OUTPUT_SIZE>;
  using in_t = std::array<T, INPUT_SIZE>;
  using out_t = std::array<T, OUTPUT_SIZE>;
  using weight_t =
      std::array<std::array<T, OUTPUT_SIZE + INPUT_SIZE>, OUTPUT_SIZE>;
  using concat_t = std::array<T, OUTPUT_SIZE + INPUT_SIZE>;

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
  LSTM() = default;
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
};
