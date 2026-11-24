#include <array>
#include <cmath>
#include <cstddef>

template <typename T, size_t M, size_t N>
T matrix_mul_scalar_max(const std::array<std::array<T, N>, M> &m1,
                        const std::array<T, N> &m2, std::array<T, M> &res) {
  // MxN (dot) Nx1 = Mx1
  for (size_t i = 0; i < M; ++i) {
#pragma clang loop vectorize(enable)
#pragma clang loop interleave(enable)
    for (size_t j = 0; j < N; ++j) {
      res[i] += m2[j] * m1[i][j];
    }
  }

  T max = res[0];
  for (size_t i = 0; i < M; ++i) {
    max = std::max(max, res[i]);
  }
  return max;
}

template <std::floating_point T, size_t N, size_t M>
std::array<T, N> attend(const std::array<T, N> &query,
                        const std::array<std::array<T, N>, M> &rows) {
  static_assert(N % 2 == 0);
  static_assert(M % 2 == 0);
  static_assert(N > 0 && M > 0);

  std::array<T, M> dot{};
  T max = matrix_mul_scalar_max(rows, query, dot);

  T total = 0;
  // Non vectorizable
  for (size_t i = 0; i < M; ++i) {
    auto value = std::exp(dot[i] - max);
    dot[i] = value;
    total += value;
  }

  // vectorizable and unrolled
#pragma clang loop vectorize(enable)
#pragma clang loop interleave(enable)
  for (size_t i = 0; i < M; ++i) {
    dot[i] /= total;
  }

  std::array<T, N> res{};
  for (size_t i = 0; i < M; ++i) {
    T scale_val = dot[i];
#pragma clang loop vectorize(enable)
#pragma clang loop interleave(enable)
    for (size_t j = 0; j < N; ++j) {
      res[j] += rows[i][j] * scale_val;
    }
  }

  return res;
}
