// Performance for matrix multiplication
//-----------------------------------------------------------
// Benchmark                 Time             CPU   Iterations
// -----------------------------------------------------------
// BM_Compute_1<N1>    2033388 ns      2022479 ns          393
// BM_Compute_1<N2>       6999 ns         6987 ns        95968
// BM_Compute_1<N3>        824 ns          822 ns       786671
// BM_Compute_2<N1>     382263 ns       379773 ns         1836
// BM_Compute_2<N2>       1049 ns         1046 ns       680695
// BM_Compute_2<N3>        257 ns          256 ns      2742549
// BM_Compute_3<N1>     407006 ns       402826 ns         1767
// BM_Compute_3<N2>       1092 ns         1088 ns       636166
// BM_Compute_3<N3>        301 ns          300 ns      2390641

#include <array>
#include <benchmark/benchmark.h>
#include <cmath>
#include <cstddef>
#include <random>

constexpr size_t N1 = 1580;
constexpr size_t N2 = 128;
constexpr size_t N3 = 64;

using output_t = float_t;
#if defined(__clang__)
using simd4_t = output_t __attribute__((ext_vector_type(4)));
#elif defined(__GNUC__)
using simd4_t = output_t __attribute__((vector_size(4 * sizeof(output_t))));
#endif

// --- Helpers
std::random_device r;
std::default_random_engine generator(r());
std::uniform_int_distribution<size_t> distribution(0, 100);

template <size_t DIM> std::array<output_t, DIM> init_res() {
  std::array<output_t, DIM> res{};
  for (size_t i = 0; i < DIM; ++i) {
    res[i] = 0;
  }
  return res;
}

template <size_t DIM> std::array<output_t, DIM> init_in() {
  std::array<output_t, DIM> res{};
  for (size_t i = 0; i < DIM; ++i) {
    res[i] = static_cast<output_t>(distribution(generator));
  }
  return res;
}

template <size_t DIM>
std::array<std::array<output_t, DIM>, DIM> init_weights() {
  std::array<std::array<output_t, DIM>, DIM> res{};
  for (size_t i = 0; i < DIM; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      res[i][j] += static_cast<output_t>(distribution(generator));
    }
  }
  return res;
}

// 1. Column major
template <size_t DIM>
void matrix_mul_1(std::array<output_t, DIM> &inputs,
                  std::array<std::array<output_t, DIM>, DIM> &weights,
                  std::array<output_t, DIM> &res) {
  auto N = weights.size();
  for (size_t j = 0; j < N; ++j) {
    for (size_t i = 0; i < N; ++i) {
      res[i] += inputs[j] * weights[i][j];
    }
  }
}

// 2. Row major
template <size_t DIM>
void matrix_mul_2(std::array<output_t, DIM> &inputs,
                  std::array<std::array<output_t, DIM>, DIM> &weights,
                  std::array<output_t, DIM> &res) {
  auto N = weights.size();
  for (size_t i = 0; i < N; ++i) {
    output_t total = 0;
    for (size_t j = 0; j < N; ++j) {
      total += inputs[j] * weights[i][j];
    }
    res[i] = total;
  }
}

// 3. Vectorize with two independent accumulators
template <size_t DIM>
void matrix_mul_3(std::array<output_t, DIM> &inputs,
                  std::array<std::array<output_t, DIM>, DIM> &weights,
                  std::array<output_t, DIM> &res) {
  static_assert(DIM % 4 == 0);
  auto N = weights.size();
  for (size_t i = 0; i < N; ++i) {
    simd4_t total0 = {};
    simd4_t total1 = {};
    size_t j = 0;
    for (; j + 8 <= N; j += 8) {
      simd4_t in0 = {inputs[j], inputs[j + 1], inputs[j + 2], inputs[j + 3]};
      simd4_t we0 = {weights[i][j], weights[i][j + 1], weights[i][j + 2],
                     weights[i][j + 3]};
      simd4_t in1 = {inputs[j + 4], inputs[j + 5], inputs[j + 6],
                     inputs[j + 7]};
      simd4_t we1 = {weights[i][j + 4], weights[i][j + 5], weights[i][j + 6],
                     weights[i][j + 7]};
      total0 += in0 * we0;
      total1 += in1 * we1;
    }
    // DIM can leave four elements after the unrolled loop (e.g. 1580).
    if (j < N) {
      simd4_t in = {inputs[j], inputs[j + 1], inputs[j + 2], inputs[j + 3]};
      simd4_t we = {weights[i][j], weights[i][j + 1], weights[i][j + 2],
                    weights[i][j + 3]};
      total0 += in * we;
    }
    simd4_t total = total0 + total1;
    res[i] = (total[0] + total[1]) + (total[2] + total[3]);
  }
}

// Benchmark
template <size_t DIM> static void BM_Compute_1(benchmark::State &state) {
  static_assert(DIM % 4 == 0);
  auto in = init_in<DIM>();
  auto weights = init_weights<DIM>();
  auto res = init_res<DIM>();
  auto *out = res.data();
  benchmark::DoNotOptimize(out);
  for (auto _ : state) {
    matrix_mul_1(in, weights, res);
    benchmark::ClobberMemory();
  }
}
BENCHMARK_TEMPLATE(BM_Compute_1, N1);
BENCHMARK_TEMPLATE(BM_Compute_1, N2);
BENCHMARK_TEMPLATE(BM_Compute_1, N3);

template <size_t DIM> static void BM_Compute_2(benchmark::State &state) {
  static_assert(DIM % 4 == 0);
  auto in = init_in<DIM>();
  auto weights = init_weights<DIM>();
  auto res = init_res<DIM>();
  auto *out = res.data();
  benchmark::DoNotOptimize(out);
  for (auto _ : state) {
    matrix_mul_2(in, weights, res);
    benchmark::ClobberMemory();
  }
}
BENCHMARK_TEMPLATE(BM_Compute_2, N1);
BENCHMARK_TEMPLATE(BM_Compute_2, N2);
BENCHMARK_TEMPLATE(BM_Compute_2, N3);

template <size_t DIM> static void BM_Compute_3(benchmark::State &state) {
  static_assert(DIM % 4 == 0);
  auto in = init_in<DIM>();
  auto weights = init_weights<DIM>();
  auto res = init_res<DIM>();
  auto *out = res.data();
  benchmark::DoNotOptimize(out);
  for (auto _ : state) {
    matrix_mul_3(in, weights, res);
    benchmark::ClobberMemory();
  }
}
BENCHMARK_TEMPLATE(BM_Compute_3, N1);
BENCHMARK_TEMPLATE(BM_Compute_3, N2);
BENCHMARK_TEMPLATE(BM_Compute_3, N3);

BENCHMARK_MAIN();
