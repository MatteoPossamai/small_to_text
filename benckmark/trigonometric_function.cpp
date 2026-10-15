// Benchmark of the performance of different ways of computing the output for
// the compressed weights for the NN
// Results:
// -------------------------------------------------------------
// Benchmark                   Time             CPU   Iterations
// -------------------------------------------------------------
// BM_Compute_1<400>      389168 ns       388329 ns         1802
// BM_Compute_1<1600>     383364 ns       382437 ns         1832
// BM_Compute_1<2000>     373667 ns       371511 ns         1884
// BM_Compute_1<4000>     352046 ns       350347 ns         2000
// BM_Compute_1<8000>     334368 ns       333811 ns         2099
// BM_Compute_2<400>      270271 ns       269730 ns         2596
// BM_Compute_2<1600>     266992 ns       266442 ns         2629
// BM_Compute_2<2000>     258482 ns       256513 ns         2746
// BM_Compute_2<4000>     238543 ns       237897 ns         2944
// BM_Compute_2<8000>     237894 ns       237413 ns         2951
// BM_Compute_3<400>       23592 ns        23552 ns        29716
// BM_Compute_3<1600>      23613 ns        23569 ns        29711
// BM_Compute_3<2000>      23598 ns        23553 ns        29739
// BM_Compute_3<4000>      23931 ns        23686 ns        29817
// BM_Compute_3<8000>      23617 ns        23573 ns        29709
// BM_Compute_4<400>        6583 ns         6572 ns       106493
// BM_Compute_4<1600>       6598 ns         6582 ns       106516
// BM_Compute_4<2000>       6602 ns         6590 ns       106348
// BM_Compute_4<4000>       6606 ns         6591 ns       106000
// BM_Compute_4<8000>       6585 ns         6574 ns       105958

#include <array>
#include <benchmark/benchmark.h>
#include <cmath>
#include <numbers>
#include <random>

// Definitions
using input_t = int32_t;
using output_t = float_t;
using trig_t = double;
#if defined(__clang__)
using simd4_t = double __attribute__((ext_vector_type(4)));
#elif defined(__GNUC__)
using simd4_t = double __attribute__((vector_size(32)));
#endif
constexpr size_t N_SAMPLES = 16000;
constexpr float N_SAMPLES_F = static_cast<float>(N_SAMPLES);

// C++20 does not make std::sin/std::cos constexpr. Reduce the angle to
// [-pi, pi], then evaluate their Taylor series at compile time.
constexpr trig_t sin_constexpr(trig_t x) {
  constexpr trig_t pi = std::numbers::pi_v<trig_t>;
  constexpr trig_t two_pi = 2 * pi;
  while (x > pi)
    x -= two_pi;
  while (x < -pi)
    x += two_pi;

  trig_t term = x;
  trig_t sum = x;
  for (int n = 1; n <= 12; ++n) {
    term *= -x * x / ((2 * n) * (2 * n + 1));
    sum += term;
  }
  return sum;
}

constexpr trig_t cos_constexpr(trig_t x) {
  constexpr trig_t pi = std::numbers::pi_v<trig_t>;
  constexpr trig_t two_pi = 2 * pi;
  while (x > pi)
    x -= two_pi;
  while (x < -pi)
    x += two_pi;

  trig_t term = 1;
  trig_t sum = 1;
  for (int n = 1; n <= 12; ++n) {
    term *= -x * x / ((2 * n - 1) * (2 * n));
    sum += term;
  }
  return sum;
}

[[gnu::always_inline]] constexpr float get_time_from_idx(size_t tick) {
  return tick / N_SAMPLES_F;
}

std::array<input_t, N_SAMPLES> generate_array() {
  std::random_device r;
  std::default_random_engine generator(r());
  std::uniform_int_distribution<input_t> distribution(0, 100);
  std::array<input_t, N_SAMPLES> res = {};
  for (std::size_t i = 0; i < N_SAMPLES; i++) {
    res[i] = distribution(generator);
  }
  return res;
}

// 1. Baseline - Good Looking code
template <size_t FREQUENCY>
output_t sin(const std::array<input_t, N_SAMPLES> &input_series) {
  output_t res = 0;
  for (size_t i = 0; i < N_SAMPLES; i++) {
    res += input_series[i] *
           std::sin(2 * std::numbers::pi * FREQUENCY * get_time_from_idx(i));
  }
  return res;
}

template <size_t FREQUENCY>
output_t cos(const std::array<input_t, N_SAMPLES> &input_series) {
  output_t res = 0;
  for (size_t i = 0; i < N_SAMPLES; i++) {
    res += input_series[i] *
           std::cos(2 * std::numbers::pi * FREQUENCY * get_time_from_idx(i));
  }
  return res;
}

template <size_t FREQUENCY>
output_t compute_1(const std::array<input_t, N_SAMPLES> &input_series) {
  return std::sqrt(
      static_cast<output_t>(std::pow(sin<FREQUENCY>(input_series), 2)) +
      static_cast<output_t>(std::pow(cos<FREQUENCY>(input_series), 2)));
}

// 2. Loop unrolling
template <size_t FREQUENCY>
output_t compute_2(const std::array<input_t, N_SAMPLES> &input_series) {
  static_assert(N_SAMPLES % 4 == 0);
  static_assert(FREQUENCY % 400 == 0);

  output_t sin_sum_1 = 0;
  output_t sin_sum_2 = 0;
  output_t cos_sum_1 = 0;
  output_t cos_sum_2 = 0;

  for (size_t i = 0; i < N_SAMPLES; i += 4) {
    sin_sum_1 +=
        (input_series[i] *
             std::sin(2 * std::numbers::pi * FREQUENCY * get_time_from_idx(i)) +
         input_series[i + 1] * std::sin(2 * std::numbers::pi * FREQUENCY *
                                        get_time_from_idx(i + 1)));
    sin_sum_2 +=
        (input_series[i + 2] * std::sin(2 * std::numbers::pi * FREQUENCY *
                                        get_time_from_idx(i + 2)) +
         input_series[i + 3] * std::sin(2 * std::numbers::pi * FREQUENCY *
                                        get_time_from_idx(i + 3)));
    cos_sum_1 +=
        (input_series[i] *
             std::cos(2 * std::numbers::pi * FREQUENCY * get_time_from_idx(i)) +
         input_series[i + 1] * std::cos(2 * std::numbers::pi * FREQUENCY *
                                        get_time_from_idx(i + 1)));
    cos_sum_2 +=
        (input_series[i + 2] * std::cos(2 * std::numbers::pi * FREQUENCY *
                                        get_time_from_idx(i + 2)) +
         input_series[i + 3] * std::cos(2 * std::numbers::pi * FREQUENCY *
                                        get_time_from_idx(i + 3)));
  }

  return std::sqrt(static_cast<output_t>(std::pow(sin_sum_1 + sin_sum_2, 2)) +
                   static_cast<output_t>(std::pow(cos_sum_1 + cos_sum_2, 2)));
}

// 3. Compile time lists
template <size_t FREQUENCY>
consteval std::array<trig_t, N_SAMPLES> get_sin_input_const() {
  std::array<trig_t, N_SAMPLES> res{};
  for (size_t i = 0; i < N_SAMPLES; ++i) {
    const size_t phase = (FREQUENCY * i) % N_SAMPLES;
    res[i] = sin_constexpr(2 * std::numbers::pi_v<trig_t> * phase / N_SAMPLES);
  }
  return res;
}

template <size_t FREQUENCY>
consteval std::array<trig_t, N_SAMPLES> get_cos_input_const() {
  std::array<trig_t, N_SAMPLES> res{};
  for (size_t i = 0; i < N_SAMPLES; ++i) {
    const size_t phase = (FREQUENCY * i) % N_SAMPLES;
    res[i] = cos_constexpr(2 * std::numbers::pi_v<trig_t> * phase / N_SAMPLES);
  }
  return res;
}

template <size_t FREQUENCY>
output_t compute_3(const std::array<input_t, N_SAMPLES> &input_series) {
  static_assert(N_SAMPLES % 4 == 0);
  static_assert(FREQUENCY % 400 == 0);

  constexpr auto sin_arr = get_sin_input_const<FREQUENCY>();
  constexpr auto cos_arr = get_cos_input_const<FREQUENCY>();

  output_t sin_sum_1 = 0;
  output_t sin_sum_2 = 0;
  output_t cos_sum_1 = 0;
  output_t cos_sum_2 = 0;

  for (size_t i = 0; i < N_SAMPLES; i += 4) {
    sin_sum_1 +=
        (input_series[i] * sin_arr[i] + input_series[i + 1] * sin_arr[i + 1]);
    sin_sum_2 += (input_series[i + 2] * sin_arr[i + 2] +
                  input_series[i + 3] * sin_arr[i + 3]);
    cos_sum_1 +=
        (input_series[i] * cos_arr[i] + input_series[i + 1] * cos_arr[i + 1]);
    cos_sum_2 += (input_series[i + 2] * cos_arr[i + 2] +
                  input_series[i + 3] * cos_arr[i + 3]);
  }

  return std::sqrt(static_cast<output_t>(std::pow(sin_sum_1 + sin_sum_2, 2)) +
                   static_cast<output_t>(std::pow(cos_sum_1 + cos_sum_2, 2)));
}

// 4. Explicit four-lane SIMD arithmetic. The compiler vector type guarantees
// that the multiply and add operations use vector instructions.
template <size_t FREQUENCY>
output_t compute_4(const std::array<input_t, N_SAMPLES> &input_series) {
  static_assert(FREQUENCY % 400 == 0);
  static_assert(N_SAMPLES % 4 == 0);

  constexpr auto sin_arr = get_sin_input_const<FREQUENCY>();
  constexpr auto cos_arr = get_cos_input_const<FREQUENCY>();
  simd4_t sin_sum = {};
  simd4_t cos_sum = {};
  for (size_t i = 0; i < N_SAMPLES; i += 4) {
    const simd4_t samples = {static_cast<trig_t>(input_series[i]),
                             static_cast<trig_t>(input_series[i + 1]),
                             static_cast<trig_t>(input_series[i + 2]),
                             static_cast<trig_t>(input_series[i + 3])};
    const simd4_t sin_values = {sin_arr[i], sin_arr[i + 1], sin_arr[i + 2],
                                sin_arr[i + 3]};
    const simd4_t cos_values = {cos_arr[i], cos_arr[i + 1], cos_arr[i + 2],
                                cos_arr[i + 3]};
    sin_sum += samples * sin_values;
    cos_sum += samples * cos_values;
  }

  const trig_t sin_total = sin_sum[0] + sin_sum[1] + sin_sum[2] + sin_sum[3];
  const trig_t cos_total = cos_sum[0] + cos_sum[1] + cos_sum[2] + cos_sum[3];
  return std::sqrt(sin_total * sin_total + cos_total * cos_total);
}

// Benchmark 1
template <size_t FREQUENCY> static void BM_Compute_1(benchmark::State &state) {
  auto in = generate_array();
  output_t res = 0;
  for (auto _ : state) {
    res = compute_1<FREQUENCY>(in);
    benchmark::DoNotOptimize(res);
  }
}

BENCHMARK_TEMPLATE(BM_Compute_1, 400);
BENCHMARK_TEMPLATE(BM_Compute_1, 1600);
BENCHMARK_TEMPLATE(BM_Compute_1, 2000);
BENCHMARK_TEMPLATE(BM_Compute_1, 4000);
BENCHMARK_TEMPLATE(BM_Compute_1, 8000);

// Benchmark 2
template <size_t FREQUENCY> static void BM_Compute_2(benchmark::State &state) {
  auto in = generate_array();
  output_t res = 0;
  for (auto _ : state) {
    res = compute_2<FREQUENCY>(in);
    benchmark::DoNotOptimize(res);
  }
}

BENCHMARK_TEMPLATE(BM_Compute_2, 400);
BENCHMARK_TEMPLATE(BM_Compute_2, 1600);
BENCHMARK_TEMPLATE(BM_Compute_2, 2000);
BENCHMARK_TEMPLATE(BM_Compute_2, 4000);
BENCHMARK_TEMPLATE(BM_Compute_2, 8000);

// Benchmark 3
template <size_t FREQUENCY> static void BM_Compute_3(benchmark::State &state) {
  auto in = generate_array();
  output_t res = 0;
  for (auto _ : state) {
    res = compute_3<FREQUENCY>(in);
    benchmark::DoNotOptimize(res);
  }
}

BENCHMARK_TEMPLATE(BM_Compute_3, 400);
BENCHMARK_TEMPLATE(BM_Compute_3, 1600);
BENCHMARK_TEMPLATE(BM_Compute_3, 2000);
BENCHMARK_TEMPLATE(BM_Compute_3, 4000);
BENCHMARK_TEMPLATE(BM_Compute_3, 8000);

// Benchmark 4
template <size_t FREQUENCY> static void BM_Compute_4(benchmark::State &state) {
  auto in = generate_array();
  output_t res = 0;
  for (auto _ : state) {
    res = compute_4<FREQUENCY>(in);
    benchmark::DoNotOptimize(res);
  }
}

BENCHMARK_TEMPLATE(BM_Compute_4, 400);
BENCHMARK_TEMPLATE(BM_Compute_4, 1600);
BENCHMARK_TEMPLATE(BM_Compute_4, 2000);
BENCHMARK_TEMPLATE(BM_Compute_4, 4000);
BENCHMARK_TEMPLATE(BM_Compute_4, 8000);

BENCHMARK_MAIN();
