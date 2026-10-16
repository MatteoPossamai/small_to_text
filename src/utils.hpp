#pragma once
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <numbers>
#include <optional>
#include <queue>
#include <string>
#include <tuple>

// --- Constants ---
constexpr size_t SAMPLE_RATE = 16000; // Samples per second in the WAV.
constexpr size_t N_SAMPLES = 16000;   // Fixed number of network input samples.
constexpr float SAMPLE_RATE_F = static_cast<float>(SAMPLE_RATE);

constexpr size_t SAMPLES_PER_BLOCK = 400;
constexpr size_t SAMPLES_SHIFT = 200;
constexpr std::array<size_t, 20> FREQUENCIES = {
    400,  8000, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000,
    4400, 4800, 5200, 5600, 6000, 6400, 6800, 7200, 7600, 8000,
};
constexpr size_t NUM_FREQUENCIES = FREQUENCIES.size();

constexpr size_t NETWORK_IN_VALUES =
    ((N_SAMPLES - SAMPLES_PER_BLOCK) / SAMPLES_SHIFT + 1) * NUM_FREQUENCIES;

[[gnu::always_inline]] constexpr size_t get_starting_padding(size_t chunk_idx) {
  return NUM_FREQUENCIES * chunk_idx;
}

// --- Type aliasing ---
using input_t = int32_t;
using output_t = float_t;
using trig_t = double;
#if defined(__clang__)
using simd4_t = double __attribute__((ext_vector_type(4)));
#elif defined(__GNUC__)
using simd4_t = double __attribute__((vector_size(32)));
#endif

// --- Util functions ---
inline std::int16_t from_bytes_to_sample(const char *buff) {
  // WAV stores the low byte first; treat each char as a raw byte.
  const auto low = static_cast<std::uint8_t>(buff[0]);
  const auto high = static_cast<std::uint8_t>(buff[1]);
  const std::uint16_t raw = low | (std::uint16_t(high) << 8); // Little endian

  // Subtract 65536 when the 16-bit sign bit is set.
  // This to convert from unsigned to signed. If done via static_cast that
  // is platform dependent. So done explicitly
  const std::int32_t value = raw < 0x8000 ? raw : std::int32_t(raw) - 0x10000;
  return static_cast<std::int16_t>(value);
}

// constexpr alternative for sin
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

// constexpr alternative for sin
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
  return tick / SAMPLE_RATE_F;
}

// Compile time sin waves function for sample computing
template <size_t FREQUENCY>
consteval std::array<trig_t, N_SAMPLES> get_sin_input_const() {
  std::array<trig_t, N_SAMPLES> res{};
  for (size_t i = 0; i < N_SAMPLES; ++i) {
    const size_t phase = (FREQUENCY * i) % SAMPLE_RATE;
    res[i] = sin_constexpr(2 * std::numbers::pi_v<trig_t> * phase / SAMPLE_RATE);
  }
  return res;
}

// Compile time cos waves function for sample computing
template <size_t FREQUENCY>
consteval std::array<trig_t, N_SAMPLES> get_cos_input_const() {
  std::array<trig_t, N_SAMPLES> res{};
  for (size_t i = 0; i < N_SAMPLES; ++i) {
    const size_t phase = (FREQUENCY * i) % SAMPLE_RATE;
    res[i] = cos_constexpr(2 * std::numbers::pi_v<trig_t> * phase / SAMPLE_RATE);
  }
  return res;
}

template <typename T> class s_queue {
  // Simple thread safe queue
  // Producing is single threaded, consumption
  // is parallel
  std::mutex m{};
  std::queue<T> queue{};

public:
  void put(T val) {
    // No need to be safe on put, since
    // is pre-populated with one producer, while no
    // one is consuming
    queue.push(val);
  }

  std::optional<T> pop() {
    std::lock_guard<std::mutex> lk(m);
    if (queue.size() > 0) {
      auto res = queue.front();
      queue.pop();
      return res;
    }
    return std::nullopt;
  }
};

using weights_queue_t = s_queue<std::tuple<std::string, std::string>>;
