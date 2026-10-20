#pragma once

#include "utils.hpp"
#include "wav_file.hpp"
#include <cmath>
#include <span>
#include <utility>

// SIMD implementation for the computation of the Fourier component for each
// chunk, in an High Performance fashion
template <size_t FREQUENCY>
output_t compute_point_wave_point(
    std::span<const input_t, SAMPLES_PER_BLOCK> input_series) {
  static_assert(FREQUENCY % FREQUENCY_STEP == 0);
  static_assert(SAMPLES_PER_BLOCK % 4 == 0);

  constexpr auto sin_arr = get_sin_input_const<FREQUENCY>();
  constexpr auto cos_arr = get_cos_input_const<FREQUENCY>();
  simd4_t sin_sum = {};
  simd4_t cos_sum = {};
  for (size_t i = 0; i < SAMPLES_PER_BLOCK; i += 4) {
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

using compute_point_wave_fn =
    output_t (*)(std::span<const input_t, SAMPLES_PER_BLOCK>);

template <std::size_t... I>
constexpr auto make_compute_point_wave_table(std::index_sequence<I...>) {
  return std::array<compute_point_wave_fn, sizeof...(I)>{
      &compute_point_wave_point<FREQUENCIES[I]>...};
}

inline constexpr auto COMPUTE_POINT_WAVE_TABLE =
    make_compute_point_wave_table(std::make_index_sequence<NUM_FREQUENCIES>{});

// Crunch files data
[[gnu::always_inline]] inline void
crunck_file_data(WavFile &wf,
                 std::array<output_t, NETWORK_IN_VALUES> &buffer_out) {
  constexpr size_t NUM_WINDOWS =
      (N_SAMPLES - SAMPLES_PER_BLOCK) / SAMPLES_SHIFT + 1;
  const auto clip = wf.read_clip();
  for (size_t window_idx = 0; window_idx < NUM_WINDOWS; ++window_idx) {
    const std::span<const input_t, SAMPLES_PER_BLOCK> samples(
        clip.data() + window_idx * SAMPLES_SHIFT, SAMPLES_PER_BLOCK);
    const auto start_idx = get_starting_padding(window_idx);
    for (size_t band = 0; band < NUM_BANDS; ++band) {
      trig_t energy = 0;
      for (size_t j = 0; j < FREQUENCIES_PER_BAND; ++j) {
        const auto frequency_idx = band * FREQUENCIES_PER_BAND + j;
        const trig_t magnitude =
            COMPUTE_POINT_WAVE_TABLE[frequency_idx](samples);
        energy += magnitude * magnitude;
      }
      buffer_out[start_idx + band] = static_cast<output_t>(std::sqrt(energy));
    }
  }
}
