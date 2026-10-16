#include "utils.hpp"
#include <cmath>

// SIMD implementation for the computation of the Fourier component for each
// chunk, in an High Performance fashion
template <size_t FREQUENCY>
output_t
compute_point_wave_point(const std::array<input_t, N_SAMPLES> &input_series) {
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
