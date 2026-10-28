#include "../src/wav_weights.hpp"
#include <cassert>
#include <stdexcept>

int main() {
  constexpr size_t rows = NETWORK_IN_VALUES / NUM_BANDS;
  Mask<NETWORK_IN_VALUES> mask;
  mask.mask_frequency(NUM_BANDS - 1);
  for (size_t row = 0; row < rows; ++row)
    for (size_t band = 0; band < NUM_BANDS; ++band)
      assert(mask.data()[row * NUM_BANDS + band] == (band == NUM_BANDS - 1));

  mask.reset();
  mask.mask_sequence(rows - 1);
  for (size_t row = 0; row < rows; ++row)
    for (size_t band = 0; band < NUM_BANDS; ++band)
      assert(mask.data()[row * NUM_BANDS + band] == (row == rows - 1));

  std::array<output_t, NETWORK_IN_VALUES> values;
  values.fill(1);
  WavWeights weights;
  weights.from_buffer(values);
  auto masked = weights.masked_data(mask);
  assert(masked.back() == 0 && masked.front() == 1);
  assert(weights.data().back() == 1);
  bool rejected = false;
  try { mask.mask_sequence(rows); } catch (const std::out_of_range &) { rejected = true; }
  assert(rejected);
  rejected = false;
  try { mask.mask_frequency(NUM_BANDS); } catch (const std::out_of_range &) { rejected = true; }
  assert(rejected);
}
