#pragma once

#include "utils.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>

template <size_t NETWORK_IN> class Mask {
  static_assert(NETWORK_IN % NUM_BANDS == 0);
  std::array<bool, NETWORK_IN> mask{};

public:
  Mask() = default;
  Mask(std::array<bool, NETWORK_IN> mask_in) : mask(mask_in) {}

  void mask_frequency(size_t band) {
    if (band >= NUM_BANDS)
      throw std::out_of_range("Frequency band outside input");
    for (size_t index = band; index < NETWORK_IN; index += NUM_BANDS)
      mask[index] = true;
  }

  void mask_sequence(size_t row) {
    if (row >= NETWORK_IN / NUM_BANDS)
      throw std::out_of_range("Time row outside input");
    const size_t start = NUM_BANDS * row;
    std::fill(mask.begin() + start, mask.begin() + start + NUM_BANDS, true);
  }

  void reset() { mask.fill(false); }
  const std::array<bool, NETWORK_IN> &data() const { return mask; }
};
