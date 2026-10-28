#pragma once

#include "utils.hpp"
#include <algorithm>
#include <array>
#include <stdexcept>

// Move the middle row by displacement, interpolating each band between rows.
template <size_t N>
std::array<output_t, N> time_warp(const std::array<output_t, N> &input,
                                  int displacement) {
  static_assert(N % NUM_BANDS == 0);
  constexpr size_t rows = N / NUM_BANDS;
  constexpr size_t middle = rows / 2;
  const int moved = int(middle) + displacement;
  if (moved <= 0 || moved >= int(rows - 1))
    throw std::out_of_range("Warp midpoint outside input");
  std::array<output_t, N> result{};
  for (size_t row = 0; row < rows; ++row) {
    const output_t source = row <= size_t(moved)
        ? output_t(row) * middle / moved
        : middle + output_t(row - moved) * (rows - 1 - middle) / (rows - 1 - moved);
    const size_t lower = size_t(source);
    const size_t upper = std::min(lower + 1, rows - 1);
    const output_t fraction = source - lower;
    for (size_t band = 0; band < NUM_BANDS; ++band)
      result[row * NUM_BANDS + band] = input[lower * NUM_BANDS + band] * (1 - fraction) +
                                       input[upper * NUM_BANDS + band] * fraction;
  }
  return result;
}
