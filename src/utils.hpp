#pragma once
#include <cstdint>

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
