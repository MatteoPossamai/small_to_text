#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace audio_decoder {

inline constexpr std::size_t feature_count = 40;

template <std::floating_point T> using Frame = std::array<T, feature_count>;

namespace detail {

inline constexpr std::uint32_t sample_rate = 16000;
inline constexpr std::size_t window_samples = 400; // 25 ms
inline constexpr std::size_t hop_samples = 160;    // 10 ms
inline constexpr std::size_t fft_size = 512;
inline constexpr std::size_t spectrum_bins = fft_size / 2 + 1;
inline constexpr double energy_floor = 1e-10;

using Complex = std::complex<double>;
using Spectrum = std::array<Complex, fft_size>;

inline std::uint16_t u16_le(const unsigned char *bytes) {
  return std::uint16_t(bytes[0]) | (std::uint16_t(bytes[1]) << 8);
}

inline std::uint32_t u32_le(const unsigned char *bytes) {
  return std::uint32_t(u16_le(bytes)) |
         (std::uint32_t(u16_le(bytes + 2)) << 16);
}

inline void read_bytes(std::istream &file, std::span<unsigned char> bytes) {
  file.read(reinterpret_cast<char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  if (!file)
    throw std::runtime_error("Unexpected end of WAV file");
}

inline std::vector<std::int16_t>
read_wav_samples(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file)
    throw std::runtime_error("Could not open WAV file: " + path.string());

  const std::streamoff file_size = file.tellg();
  file.seekg(0);
  std::array<unsigned char, 12> riff;
  read_bytes(file, riff);
  if (std::memcmp(riff.data(), "RIFF", 4) != 0 ||
      std::memcmp(riff.data() + 8, "WAVE", 4) != 0)
    throw std::runtime_error("Expected a RIFF/WAVE file");

  const auto riff_end = std::uint64_t{8} + u32_le(riff.data() + 4);
  if (file_size < 0 || riff_end < riff.size() ||
      riff_end > static_cast<std::uint64_t>(file_size))
    throw std::runtime_error("Invalid WAV file size");

  bool found_format = false;
  std::uint64_t data_offset = 0;
  std::uint32_t data_size = 0;
  std::uint64_t offset = riff.size();
  while (riff_end - offset >= 8 && (!found_format || data_offset == 0)) {
    std::array<unsigned char, 8> chunk;
    read_bytes(file, chunk);
    offset += chunk.size();
    const auto size = u32_le(chunk.data() + 4);
    const auto padded_size = std::uint64_t{size} + (size & 1u);
    if (padded_size > riff_end - offset)
      throw std::runtime_error("WAV chunk exceeds RIFF size");

    if (std::memcmp(chunk.data(), "fmt ", 4) == 0 && !found_format) {
      std::array<unsigned char, 16> format;
      if (size < format.size())
        throw std::runtime_error("WAV fmt chunk is too small");
      read_bytes(file, format);
      if (u16_le(format.data()) != 1 || u16_le(format.data() + 2) != 1 ||
          u32_le(format.data() + 4) != sample_rate ||
          u16_le(format.data() + 12) != 2 || u16_le(format.data() + 14) != 16)
        throw std::runtime_error("Expected 16 kHz mono PCM16 WAV audio");
      found_format = true;
    } else if (std::memcmp(chunk.data(), "data", 4) == 0 && data_offset == 0) {
      data_offset = offset;
      data_size = size;
    }

    offset += padded_size;
    file.seekg(static_cast<std::streamoff>(offset));
  }

  if (!found_format || data_offset == 0)
    throw std::runtime_error("WAV file is missing its fmt or data chunk");
  if (data_size % 2 != 0)
    throw std::runtime_error("WAV data ends in a partial PCM16 sample");

  file.seekg(static_cast<std::streamoff>(data_offset));
  std::vector<std::int16_t> samples;
  samples.reserve(data_size / 2);
  std::array<unsigned char, 8192> buffer;
  for (std::size_t remaining = data_size; remaining > 0;) {
    const auto count = std::min(remaining, buffer.size());
    read_bytes(file, std::span{buffer}.first(count));
    for (std::size_t i = 0; i < count; i += 2)
      samples.push_back(std::bit_cast<std::int16_t>(u16_le(buffer.data() + i)));
    remaining -= count;
  }
  return samples;
}

inline const auto &hann_window() {
  static const auto window = [] {
    std::array<double, window_samples> values;
    for (std::size_t i = 0; i < values.size(); ++i) {
      const double phase = 2 * std::numbers::pi * i / (values.size() - 1);
      values[i] = 0.5 - 0.5 * std::cos(phase);
    }
    return values;
  }();
  return window;
}

inline void fft(Spectrum &values) {
  static const auto twiddles = [] {
    std::array<Complex, fft_size / 2> values;
    for (std::size_t i = 0; i < values.size(); ++i) {
      const double phase = -2 * std::numbers::pi * i / fft_size;
      values[i] = {std::cos(phase), std::sin(phase)};
    }
    return values;
  }();

  // Bit-reverse the input order for the iterative radix-2 FFT.
  for (std::size_t i = 1, reversed = 0; i < fft_size; ++i) {
    std::size_t bit = fft_size / 2;
    for (; reversed & bit; bit >>= 1)
      reversed ^= bit;
    reversed ^= bit;
    if (i < reversed)
      std::swap(values[i], values[reversed]);
  }

  for (std::size_t width = 2; width <= fft_size; width *= 2) {
    const std::size_t half = width / 2;
    const std::size_t stride = fft_size / width;
    for (std::size_t start = 0; start < fft_size; start += width) {
      for (std::size_t i = 0; i < half; ++i) {
        const auto even = values[start + i];
        const auto odd = twiddles[i * stride] * values[start + i + half];
        values[start + i] = even + odd;
        values[start + i + half] = even - odd;
      }
    }
  }
}

struct MelFilter {
  std::size_t first_bin;
  std::size_t end_bin;
  std::array<double, spectrum_bins> weights;
};

inline const auto &mel_filters() {
  static const auto filters = [] {
    constexpr double mel_break_hz = 700;
    constexpr double bin_hz = static_cast<double>(sample_rate) / fft_size;
    const double max_mel = std::log1p(sample_rate / (2 * mel_break_hz));
    std::array<double, feature_count + 2> edges;
    for (std::size_t i = 0; i < edges.size(); ++i)
      edges[i] = mel_break_hz * std::expm1(max_mel * i / (edges.size() - 1));

    std::array<MelFilter, feature_count> result{};
    for (std::size_t i = 0; i < result.size(); ++i) {
      const double left = edges[i];
      const double center = edges[i + 1];
      const double right = edges[i + 2];
      auto &filter = result[i];
      filter.first_bin = static_cast<std::size_t>(std::ceil(left / bin_hz));
      filter.end_bin =
          std::min(spectrum_bins, static_cast<std::size_t>(right / bin_hz) + 1);
      for (std::size_t bin = filter.first_bin; bin < filter.end_bin; ++bin) {
        const double hz = bin * bin_hz;
        filter.weights[bin] = std::min((hz - left) / (center - left),
                                       (right - hz) / (right - center));
      }
    }
    return result;
  }();
  return filters;
}

template <std::floating_point T>
Frame<T> log_mel_frame(std::span<const std::int16_t> samples) {
  Spectrum spectrum{};
  const auto &window = hann_window();
  for (std::size_t i = 0; i < samples.size(); ++i)
    spectrum[i] = (samples[i] / 32768.0) * window[i];
  fft(spectrum);

  std::array<double, spectrum_bins> power;
  for (std::size_t bin = 0; bin < power.size(); ++bin)
    power[bin] = std::norm(spectrum[bin]) / fft_size;

  Frame<T> frame;
  const auto &filters = mel_filters();
  for (std::size_t i = 0; i < frame.size(); ++i) {
    const auto &filter = filters[i];
    double energy = 0;
    for (std::size_t bin = filter.first_bin; bin < filter.end_bin; ++bin)
      energy += power[bin] * filter.weights[bin];
    frame[i] = static_cast<T>(std::log(std::max(energy, energy_floor)));
  }
  return frame;
}

} // namespace detail

// Return a frame every 10 ms, zero-padding any window that extends past EOF.
template <std::floating_point T = double>
std::vector<Frame<T>> decode_wav(const std::filesystem::path &path) {
  const auto samples = detail::read_wav_samples(path);
  const std::span<const std::int16_t> view{samples};
  std::vector<Frame<T>> frames;
  frames.reserve(view.size() / detail::hop_samples +
                 (view.size() % detail::hop_samples != 0));
  for (std::size_t start = 0; start < view.size();
       start += detail::hop_samples) {
    const auto count = std::min(detail::window_samples, view.size() - start);
    frames.push_back(detail::log_mel_frame<T>(view.subspan(start, count)));
  }
  return frames;
}

// Select a window for a fixed-size model and pad missing frames with silence.
template <std::size_t FrameCount, std::floating_point T>
std::array<Frame<T>, FrameCount>
input_window(const std::vector<Frame<T>> &frames, std::size_t first_frame = 0) {
  if (first_frame > frames.size())
    throw std::out_of_range("First input frame is past the end of the audio");

  std::array<Frame<T>, FrameCount> input;
  const auto count = std::min(FrameCount, frames.size() - first_frame);
  for (std::size_t i = 0; i < count; ++i)
    input[i] = frames[first_frame + i];
  const auto silence = static_cast<T>(std::log(detail::energy_floor));
  for (std::size_t i = count; i < input.size(); ++i)
    input[i].fill(silence);
  return input;
}

} // namespace audio_decoder
