#pragma once

#include "utils.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

class WavFile {
  std::ifstream file;
  std::uint32_t sample_rate_value = 0;
  std::uint16_t bytes_per_sample_frame_value = 0;
  std::streamoff data_offset = 0;
  std::uint32_t data_size = 0;
  std::uint32_t data_remaining = 0;

  static std::uint32_t read_u32_le(std::istream &in) {
    unsigned char bytes[4];
    in.read(reinterpret_cast<char *>(bytes), 4);
    if (!in)
      throw std::runtime_error("Unexpected end of WAV file");
    return std::uint32_t(bytes[0]) | (std::uint32_t(bytes[1]) << 8) |
           (std::uint32_t(bytes[2]) << 16) | (std::uint32_t(bytes[3]) << 24);
  }

  static std::uint16_t read_u16_le(std::istream &in) {
    unsigned char bytes[2];
    in.read(reinterpret_cast<char *>(bytes), 2);
    if (!in)
      throw std::runtime_error("Unexpected end of WAV file");
    return std::uint16_t(bytes[0]) | (std::uint16_t(bytes[1]) << 8);
  }

public:
  WavFile() = delete;
  explicit WavFile(const std::string &filename)
      : file(filename, std::ios::binary) {
    if (!file)
      throw std::runtime_error("Could not open WAV file: " + filename);

    char id[4];
    file.read(id, 4);
    if (!file || std::string(id, 4) != "RIFF")
      throw std::runtime_error("Invalid WAV file: missing RIFF");

    const std::uint32_t riff_size = read_u32_le(file);
    file.read(id, 4);
    if (!file || std::string(id, 4) != "WAVE" || riff_size < 4)
      throw std::runtime_error("Invalid WAV file: missing WAVE");

    // RIFF size includes "WAVE" and all chunks after it.
    std::uint32_t remaining = riff_size - 4;
    bool found_fmt = false;
    bool found_data = false;

    while (remaining >= 8 && !(found_fmt && found_data)) {
      file.read(id, 4);
      if (!file)
        throw std::runtime_error("Unexpected end of WAV file");
      const std::uint32_t chunk_size = read_u32_le(file);
      remaining -= 8;

      const std::uint32_t padded_size = chunk_size + (chunk_size & 1u);
      if (padded_size > remaining)
        throw std::runtime_error("Invalid WAV file: chunk exceeds RIFF size");

      const auto payload = file.tellg();
      const std::string chunk_id(id, 4);
      if (chunk_id == "fmt ") {
        if (chunk_size < 14)
          throw std::runtime_error("Invalid WAV file: fmt chunk is too small");
        file.seekg(payload +
                   std::streamoff(4)); // sample rate is 4 bytes into fmt
        sample_rate_value = read_u32_le(file);
        if (sample_rate_value != SAMPLE_RATE) {
          throw std::runtime_error("Unsupported WAV sample rate: " +
                                   std::to_string(sample_rate_value) +
                                   " Hz (expected " +
                                   std::to_string(SAMPLE_RATE) + " Hz)");
        }
        file.seekg(payload +
                   std::streamoff(12)); // block alignment is at byte 12
        bytes_per_sample_frame_value = read_u16_le(file);
        found_fmt = true;
      } else if (chunk_id == "data" && !found_data) {
        data_offset = payload;
        data_size = chunk_size;
        found_data = true;
      }

      file.seekg(payload + static_cast<std::streamoff>(padded_size));
      if (!file)
        throw std::runtime_error("Could not skip WAV chunk");
      remaining -= padded_size;
    }

    if (!found_fmt || !found_data)
      throw std::runtime_error("Invalid WAV file: missing fmt or data chunk");

    seek_to_data();
  }

  std::uint32_t sample_rate() const { return sample_rate_value; }
  std::uint16_t bytes_per_sample_frame() const {
    return bytes_per_sample_frame_value;
  }

  void seek_to_data() {
    file.clear();
    file.seekg(data_offset);
    if (!file)
      throw std::runtime_error("Could not seek to WAV data");
    data_remaining =
        std::min<std::uint32_t>(data_size, N_SAMPLES * sizeof(std::int16_t));
  }

  void rewind_data(std::uint32_t byte_count) {
    const auto readable_size =
        std::min<std::uint32_t>(data_size, N_SAMPLES * sizeof(std::int16_t));
    const auto bytes_read = readable_size - data_remaining;
    if (byte_count > bytes_read)
      throw std::runtime_error("Cannot rewind before WAV data");

    file.clear();
    file.seekg(-static_cast<std::streamoff>(byte_count), std::ios::cur);
    if (!file)
      throw std::runtime_error("Could not rewind WAV data");
    data_remaining += byte_count;
  }

  std::streamsize read_data(char *buffer, std::streamsize size) {
    if (size <= 0 || data_remaining == 0 || file.eof())
      return 0;
    const auto remaining = static_cast<std::streamsize>(data_remaining);
    const auto count = size < remaining ? size : remaining;
    file.read(buffer, count);
    const auto bytes_read = file.gcount();
    data_remaining -= static_cast<std::uint32_t>(bytes_read);
    if (file.bad())
      throw std::runtime_error("Could not read WAV data");
    return bytes_read;
  }

  std::size_t samples_read() const {
    const auto readable_size =
        std::min<std::uint32_t>(data_size, N_SAMPLES * sizeof(std::int16_t));
    return (readable_size - data_remaining) / sizeof(std::int16_t);
  }

  // Read at most 16,000 samples and fill missing samples with silence.
  std::array<input_t, N_SAMPLES> read_clip() {
    if (bytes_per_sample_frame_value != sizeof(std::int16_t))
      throw std::runtime_error("Reading clips requires 16-bit mono WAV data");

    std::array<char, N_SAMPLES * sizeof(std::int16_t)> raw{};
    const auto bytes_read = read_data(raw.data(), raw.size());
    std::array<input_t, N_SAMPLES> samples{};
    const auto sample_count = bytes_read / sizeof(std::int16_t);
    for (size_t i = 0; i < sample_count; ++i)
      samples[i] = from_bytes_to_sample(raw.data() + i * sizeof(std::int16_t));
    return samples;
  }
};
