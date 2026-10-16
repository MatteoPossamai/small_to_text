#include "utils.hpp"
#include <array>
#include <filesystem>
#include <fstream>
#include <string>

class WavWeights {
  std::array<input_t, NETWORK_IN_VALUES> input_data{};
  bool initialized = false;

public:
  void from_file(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
      throw std::runtime_error("Could not open WAV file: " + filename);
    }

    file.read(reinterpret_cast<char *>(input_data.data()),
              input_data.size() * sizeof(input_t));

    if (!file || file.gcount() != static_cast<std::streamsize>(
                                      input_data.size() * sizeof(input_t))) {
      throw std::runtime_error("WAV file too short: " + filename);
    }
    initialized = true;
  }

  bool to_file(const std::string &filename) {
    if (!initialized) {
      return false;
    }

    std::filesystem::path path(filename);
    if (!path.parent_path().empty()) {
      std::filesystem::create_directories(path.parent_path());
    }

    std::ofstream file(filename, std::ios::binary);
    if (!file) {
      throw std::runtime_error("Could not open Dump file: " + filename);
    }

    file.write(reinterpret_cast<const char *>(input_data.data()),
               input_data.size() * sizeof(input_t));

    file.close();
    return file.good();
  }

  void from_buffer(const std::array<input_t, NETWORK_IN_VALUES> &buffer_in) {
    input_data = buffer_in;
    initialized = true;
  }

  const std::array<input_t, NETWORK_IN_VALUES> &data() const {
    return input_data;
  }
};
