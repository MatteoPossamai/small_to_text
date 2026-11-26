#include "audio_decoder.hpp"
#include "las.hpp"
#include <exception>
#include <iostream>
#include <string>

int main() {
  std::string filename;
  if (!std::getline(std::cin, filename) || filename.empty()) {
    std::cerr << "Expected a WAV filename on stdin\n";
    return 1;
  }

  try {
    const auto frames = audio_decoder::decode_wav<double>(filename);
    auto las = LAS<double, false>::load_weights();
    std::cout << las->inference(frames) << '\n';
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
