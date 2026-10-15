#include "utils.hpp"
#include "wav_file.hpp"
#include <iostream>

int main() {
  WavFile file("/home/mpossamai/Downloads/speech_commands_v0.02/bird/"
               "00970ce1_nohash_0.wav");
  std::cout << file.sample_rate() << std::endl;
  std::cout << file.bytes_per_sample_frame() << std::endl;
  file.seek_to_data();
  char test[16];
  file.read_data(test, 16);

  for (size_t i = 0; i < 16; i += 2) {
    std::cout << from_bytes_to_sample(test + i) << "\t";
  }
  return 0;
}
