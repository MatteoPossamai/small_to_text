#include "network.hpp"
#include "network_utils.hpp"
#include "trigonometric_functions.hpp"
#include "utils.hpp"
#include "wav_file.hpp"
#include <iostream>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " <path/to/input.wav>" << std::endl;
    return -1;
  }
  std::string source_file = static_cast<std::string>(argv[1]);
  Network<NETWORK_IN_VALUES, L1_NEURONS, L2_NEURONS, N_LABELS> network = {};
  if (!network.from_file("../network/network_weights.bin")) {
    std::cout << "Unable to load network weights" << std::endl;
    return -1;
  }

  // 1. convert wav into inputs
  WavFile wf(source_file);
  std::array<output_t, NETWORK_IN_VALUES> buffer{};

  crunck_file_data(wf, buffer);

  // 2. Feed into network
  network.load(buffer);

  // 3. Network run inference
  network.forward_pass();

  // 4. Print outcome
  auto out = network.output();
  size_t max_idx = 0;
  output_t max_val = out[0];

  for (size_t i = 0; i < out.size(); i++) {
    if (max_val < out[i]) {
      max_val = out[i];
      max_idx = i;
    }
  }
  std::cout << "Network inference output:" << std::endl
            << "\t" << LABELS_MAPPINGS[max_idx] << std::endl;
}
