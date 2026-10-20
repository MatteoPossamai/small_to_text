#include "network.hpp"
#include "network_utils.hpp"
#include "trigonometric_functions.hpp"
#include "utils.hpp"
#include "wav_file.hpp"
#include "wav_weights.hpp"
#include <iostream>

int main(int argc, char **argv) {
  if (argc != 2 && argc != 3) {
    std::cout << "Usage: " << argv[0] << " <path/to/input.wav> [path/to/model]"
              << std::endl;
    return -1;
  }
  std::string source_file = static_cast<std::string>(argv[1]);
  const std::string model_path = argc == 3 ? argv[2] : NETWORK_MODEL_PATH;
  Network<NETWORK_IN_VALUES, L1_NEURONS, L2_NEURONS, N_LABELS> network = {};
  if (!network.from_file(model_path)) {
    std::cerr << "Unable to load network weights: " << model_path << std::endl;
    return -1;
  }

  // 1. convert wav into inputs
  std::array<output_t, NETWORK_IN_VALUES> buffer{};
  try {
    WavFile wf(source_file);
    crunck_file_data(wf, buffer);
  } catch (const std::exception &error) {
    std::cerr << "Unable to read audio: " << error.what() << std::endl;
    return 1;
  }

  // 2. Feed into network
  WavWeights input;
  input.from_buffer(buffer);
  input.normalize();
  network.load(input.data());

  // 3. Network run inference
  network.forward_pass();

  // 4. Print outcome
  const auto &out = network.output();
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
