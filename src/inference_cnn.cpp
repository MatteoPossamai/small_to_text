#include "cnn.hpp"
#include "network_utils.hpp"
#include "trigonometric_functions.hpp"
#include "wav_file.hpp"
#include "wav_weights.hpp"
#include <iostream>

int main(int argc, char **argv) {
  if (argc != 2 && argc != 3) {
    std::cout << "Usage: " << argv[0]
              << " <path/to/input.wav> [path/to/cnn/model]" << std::endl;
    return 1;
  }

  const std::string model_path = argc == 3 ? argv[2] : CNN_MODEL_PATH;
  CNN<NETWORK_IN_VALUES, NUM_BANDS, 32, 5, 3, 128, 64, N_LABELS> network;
  if (!network.from_file(model_path)) {
    std::cerr << "Unable to load 32-neuron CNN model: " << model_path
              << std::endl;
    return 1;
  }

  // Use the same band extraction and normalization as the training recordings.
  std::array<output_t, NETWORK_IN_VALUES> buffer{};
  try {
    WavFile wav(argv[1]);
    crunck_file_data(wav, buffer);
  } catch (const std::exception &error) {
    std::cerr << "Unable to read audio: " << error.what() << std::endl;
    return 1;
  }

  WavWeights input;
  input.from_buffer(buffer);
  input.normalize();
  network.load_inputs(input.data());
  network.forward_pass();

  const auto &output = network.output();
  const auto predicted =
      std::max_element(output.begin(), output.end()) - output.begin();
  std::cout << "CNN inference output:\n\t" << LABELS_MAPPINGS[predicted]
            << std::endl;
  return 0;
}
