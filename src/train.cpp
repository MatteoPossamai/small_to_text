#include "network.hpp"
#include "network_utils.hpp"
#include "utils.hpp"
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

void build_training_set(const std::string &path,
                        training_set &training_dataset) {
  for (const auto &entry : fs::directory_iterator(path)) {
    std::string label = entry.path().filename().string();

    for (const auto &file : fs::directory_iterator(entry.path())) {
      if (!file.is_regular_file())
        continue;

      WavWeights weights;
      weights.from_file(file.path().string());
      std::array<output_t, N_LABELS> exp_output{};
      auto it =
          std::find(LABELS_MAPPINGS.begin(), LABELS_MAPPINGS.end(), label);
      if (it != LABELS_MAPPINGS.end()) {
        auto idx = std::distance(LABELS_MAPPINGS.begin(), it);
        exp_output[idx] = 1.0;
      }

      training_dataset.emplace_back(std::move(weights), exp_output);
    }
  }
}

int main(int argc, char **argv) {
  constexpr size_t TRAINING_EPOCHS = 10000;
  constexpr output_t LEARNING_RATE = 0.001;
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " <path/to/training/set>" << std::endl;
    return -1;
  }

  // Training dataset: weights -> Label
  training_set training_dataset;
  std::string path = static_cast<std::string>(argv[1]);
  build_training_set(path, training_dataset);
  std::cout << "Training dataset loaded." << std::endl;

  Network<NETWORK_IN_VALUES, L1_NEURONS, L2_NEURONS, N_LABELS> network = {};
  network.initialize_random_weights();
  train_network(network, TRAINING_EPOCHS, LEARNING_RATE, training_dataset,
                LABELS_MAPPINGS);

  return 0;
}
