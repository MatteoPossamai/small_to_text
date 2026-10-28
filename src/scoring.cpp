
#include "network_utils.hpp"
#include "utils.hpp"
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;

void build_scoring_set(const std::string &path,
                       training_set &training_dataset) {
  if (!fs::is_directory(path)) {
    throw std::runtime_error("Scoring dataset is not a directory: " + path);
  }

  for (const auto &entry : fs::directory_iterator(path)) {
    if (!entry.is_directory())
      continue;

    std::string label = entry.path().filename().string();
    auto it = std::find(LABELS_MAPPINGS.begin(), LABELS_MAPPINGS.end(), label);
    if (it == LABELS_MAPPINGS.end()) {
      throw std::runtime_error("Unknown scoring label: " + label);
    }

    std::array<output_t, N_LABELS> exp_output{};
    exp_output[std::distance(LABELS_MAPPINGS.begin(), it)] = 1;

    for (const auto &file : fs::directory_iterator(entry.path())) {
      if (!file.is_regular_file())
        continue;

      WavWeights weights;
      weights.from_file(file.path().string());
      weights.normalize();

      training_dataset.emplace_back(std::move(weights), exp_output);
    }
  }
}

int main(int argc, char **argv) {
  if (argc != 2 && argc != 3) {
    std::cout << "Usage: " << argv[0]
              << " [path/to/network/weights] <path/to/encoded/scoring/set>"
              << std::endl;
    return 1;
  }

  training_set scoring_dataset = {};
  const std::string nn_path = argc == 3 ? argv[1] : NETWORK_MODEL_PATH;
  const std::string scoring_path = argv[argc - 1];

  Network<NETWORK_IN_VALUES, L1_NEURONS, L2_NEURONS, N_LABELS> network = {};
  if (!network.from_file(nn_path)) {
    std::cerr << "Unable to load model: " << nn_path << std::endl;
    return 1;
  }

  try {
    build_scoring_set(scoring_path, scoring_dataset);
  } catch (const std::exception &error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
  if (scoring_dataset.empty()) {
    std::cerr << "Scoring dataset contains no encoded samples: " << scoring_path
              << std::endl;
    return 1;
  }

  std::cout << "Scoring dataset loaded." << std::endl;

  size_t total_scoring = 0;
  size_t total_correct = 0;

  for (const auto &element : scoring_dataset) {
    const auto &[weights, result] = element;
    network.load(weights.data());
    network.forward_pass();
    const auto &arr = network.output();
    auto idx =
        std::distance(arr.begin(), std::max_element(arr.begin(), arr.end()));
    total_scoring++;
    total_correct += result[idx] == 1 ? 1 : 0;
  }

  std::cout << std::endl;
  std::cout << "Total: " << total_scoring << "\tCorrect: " << total_correct
            << std::endl;
  std::cout << "Accuracy: "
            << std::fixed << std::setprecision(2)
            << (100.0 * total_correct / total_scoring) << "%" << std::endl;

  return 0;
}
