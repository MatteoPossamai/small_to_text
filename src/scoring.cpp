
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
  std::array<std::array<size_t, N_LABELS>, N_LABELS> confusion{};
  std::array<size_t, N_LABELS> label_totals{};

  for (const auto &element : scoring_dataset) {
    const auto &[weights, result] = element;
    network.load(weights.data());
    network.forward_pass();
    const auto &arr = network.output();
    auto idx =
        std::distance(arr.begin(), std::max_element(arr.begin(), arr.end()));
    auto actual = std::distance(
        result.begin(), std::max_element(result.begin(), result.end()));
    ++confusion[actual][idx];
    ++label_totals[actual];
    total_scoring++;
    total_correct += result[idx] == 1 ? 1 : 0;
  }

  std::cout << std::endl;
  std::cout << "Total: " << total_scoring << "\tCorrect: " << total_correct
            << std::endl;
  std::cout << "Accuracy: "
            << std::fixed << std::setprecision(2)
            << (100.0 * total_correct / total_scoring) << "%" << std::endl;

  std::array<size_t, N_LABELS> label_order{};
  for (size_t i = 0; i < N_LABELS; ++i)
    label_order[i] = i;
  std::stable_sort(label_order.begin(), label_order.end(),
                   [&](size_t left, size_t right) {
                     if (label_totals[left] == 0)
                       return false;
                     if (label_totals[right] == 0)
                       return true;
                     return static_cast<double>(confusion[left][left]) /
                                label_totals[left] <
                            static_cast<double>(confusion[right][right]) /
                                label_totals[right];
                   });

  std::cout << "\nPer-label results (weakest accuracy first):\n"
            << std::left << std::setw(21) << "Label" << std::right
            << std::setw(8) << "Total" << std::setw(9) << "Correct"
            << std::setw(8) << "Wrong" << std::setw(12) << "Accuracy (%)"
            << "  " << std::left << std::setw(29)
            << "Most common wrong prediction" << std::right << "Count\n";

  for (const auto label : label_order) {
    const auto total = label_totals[label];
    const auto correct = confusion[label][label];
    size_t wrong_prediction = 0;
    size_t wrong_count = 0;
    for (size_t predicted = 0; predicted < N_LABELS; ++predicted) {
      if (predicted != label && confusion[label][predicted] > wrong_count) {
        wrong_prediction = predicted;
        wrong_count = confusion[label][predicted];
      }
    }

    std::cout << std::left << std::setw(21) << LABELS_MAPPINGS[label]
              << std::right << std::setw(8) << total << std::setw(9) << correct
              << std::setw(8) << total - correct << std::setw(12);
    if (total == 0)
      std::cout << "N/A";
    else
      std::cout << 100.0 * correct / total;
    std::cout << "  " << std::left << std::setw(29)
              << (wrong_count == 0 ? std::string_view{"-"}
                                   : LABELS_MAPPINGS[wrong_prediction])
              << std::right << wrong_count << '\n';
  }
  return 0;
}
