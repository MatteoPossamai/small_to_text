#include "io_utils.hpp"
#include "network.hpp"
#include "network_utils.hpp"
#include "utils.hpp"
#include <charconv>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>

namespace fs = std::filesystem;

void build_training_set(const std::string &path,
                        training_set &training_dataset) {
  for (const auto &entry : fs::directory_iterator(path)) {
    if (!entry.is_directory())
      continue;
    std::string label = entry.path().filename().string();
    const auto it =
        std::find(LABELS_MAPPINGS.begin(), LABELS_MAPPINGS.end(), label);
    if (it == LABELS_MAPPINGS.end())
      throw std::runtime_error("Unknown training label: " + label);

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
  constexpr size_t TRAINING_EPOCHS = 10000;
  constexpr output_t INITIAL_LEARNING_RATE = 0.005;
  constexpr output_t MIN_LEARNING_RATE = 0.00001;
  constexpr output_t MIN_RELATIVE_IMPROVEMENT = 0.001;
  constexpr size_t LEARNING_RATE_PATIENCE = 10;
  constexpr size_t CHECKPOINT_INTERVAL = 10;
  if (argc < 2 || argc > 4) {
    std::cout << "Usage: " << argv[0]
              << " <path/to/training/set> [epochs] [path/to/model]"
              << std::endl;
    return -1;
  }

  size_t epochs = TRAINING_EPOCHS;
  if (argc >= 3) {
    const std::string_view value = argv[2];
    const auto [end, error] =
        std::from_chars(value.data(), value.data() + value.size(), epochs);
    if (error != std::errc{} || end != value.data() + value.size() ||
        epochs == 0) {
      std::cerr << "Epochs must be a positive integer" << std::endl;
      return 1;
    }
  }

  const fs::path model_path = argc == 4 ? argv[3] : NETWORK_MODEL_PATH;
  std::error_code error;
  if (!model_path.parent_path().empty())
    fs::create_directories(model_path.parent_path(), error);
  if (error) {
    std::cerr << "Unable to create model directory: " << error.message()
              << std::endl;
    return 1;
  }

  training_set training_dataset;
  std::string path = static_cast<std::string>(argv[1]);
  try {
    build_training_set(path, training_dataset);
  } catch (const std::exception &error) {
    std::cerr << "Unable to load training dataset: " << error.what()
              << std::endl;
    return 1;
  }
  if (training_dataset.empty()) {
    std::cerr << "Training dataset is empty" << std::endl;
    return 1;
  }
  std::cout << "Training dataset loaded." << std::endl;

  Network<NETWORK_IN_VALUES, L1_NEURONS, L2_NEURONS, N_LABELS> network = {};
  network.initialize_random_weights();
  std::mt19937 shuffle_generator(std::random_device{}());

  auto save_model = [&]() {
    fs::path temporary_path = model_path;
    temporary_path += ".tmp";
    if (!network.to_file(temporary_path.string())) {
      std::cerr << "Unable to save model: " << temporary_path << std::endl;
      return false;
    }
    std::error_code save_error;
    fs::rename(temporary_path, model_path, save_error);
    if (save_error) {
      std::cerr << "Unable to replace model: " << save_error.message()
                << std::endl;
      return false;
    }
    return true;
  };

  output_t learning_rate = INITIAL_LEARNING_RATE;
  output_t reference_loss = 0;
  size_t epochs_without_improvement = 0;

  for (size_t i = 0; i < epochs; i++) {
    std::shuffle(training_dataset.begin(), training_dataset.end(),
                 shuffle_generator);
    auto loss = train_network(network, learning_rate, training_dataset,
                              LABELS_MAPPINGS);
    const auto mean_loss = loss / training_dataset.size();
    progress_bar(i + 1, epochs);
    std::cout << " Loss: " << loss << " Mean: " << mean_loss
              << " LR: " << learning_rate << "\033[K" << std::flush;

    if (i == 0 ||
        mean_loss < reference_loss * (1 - MIN_RELATIVE_IMPROVEMENT)) {
      reference_loss = mean_loss;
      epochs_without_improvement = 0;
    } else {
      ++epochs_without_improvement;
    }

    if (epochs_without_improvement >= LEARNING_RATE_PATIENCE &&
        learning_rate > MIN_LEARNING_RATE) {
      learning_rate = std::max(MIN_LEARNING_RATE, learning_rate * output_t{0.5});
      reference_loss = mean_loss;
      epochs_without_improvement = 0; }

    if ((i + 1) % CHECKPOINT_INTERVAL == 0 && i + 1 < epochs) {
      if (!save_model())
        return 1;
    }
  }
  std::cout << std::endl;
  if (!save_model())
    return 1;
  std::cout << "Saved model to " << model_path << std::endl;
  return 0;
}
