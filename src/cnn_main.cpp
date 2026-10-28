#include "utils.hpp"
#include <algorithm>
#include <cstdlib>
#ifdef CNN_EXPERIMENTAL
#include "cnn_2.hpp"
#else
#include "cnn.hpp"
#endif
#include "mask.hpp"
#include "network_utils.hpp"
#include <charconv>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

namespace fs = std::filesystem;
constexpr size_t CONV_NEURONS = CNN_NEURONS;
constexpr size_t WINDOW_SIZE = CNN_WINDOW;
constexpr size_t COMPRESS_RATIO = 3;
constexpr size_t POOLED_INPUTS =
    (NETWORK_IN_VALUES / NUM_BANDS - WINDOW_SIZE + 1) / COMPRESS_RATIO *
    CONV_NEURONS;
using SpeechCNN = CNN<NETWORK_IN_VALUES, NUM_BANDS, CONV_NEURONS, WINDOW_SIZE,
                      COMPRESS_RATIO, 128, 64, N_LABELS>;

struct Options {
  std::string command;
  fs::path dataset;
  fs::path model;
  size_t epochs = 500;
  output_t learning_rate = 0.005f;
  std::uint32_t seed = 12345;
  int shift = 4;
  bool resume = false;
  bool mask = false;
#ifdef CNN_EXPERIMENTAL
  output_t l2 = 0;
#endif
};

void usage(const char *program) {
  std::cout
      << "Usage:\n  " << program
      << " train DATASET_DIR MODEL_PATH [options]\n  " << program
      << " score DATASET_DIR MODEL_PATH\n\n"
      << "Training options:\n"
      << "  --epochs N           Additional epochs (default: 500)\n"
      << "  --learning-rate RATE Initial learning rate (default: 0.005)\n"
      << "  --seed N             Initialization/shuffle/shift seed (default: "
         "12345)\n"
      << "  --shift N            Maximum shift in time rows (default: 4; 0 "
         "disables)\n"
#ifdef CNN_EXPERIMENTAL
      << "  --l2 VALUE           L2 penalty strength (default: 0)\n"
#endif
      << "  --mask               Mask one random band or time row during training\n"
      << "  --resume             Continue from MODEL_PATH instead of "
         "initializing\n\n"
      << "Uses " << NUM_BANDS << " bands, " << CONV_NEURONS
      << " convolution neurons, a " << WINDOW_SIZE << "-row window,\n"
      << "averaging by three, and ordinary layers " << POOLED_INPUTS
      << " -> 128 -> 64 -> " << N_LABELS << ".\n";
}

template <typename T>
T number(std::string_view value, std::string_view option) {
  T result{};
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), result);
  if (error != std::errc{} || end != value.data() + value.size())
    throw std::runtime_error("Invalid value for " + std::string(option));
  return result;
}

Options parse_options(int argc, char **argv) {
  if (argc < 4)
    throw std::runtime_error(
        "Expected a command, dataset directory and model path; use --help");
  Options options;
  options.command = argv[1];
  options.dataset = argv[2];
  options.model = argv[3];
  if (options.command != "train" && options.command != "score")
    throw std::runtime_error("Command must be train or score");
  if (options.command == "score" && argc != 4)
    throw std::runtime_error("Scoring expects only DATASET_DIR and MODEL_PATH");

  for (int i = 4; i < argc; ++i) {
    const std::string_view option = argv[i];
    if (option == "--mask") {
      options.mask = true;
      continue;
    }
    if (option == "--resume") {
      options.resume = true;
      continue;
    }
#ifdef CNN_EXPERIMENTAL
    if (option == "--l2") {
      if (++i == argc) throw std::runtime_error("Missing value for --l2");
      options.l2 = number<output_t>(argv[i], option);
      if (!std::isfinite(options.l2) || options.l2 < 0)
        throw std::runtime_error("L2 strength must be finite and non-negative");
      continue;
    }
#endif
    if (option != "--epochs" && option != "--learning-rate" &&
        option != "--seed" && option != "--shift")
      throw std::runtime_error("Unknown option: " + std::string(option));
    if (++i == argc)
      throw std::runtime_error("Missing value for " + std::string(option));
    const std::string_view value = argv[i];
    if (option == "--epochs")
      options.epochs = number<size_t>(value, option);
    else if (option == "--learning-rate")
      options.learning_rate = number<output_t>(value, option);
    else if (option == "--seed")
      options.seed = number<std::uint32_t>(value, option);
    else
      options.shift = number<int>(value, option);
  }
  if (options.epochs == 0)
    throw std::runtime_error("Epochs must be positive");
  if (!std::isfinite(options.learning_rate) || options.learning_rate <= 0)
    throw std::runtime_error("Learning rate must be finite and positive");
  if (options.shift < 0 || options.shift >= int(NETWORK_IN_VALUES / NUM_BANDS))
    throw std::runtime_error("Shift must be between 0 and " +
                             std::to_string(NETWORK_IN_VALUES / NUM_BANDS - 1));
  return options;
}

struct Sample {
  WavWeights weights;
  size_t label;
};

std::vector<Sample> load_dataset(const fs::path &path) {
  if (!fs::is_directory(path))
    throw std::runtime_error("Dataset is not a directory: " + path.string());
  std::vector<std::pair<fs::path, size_t>> files;
  for (const auto &entry : fs::directory_iterator(path)) {
    if (!entry.is_directory())
      continue;
    const std::string label = entry.path().filename().string();
    const auto found =
        std::find(LABELS_MAPPINGS.begin(), LABELS_MAPPINGS.end(), label);
    if (found == LABELS_MAPPINGS.end())
      throw std::runtime_error("Unknown label: " + label);
    for (const auto &file : fs::directory_iterator(entry.path()))
      if (file.is_regular_file())
        files.emplace_back(file.path(), found - LABELS_MAPPINGS.begin());
  }
  if (files.empty())
    throw std::runtime_error("Dataset contains no encoded samples: " +
                             path.string());
  // Stable ordering makes the seed independent of filesystem iteration order.
  std::sort(files.begin(), files.end());
  std::vector<Sample> samples;
  samples.reserve(files.size());
  for (const auto &[file_path, label] : files) {
    std::array<output_t, NETWORK_IN_VALUES> values{};
    if (fs::file_size(file_path) != sizeof(values))
      throw std::runtime_error("Wrong encoded sample size (expected " +
                               std::to_string(sizeof(values)) + " bytes for " +
                               std::to_string(NUM_BANDS) +
                               " bands): " + file_path.string());
    std::ifstream file(file_path, std::ios::binary);
    if (!file.read(reinterpret_cast<char *>(values.data()), sizeof(values)))
      throw std::runtime_error("Unable to read sample: " + file_path.string());
    for (const auto value : values)
      if (!std::isfinite(value))
        throw std::runtime_error("Non-finite value in sample: " +
                                 file_path.string());
    WavWeights weights;
    weights.from_buffer(values);
    weights.normalize();
    samples.push_back({std::move(weights), label});
  }
  return samples;
}

void save_model(SpeechCNN &model, const fs::path &path) {
  if (!path.parent_path().empty())
    fs::create_directories(path.parent_path());
  fs::path temporary = path;
  temporary += ".tmp";
  if (!model.to_file(temporary.string()))
    throw std::runtime_error("Unable to save model: " + temporary.string());
  // Replace the checkpoint only once writing it has succeeded.
  fs::rename(temporary, path);
}

void train(SpeechCNN &model, const Options &options,
           const std::vector<Sample> &samples) {
  std::mt19937 random(options.seed);
  std::mt19937 mask_random(options.seed ^ 0x9e3779b9u);
  std::uniform_int_distribution<int> shift(-options.shift, options.shift);
  std::bernoulli_distribution mask_frequency(0.5);
  std::uniform_int_distribution<size_t> band(0, NUM_BANDS - 1);
  std::uniform_int_distribution<size_t> row(0, NETWORK_IN_VALUES / NUM_BANDS - 1);
  std::vector<size_t> order(samples.size());
  std::iota(order.begin(), order.end(), 0);
  output_t rate = options.learning_rate;
  const output_t minimum_rate = std::min(rate, output_t{0.00001});
  double reference_loss = 0;
  size_t stalled_epochs = 0;
  for (size_t index = 0; index < options.epochs; ++index) {
    const size_t epoch = index + 1;
    std::shuffle(order.begin(), order.end(), random);
    double total_loss = 0;
    for (const size_t index_i : order) {
      const auto &sample = samples[index_i];
      if (options.mask) {
        Mask<NETWORK_IN_VALUES> mask{};
        if (mask_frequency(mask_random)) {
          mask.mask_frequency(band(mask_random));
        } else {
          mask.mask_sequence(row(mask_random));
        }
        model.load_inputs(time_shift(sample.weights.masked_data(mask), shift(random)));
      } else {
        model.load_inputs(time_shift(sample.weights.data(), shift(random)));
      }
      model.forward_pass();
      std::array<output_t, N_LABELS> target{};
      target[sample.label] = 1;
#ifdef CNN_EXPERIMENTAL
      const auto loss = model.backprop(target, rate, options.l2);
#else
      const auto loss = model.backprop(target, rate);
#endif
      if (!std::isfinite(loss))
        throw std::runtime_error(
            "Training diverged; try a lower --learning-rate");
      total_loss += loss;
    }
    const double mean_loss = total_loss / samples.size();
    std::cout << "Epoch " << epoch << '/' << options.epochs
              << " Mean loss: " << mean_loss << " LR: " << rate << std::endl;

    // Match the ordinary trainer: halve the rate after ten stalled epochs.
    if (epoch == 1 || mean_loss < reference_loss * 0.999) {
      reference_loss = mean_loss;
      stalled_epochs = 0;
    } else {
      ++stalled_epochs;
    }
    if (stalled_epochs >= 10 && rate > minimum_rate) {
      rate = std::max(minimum_rate, rate * output_t{0.5});
      reference_loss = mean_loss;
      stalled_epochs = 0;
    }
    if (epoch % 10 == 0 || epoch == options.epochs)
      save_model(model, options.model);
  }
  std::cout << "Saved model to " << options.model << '\n';
}

void score(SpeechCNN &model, const std::vector<Sample> &samples) {
  std::array<std::array<size_t, N_LABELS>, N_LABELS> confusion{};
  std::array<size_t, N_LABELS> totals{};
  size_t correct = 0;
  for (const auto &sample : samples) {
    model.load_inputs(sample.weights.data());
    model.forward_pass();
    const auto &output = model.output();
    const size_t predicted =
        std::max_element(output.begin(), output.end()) - output.begin();
    ++confusion[sample.label][predicted];
    ++totals[sample.label];
    correct += predicted == sample.label;
  }
  std::cout << "\nTotal: " << samples.size() << "\tCorrect: " << correct
            << "\nAccuracy: " << std::fixed << std::setprecision(2)
            << 100.0 * correct / samples.size() << "%\n";

  std::array<size_t, N_LABELS> order{};
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](size_t left, size_t right) {
    if (totals[left] == 0)
      return false;
    if (totals[right] == 0)
      return true;
    return double(confusion[left][left]) / totals[left] <
           double(confusion[right][right]) / totals[right];
  });
  std::cout << "\nPer-label results (weakest accuracy first):\n"
            << std::left << std::setw(21) << "Label" << std::right
            << std::setw(8) << "Total" << std::setw(9) << "Correct"
            << std::setw(8) << "Wrong" << std::setw(12) << "Accuracy (%)"
            << "  " << std::left << std::setw(29)
            << "Most common wrong prediction" << std::right << "Count\n";
  for (const size_t label : order) {
    size_t wrong_label = 0, wrong_count = 0;
    for (size_t predicted = 0; predicted < N_LABELS; ++predicted)
      if (predicted != label && confusion[label][predicted] > wrong_count) {
        wrong_label = predicted;
        wrong_count = confusion[label][predicted];
      }
    const size_t right = confusion[label][label];
    std::cout << std::left << std::setw(21) << LABELS_MAPPINGS[label]
              << std::right << std::setw(8) << totals[label] << std::setw(9)
              << right << std::setw(8) << totals[label] - right
              << std::setw(12);
    if (totals[label] == 0)
      std::cout << "N/A";
    else
      std::cout << 100.0 * right / totals[label];
    std::cout << "  " << std::left << std::setw(29)
              << (wrong_count ? LABELS_MAPPINGS[wrong_label]
                              : std::string_view{"-"})
              << std::right << wrong_count << '\n';
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--help") {
    usage(argv[0]);
    return 0;
  }
  try {
    const auto options = parse_options(argc, argv);
    SpeechCNN model;
    if (options.command == "score" || options.resume) {
      if (!model.from_file(options.model.string()))
        throw std::runtime_error("Unable to load compatible CNN model: " +
                                 options.model.string());
    } else {
      if (fs::exists(options.model))
        throw std::runtime_error("Model already exists; use --resume or choose "
                                 "a different model path");
      model.initialize_random_weights(options.seed);
    }
    const auto samples = load_dataset(options.dataset);
    std::cout << "Loaded " << samples.size() << " samples (" << NUM_BANDS
              << " bands)." << std::endl;
    if (options.command == "train") {
      std::cout << "Seed: " << options.seed << std::endl;
#ifdef CNN_EXPERIMENTAL
      std::cout << "L2 strength: " << options.l2 << std::endl;
#endif
      train(model, options, samples);
    } else {
      score(model, samples);
    }
  } catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
