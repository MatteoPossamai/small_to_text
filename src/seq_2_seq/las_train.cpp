#include "audio_decoder.hpp"
#include "las.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

namespace {

using Model = LAS<float, true>;

constexpr char USAGE[] =
    "Usage: las_train TRAIN.tsv DEV.tsv EPOCHS [--resume DIRECTORY] "
    "[--output DIRECTORY] [--learning-rate RATE] [--max-tokens N] "
    "[--checkpoint-every N]\n"
    "Each manifest line contains a WAV path, a tab, and the transcript.\n"
    "An interval of 0 saves only after the final epoch.\n"
    "Use - DEV.tsv 0 --resume DIRECTORY to score a checkpoint.\n";

struct Options {
  std::filesystem::path train_manifest;
  std::filesystem::path dev_manifest;
  std::size_t epochs = 0;
  std::filesystem::path resume;
  std::filesystem::path output = las_weights::directory;
  float learning_rate = 0.05f;
  std::size_t max_tokens = MAX_LEN - 1;
  std::size_t checkpoint_every = 0;
};

template <typename T> T parse_number(std::string_view text) {
  T value{};
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size())
    throw std::invalid_argument("Invalid number: " + std::string(text));
  return value;
}

Options parse_options(int argc, char **argv) {
  if (argc < 4)
    throw std::invalid_argument(
        "Expected training manifest, validation manifest, and epochs");

  Options options;
  options.train_manifest = argv[1];
  options.dev_manifest = argv[2];
  options.epochs = parse_number<std::size_t>(argv[3]);
  for (int i = 4; i < argc; i += 2) {
    const std::string_view option = argv[i];
    if (i + 1 == argc)
      throw std::invalid_argument("Missing value for " + std::string(option));
    const std::string_view value = argv[i + 1];
    if (option == "--resume")
      options.resume = value;
    else if (option == "--output")
      options.output = value;
    else if (option == "--learning-rate")
      options.learning_rate = parse_number<float>(value);
    else if (option == "--max-tokens")
      options.max_tokens = parse_number<std::size_t>(value);
    else if (option == "--checkpoint-every")
      options.checkpoint_every = parse_number<std::size_t>(value);
    else
      throw std::invalid_argument("Unknown option: " + std::string(option));
  }
  if (options.epochs == 0 && options.resume.empty())
    throw std::invalid_argument("0 epochs requires --resume");
  if (!std::isfinite(options.learning_rate) || options.learning_rate <= 0)
    throw std::invalid_argument("Learning rate must be positive and finite");
  if (options.max_tokens == 0)
    throw std::invalid_argument("--max-tokens must be positive");
  return options;
}

class Progress {
  using Clock = std::chrono::steady_clock;
  const bool terminal = isatty(STDOUT_FILENO) != 0;
  bool active = false;
  Clock::time_point last_update{};

public:
  bool enabled() const { return terminal; }

  void update(std::string_view label, std::size_t done, std::size_t total,
              std::optional<double> loss = {}) {
    if (!terminal)
      return;
    const auto now = Clock::now();
    if (done != 0 && done != total &&
        now - last_update < std::chrono::milliseconds(250))
      return;

    constexpr std::size_t width = 24;
    const auto filled = static_cast<std::size_t>(double(done) / total * width);
    std::cout << "\r\033[K" << label << " [" << std::string(filled, '=')
              << std::string(width - filled, '-') << "] " << done << '/'
              << total;
    if (loss)
      std::cout << ", mean loss " << std::fixed << std::setprecision(4)
                << *loss;
    std::cout << std::flush;
    active = true;
    last_update = now;
  }

  void finish() {
    if (active) {
      std::cout << '\n' << std::flush;
      active = false;
    }
  }
};

struct Utterance {
  std::filesystem::path audio;
  std::string transcript;
};

std::vector<Utterance> read_manifest(const std::filesystem::path &path) {
  std::ifstream file(path);
  if (!file)
    throw std::runtime_error("Cannot open manifest " + path.string());

  std::vector<Utterance> utterances;
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(file, line)) {
    ++line_number;
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    if (line.empty())
      continue;

    const auto tab = line.find('\t');
    if (tab == std::string::npos || tab == 0 || tab + 1 == line.size() ||
        line.find('\t', tab + 1) != std::string::npos)
      throw std::runtime_error("Expected WAV path and transcript at " +
                               path.string() + ":" +
                               std::to_string(line_number));
    utterances.push_back(
        {path.parent_path() / line.substr(0, tab), line.substr(tab + 1)});
  }
  if (file.bad())
    throw std::runtime_error("Cannot read manifest " + path.string());
  if (utterances.empty())
    throw std::runtime_error("Manifest is empty: " + path.string());
  return utterances;
}

std::vector<std::string_view> words(std::string_view text) {
  std::vector<std::string_view> result;
  std::size_t start = text.find_first_not_of(' ');
  while (start != std::string_view::npos) {
    const auto end = text.find(' ', start);
    result.push_back(text.substr(start, end - start));
    start = text.find_first_not_of(' ', end);
  }
  return result;
}

template <typename Sequence>
std::size_t edit_distance(const Sequence &reference,
                          const Sequence &prediction) {
  std::vector<std::size_t> previous(prediction.size() + 1);
  std::vector<std::size_t> current(prediction.size() + 1);
  std::iota(previous.begin(), previous.end(), std::size_t{0});
  for (std::size_t i = 0; i < reference.size(); ++i) {
    current[0] = i + 1;
    for (std::size_t j = 0; j < prediction.size(); ++j) {
      current[j + 1] =
          std::min({previous[j + 1] + 1, current[j] + 1,
                    previous[j] + (reference[i] != prediction[j])});
    }
    previous.swap(current);
  }
  return previous.back();
}

void score(Model &model, const std::vector<Utterance> &validation,
           std::size_t max_tokens, Progress &progress) {
  std::size_t word_errors = 0;
  std::size_t word_count = 0;
  std::size_t character_errors = 0;
  std::size_t character_count = 0;
  std::size_t done = 0;
  progress.update("Validation", done, validation.size());
  for (const auto &utterance : validation) {
    const auto frames = audio_decoder::decode_wav<float>(utterance.audio);
    const auto prediction = model.inference(frames, max_tokens);
    const auto reference_words = words(utterance.transcript);
    word_errors += edit_distance(reference_words, words(prediction));
    word_count += reference_words.size();
    character_errors += edit_distance(utterance.transcript, prediction);
    character_count += utterance.transcript.size();
    progress.update("Validation", ++done, validation.size());
  }
  progress.finish();
  if (word_count == 0 || character_count == 0)
    throw std::runtime_error("Validation has no reference text");

  std::cout << std::fixed << std::setprecision(2) << "Validation WER "
            << 100.0 * word_errors / word_count << "% (" << word_errors << '/'
            << word_count << "), CER "
            << 100.0 * character_errors / character_count << "% ("
            << character_errors << '/' << character_count << ")\n"
            << std::flush;
}

} // namespace

int main(int argc, char **argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--help") {
    std::cout << USAGE;
    return 0;
  }

  Progress progress;
  try {
    const auto options = parse_options(argc, argv);
    const auto validation = read_manifest(options.dev_manifest);
    auto training = options.epochs > 0 ? read_manifest(options.train_manifest)
                                       : std::vector<Utterance>{};
    auto model = options.resume.empty() ? std::make_unique<Model>()
                                        : Model::load_weights(options.resume);
    std::cout << "Training utterances: " << training.size()
              << ", validation utterances: " << validation.size() << '\n'
              << std::flush;
    score(*model, validation, options.max_tokens, progress);

    std::mt19937 generator(12345);
    for (std::size_t previous_epoch = 0; previous_epoch < options.epochs;
         ++previous_epoch) {
      const auto epoch = previous_epoch + 1;
      const auto label = "Epoch " + std::to_string(epoch) + '/' +
                         std::to_string(options.epochs);
      std::shuffle(training.begin(), training.end(), generator);
      double loss = 0;
      std::size_t done = 0;
      progress.update(label, done, training.size());
      for (const auto &utterance : training) {
        const auto frames = audio_decoder::decode_wav<float>(utterance.audio);
        loss += model->train_utterance(frames, utterance.transcript,
                                       options.learning_rate);
        ++done;
        progress.update(label, done, training.size(), loss / done);
      }
      if (!progress.enabled())
        std::cout << std::fixed << std::setprecision(4) << label
                  << ", mean loss " << loss / training.size() << '\n'
                  << std::flush;

      if (epoch == options.epochs ||
          (options.checkpoint_every && epoch % options.checkpoint_every == 0)) {
        progress.finish();
        model->save_weights(options.output);
        std::cout << "Saved checkpoint after epoch " << epoch << " in "
                  << options.output << '\n'
                  << std::flush;
      }
    }

    if (options.epochs > 0)
      score(*model, validation, options.max_tokens, progress);
    return 0;
  } catch (const std::exception &error) {
    progress.finish();
    std::cerr << error.what() << '\n';
    return 1;
  }
}
