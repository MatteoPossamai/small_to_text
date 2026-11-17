#include "lstm.hpp"

#include <iostream>
#include <vector>

// Toy data: read four numbers and eventually predict the first one's sign.
constexpr size_t INPUT_SIZE = 1;
constexpr size_t HIDDEN_SIZE = 8;
constexpr size_t SEQUENCE_LENGTH = 4;
constexpr double LEARNING_RATE = 0.05;
constexpr size_t EPOCHS = 40;
constexpr size_t TRAINING_SAMPLES = 512;
constexpr std::uint32_t SEED = 12345;

using RecurrentModel = LSTM<double, INPUT_SIZE, HIDDEN_SIZE, Platform::ANY>;

struct Sample {
  std::array<RecurrentModel::in_t, SEQUENCE_LENGTH> frames{};
  double target = 0.0; // 1 = first number is positive; 0 = negative.
};
using Dataset = std::vector<Sample>;

Dataset make_dataset(size_t count, std::uint32_t seed) {
  std::mt19937 generator(seed);
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  Dataset dataset(count);
  for (auto &sample : dataset) {
    for (auto &frame : sample.frames)
      frame[0] = distribution(generator);
    sample.target = sample.frames[0][0] > 0.0 ? 1.0 : 0.0;
  }
  return dataset;
}


int main() {
  const auto dataset = make_dataset(TRAINING_SAMPLES, SEED + 1);
  RecurrentModel lstm(SEED);
  lstm.train(EPOCHS, LEARNING_RATE, dataset);
  std::cout << "Forward-only scaffold completed: " << EPOCHS << " epochs, "
            << dataset.size() << " sequences per epoch.\n"
            << "Loss, backward, and weight updates are TODO.\n";
  return 0;
}
