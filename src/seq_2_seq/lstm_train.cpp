#include "lstm.hpp"

#include <iomanip>
#include <iostream>
#include <string>
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
  std::array<double, 8> target{}; // 1 = first number is positive; 0 = negative.
};
using Dataset = std::vector<Sample>;

Dataset make_dataset(size_t count, std::uint32_t seed) {
  std::mt19937 generator(seed);
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  Dataset dataset(count);
  for (auto &sample : dataset) {
    for (auto &frame : sample.frames)
      frame[0] = distribution(generator);
    std::array<double, 8> target_1 = {1.0, 0, 0, 0, 0, 0, 0, 0};
    std::array<double, 8> target_2 = {0, 0, 0, 0, 0, 0, 0, 0};
    sample.target = sample.frames[0][0] > 0.0 ? target_1 : target_2;
  }
  return dataset;
}

int main() {
  const auto dataset = make_dataset(TRAINING_SAMPLES, SEED + 1);
  RecurrentModel lstm(SEED);
  constexpr size_t BAR_WIDTH = 20;
  const size_t total_updates = EPOCHS * dataset.size();
  const auto count_width = std::to_string(total_updates).size();
  const auto epoch_width = std::to_string(EPOCHS).size();
  size_t completed_updates = 0;

  for (size_t i = 0; i < EPOCHS; ++i) {
    for (const auto &sample : dataset) {
      lstm.reset_state();

      // Forward pass
      RecurrentModel::out_t final_output;
      for (const auto &frame : sample.frames) {
        final_output = lstm.forward_pass_train(frame);
      }

      // Loss and gradient
      RecurrentModel::out_t error{};
      for (size_t i = 0; i < 8; ++i) {
        error[i] = final_output[i] - sample.target[i];
      }

      RecurrentModel::out_t hidden_gradient = error;
      RecurrentModel::out_t cell_gradient{};

      // Accumulate gradients
      for (const auto &frame : sample.frames) {
        auto res = lstm.backward_pass(hidden_gradient, cell_gradient);
        hidden_gradient = res.previous_hidden_gradient;
        cell_gradient = res.previous_cell_gradient;
      }

      // Apply gradients
      lstm.apply_gradient(LEARNING_RATE);

      ++completed_updates;
      const size_t filled = BAR_WIDTH * completed_updates / total_updates;
      std::cout << '\r' << "Epoch " << std::setw(epoch_width) << i + 1 << '/'
                << EPOCHS << " [" << std::string(filled, '#')
                << std::string(BAR_WIDTH - filled, '-') << "] "
                << std::setw(count_width) << total_updates - completed_updates
                << " updates left" << std::flush;
    }
  }

  std::cout << '\n';
  return 0;
}
