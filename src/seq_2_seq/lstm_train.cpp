#include "lstm.hpp"

#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

// Small enough to inspect: read four numbers, then remember the first one's sign.
constexpr size_t INPUT_SIZE = 1;
constexpr size_t HIDDEN_SIZE = 8;
constexpr size_t SEQUENCE_LENGTH = 4;
constexpr double LEARNING_RATE = 0.05;
constexpr size_t EPOCHS = 40;
constexpr size_t TRAINING_SAMPLES = 512;
constexpr size_t VALIDATION_SAMPLES = 256;
constexpr std::uint32_t SEED = 12345;

using RecurrentModel = LSTM<double, INPUT_SIZE, HIDDEN_SIZE, Platform::ANY>;

struct Hyperparameters {
  double learning_rate = LEARNING_RATE;
  size_t epochs = EPOCHS;
  std::uint32_t shuffle_seed = SEED + 3;
};

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

struct Model {
  RecurrentModel lstm;
  // A learned scalar output: score = weights dot final_hidden + bias.
  RecurrentModel::out_t output_weights{};
  double output_bias = 0.0;

  explicit Model(std::uint32_t seed) : lstm(seed) {
    std::mt19937 generator(seed + 1);
    std::uniform_real_distribution<double> distribution(-0.3, 0.3);
    for (auto &weight : output_weights)
      weight = distribution(generator);
  }

  double score(const RecurrentModel::out_t &hidden) const {
    return std::inner_product(output_weights.begin(), output_weights.end(),
                              hidden.begin(), output_bias);
  }
};

double probability(double score) {
  // Stable sigmoid, including for large positive or negative scores.
  if (score >= 0.0)
    return 1.0 / (1.0 + std::exp(-score));
  const double exponential = std::exp(score);
  return exponential / (1.0 + exponential);
}

double loss(double score, double target) {
  // Binary cross entropy computed directly from the score for stability.
  return std::max(score, 0.0) - score * target +
         std::log1p(std::exp(-std::abs(score)));
}

double predict_score(Model &model, const Sample &sample) {
  model.lstm.reset_state();
  RecurrentModel::out_t hidden{};
  for (const auto &frame : sample.frames)
    hidden = model.lstm.forward_pass(frame);
  return model.score(hidden);
}

struct Metrics {
  double loss = 0.0;
  double accuracy = 0.0;
};

Metrics evaluate(Model &model, const Dataset &dataset) {
  Metrics metrics;
  for (const auto &sample : dataset) {
    const double score = predict_score(model, sample);
    metrics.loss += loss(score, sample.target);
    metrics.accuracy += (score >= 0.0) == (sample.target == 1.0);
  }
  metrics.loss /= dataset.size();
  metrics.accuracy /= dataset.size();
  return metrics;
}

void print_metrics(size_t epoch, Model &model, const Dataset &training,
                   const Dataset &validation) {
  const auto train = evaluate(model, training);
  const auto valid = evaluate(model, validation);
  std::cout << std::setw(5) << epoch << " | " << train.loss << " | "
            << std::setw(6) << train.accuracy * 100.0 << "% | " << valid.loss
            << " | " << std::setw(6) << valid.accuracy * 100.0 << "%\n";
}

void train(Model &model, const Dataset &training, const Dataset &validation,
           const Hyperparameters &hyperparameters) {
  std::vector<size_t> order(training.size());
  std::iota(order.begin(), order.end(), 0);
  std::mt19937 generator(hyperparameters.shuffle_seed);

  std::cout << "Epoch | Train loss | Accuracy | Valid loss | Accuracy\n";
  print_metrics(0, model, training, validation);

  for (size_t epoch = 1; epoch <= hyperparameters.epochs; ++epoch) {
    std::shuffle(order.begin(), order.end(), generator);
    for (const size_t index : order) {
      const auto &sample = training[index];

      // 1. Forward through time. Keep each step's values for the backward pass.
      model.lstm.reset_state();
      std::array<RecurrentModel::StepCache, SEQUENCE_LENGTH> history{};
      RecurrentModel::out_t hidden{};
      for (size_t step = 0; step < SEQUENCE_LENGTH; ++step)
        hidden = model.lstm.forward_pass(sample.frames[step], history[step]);

      // 2. For sigmoid + binary cross entropy: d_loss/d_score = prediction-target.
      const double score_gradient =
          probability(model.score(hidden)) - sample.target;
      RecurrentModel::out_t hidden_gradient{}, cell_gradient{};
      for (size_t i = 0; i < HIDDEN_SIZE; ++i)
        hidden_gradient[i] = model.output_weights[i] * score_gradient;

      // 3. Backward through time. Shared weights collect gradients from all steps.
      RecurrentModel::Gradients gradients{};
      for (size_t step = SEQUENCE_LENGTH; step-- > 0;) {
        const auto previous = model.lstm.backward_pass(
            history[step], hidden_gradient, cell_gradient, gradients);
        hidden_gradient = previous.previous_hidden;
        cell_gradient = previous.previous_cell;
      }

      // 4. Plain stochastic gradient descent: parameter -= learning_rate*gradient.
      // All gradients used the old parameters; update only after backward finishes.
      model.lstm.apply_gradients(gradients, hyperparameters.learning_rate);
      for (size_t i = 0; i < HIDDEN_SIZE; ++i)
        model.output_weights[i] -=
            hyperparameters.learning_rate * score_gradient * hidden[i];
      model.output_bias -= hyperparameters.learning_rate * score_gradient;
    }

    if (epoch == 1 || epoch % 5 == 0 || epoch == hyperparameters.epochs)
      print_metrics(epoch, model, training, validation);
  }
}

void print_examples(Model &model, const Dataset &dataset) {
  for (size_t index = 0; index < 6 && index < dataset.size(); ++index) {
    const auto &sample = dataset[index];
    const double predicted = probability(predict_score(model, sample));
    std::cout << "  [";
    for (size_t step = 0; step < SEQUENCE_LENGTH; ++step)
      std::cout << (step == 0 ? "" : ", ") << sample.frames[step][0];
    std::cout << "]  expected=" << (sample.target == 1.0 ? "positive" : "negative")
              << "  predicted=" << (predicted >= 0.5 ? "positive" : "negative")
              << "  P(positive)=" << predicted << '\n';
  }
}

int main() {
  const Hyperparameters hyperparameters;
  const auto training = make_dataset(TRAINING_SAMPLES, SEED + 1);
  const auto validation = make_dataset(VALIDATION_SAMPLES, SEED + 2);
  Model model(SEED);

  std::cout << std::fixed << std::setprecision(4)
            << "Task: after four random numbers, predict the first one's sign.\n"
            << "Training samples: " << training.size()
            << "; independent validation samples: " << validation.size()
            << "; learning rate: " << hyperparameters.learning_rate << "\n\n"
            << "Before training:\n";
  print_examples(model, validation);
  std::cout << '\n';
  train(model, training, validation, hyperparameters);
  std::cout << "\nAfter training (same validation examples):\n";
  print_examples(model, validation);
  return 0;
}
