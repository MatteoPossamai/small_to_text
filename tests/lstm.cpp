#include "../src/seq_2_seq/lstm.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

template <typename T> void check_initialization_and_reset() {
  using Model = LSTM<T, 2, 8, Platform::ANY>;
  Model first(12345), same_seed(12345), other_seed(67890);
  const std::array<T, 2> input{T{0.5}, T{-0.25}};
  const auto initial = first.forward_pass(input);
  if (initial != same_seed.forward_pass(input))
    throw std::runtime_error("Same seed produced different outputs");
  if (initial == other_seed.forward_pass(input))
    throw std::runtime_error("Different seeds produced identical outputs");

  bool nonzero = false;
  for (const T value : initial) {
    if (!std::isfinite(value) || std::abs(value) > T{1})
      throw std::runtime_error("Invalid hidden state");
    nonzero |= value != T{0};
  }
  if (!nonzero)
    throw std::runtime_error("Initialized model produced only zeros");

  const auto second = first.forward_pass(input);
  if (second == initial)
    throw std::runtime_error("Model did not retain sequence memory");
  first.reset_state();
  if (first.forward_pass(input) != initial ||
      first.forward_pass(input) != second)
    throw std::runtime_error("Reset did not replay the sequence exactly");

  // Default construction also initializes weights and supports state reset.
  Model default_model;
  const auto default_initial = default_model.forward_pass(input);
  default_model.reset_state();
  if (default_model.forward_pass(input) != default_initial)
    throw std::runtime_error("Default model reset changed its parameters");
}

void check_supplied_parameters() {
  std::array<std::array<double, 10>, 8> weights{};
  std::array<double, 8> zero{}, candidate_bias{};
  candidate_bias.fill(1.0);
  LSTM<double, 2, 8, Platform::ANY> model(weights, weights, weights, weights,
                                          zero, zero, candidate_bias, zero);
  const double expected = 0.5 * std::tanh(0.5 * std::tanh(1.0));
  for (const double value : model.forward_pass({0.0, 0.0})) {
    if (std::abs(value - expected) > 1e-12)
      throw std::runtime_error("Constructor changed supplied parameters");
  }
}

using GradientModel = LSTM<double, 2, 8, Platform::ANY>;
using Parameters = GradientModel::Gradients;
using Sequence = std::array<GradientModel::in_t, 3>;

GradientModel make_model(Parameters &parameters) {
  return GradientModel(parameters.W_f, parameters.W_i, parameters.W_c,
                       parameters.W_o, parameters.B_f, parameters.B_i,
                       parameters.B_c, parameters.B_o);
}

// Include a loss at every time step and on the final cell memory. This
// exercises both recurrent gradient paths, rather than checking one isolated
// LSTM step.
double hidden_coefficient(size_t step, size_t neuron) {
  return double(step + 1) * (neuron % 2 == 0 ? 0.1 : -0.07);
}

double objective(GradientModel &model, const Sequence &sequence) {
  model.reset_state();
  double result = 0.0;
  GradientModel::StepCache cache{};
  for (size_t step = 0; step < sequence.size(); ++step) {
    const auto hidden = model.forward_pass(sequence[step], cache);
    for (size_t neuron = 0; neuron < hidden.size(); ++neuron)
      result += hidden_coefficient(step, neuron) * hidden[neuron];
  }
  for (size_t neuron = 0; neuron < cache.cell.size(); ++neuron)
    result += 0.03 * double(neuron + 1) * cache.cell[neuron];
  return result;
}

double objective(Parameters &parameters, const Sequence &sequence) {
  auto model = make_model(parameters);
  return objective(model, sequence);
}

Sequence backward(GradientModel &model, const Sequence &sequence,
                  Parameters &gradients) {
  model.reset_state();
  std::array<GradientModel::StepCache, 3> caches{};
  for (size_t step = 0; step < sequence.size(); ++step)
    model.forward_pass(sequence[step], caches[step]);

  GradientModel::out_t d_hidden{}, d_cell{};
  for (size_t neuron = 0; neuron < d_cell.size(); ++neuron)
    d_cell[neuron] = 0.03 * double(neuron + 1);
  Sequence d_inputs{};
  for (size_t step = sequence.size(); step-- > 0;) {
    for (size_t neuron = 0; neuron < d_hidden.size(); ++neuron)
      d_hidden[neuron] += hidden_coefficient(step, neuron);
    const auto result =
        model.backward_pass(caches[step], d_hidden, d_cell, gradients);
    d_inputs[step] = result.input;
    d_hidden = result.previous_hidden;
    d_cell = result.previous_cell;
  }
  return d_inputs;
}

void check_close(double actual, double expected, const std::string &name) {
  const double tolerance =
      2e-8 + 2e-6 * std::max(std::abs(actual), std::abs(expected));
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::abs(actual - expected) > tolerance)
    throw std::runtime_error(name + ": actual " + std::to_string(actual) +
                             ", expected " + std::to_string(expected));
}

void check_gradients() {
  Parameters parameters{};
  size_t index = 0;
  for (auto *matrix :
       {&parameters.W_f, &parameters.W_i, &parameters.W_c, &parameters.W_o}) {
    for (auto &row : *matrix) {
      for (double &value : row)
        value = 0.2 * std::sin(double(++index));
    }
  }
  for (auto *bias :
       {&parameters.B_f, &parameters.B_i, &parameters.B_c, &parameters.B_o}) {
    for (double &value : *bias)
      value = 0.1 * std::cos(double(++index));
  }
  Sequence sequence{{{{0.4, -0.3}}, {{-0.2, 0.7}}, {{0.6, 0.1}}}};
  auto model = make_model(parameters);
  Parameters gradients{};
  const auto d_inputs = backward(model, sequence, gradients);
  constexpr double epsilon = 1e-6;

  // Compare every matrix weight and bias against a central finite difference.
  // Reconstruct the model for each perturbation so its memory always starts at
  // 0.
  auto check_parameter = [&](double &parameter, double derivative) {
    const double original = parameter;
    parameter = original + epsilon;
    const double plus = objective(parameters, sequence);
    parameter = original - epsilon;
    const double minus = objective(parameters, sequence);
    parameter = original;
    check_close(derivative, (plus - minus) / (2.0 * epsilon),
                "Parameter gradient");
  };
  const std::array weight_members{&Parameters::W_f, &Parameters::W_i,
                                  &Parameters::W_c, &Parameters::W_o};
  const std::array bias_members{&Parameters::B_f, &Parameters::B_i,
                                &Parameters::B_c, &Parameters::B_o};
  for (const auto member : weight_members) {
    for (size_t row = 0; row < (parameters.*member).size(); ++row) {
      for (size_t col = 0; col < (parameters.*member)[row].size(); ++col)
        check_parameter((parameters.*member)[row][col],
                        (gradients.*member)[row][col]);
    }
  }
  for (const auto member : bias_members) {
    for (size_t row = 0; row < (parameters.*member).size(); ++row)
      check_parameter((parameters.*member)[row], (gradients.*member)[row]);
  }

  for (size_t step = 0; step < sequence.size(); ++step) {
    for (size_t feature = 0; feature < sequence[step].size(); ++feature) {
      const double original = sequence[step][feature];
      sequence[step][feature] = original + epsilon;
      const double plus = objective(parameters, sequence);
      sequence[step][feature] = original - epsilon;
      const double minus = objective(parameters, sequence);
      sequence[step][feature] = original;
      check_close(d_inputs[step][feature], (plus - minus) / (2.0 * epsilon),
                  "Input gradient");
    }
  }

  Parameters accumulated = gradients;
  backward(model, sequence, accumulated);
  for (const auto member : weight_members) {
    for (size_t row = 0; row < (gradients.*member).size(); ++row) {
      for (size_t col = 0; col < (gradients.*member)[row].size(); ++col)
        check_close((accumulated.*member)[row][col],
                    2.0 * (gradients.*member)[row][col], "Accumulated weight");
    }
  }
  for (const auto member : bias_members) {
    for (size_t row = 0; row < (gradients.*member).size(); ++row)
      check_close((accumulated.*member)[row], 2.0 * (gradients.*member)[row],
                  "Accumulated bias");
  }

  const double before = objective(model, sequence);
  constexpr double learning_rate = 0.01;
  model.apply_gradients(gradients, learning_rate);
  for (const auto member : weight_members) {
    for (size_t row = 0; row < (parameters.*member).size(); ++row) {
      for (size_t col = 0; col < (parameters.*member)[row].size(); ++col)
        (parameters.*member)[row][col] -=
            learning_rate * (gradients.*member)[row][col];
    }
  }
  for (const auto member : bias_members) {
    for (size_t row = 0; row < (parameters.*member).size(); ++row)
      (parameters.*member)[row] -= learning_rate * (gradients.*member)[row];
  }
  const double after = objective(model, sequence);
  check_close(after, objective(parameters, sequence),
              "Gradient descent update");
  if (!(after < before))
    throw std::runtime_error("Gradient descent did not reduce the objective");
}

} // namespace

int main() {
  try {
    check_initialization_and_reset<float>();
    check_initialization_and_reset<double>();
    check_supplied_parameters();
    check_gradients();
    std::cout << "LSTM initialization, reset, and gradient tests passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
