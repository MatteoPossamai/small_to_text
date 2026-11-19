#include "../src/seq_2_seq/lstm.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
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
using Sequence = std::array<GradientModel::in_t, 3>;

// Test-owned parameters: the LSTM keeps its parameters and gradients private.
struct Parameters {
  GradientModel::weight_t W_f{}, W_i{}, W_c{}, W_o{};
  GradientModel::out_t B_f{}, B_i{}, B_c{}, B_o{};
};

GradientModel make_model(Parameters &parameters) {
  return GradientModel(parameters.W_f, parameters.W_i, parameters.W_c,
                       parameters.W_o, parameters.B_f, parameters.B_i,
                       parameters.B_c, parameters.B_o);
}

// A loss at every step checks that shared weight gradients accumulate across
// time, and that later errors flow through both hidden and cell memory.
double hidden_coefficient(size_t step, size_t neuron) {
  return double(step + 1) * (neuron % 2 == 0 ? 0.1 : -0.07);
}

double objective(GradientModel &model, const Sequence &sequence) {
  model.reset_state();
  double result = 0.0;
  for (size_t step = 0; step < sequence.size(); ++step) {
    const auto hidden = model.forward_pass(sequence[step]);
    for (size_t neuron = 0; neuron < hidden.size(); ++neuron)
      result += hidden_coefficient(step, neuron) * hidden[neuron];
  }
  return result;
}

double objective(Parameters &parameters, const Sequence &sequence) {
  auto model = make_model(parameters);
  return objective(model, sequence);
}

Sequence backward(GradientModel &model, const Sequence &sequence) {
  model.reset_state();
  for (const auto &input : sequence)
    model.forward_pass_train(input);

  GradientModel::out_t d_hidden{}, d_cell{};
  Sequence d_inputs{};
  for (size_t step = sequence.size(); step-- > 0;) {
    for (size_t neuron = 0; neuron < d_hidden.size(); ++neuron)
      d_hidden[neuron] += hidden_coefficient(step, neuron);
    const auto result = model.backward_pass(d_hidden, d_cell);
    d_inputs[step] = result.input_gradient;
    d_hidden = result.previous_hidden_gradient;
    d_cell = result.previous_cell_gradient;
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

void check_forward_modes() {
  GradientModel inference(12345), training(12345);
  const Sequence sequence{{{{0.4, -0.3}}, {{-0.2, 0.7}}, {{0.6, 0.1}}}};
  for (const auto &input : sequence) {
    const auto expected = inference.forward_pass(input);
    const auto actual = training.forward_pass_train(input);
    for (size_t neuron = 0; neuron < actual.size(); ++neuron)
      check_close(actual[neuron], expected[neuron], "Training forward output");
  }
}

// Compare actual updates against a separately constructed model whose
// parameters were updated using numerical gradients. Probe different input
// histories so agreement is checked on more than a single scalar loss.
void check_model_outputs(GradientModel &actual, Parameters &parameters) {
  const std::array<Sequence, 3> probes{{
      {{{{0.4, -0.3}}, {{-0.2, 0.7}}, {{0.6, 0.1}}}},
      {{{{-0.5, 0.2}}, {{0.8, -0.6}}, {{-0.1, 0.9}}}},
      {{{{0.0, 0.0}}, {{0.0, 0.0}}, {{0.0, 0.0}}}},
  }};
  for (const auto &sequence : probes) {
    auto expected = make_model(parameters);
    actual.reset_state();
    for (const auto &input : sequence) {
      const auto actual_output = actual.forward_pass(input);
      const auto expected_output = expected.forward_pass(input);
      for (size_t neuron = 0; neuron < actual_output.size(); ++neuron)
        check_close(actual_output[neuron], expected_output[neuron],
                    "Model output after parameter update");
    }
  }
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
  const auto d_inputs = backward(model, sequence);
  constexpr double epsilon = 1e-6;

  auto numerical_gradient = [&](double &value) {
    const double original = value;
    value = original + epsilon;
    const double plus = objective(parameters, sequence);
    value = original - epsilon;
    const double minus = objective(parameters, sequence);
    value = original;
    return (plus - minus) / (2.0 * epsilon);
  };

  // Calculate independent numerical gradients for every weight and bias.
  Parameters gradients{};
  const std::array weight_members{&Parameters::W_f, &Parameters::W_i,
                                  &Parameters::W_c, &Parameters::W_o};
  const std::array bias_members{&Parameters::B_f, &Parameters::B_i,
                                &Parameters::B_c, &Parameters::B_o};
  for (const auto member : weight_members) {
    for (size_t row = 0; row < (parameters.*member).size(); ++row) {
      for (size_t col = 0; col < (parameters.*member)[row].size(); ++col)
        (gradients.*member)[row][col] =
            numerical_gradient((parameters.*member)[row][col]);
    }
  }
  for (const auto member : bias_members) {
    for (size_t row = 0; row < (parameters.*member).size(); ++row)
      (gradients.*member)[row] = numerical_gradient((parameters.*member)[row]);
  }

  for (size_t step = 0; step < sequence.size(); ++step) {
    for (size_t feature = 0; feature < sequence[step].size(); ++feature)
      check_close(d_inputs[step][feature],
                  numerical_gradient(sequence[step][feature]),
                  "Input gradient");
  }

  // Backward accumulates gradients but must not change any parameters yet.
  auto before_update = model;
  check_model_outputs(before_update, parameters);

  const double before = objective(parameters, sequence);
  constexpr double learning_rate = 0.1;
  model.apply_gradient(learning_rate);
  auto expected_parameters = parameters;
  for (const auto member : weight_members) {
    for (size_t row = 0; row < (parameters.*member).size(); ++row) {
      for (size_t col = 0; col < (parameters.*member)[row].size(); ++col)
        (expected_parameters.*member)[row][col] -=
            learning_rate * (gradients.*member)[row][col];
    }
  }
  for (const auto member : bias_members) {
    for (size_t row = 0; row < (parameters.*member).size(); ++row)
      (expected_parameters.*member)[row] -=
          learning_rate * (gradients.*member)[row];
  }
  check_model_outputs(model, expected_parameters);
  if (!(objective(model, sequence) < before))
    throw std::runtime_error("Gradient descent did not reduce the objective");

  // reset_state clears pending gradients as well as recurrent state and
  // history.
  auto reset_model = make_model(parameters);
  backward(reset_model, sequence);
  reset_model.reset_state();
  reset_model.apply_gradient(learning_rate);
  check_model_outputs(reset_model, parameters);
}

struct TemporaryDirectory {
  std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("small_to_text_lstm_" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));

  TemporaryDirectory() {
    if (!std::filesystem::create_directory(path))
      throw std::runtime_error("Could not create temporary test directory");
  }
  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
};

std::string file_contents(const std::filesystem::path &path) {
  std::ifstream input(path);
  if (!input)
    throw std::runtime_error("Could not read test model");
  std::ostringstream result;
  result << input.rdbuf();
  return result.str();
}

template <typename T> void check_serialization() {
  using Model = LSTM<T, 2, 8, Platform::ANY>;
  const std::array<typename Model::in_t, 3> sequence{
      {{{T{0.4}, T{-0.3}}}, {{T{-0.2}, T{0.7}}}, {{T{0.6}, T{0.1}}}}};
  typename Model::out_t d_hidden{}, d_cell{};
  d_hidden.fill(T{0.2});
  d_cell.fill(T{-0.03});
  const auto accumulate = [&](Model &model) {
    model.reset_state();
    for (const auto &input : sequence)
      model.forward_pass_train(input);
    auto hidden = d_hidden, cell = d_cell;
    for (size_t step = sequence.size(); step-- > 0;) {
      const auto result = model.backward_pass(hidden, cell);
      hidden = result.previous_hidden_gradient;
      cell = result.previous_cell_gradient;
    }
  };

  TemporaryDirectory directory;
  const auto saved = directory.path / "model.txt";
  const auto rewritten = directory.path / "rewritten.txt";
  const auto save_text = [&](const Model &model,
                             const std::filesystem::path &path) {
    if (!model.to_file(path.string()))
      throw std::runtime_error("Could not save test model");
    return file_contents(path);
  };
  Model trained(12345);
  accumulate(trained);
  trained.apply_gradient(T{0.05});
  const auto text = save_text(trained, saved);

  // Load into a model that already has memory, history and pending gradients.
  Model restored(67890);
  accumulate(restored);
  restored.forward_pass_train(sequence[0]);
  if (!restored.from_file(saved.string()) ||
      save_text(restored, rewritten) != text)
    throw std::runtime_error("Model serialization lost parameter precision");
  restored.apply_gradient(T{0.05});
  if (save_text(restored, rewritten) != text)
    throw std::runtime_error("Loading preserved stale parameter gradients");
  trained.reset_state();
  for (const auto &input : sequence) {
    if (trained.forward_pass(input) != restored.forward_pass(input))
      throw std::runtime_error("Loaded model did not start with empty memory");
  }

  // Continuing training after a roundtrip must produce identical parameters.
  accumulate(trained);
  accumulate(restored);
  trained.apply_gradient(T{0.05});
  restored.apply_gradient(T{0.05});
  if (save_text(trained, saved) != save_text(restored, rewritten))
    throw std::runtime_error(
        "Training changed after a serialization roundtrip");

  // Preserve a partly consumed training history when loading fails.
  trained.reset_state();
  trained.forward_pass_train(sequence[0]);
  trained.forward_pass_train(sequence[1]);
  trained.backward_pass(d_hidden, d_cell);
  const auto malformed = directory.path / "malformed.txt";
  const std::array<std::string, 7> invalid{"LSTM_V2 2 8\n",
                                           "LSTM_V1 3 8\n",
                                           "LSTM_V1 2 16\n",
                                           text.substr(0, text.size() / 2),
                                           "LSTM_V1 2 8\nnonnumeric\n",
                                           "LSTM_V1 2 8\ninf\n",
                                           text + "trailing_token\n"};
  const auto check_rejected = [&](const std::filesystem::path &path) {
    auto actual = trained, expected = trained;
    if (actual.from_file(path.string()))
      throw std::runtime_error("Invalid model was accepted");
    if (actual.forward_pass(sequence[2]) != expected.forward_pass(sequence[2]))
      throw std::runtime_error("Failed load changed recurrent state");
    const auto actual_back = actual.backward_pass(d_hidden, d_cell);
    const auto expected_back = expected.backward_pass(d_hidden, d_cell);
    if (actual_back.input_gradient != expected_back.input_gradient ||
        actual_back.previous_hidden_gradient !=
            expected_back.previous_hidden_gradient ||
        actual_back.previous_cell_gradient !=
            expected_back.previous_cell_gradient)
      throw std::runtime_error("Failed load changed cached training history");
    actual.apply_gradient(T{0.05});
    expected.apply_gradient(T{0.05});
    if (save_text(actual, saved) != save_text(expected, rewritten))
      throw std::runtime_error("Failed load changed parameters or gradients");
  };
  for (const auto &contents : invalid) {
    {
      std::ofstream output(malformed);
      output << contents;
    }
    check_rejected(malformed);
  }
  check_rejected(directory.path / "missing.txt");
  if (trained.to_file((directory.path / "missing" / "model.txt").string()))
    throw std::runtime_error("Saving under a missing parent succeeded");
}

} // namespace

int main() {
  try {
    check_initialization_and_reset<float>();
    check_initialization_and_reset<double>();
    check_supplied_parameters();
    check_forward_modes();
    check_gradients();
    check_serialization<float>();
    check_serialization<double>();
    std::cout
        << "LSTM initialization, gradient, and serialization tests passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
