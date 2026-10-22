#include "../src/cnn.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

using Model = CNN<12, 2, 3, 3, 2, 4, 3, 2>;

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

struct TemporaryDirectory {
  std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("small_to_text_cnn_" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  TemporaryDirectory() { std::filesystem::create_directory(path); }
  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
};

std::string read_file(const std::filesystem::path &path) {
  std::ifstream file(path);
  std::ostringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

void test_forward() {
  CNN<12, 2, 2, 3, 2, 2, 2, 2> model;
  model.load_weights({{{1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 1}}},
                     {{{1, 0}, {0, 0.01f}, {1, 0}, {0, 0.01f}}},
                     {{{1, 0}, {0, 1}}}, {{{0.25f, -0.25f}, {0.5f, 0.25f}}},
                     {-2, 1}, {0.5f, 0.08f}, {-0.5f, 0.5f}, {0.125f, -0.25f});

  const auto check = [&](double first_logit, double second_logit) {
    model.forward_pass();
    const double expected = 1.0 / (1.0 + std::exp(second_logit - first_logit));
    require(std::isfinite(model.output()[0]) &&
                std::isfinite(model.output()[1]) &&
                std::abs(model.output()[0] - expected) < 1e-6 &&
                std::abs(model.output()[1] - (1.0 - expected)) < 1e-6,
            "CNN forward pass differs from hand-calculated probabilities");
  };

  // Two bands per row. The neurons select the window's first and last values.
  // After biases and ReLU: [0,31], [1,41], [3,51], [5,61].
  // Averaging gives [0.5,36,4,56]; ordinary layers produce logits [2,-1].
  model.load_inputs({1, 10, 3, 20, 5, 30, 7, 40, 9, 50, 11, 60});
  check(2, -1);
  check(2, -1); // Repeating a call must not accumulate old results.

  // Zero input still exercises convolution biases: pooled [0,1,0,1].
  model.load_inputs({});
  check(0.425, -0.1);

  model.load_inputs({1, 10, 3, 20, 5, 30, 7, 40, 9, 50, 11, 60});
  check(2, -1); // A different recording must not leave stale values either.
}

void test_backprop() {
  using GradientModel = CNN<16, 2, 2, 3, 3, 3, 2, 2>;
  GradientModel model;
  model.load_weights({{{0.2f, -0.1f},
                       {-0.1f, 0.15f},
                       {0.3f, 0.1f},
                       {0.1f, -0.2f},
                       {-0.15f, 0.25f},
                       {0.2f, 0.05f}}},
                     {{{0.2f, -0.1f, 0.3f},
                       {-0.15f, 0.25f, 0.1f},
                       {0.1f, 0.2f, -0.2f},
                       {0.3f, 0.1f, 0.15f}}},
                     {{{0.3f, -0.2f}, {-0.1f, 0.25f}, {0.2f, 0.1f}}},
                     {{{0.4f, -0.3f}, {-0.2f, 0.35f}}}, {0.1f, 0.05f},
                     {0.4f, -0.5f, 0.3f}, {0.2f, 0.1f}, {0.05f, -0.1f});
  model.load_inputs({0.2f, -0.4f, 0.8f, 0.1f, -0.3f, 0.7f, 0.6f, -0.2f, -0.5f,
                     0.4f, 0.9f, -0.6f, 0.3f, 0.2f, -0.7f, 0.5f});

  TemporaryDirectory directory;
  const auto path = directory.path / "gradients.txt";
  const auto parameters = [&]() {
    require(model.to_file(path.string()), "Gradient model save failed");
    std::ifstream file(path);
    std::string header;
    std::getline(file, header);
    std::vector<output_t> values;
    output_t value;
    while (file >> value)
      values.push_back(value);
    return values;
  };
  const auto load_parameters = [&](const std::vector<output_t> &values) {
    {
      std::ofstream file(path);
      file << "CNN_V1 16 2 2 3 3 3 2 2\n"
           << std::setprecision(std::numeric_limits<output_t>::max_digits10);
      for (const auto value : values)
        file << value << ' ';
    }
    require(model.from_file(path.string()), "Gradient model load failed");
  };
  const auto original = parameters();
  require(original.size() == 43, "Unexpected gradient fixture parameter count");

  const auto objective = [&](const std::array<output_t, 2> &target) {
    model.forward_pass();
    double value = 0;
    for (size_t i = 0; i < target.size(); ++i)
      if (target[i] != 0)
        value -= target[i] * std::log(static_cast<double>(model.output()[i]));
    return value;
  };

  // Compare every bias/weight update against independent finite differences.
  // The fixture includes shared overlapping windows and inactive ReLUs.
  for (const std::array<output_t, 2> target :
       {std::array<output_t, 2>{1, 0}, std::array<output_t, 2>{0.25f, 0.75f}}) {
    load_parameters(original);
    const double before = objective(target);
    constexpr output_t rate = 0.05f;
    const auto reported_loss = model.backprop(target, rate);
    require(std::abs(reported_loss - before) < 1e-6,
            "Backprop returned the wrong pre-update loss");
    const auto updated = parameters();
    require(objective(target) < before,
            "CNN learning step did not reduce loss");

    for (size_t i = 0; i < original.size(); ++i) {
      auto perturbed = original;
      constexpr output_t epsilon = 0.001f;
      perturbed[i] = original[i] + epsilon;
      load_parameters(perturbed);
      const double plus = objective(target);
      const auto upper = perturbed[i];
      perturbed[i] = original[i] - epsilon;
      load_parameters(perturbed);
      const double minus = objective(target);
      const double numeric = (plus - minus) / (upper - perturbed[i]);
      const double analytic = (original[i] - updated[i]) / rate;
      const double tolerance =
          0.0002 + 0.01 * (std::abs(numeric) + std::abs(analytic));
      if (std::abs(numeric - analytic) > tolerance)
        throw std::runtime_error("CNN gradient mismatch at parameter " +
                                 std::to_string(i));
    }
  }

  load_parameters(original);
  objective({1, 0});
  model.backprop({1, 0}, 0);
  require(parameters() == original, "Zero learning rate changed parameters");

  // A tiny target probability can underflow; the loss should still be finite.
  model.load_weights({}, {}, {}, {}, {}, {}, {}, {-1000, 1000});
  model.forward_pass();
  const auto extreme_loss = model.backprop({1, 0}, 0.1f);
  require(std::isfinite(extreme_loss) && std::abs(extreme_loss - 2000) < 1e-5,
          "Extreme logits produced an invalid loss");
  for (const auto value : parameters())
    require(std::isfinite(value), "Extreme logits produced invalid parameters");
}

void test_files() {
  TemporaryDirectory directory;
  const auto input = directory.path / "input.txt";
  const auto output = directory.path / "output.txt";
  const std::string header = "CNN_V1 12 2 3 3 2 4 3 2\n";

  // Independent fixture: 12 biases, then 18 + 24 + 12 + 6 weights.
  std::vector<output_t> expected;
  std::ostringstream fixture;
  fixture << header
          << std::setprecision(std::numeric_limits<output_t>::max_digits10);
  for (int i = 0; i < 72; ++i) {
    expected.push_back(static_cast<output_t>((i - 36) / 7.0));
    fixture << expected.back() << ' ';
  }
  std::ofstream(input) << fixture.str();

  Model model;
  require(model.from_file(input.string()), "Valid CNN file rejected");
  require(model.to_file(output.string()), "CNN save failed");
  const auto saved = read_file(output);
  std::istringstream parameters(saved);
  std::string saved_header;
  std::getline(parameters, saved_header);
  require(saved_header + '\n' == header, "CNN dimensions changed");
  for (const auto value : expected) {
    output_t actual;
    require(bool(parameters >> actual) && actual == value,
            "CNN save changed a bias or weight");
  }
  parameters >> std::ws;
  require(parameters.eof(), "CNN save added unexpected parameters");

  Model restored;
  require(restored.from_file(output.string()), "Saved CNN file rejected");
  require(restored.to_file(input.string()), "CNN rewrite failed");
  require(read_file(input) == saved, "CNN round trip changed parameters");

  const auto reject = [&](const std::string &contents) {
    std::ofstream(input) << contents;
    require(!restored.from_file(input.string()), "Invalid CNN file accepted");
    require(restored.to_file(output.string()),
            "CNN save after rejection failed");
    require(read_file(output) == saved,
            "Failed load changed existing parameters");
  };
  reject("");
  reject("6 4 3 2\n"); // Ordinary Network format.
  reject("CNN_V2" + fixture.str().substr(6));
  const std::array<size_t, 8> dimensions{12, 2, 3, 3, 2, 4, 3, 2};
  for (size_t changed = 0; changed < dimensions.size(); ++changed) {
    std::ostringstream mismatch;
    mismatch << "CNN_V1";
    for (size_t i = 0; i < dimensions.size(); ++i)
      mismatch << ' ' << dimensions[i] + (i == changed ? 1 : 0);
    mismatch << '\n' << fixture.str().substr(header.size());
    reject(mismatch.str());
  }
  reject(header); // Missing all parameters.
  auto truncated = fixture.str();
  truncated.erase(truncated.find_last_of(' ', truncated.size() - 2));
  reject(truncated); // Missing final weight, after otherwise valid parameters.
  reject(header + "invalid\n");
  reject(header + "nan\n");
  reject(header + "1e1000\n");
  reject(fixture.str() + "unexpected");
  reject(fixture.str() + "42");
  require(!restored.from_file((directory.path / "missing").string()),
          "Missing CNN file accepted");
  require(!restored.to_file(directory.path.string()),
          "Invalid save path accepted");

  // Both convolution and ordinary parameters participate in load_weights().
  restored.load_weights({}, {}, {}, {}, {1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10},
                        {11, 12});
  require(restored.to_file(output.string()),
          "Explicit CNN parameter save failed");
  std::istringstream explicit_parameters(read_file(output));
  std::getline(explicit_parameters, saved_header);
  for (int i = 1; i <= 72; ++i) {
    output_t value;
    require(bool(explicit_parameters >> value) && value == (i <= 12 ? i : 0),
            "Explicit CNN weights or biases were not loaded");
  }

  restored.initialize_random_weights();
  require(restored.to_file(output.string()), "Initialized CNN save failed");
  std::istringstream initialized(read_file(output));
  std::getline(initialized, saved_header);
  bool nonzero_convolution = false;
  for (int i = 0; i < 72; ++i) {
    output_t value;
    require(bool(initialized >> value) && std::isfinite(value),
            "Initialization produced an invalid CNN parameter");
    if (i < 12)
      require(value == 0, "Initialization did not reset CNN biases");
    if (i >= 12 && i < 30)
      nonzero_convolution |= value != 0;
  }
  require(nonzero_convolution, "Convolution weights were not initialized");
}

} // namespace

int main() {
  try {
    test_files();
    test_forward();
    test_backprop();
    std::cout << "Passed: CNN files, dimensions, malformed files and "
                 "initialization, convolution, averaging, forward pass and "
                 "backprop\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
