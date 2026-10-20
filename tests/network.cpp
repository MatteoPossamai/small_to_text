#include "../src/network.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {

void check_output(const Network<2, 3, 2, 2> &network,
                  double first_logit, double second_logit) {
  const double max = std::max(first_logit, second_logit);
  const double first = std::exp(first_logit - max);
  const double second = std::exp(second_logit - max);
  const auto &actual = network.output();
  if (!std::isfinite(actual[0]) || !std::isfinite(actual[1]) ||
      std::abs(actual[0] - first / (first + second)) > 1e-6 ||
      std::abs(actual[1] - second / (first + second)) > 1e-6)
    throw std::runtime_error("Unexpected network probabilities");
}

void test_files() {
  struct TemporaryFile {
    std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("small_to_text_network_" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()) +
         ".txt");
    ~TemporaryFile() {
      std::error_code error;
      std::filesystem::remove(path, error);
    }
  } saved, rewritten;

  Network<2, 3, 2, 2> original, restored;
  original.load_weights({{{1.25, -1, 2}, {3, 1, 1}}},
                        {{{1, 2}, {3, 4}, {-1, 2}}},
                        {{{10, 20}, {1, -1}}},
                        {2, 4.5, -4}, {-5, 1}, {-8, 7.125});
  if (!original.to_file(saved.path.string()) ||
      !restored.from_file(saved.path.string()) ||
      !restored.to_file(rewritten.path.string()))
    throw std::runtime_error("Network file round trip failed");

  // Rewriting must preserve every serialized weight and bias exactly.
  std::ifstream first(saved.path), second(rewritten.path);
  std::ostringstream first_contents, second_contents;
  first_contents << first.rdbuf();
  second_contents << second.rdbuf();
  if (first_contents.str() != second_contents.str())
    throw std::runtime_error("Network file round trip changed parameters");

  original.load({2, -1});
  restored.load({2, -1});
  original.forward_pass();
  restored.forward_pass();
  if (original.output() != restored.output())
    throw std::runtime_error("Restored network output differs");

  const auto reject = [&](const std::string &contents) {
    {
      std::ofstream file(saved.path);
      file << contents;
    }
    if (restored.from_file(saved.path.string()))
      throw std::runtime_error("Invalid network file accepted");
  };
  reject("");
  reject("2 4 2 2\n"); // Incompatible dimensions.
  reject("2 3 2 2\n2 4 -4\n-5 1\n-8 7\n"); // Missing weights.
  reject("2 3 2 2\ninvalid\n");
  auto truncated = first_contents.str();
  truncated.erase(truncated.find_last_of(
      " \n\t\r", truncated.find_last_not_of(" \n\t\r")));
  reject(truncated); // Missing the final weight.

  std::filesystem::remove(saved.path);
  if (restored.from_file(saved.path.string()))
    throw std::runtime_error("Missing network file accepted");
}

void test_backprop() {
  Network<2, 3, 2, 2> network;
  network.load_weights({{{1, 0, -1}, {0, 1, 0}}},
                       {{{1, 0}, {0, 1}, {1, -1}}},
                       {{{1, -1}, {-1, 1}}});
  network.load({1, 2});
  network.forward_pass();
  check_output(network, -1, 1);
  const auto initial_probability = network.output()[0];

  network.backprop({1, 0}, 0.1f);
  network.forward_pass();
  // Independent reference from the numerical training walkthrough.
  check_output(network, 3.214125517, -3.214125517);
  if (network.output()[0] <= initial_probability)
    throw std::runtime_error("Backprop did not improve the example prediction");
}

void run_tests() {
  test_files();
  test_backprop();
  const std::array<std::array<output_t, 3>, 2> w0{{{1, -1, 2}, {3, 1, 1}}};
  const std::array<std::array<output_t, 2>, 3> w1{{{1, 2}, {3, 4}, {-1, 2}}};
  const std::array<std::array<output_t, 2>, 2> w2{{{10, 20}, {1, -1}}};
  Network<2, 3, 2, 2> network;
  network.load({2, -1});
  network.load_weights(w0, w1, w2, {2, 4, -4}, {-5, 1}, {-8, 7});

  // By hand: hidden layers [1, 1, 0], [0, 7], output logits [-1, 0].
  // This also checks that negative logits reach softmax without ReLU.
  for (int pass = 0; pass < 2; ++pass) {
    network.forward_pass();
    check_output(network, -1, 0);
  }

  // Loading weights alone resets biases to zero, including previously loaded ones.
  network.load_weights(w0, w1, w2);
  network.forward_pass();
  check_output(network, 6, -6);

  // Zero weights isolate output biases and exercise stable softmax.
  network.load_weights({}, {}, {}, {}, {}, {1000, 1001});
  network.forward_pass();
  check_output(network, 1000, 1001);

  // Random initialization clears loaded biases. Zero input then gives zero logits.
  network.load_weights(w0, w1, w2, {1, 2, 3}, {4, 5}, {6, 7});
  network.initialize_random_weights();
  network.load({0, 0});
  network.forward_pass();
  check_output(network, 0, 0);
}

} // namespace

int main() {
  try {
    run_tests();
    std::cout << "Passed: network files, biases, initialization, forward pass and backprop\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
