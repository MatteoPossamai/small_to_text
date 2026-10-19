#include "../src/network.hpp"
#include <iostream>
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

void run_tests() {
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
    std::cout << "Passed: network biases, loading, initialization and forward pass\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
