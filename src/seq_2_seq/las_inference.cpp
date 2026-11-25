#include "las.hpp"
#include <iostream>

int main() {
  std::unique_ptr<LAS<double, false>> las =
      std::make_unique<LAS<double, false>>();
  // TODO: get some input
  const std::array<std::array<double, SAMPLE_SIZE>, INPUT_SAMPLE> input{};
  std::cout << las->inference(input) << std::endl;
  return 0;
}
