#include "lstm.hpp"
#include <iostream>

int main() {
  std::array<double, 40> in{};
  auto lstm = LSTM<double, 40, 256, Platform::ANY, false>();
  auto out_i = lstm.forward_pass(in);

  for (size_t i = 0; i < 256; ++i)
    std::cout << out_i[i] << std::endl;
  return 0;
}
