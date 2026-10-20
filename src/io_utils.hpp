#include <cmath>
#include <cstddef>
#include <iostream>

inline void progress_bar(size_t value, size_t total) {
  constexpr size_t BAR_LEN = 40;
  size_t percentage =
      static_cast<size_t>((static_cast<float_t>(value) / total) * 100);
  size_t filled = value * BAR_LEN / total;
  std::cout << "\r[";
  for (size_t i = 0; i < BAR_LEN; i++) {
    if (i < filled) {
      std::cout << "#";
    } else {
      std::cout << " ";
    }
  }
  std::cout << "] Completed " << percentage << "%. " << value << " out of "
            << total << std::flush;
}
