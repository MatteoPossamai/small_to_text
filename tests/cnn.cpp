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
    std::cout << "Passed: CNN files, dimensions, malformed files and "
                 "initialization\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
