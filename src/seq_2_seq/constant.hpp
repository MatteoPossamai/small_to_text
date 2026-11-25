#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

namespace las_weights {

inline constexpr char directory[] = "las";
inline constexpr char current_filename[] = "current";
inline constexpr std::string_view checkpoint_prefix = "checkpoint-";
inline constexpr std::array<const char *, 2> speller_filenames{"speller_1.txt",
                                                               "speller_2.txt"};
inline constexpr char mlp_filename[] = "char_mlp.txt";

enum class Direction { forward, backward };

inline std::string listener_filename(std::size_t layer, Direction direction) {
  return "listener_" + std::to_string(layer + 1) +
         (direction == Direction::forward ? "_forward.txt" : "_backward.txt");
}

template <typename Model>
void save_network(const Model &model, const std::filesystem::path &file) {
  if (!model.to_file(file.string()))
    throw std::runtime_error("Could not save weights to " + file.string());
}

template <typename Model>
void load_network(Model &model, const std::filesystem::path &file) {
  if (!model.from_file(file.string()))
    throw std::runtime_error("Could not load weights from " + file.string());
}

} // namespace las_weights
