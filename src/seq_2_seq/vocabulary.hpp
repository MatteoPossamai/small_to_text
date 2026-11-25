#pragma once
#include <array>
#include <cstdint>
#include <string_view>

constexpr std::array<std::string_view, 43> VOCABULARY = {
    "a", "b", "c", "d", "e", "f", "g", "h",     "i",     "j",    "k",
    "l", "m", "n", "o", "p", "q", "r", "s",     "t",     "u",    "v",
    "w", "x", "y", "z", "0", "1", "2", "3",     "4",     "5",    "6",
    "7", "8", "9", " ", ",", ".", "'", "<UNK>", "<EOS>", "<SOS>"};

constexpr auto CHAR_TO_INDEX = [] {
  std::array<uint8_t, 256> lut{};
  lut.fill(40); // Default to <UNK>

  for (size_t i = 0; i < VOCABULARY.size(); ++i) {
    if (VOCABULARY[i].size() == 1) {
      lut[static_cast<unsigned char>(VOCABULARY[i][0])] = i;
    }
  }

  return lut;
}();

constexpr uint8_t token_to_index(std::string_view token) {
  if (token.size() == 1)
    return CHAR_TO_INDEX[static_cast<unsigned char>(token[0])];

  if (token == "<EOS>")
    return 41;
  if (token == "<SOS>")
    return 42;

  return 40; // <UNK>
}
