// Exercise the same encoding entry point as the command-line program.
#define main compile_audio_main
#include "../src/compile_audio_to_weights.cpp"
#undef main

#include <filesystem>
#include <random>

namespace {

using EncodedClip = std::array<output_t, NETWORK_IN_VALUES>;

void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

// A real, little-endian PCM WAV keeps these tests independent of existing data.
void write_wav(const std::filesystem::path &path,
               const std::vector<input_t> &samples) {
  std::ofstream file(path, std::ios::binary);
  auto write_number = [&](std::uint32_t value, int bytes) {
    for (int i = 0; i < bytes; ++i)
      file.put(static_cast<char>((value >> (8 * i)) & 0xff));
  };
  file.write("RIFF", 4);
  write_number(36 + samples.size() * 2, 4);
  file.write("WAVEfmt ", 8);
  write_number(16, 4); // PCM format header size.
  write_number(1, 2);  // PCM.
  write_number(1, 2);  // Mono.
  write_number(16000, 4);
  write_number(32000, 4); // Bytes per second.
  write_number(2, 2);     // Bytes per sample.
  write_number(16, 2);    // Bits per sample.
  file.write("data", 4);
  write_number(samples.size() * 2, 4);
  for (auto sample : samples)
    write_number(static_cast<std::uint16_t>(sample), 2);
  require(bool(file), "Could not write test WAV");
}

// Independent scalar DFT: no production frequency list or lookup tables.
EncodedClip reference_encode(const std::vector<input_t> &samples) {
  static const auto sine = [] {
    std::vector<std::array<long double, 400>> table(200);
    for (size_t bin = 0; bin < 200; ++bin)
      for (size_t i = 0; i < 400; ++i)
        table[bin][i] = std::sin(2 * std::numbers::pi_v<long double> *
                                (bin + 1) * 40 * i / 16000);
    return table;
  }();
  static const auto cosine = [] {
    std::vector<std::array<long double, 400>> table(200);
    for (size_t bin = 0; bin < 200; ++bin)
      for (size_t i = 0; i < 400; ++i)
        table[bin][i] = std::cos(2 * std::numbers::pi_v<long double> *
                                (bin + 1) * 40 * i / 16000);
    return table;
  }();

  EncodedClip result{};
  for (size_t window = 0; window < 79; ++window) {
    for (size_t band = 0; band < 20; ++band) {
      long double energy = 0;
      for (size_t bin = band * 10; bin < (band + 1) * 10; ++bin) {
        long double real = 0;
        long double imaginary = 0;
        for (size_t i = 0; i < 400; ++i) {
          const size_t sample_idx = window * 200 + i;
          const input_t sample =
              sample_idx < samples.size() ? samples[sample_idx] : 0;
          real += sample * cosine[bin][i];
          imaginary += sample * sine[bin][i];
        }
        energy += real * real + imaginary * imaginary;
      }
      result[window * 20 + band] = std::sqrt(energy);
    }
  }
  return result;
}

std::vector<input_t> tone(size_t frequency, double amplitude = 10000,
                          size_t count = 16000) {
  std::vector<input_t> samples(count);
  for (size_t i = 0; i < count; ++i)
    samples[i] = std::lround(amplitude * std::sin(
        2 * std::numbers::pi * frequency * i / 16000));
  return samples;
}

EncodedClip check_encoding(const std::filesystem::path &path,
                           const std::string &name,
                           const std::vector<input_t> &samples) {
  write_wav(path, samples);
  WavFile wav(path.string());
  EncodedClip actual{};
  crunck_file_data(wav, actual);
  const auto expected = reference_encode(samples);
  for (size_t i = 0; i < actual.size(); ++i) {
    // Float output rounds large magnitudes; near-silent bins need an absolute
    // tolerance because a relative tolerance alone approaches zero.
    const double tolerance = 0.002 + std::abs(expected[i]) * 0.000002;
    require(std::isfinite(actual[i]) &&
                std::abs(actual[i] - expected[i]) <= tolerance,
            name + ": output " + std::to_string(i) + " was " +
                std::to_string(actual[i]) + ", expected " +
                std::to_string(expected[i]));
  }
  std::cout << "Passed: " << name << '\n';
  return actual;
}

void run_tests(const std::filesystem::path &path) {
  static_assert(NETWORK_IN_VALUES == 79 * 20);
  static_assert(sizeof(EncodedClip) == 6320);

  check_encoding(path, "silence", std::vector<input_t>(16000));

  const auto middle_tone = check_encoding(path, "1000 Hz tone", tone(1000));
  require(middle_tone[2] > 1999000,
          "1000 Hz tone must be captured in the third band");

  const auto lower_edge = check_encoding(path, "400 Hz boundary", tone(400));
  require(lower_edge[0] > 1999000 && lower_edge[1] < 100,
          "400 Hz must belong to the first band");
  const auto upper_edge = check_encoding(path, "440 Hz boundary", tone(440));
  require(upper_edge[0] < 100 && upper_edge[1] > 1999000,
          "440 Hz must belong to the second band");

  auto mixed = tone(80, 6000);
  const auto second_tone = tone(320, 8000);
  for (size_t i = 0; i < mixed.size(); ++i)
    mixed[i] += second_tone[i];
  const auto combined = check_encoding(path, "two tones in one band", mixed);
  require(std::abs(combined[0] - 2000000) < 200,
          "Band strength must combine squared magnitudes");

  std::vector<input_t> nyquist(16000);
  for (size_t i = 0; i < nyquist.size(); ++i)
    nyquist[i] = i % 2 == 0 ? 10000 : -10000;
  const auto highest = check_encoding(path, "8000 Hz Nyquist tone", nyquist);
  require(std::abs(highest[19] - 4000000) < 1,
          "8000 Hz must be included in the last band");

  std::mt19937 random(12345);
  std::uniform_int_distribution<input_t> sample(-32768, 32767);
  std::vector<input_t> noise(16000);
  for (auto &value : noise)
    value = sample(random);
  check_encoding(path, "seeded random samples", noise);

  const auto short_clip =
      check_encoding(path, "short clip with silence padding", tone(1000, 10000, 275));
  require(short_clip[2] > 0 && short_clip[20 + 2] > 0,
          "Both overlapping windows must read the short clip");
  for (size_t i = 40; i < short_clip.size(); ++i)
    require(short_clip[i] == 0, "Missing samples must be padded with silence");

  std::vector<input_t> final_sample(16000);
  final_sample.back() = 10000;
  const auto last_window =
      check_encoding(path, "last sample and last window", final_sample);
  for (size_t i = 0; i < 78 * 20; ++i)
    require(last_window[i] == 0, "Last sample leaked into an earlier window");
  for (size_t i = 78 * 20; i < last_window.size(); ++i)
    require(std::abs(last_window[i] - std::sqrt(10.) * 10000) < 0.1,
            "Last window must contain the final sample in every band");
}

} // namespace

int main() {
  // A unique temporary directory allows concurrent test runs.
  auto path = std::filesystem::temp_directory_path() /
              ("small-to-text-test-" + std::to_string(std::random_device{}()));
  if (!std::filesystem::create_directory(path))
    return 1;
  try {
    run_tests(path / "input.wav");
    std::filesystem::remove_all(path);
    return 0;
  } catch (const std::exception &error) {
    std::filesystem::remove_all(path);
    std::cerr << "Encoding test failed: " << error.what() << '\n';
    return 1;
  }
}
