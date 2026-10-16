#include "utils.hpp"
#include "wav_file.hpp"
#include "wav_weights.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifndef DATASET_METADATA_DIR
#define DATASET_METADATA_DIR "data"
#endif

void enqueue_dataset(const std::string &metadata_file,
                     const std::string &source_dir, const std::string &dest_dir,
                     weights_queue_t &work_queue) {
  std::ifstream input(metadata_file);
  if (!input) {
    throw std::runtime_error("Could not open dataset list: " + metadata_file);
  }

  std::string line;
  while (std::getline(input, line)) {
    std::istringstream row(line);
    std::string file_name;
    std::string label;
    std::string extra;
    if (!(row >> file_name >> label) || (row >> extra)) {
      throw std::runtime_error("Malformed dataset row in " + metadata_file +
                               ": " + line);
    }
    work_queue.put({source_dir + "/" + label + "/" + file_name,
                    dest_dir + "/" + label + "/" + file_name});
  }
  if (input.bad()) {
    throw std::runtime_error("Error reading dataset list: " + metadata_file);
  }
}

[[gnu::always_inline]] inline void
crunck_file_data(size_t start_point, WavFile &wf, char *f_buffer) {
  for (size_t i = start_point; i < N_SAMPLES; i += 2) {
    wf.read_data(f_buffer, SAMPLES_PER_BLOCK);
  }
}

void thread_fn(weights_queue_t &work_queue) {
  auto val = work_queue.pop();
  while (val.has_value()) {
    // Collect work and initialization
    auto source_file = get<0>(val.value());
    auto dest_file = get<1>(val.value());
    std::array<input_t, NETWORK_IN_VALUES> buffer{};
    WavFile wf(source_file);
    wf.seek_to_data();
    std::array<output_t, NETWORK_IN_VALUES> buffer_out;

    // Actual work
    char f_buffer[SAMPLES_PER_BLOCK];
    crunck_file_data(0, wf, f_buffer);
    wf.seek_to_data();
    wf.read_data(f_buffer, SAMPLES_SHIFT);
    crunck_file_data(1, wf, f_buffer);

    WavWeights ww{};
    ww.from_buffer(std::move(buffer_out));
    ww.to_file(dest_file);
    val = work_queue.pop();
  }
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " " << "<SOURCE_DIR> <DEST_DIR>"
              << std::endl;
    return 1;
  }
  auto source_dir = static_cast<std::string>(argv[1]);
  auto dest_dir = static_cast<std::string>(argv[2]);
  std::string training_dir = dest_dir + "/training";
  std::string scoring_dir = dest_dir + "/scoring";

  weights_queue_t work_queue;

  const std::string metadata_dir = std::string(DATASET_METADATA_DIR) + "/";
  enqueue_dataset(metadata_dir + "training.txt", source_dir, training_dir,
                  work_queue);
  enqueue_dataset(metadata_dir + "scoring.txt", source_dir, scoring_dir,
                  work_queue);

  std::vector<std::jthread> threads;

  for (unsigned int i = 0; i < std::thread::hardware_concurrency(); ++i) {
    std::jthread thread(thread_fn, std::ref(work_queue));
  }

  return 0;
}
