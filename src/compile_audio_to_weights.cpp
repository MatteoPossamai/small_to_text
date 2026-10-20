#include "trigonometric_functions.hpp"
#include "utils.hpp"
#include "wav_file.hpp"
#include "wav_weights.hpp"
#include <algorithm>
#include <atomic>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <syncstream>
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

struct CompileStats {
  std::atomic<size_t> completed{0};
  std::atomic<size_t> short_clips{0};
  std::atomic<size_t> failed{0};
};

void thread_fn(weights_queue_t &work_queue, CompileStats &stats) {
  while (auto val = work_queue.pop()) {
    const auto &[source_file, dest_file] = *val;
    try {
      WavFile wf(source_file);
      std::array<output_t, NETWORK_IN_VALUES> buffer{};

      crunck_file_data(wf, buffer);

      WavWeights ww{};
      ww.from_buffer(buffer);
      if (!ww.to_file(dest_file))
        throw std::runtime_error("Could not write weights: " + dest_file);
      stats.completed.fetch_add(1, std::memory_order_relaxed);
      if (wf.samples_read() < N_SAMPLES)
        stats.short_clips.fetch_add(1, std::memory_order_relaxed);
    } catch (const std::exception &error) {
      stats.failed.fetch_add(1, std::memory_order_relaxed);
      std::osyncstream(std::cerr)
          << "Failed: " << source_file << ": " << error.what() << '\n';
    }
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

  try {
    const std::string metadata_dir = std::string(DATASET_METADATA_DIR) + "/";
    enqueue_dataset(metadata_dir + "training.txt", source_dir, training_dir,
                    work_queue);
    enqueue_dataset(metadata_dir + "scoring.txt", source_dir, scoring_dir,
                    work_queue);
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }

  CompileStats stats;
  {
    std::vector<std::jthread> threads;
    const auto worker_count = std::max(1u, std::thread::hardware_concurrency());
    for (unsigned int i = 0; i < worker_count; ++i)
      threads.emplace_back(thread_fn, std::ref(work_queue), std::ref(stats));
  }

  std::cout << "Completed: " << stats.completed.load()
            << ", short clips: " << stats.short_clips.load()
            << ", failed: " << stats.failed.load() << '\n';
  return stats.failed.load() == 0 ? 0 : 1;
}
