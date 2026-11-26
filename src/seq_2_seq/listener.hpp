#pragma once
#include "constant.hpp"
#include "lstm.hpp"
#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

template <std::floating_point T, size_t SAMPLE_SIZE, size_t LAYERS,
          size_t HIDDEN_VALS, bool TRAIN_ENABLED = true>
class Listener {
  static_assert(LAYERS > 0);

public:
  using lstm_f_t = LSTM<T, SAMPLE_SIZE, HIDDEN_VALS, TRAIN_ENABLED>;
  using lstm_h_t = LSTM<T, HIDDEN_VALS * 4, HIDDEN_VALS, TRAIN_ENABLED>;
  using l_int_t = typename lstm_h_t::in_t;
  using input_row = std::array<T, SAMPLE_SIZE>;
  using output_row = std::array<T, 2 * HIDDEN_VALS>;

private:
  std::pair<lstm_f_t, lstm_f_t> first_layer;
  std::array<std::pair<lstm_h_t, lstm_h_t>, LAYERS - 1> layers;
  std::optional<size_t> training_frames;

  template <typename Self, typename Fn>
  static void for_each_lstm(Self &self, const Fn &fn) {
    using las_weights::Direction;
    const auto visit_pair = [&fn](auto &pair, std::size_t layer) {
      fn(pair.first, las_weights::listener_filename(layer, Direction::forward));
      fn(pair.second,
         las_weights::listener_filename(layer, Direction::backward));
    };
    visit_pair(self.first_layer, 0);
    for (std::size_t i = 0; i < self.layers.size(); ++i)
      visit_pair(self.layers[i], i + 1);
  }

  template <bool Training, typename Network>
  static auto forward_step(Network &network,
                           const typename Network::in_t &row) {
    if constexpr (Training)
      return network.forward_pass_train(row);
    else
      return network.forward_pass(row);
  }

  template <bool Training, typename Network>
  static std::vector<output_row>
  forward_layer(std::pair<Network, Network> &networks,
                std::span<const typename Network::in_t> input) {
    std::vector<output_row> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
      const size_t j = input.size() - 1 - i;
      const auto forward = forward_step<Training>(networks.first, input[i]);
      const auto backward = forward_step<Training>(networks.second, input[j]);
      std::copy(forward.begin(), forward.end(), output[i].begin());
      std::copy(backward.begin(), backward.end(),
                output[j].begin() + HIDDEN_VALS);
    }
    return output;
  }

  static size_t merged_size(size_t rows) { return rows / 2 + rows % 2; }

  static std::vector<l_int_t> merge_rows(std::span<const output_row> rows) {
    // Merge neighboring activations after the recurrent pass. The second half
    // of the final row stays zero when the sequence has an odd length.
    std::vector<l_int_t> merged(merged_size(rows.size()));
    for (size_t i = 0; i < rows.size(); ++i)
      std::copy(rows[i].begin(), rows[i].end(),
                merged[i / 2].begin() + (i % 2) * rows[i].size());
    return merged;
  }

  template <bool Training>
  std::vector<output_row> forward(std::span<const input_row> input) {
    auto output = forward_layer<Training>(first_layer, input);
    for (auto &layer : layers) {
      const auto merged = merge_rows(output);
      output = forward_layer<Training>(layer, merged);
    }
    return output;
  }

  template <typename Network>
  static std::vector<typename Network::in_t>
  backward_layer(std::pair<Network, Network> &networks,
                 std::span<const output_row> output_gradient) {
    std::vector<typename Network::in_t> input_gradient(output_gradient.size());
    typename Network::out_t forward_hidden{}, forward_cell{};
    typename Network::out_t backward_hidden{}, backward_cell{};

    const auto propagate = [&](Network &network, size_t frame, size_t offset,
                               auto &hidden, auto &cell) {
      for (size_t i = 0; i < HIDDEN_VALS; ++i)
        hidden[i] += output_gradient[frame][offset + i];
      const auto step = network.backward_pass(hidden, cell);
      hidden = step.previous_hidden_gradient;
      cell = step.previous_cell_gradient;
      for (size_t i = 0; i < step.input_gradient.size(); ++i)
        input_gradient[frame][i] += step.input_gradient[i];
    };

    // Consume each direction's saved activations in reverse order.
    for (size_t i = 0; i < output_gradient.size(); ++i) {
      const size_t reverse_frame = output_gradient.size() - 1 - i;
      propagate(networks.first, reverse_frame, 0, forward_hidden, forward_cell);
      propagate(networks.second, i, HIDDEN_VALS, backward_hidden,
                backward_cell);
    }
    return input_gradient;
  }

  static std::vector<output_row>
  split_gradients(std::span<const l_int_t> merged, size_t rows) {
    // Only the original rows receive gradients. Any zero-filled half is
    // dropped.
    std::vector<output_row> output(rows);
    for (size_t i = 0; i < rows; ++i)
      std::copy_n(merged[i / 2].begin() + (i % 2) * output[i].size(),
                  output[i].size(), output[i].begin());
    return output;
  }

public:
  Listener() = default;
  explicit Listener(const std::filesystem::path &directory) : Listener() {
    for_each_lstm(*this, [&directory](auto &network, const auto &filename) {
      las_weights::load_network(network, directory / filename);
    });
  }

  Listener(const Listener &) = delete;
  Listener(Listener &&) = delete;
  Listener &operator=(const Listener &) = delete;
  Listener &operator=(Listener &&) = delete;

  void save_weights(const std::filesystem::path &directory) const {
    for_each_lstm(*this, [&directory](auto &network, const auto &filename) {
      las_weights::save_network(network, directory / filename);
    });
  }

  void reset_state() {
    first_layer.first.reset_state();
    first_layer.second.reset_state();
    for (auto &layer : layers) {
      layer.first.reset_state();
      layer.second.reset_state();
    }
    training_frames.reset();
  }

  std::vector<output_row> forward_pass(std::span<const input_row> input) {
    return forward<false>(input);
  }

  std::vector<output_row> forward_pass_train(std::span<const input_row> input)
    requires(TRAIN_ENABLED)
  {
    if (training_frames)
      throw std::logic_error(
          "Listener already has a pending training sequence");
    auto output = forward<true>(input);
    training_frames = input.size();
    return output;
  }

  std::vector<input_row>
  backward_pass(std::span<const output_row> output_gradient)
    requires(TRAIN_ENABLED)
  {
    if (!training_frames)
      throw std::logic_error("Listener has no pending training sequence");

    std::array<size_t, LAYERS> sizes{*training_frames};
    for (size_t i = 1; i < sizes.size(); ++i)
      sizes[i] = merged_size(sizes[i - 1]);
    if (output_gradient.size() != sizes.back())
      throw std::invalid_argument(
          "Listener gradient length does not match the sequence");

    std::vector<output_row> previous_gradient;
    for (size_t layer = layers.size(); layer > 0; --layer) {
      const auto merged = backward_layer(layers[layer - 1], output_gradient);
      previous_gradient = split_gradients(merged, sizes[layer - 1]);
      output_gradient = previous_gradient;
    }
    auto input_gradient = backward_layer(first_layer, output_gradient);
    training_frames.reset();
    return input_gradient;
  }

  void apply_gradient(T learning_rate)
    requires(TRAIN_ENABLED)
  {
    if (training_frames)
      throw std::logic_error(
          "Listener must backpropagate before applying gradients");
    first_layer.first.apply_gradient(learning_rate);
    first_layer.second.apply_gradient(learning_rate);
    for (auto &layer : layers) {
      layer.first.apply_gradient(learning_rate);
      layer.second.apply_gradient(learning_rate);
    }
  }
};
