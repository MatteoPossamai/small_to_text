#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace attention_detail {

template <std::floating_point T, std::size_t N>
T dot(const std::array<T, N> &left, const std::array<T, N> &right) {
  T sum{};
#pragma clang loop vectorize(enable)
#pragma clang loop interleave(enable)
  for (std::size_t i = 0; i < N; ++i)
    sum += left[i] * right[i];
  return sum;
}

template <std::floating_point T, std::size_t N>
std::array<T, N> attend(const std::array<T, N> &query,
                        std::span<const std::array<T, N>> rows,
                        std::span<T> weights) {
  if (rows.empty())
    throw std::invalid_argument("Attention requires at least one listener row");

  for (std::size_t i = 0; i < rows.size(); ++i)
    weights[i] = dot(query, rows[i]);

  // Softmax over the scores, subtracting the maximum to keep exp finite.
  const T max_score = *std::max_element(weights.begin(), weights.end());
  T total{};
  for (T &weight : weights) {
    weight = std::exp(weight - max_score);
    total += weight;
  }

#pragma clang loop vectorize(enable)
#pragma clang loop interleave(enable)
  for (T &weight : weights)
    weight /= total;

  std::array<T, N> context{};
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const T scale = weights[i];
#pragma clang loop vectorize(enable)
#pragma clang loop interleave(enable)
    for (std::size_t j = 0; j < N; ++j)
      context[j] += scale * rows[i][j];
  }
  return context;
}

} // namespace attention_detail

template <std::floating_point T, std::size_t N>
std::array<T, N> attend(const std::array<T, N> &query,
                        const std::vector<std::array<T, N>> &rows,
                        std::vector<T> &weights) {
  weights.resize(rows.size());
  return attention_detail::attend<T, N>(query, rows, weights);
}

template <std::floating_point T, std::size_t N>
std::array<T, N> attend(const std::array<T, N> &query,
                        const std::vector<std::array<T, N>> &rows) {
  std::vector<T> weights;
  return attend(query, rows, weights);
}

template <std::floating_point T, std::size_t N, std::size_t M>
std::array<T, N> attend(const std::array<T, N> &query,
                        const std::array<std::array<T, N>, M> &rows) {
  std::array<T, M> weights;
  return attention_detail::attend<T, N>(query, rows, weights);
}

template <std::floating_point T, std::size_t N>
std::array<T, N> attend_backward(const std::array<T, N> &query,
                                 const std::vector<std::array<T, N>> &rows,
                                 const std::vector<T> &weights,
                                 const std::array<T, N> &context_gradient,
                                 std::vector<std::array<T, N>> &row_gradient) {
  if (rows.empty() || rows.size() != weights.size() ||
      rows.size() != row_gradient.size())
    throw std::invalid_argument(
        "Attention rows and gradients have incompatible sizes");

  // A row's weight scales its contribution to the context.
  std::vector<T> weight_gradient(rows.size());
  T mean_weight_gradient{};
  for (std::size_t i = 0; i < rows.size(); ++i) {
    weight_gradient[i] = attention_detail::dot(rows[i], context_gradient);
    mean_weight_gradient += weights[i] * weight_gradient[i];
  }

  std::array<T, N> query_gradient{};
  for (std::size_t i = 0; i < rows.size(); ++i) {
    // Softmax scores affect every normalized weight.
    const T score_gradient =
        weights[i] * (weight_gradient[i] - mean_weight_gradient);
    for (std::size_t j = 0; j < N; ++j) {
      query_gradient[j] += score_gradient * rows[i][j];
      // A row contributes both to its score and directly to the context.
      row_gradient[i][j] +=
          score_gradient * query[j] + weights[i] * context_gradient[j];
    }
  }
  return query_gradient;
}
