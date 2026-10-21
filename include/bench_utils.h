#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

struct Shape { int M, N, K; };

inline std::vector<Shape> default_shapes() {
  return {{256,256,256}, {1024,1024,1024}, {2048,2048,2048},
          {4096,4096,2048}, {4096,1024,4096}, {1024,4096,4096}, {4096,4096,4096}};
}

inline int parse_integer(const std::string& text, int minimum = 1) {
  if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
    throw std::invalid_argument("invalid integer: " + text);
  const auto value = std::stoull(text);
  if (value < static_cast<unsigned long long>(minimum) || value > std::numeric_limits<int>::max())
    throw std::invalid_argument("integer outside supported range: " + text);
  return static_cast<int>(value);
}

inline Shape parse_shape(std::string text) {
  std::replace(text.begin(), text.end(), 'x', ',');
  std::replace(text.begin(), text.end(), 'X', ',');
  const auto first = text.find(','), last = text.rfind(',');
  if (first == std::string::npos || first == last || text.find(',', first + 1) != last)
    throw std::invalid_argument("shape must be M,N,K or MxNxK: " + text);
  return {parse_integer(text.substr(0, first)),
          parse_integer(text.substr(first + 1, last - first - 1)),
          parse_integer(text.substr(last + 1))};
}

inline void check_shape(Shape s, int BM, int BN, int BK) {
  if (s.M <= 0 || s.N <= 0 || s.K <= 0 || s.M % BM || s.N % BN || s.K % BK)
    throw std::invalid_argument("positive M,N,K must be divisible by " +
        std::to_string(BM) + "," + std::to_string(BN) + "," + std::to_string(BK));
  if (s.M / BM > 65535) throw std::invalid_argument("grid.y exceeds 65535");
}

template<typename T>
inline std::size_t matrix_bytes(int rows, int cols) {
  if (rows <= 0 || cols <= 0 ||
      static_cast<std::size_t>(rows) > std::numeric_limits<std::size_t>::max() / sizeof(T) / cols)
    throw std::invalid_argument("matrix size is invalid or overflows size_t");
  return static_cast<std::size_t>(rows) * cols * sizeof(T);
}

struct Stats { double mean, median, minimum, maximum; };
inline Stats summarize(std::vector<double> values) {
  if (values.empty()) throw std::invalid_argument("no timing samples");
  for (double v : values)
    if (!std::isfinite(v) || v <= 0) throw std::runtime_error("invalid CUDA timing sample");
  std::sort(values.begin(), values.end());
  const auto n = values.size();
  return {std::accumulate(values.begin(), values.end(), 0.0) / n,
          (values[(n - 1) / 2] + values[n / 2]) / 2, values.front(), values.back()};
}

struct Accuracy { std::size_t failures = 0; double max_abs_error = 0; };
inline Accuracy compare(const std::vector<float>& actual, const std::vector<double>& expected) {
  if (actual.size() != expected.size()) throw std::invalid_argument("output size mismatch");
  Accuracy result;
  for (std::size_t i = 0; i < actual.size(); i++) {
    const double error = std::abs(static_cast<double>(actual[i]) - expected[i]);
    if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
      ++result.failures;
      result.max_abs_error = std::numeric_limits<double>::infinity();
    } else {
      result.max_abs_error = std::max(result.max_abs_error, error);
      if (error > 1e-3 + 1e-3 * std::abs(expected[i])) ++result.failures;
    }
  }
  return result;
}

// A is physically [K,M], B/C are [K,N]/[M,N]. Accumulate in FP64 on the CPU.
inline double reference_element(const std::vector<float>& a, const std::vector<float>& b,
                                Shape s, int row, int col) {
  double sum = 0;
  for (int k = 0; k < s.K; k++)
    sum += static_cast<double>(a[static_cast<std::size_t>(k) * s.M + row]) *
           static_cast<double>(b[static_cast<std::size_t>(k) * s.N + col]);
  return sum;
}
