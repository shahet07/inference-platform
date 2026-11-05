#include "stream/window.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace stream {
namespace {

// Rounds down, including for timestamps before the epoch.
std::int64_t windowStart(std::int64_t timestamp_ms, std::int64_t window_ms) {
  std::int64_t remainder = timestamp_ms % window_ms;
  if (remainder < 0) {
    remainder += window_ms;
  }
  return timestamp_ms - remainder;
}

}  // namespace

WindowAggregator::WindowAggregator(const WindowConfig& config)
    : config_(config),
      max_timestamp_ms_(std::numeric_limits<std::int64_t>::min()),
      closed_up_to_ms_(std::numeric_limits<std::int64_t>::min()) {
  if (config.window_ms <= 0) {
    throw std::invalid_argument("window_ms must be positive");
  }
}

void WindowAggregator::add(std::string_view key, double value, std::int64_t timestamp_ms) {
  const std::int64_t start = windowStart(timestamp_ms, config_.window_ms);
  if (start + config_.window_ms <= closed_up_to_ms_) {
    ++late_events_;
    return;
  }
  max_timestamp_ms_ = std::max(max_timestamp_ms_, timestamp_ms);
  KeyStats& keys = open_[start];
  // TODO(C++26): heterogeneous lookup would avoid building this string for existing keys.
  auto it = keys.find(std::string(key));
  if (it == keys.end()) {
    if (keys.size() >= config_.max_keys_per_window) {
      ++overflow_events_;
      return;
    }
    keys.emplace(std::string(key), Stats{.count = 1, .sum = value, .min = value, .max = value});
    return;
  }
  Stats& stats = it->second;
  ++stats.count;
  stats.sum += value;
  stats.min = std::min(stats.min, value);
  stats.max = std::max(stats.max, value);
}

void WindowAggregator::advanceTo(std::int64_t timestamp_ms) {
  max_timestamp_ms_ = std::max(max_timestamp_ms_, timestamp_ms);
}

std::vector<WindowResult> WindowAggregator::takeClosed() {
  if (max_timestamp_ms_ == std::numeric_limits<std::int64_t>::min()) {
    return {};
  }
  const std::int64_t watermark = max_timestamp_ms_ - config_.allowed_lateness_ms;
  closed_up_to_ms_ = std::max(closed_up_to_ms_, watermark);
  return takeUpTo(watermark);
}

std::vector<WindowResult> WindowAggregator::flush() {
  return takeUpTo(std::numeric_limits<std::int64_t>::max());
}

std::vector<WindowResult> WindowAggregator::takeUpTo(std::int64_t watermark) {
  std::vector<WindowResult> results;
  auto it = open_.begin();
  while (it != open_.end() && (watermark == std::numeric_limits<std::int64_t>::max() ||
                               it->first + config_.window_ms <= watermark)) {
    for (auto& [key, stats] : it->second) {
      results.push_back({.key = key,
                         .window_start_ms = it->first,
                         .window_end_ms = it->first + config_.window_ms,
                         .count = stats.count,
                         .sum = stats.sum,
                         .min = stats.min,
                         .max = stats.max});
    }
    it = open_.erase(it);
  }
  return results;
}

}  // namespace stream
