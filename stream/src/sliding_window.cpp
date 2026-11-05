#include "stream/sliding_window.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace stream {
namespace {

constexpr std::int64_t kMin = std::numeric_limits<std::int64_t>::min();
constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();

std::int64_t addSaturating(std::int64_t a, std::int64_t b) {
  std::int64_t out = 0;
  return __builtin_add_overflow(a, b, &out) ? (b > 0 ? kMax : kMin) : out;
}

std::int64_t subSaturating(std::int64_t a, std::int64_t b) {
  std::int64_t out = 0;
  return __builtin_sub_overflow(a, b, &out) ? (b > 0 ? kMin : kMax) : out;
}

std::int64_t floorTo(std::int64_t timestamp, std::int64_t step) {
  std::int64_t remainder = timestamp % step;
  if (remainder < 0) {
    remainder += step;
  }
  return subSaturating(timestamp, remainder);
}

}  // namespace

SlidingWindowAggregator::SlidingWindowAggregator(const SlidingWindowConfig& config)
    : config_(config), max_timestamp_ms_(kMin), next_window_end_(kMin), closed_up_to_(kMin) {
  if (config.slide_ms <= 0 || config.window_ms <= 0 || config.window_ms % config.slide_ms != 0) {
    throw std::invalid_argument("window_ms must be a positive multiple of a positive slide_ms");
  }
}

void SlidingWindowAggregator::add(std::string_view key, double value, std::int64_t timestamp_ms) {
  const std::int64_t pane_start = floorTo(timestamp_ms, config_.slide_ms);
  // The last window holding this pane ends window_ms after it starts. If that has closed, nothing
  // can use this event.
  if (addSaturating(pane_start, config_.window_ms) <= closed_up_to_) {
    ++late_events_;
    return;
  }
  // Windows that already closed cannot reopen, so this event only reaches the ones still to come.
  const std::int64_t first_end =
      std::max(addSaturating(pane_start, config_.slide_ms),
               closed_up_to_ == kMin ? kMin : addSaturating(closed_up_to_, config_.slide_ms));
  next_window_end_ = has_events_ ? std::min(next_window_end_, first_end) : first_end;
  has_events_ = true;
  max_timestamp_ms_ = std::max(max_timestamp_ms_, timestamp_ms);
  Pane& pane = panes_[pane_start];
  const auto it = pane.find(std::string(key));
  if (it == pane.end()) {
    if (pane.size() >= config_.max_keys_per_pane) {
      ++overflow_events_;
      return;
    }
    pane.emplace(std::string(key), Stats{.count = 1, .sum = value, .min = value, .max = value});
    return;
  }
  Stats& stats = it->second;
  ++stats.count;
  stats.sum += value;
  stats.min = std::min(stats.min, value);
  stats.max = std::max(stats.max, value);
}

void SlidingWindowAggregator::advanceTo(std::int64_t timestamp_ms) {
  max_timestamp_ms_ = std::max(max_timestamp_ms_, timestamp_ms);
}

std::vector<WindowResult> SlidingWindowAggregator::takeClosed() {
  std::vector<WindowResult> results;
  if (!has_events_) {
    return results;
  }
  const std::int64_t watermark = subSaturating(max_timestamp_ms_, config_.allowed_lateness_ms);
  while (next_window_end_ <= watermark && next_window_end_ != kMax) {
    const std::int64_t end = next_window_end_;
    const std::int64_t start = subSaturating(end, config_.window_ms);
    std::unordered_map<std::string, Stats> merged;
    for (auto pane = panes_.lower_bound(start); pane != panes_.end() && pane->first < end; ++pane) {
      for (const auto& [key, stats] : pane->second) {
        const auto [entry, inserted] = merged.try_emplace(key, stats);
        if (!inserted) {
          entry->second.count += stats.count;
          entry->second.sum += stats.sum;
          entry->second.min = std::min(entry->second.min, stats.min);
          entry->second.max = std::max(entry->second.max, stats.max);
        }
      }
    }
    for (const auto& [key, stats] : merged) {
      results.push_back({.key = key,
                         .window_start_ms = start,
                         .window_end_ms = end,
                         .count = stats.count,
                         .sum = stats.sum,
                         .min = stats.min,
                         .max = stats.max});
    }
    closed_up_to_ = end;
    next_window_end_ = addSaturating(end, config_.slide_ms);
    // Panes before the next window's start cannot be used again.
    panes_.erase(panes_.begin(), panes_.lower_bound(addSaturating(start, config_.slide_ms)));
    if (panes_.empty()) {
      has_events_ = false;
      break;
    }
    // Skips idle gaps without stepping through every empty window.
    next_window_end_ =
        std::max(next_window_end_, addSaturating(panes_.begin()->first, config_.slide_ms));
  }
  return results;
}

}  // namespace stream
