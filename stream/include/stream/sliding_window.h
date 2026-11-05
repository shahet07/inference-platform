#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "stream/window.h"

namespace stream {

struct SlidingWindowConfig {
  std::int64_t window_ms = 60000;  // Must be a whole multiple of slide_ms.
  std::int64_t slide_ms = 10000;
  std::int64_t allowed_lateness_ms = 0;
  std::size_t max_keys_per_pane = 100000;
};

// Windows of window_ms, starting every slide_ms, so one event falls in several. Events are counted
// once, into a pane of slide_ms, and each window merges its panes when it closes. That makes adding
// an event cost the same however many windows overlap.
//
// Not thread-safe. As with WindowAggregator, each shard's consumer owns one.
class SlidingWindowAggregator {
 public:
  // Throws std::invalid_argument if window_ms is not a positive multiple of a positive slide_ms.
  explicit SlidingWindowAggregator(const SlidingWindowConfig& config);

  void add(std::string_view key, double value, std::int64_t timestamp_ms);
  void advanceTo(std::int64_t timestamp_ms);
  // Returns windows that have closed since the last call, oldest first. Windows with no events are
  // left out.
  [[nodiscard]] std::vector<WindowResult> takeClosed();

  [[nodiscard]] std::uint64_t lateEvents() const { return late_events_; }
  [[nodiscard]] std::uint64_t overflowEvents() const { return overflow_events_; }

 private:
  struct Stats {
    std::uint64_t count = 0;
    double sum = 0.0;
    double min = 0.0;
    double max = 0.0;
  };
  using Pane = std::unordered_map<std::string, Stats>;

  SlidingWindowConfig config_;
  std::map<std::int64_t, Pane> panes_;  // Keyed by pane start.
  std::int64_t max_timestamp_ms_;
  std::int64_t next_window_end_;  // The next window to close, once there is one.
  std::int64_t closed_up_to_;     // The end of the newest window that has closed.
  bool has_events_ = false;
  std::uint64_t late_events_ = 0;
  std::uint64_t overflow_events_ = 0;
};

}  // namespace stream
