#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace stream {

struct WindowResult {
  std::string key;
  std::int64_t window_start_ms = 0;
  std::int64_t window_end_ms = 0;
  std::uint64_t count = 0;
  double sum = 0.0;
  double min = 0.0;
  double max = 0.0;
};

struct WindowConfig {
  std::int64_t window_ms = 1000;
  std::int64_t allowed_lateness_ms = 0;
  // Keys tracked per window. Events for further keys are dropped and counted, so a flood of
  // unique keys cannot exhaust memory.
  std::size_t max_keys_per_window = 100000;
};

// Tumbling-window aggregation per key. A window closes once the watermark, the newest timestamp
// seen minus the allowed lateness, reaches its end. Later events for it are dropped and counted.
//
// Not thread-safe. Each shard's consumer owns one, which is safe because a key maps to one shard.
class WindowAggregator {
 public:
  explicit WindowAggregator(const WindowConfig& config);

  void add(std::string_view key, double value, std::int64_t timestamp_ms);
  // Moves the watermark without an event, so an idle shard still closes its windows.
  void advanceTo(std::int64_t timestamp_ms);
  // Returns and forgets windows that have closed since the last call, oldest first.
  [[nodiscard]] std::vector<WindowResult> takeClosed();
  // Closes everything still open, for shutdown.
  [[nodiscard]] std::vector<WindowResult> flush();

  [[nodiscard]] std::uint64_t lateEvents() const { return late_events_; }
  [[nodiscard]] std::uint64_t overflowEvents() const { return overflow_events_; }

 private:
  struct Stats {
    std::uint64_t count = 0;
    double sum = 0.0;
    double min = 0.0;
    double max = 0.0;
  };
  using KeyStats = std::unordered_map<std::string, Stats>;

  std::vector<WindowResult> takeUpTo(std::int64_t watermark);

  WindowConfig config_;
  std::map<std::int64_t, KeyStats> open_;  // Keyed by window start.
  std::int64_t max_timestamp_ms_;
  std::int64_t closed_up_to_ms_;
  std::uint64_t late_events_ = 0;
  std::uint64_t overflow_events_ = 0;
};

}  // namespace stream
