#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace stream {

enum class JoinSide : std::uint8_t { kLeft, kRight };

struct JoinConfig {
  // Events match when they share a key and their timestamps are at most this far apart.
  std::int64_t max_gap_ms = 1000;
  // Buffered events per key and side. Beyond this the oldest is dropped and counted, so one very
  // busy key cannot exhaust memory.
  std::size_t max_buffered_per_key = 10000;
};

struct JoinedPair {
  std::string key;
  double left_value = 0.0;
  double right_value = 0.0;
  std::int64_t left_timestamp_ms = 0;
  std::int64_t right_timestamp_ms = 0;
};

// Joins two event streams by key and event time, in the order events arrive, so either side may run
// ahead. Each matching pair is produced exactly once. An event is kept until both sides' watermarks
// show no future match is possible, so a stalled side holds events back, up to the per-key limit.
//
// Not thread-safe. Both streams must be partitioned the same way, so a key always reaches one
// joiner.
class IntervalJoin {
 public:
  explicit IntervalJoin(const JoinConfig& config) : config_(config) {}

  // Returns the pairs this event completes.
  [[nodiscard]] std::vector<JoinedPair> add(JoinSide side, std::string_view key, double value,
                                            std::int64_t timestamp_ms);

  [[nodiscard]] std::uint64_t lateEvents() const { return late_events_; }
  [[nodiscard]] std::uint64_t droppedEvents() const { return dropped_events_; }
  [[nodiscard]] std::size_t bufferedEvents() const { return buffered_; }

 private:
  struct Entry {
    double value = 0.0;
    std::int64_t timestamp_ms = 0;
  };
  struct KeyBuffers {
    std::deque<Entry> left;
    std::deque<Entry> right;
  };

  void evict();

  JoinConfig config_;
  std::unordered_map<std::string, KeyBuffers> buffers_;
  std::int64_t left_watermark_ = std::numeric_limits<std::int64_t>::min();
  std::int64_t right_watermark_ = std::numeric_limits<std::int64_t>::min();
  std::uint64_t events_since_eviction_ = 0;
  std::size_t buffered_ = 0;
  std::uint64_t late_events_ = 0;
  std::uint64_t dropped_events_ = 0;
};

}  // namespace stream
