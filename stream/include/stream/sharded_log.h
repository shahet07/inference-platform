#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace stream {

struct Event {
  std::string key;
  double value = 0.0;
  std::int64_t timestamp_ms = 0;
};

// What append does when a shard is full.
enum class BackpressurePolicy : std::uint8_t {
  kBlock,       // Wait for consumers to commit. Loses nothing, but slows producers.
  kDropNewest,  // Reject the new event. Never slows producers; keeps the history.
  kDropOldest,  // Evict the oldest event. Never slows producers; keeps the latest data.
};

struct LogConfig {
  std::size_t shards = 8;
  std::size_t shard_capacity = 65536;
  BackpressurePolicy policy = BackpressurePolicy::kDropNewest;
};

enum class AppendStatus : std::uint8_t { kAppended, kDropped, kClosed };

struct AppendResult {
  AppendStatus status = AppendStatus::kClosed;
  std::size_t shard = 0;
  std::uint64_t offset = 0;
};

struct ReadResult {
  std::vector<Event> events;
  std::uint64_t next_offset = 0;
  std::uint64_t skipped = 0;  // Events evicted before they could be read.
  bool closed = false;        // The log is closed and this shard has nothing more to read.
};

struct LogStats {
  std::uint64_t appended = 0;
  std::uint64_t dropped_newest = 0;
  std::uint64_t evicted_oldest = 0;
  std::uint64_t blocked_appends = 0;
  std::uint64_t retained = 0;
};

// An in-memory log split into fixed-capacity shards. Events with the same key always go to the
// same shard, in append order. Memory stays bounded however slow the consumers are: each shard
// is a preallocated ring, and a full ring is handled by the configured policy.
//
// Events leave a ring once every registered group has committed past them. With no groups
// registered nothing is ever committed, so kBlock would wait until close().
class ShardedLog {
 public:
  using GroupId = std::size_t;

  explicit ShardedLog(const LogConfig& config);

  [[nodiscard]] AppendResult append(Event event);

  // Copies up to `max_events` starting at `offset`. If fewer than `min_events` are available,
  // waits up to `wait` for that many, then returns what there is. Raising `min_events` lets a
  // consumer that has caught up take batches instead of being woken for every event, at the cost
  // of up to `wait` extra delay. An offset that was already evicted reads from the oldest
  // retained event instead.
  [[nodiscard]] ReadResult read(std::size_t shard, std::uint64_t offset, std::size_t max_events,
                                std::chrono::milliseconds wait = std::chrono::milliseconds(0),
                                std::size_t min_events = 1);

  // A new group starts at each shard's oldest retained event.
  [[nodiscard]] GroupId registerGroup();
  void unregisterGroup(GroupId group);
  // Offsets only move forward; committing an older or out-of-range offset is ignored or clamped.
  void commit(GroupId group, std::size_t shard, std::uint64_t next_offset);
  [[nodiscard]] std::uint64_t committed(GroupId group, std::size_t shard) const;

  // Wakes all blocked callers. Later appends fail, and reads drain what remains.
  void close();

  [[nodiscard]] std::size_t shardCount() const { return shards_.size(); }
  [[nodiscard]] std::size_t shardFor(std::string_view key) const;
  [[nodiscard]] LogStats stats() const;
  [[nodiscard]] std::string renderPrometheus() const;

 private:
  static constexpr std::uint64_t kNoGroup = UINT64_MAX;
  static constexpr std::uint64_t kNoWake = UINT64_MAX;

  // Aligned so that producers on different shards do not share a cache line.
  struct alignas(64) Shard {
    mutable std::mutex mutex;
    std::condition_variable has_events;
    std::condition_variable has_room;
    std::vector<Event> ring;
    std::vector<std::uint64_t> group_offsets;  // kNoGroup marks an unregistered slot.
    std::uint64_t head = 0;                    // Oldest retained offset.
    std::uint64_t tail = 0;                    // Next offset to assign.
    std::uint64_t wake_readers_at = kNoWake;   // Tail at which to signal has_events.
    std::size_t waiting_writers = 0;
    bool closed = false;
    LogStats stats;
  };

  void trim(Shard& shard) const;

  std::size_t capacity_;
  BackpressurePolicy policy_;
  std::vector<std::unique_ptr<Shard>> shards_;
};

}  // namespace stream
