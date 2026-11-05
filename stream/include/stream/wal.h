#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "stream/event.h"

namespace stream {

// When appended events reach stable storage. Each step trades throughput for what survives.
enum class FsyncPolicy : std::uint8_t {
  kEveryAppend,  // Append returns only after a sync. Concurrent appends share one (group commit).
  kInterval,     // Sync every flush_interval. A power failure can lose that much acknowledged data.
  kNever,        // Leave it to the OS. Survives a process crash, but not a power failure.
};

// Stores sealed segments outside the local log directory, so that old events can leave local disk
// and still be read. Implementations must make `put` atomic: a segment is fully there or not at
// all.
class SegmentStore {
 public:
  SegmentStore() = default;
  virtual ~SegmentStore() = default;
  SegmentStore(const SegmentStore&) = delete;
  SegmentStore& operator=(const SegmentStore&) = delete;
  SegmentStore(SegmentStore&&) = delete;
  SegmentStore& operator=(SegmentStore&&) = delete;

  virtual void put(const std::filesystem::path& local_file, const std::string& name) = 0;
  // Names in ascending order.
  [[nodiscard]] virtual std::vector<std::string> list(const std::string& prefix) const = 0;
  [[nodiscard]] virtual std::string get(const std::string& name) const = 0;
};

// A SegmentStore on a directory. Object storage such as S3 would implement the same interface.
class DirectorySegmentStore final : public SegmentStore {
 public:
  explicit DirectorySegmentStore(std::filesystem::path dir);

  void put(const std::filesystem::path& local_file, const std::string& name) override;
  [[nodiscard]] std::vector<std::string> list(const std::string& prefix) const override;
  [[nodiscard]] std::string get(const std::string& name) const override;

 private:
  std::filesystem::path dir_;
};

struct WalConfig {
  std::filesystem::path dir;
  FsyncPolicy fsync = FsyncPolicy::kEveryAppend;
  std::chrono::milliseconds flush_interval{10};
  std::size_t segment_bytes = std::size_t{64} * 1024 * 1024;
  // Optional. Sealed segments that consumers have passed move here instead of being deleted.
  std::shared_ptr<SegmentStore> archive;
};

struct RecoveredShard {
  std::vector<Event> events;  // The newest events on local disk, up to the requested maximum.
  std::uint64_t first_offset = 0;
  std::uint64_t next_offset = 0;
  std::uint64_t truncated_bytes = 0;  // Torn data removed from the end of the log.
};

// Parses as many valid records as `data` holds, starting at `expected_offset`. Returns the number
// of bytes they used. Stops at the first torn or corrupt record, and never reads out of bounds.
// `on_event` receives each record's offset and event.
std::size_t parseRecords(std::string_view data, std::uint64_t expected_offset,
                         const std::function<void(std::uint64_t, Event)>& on_event);

// The write-ahead log for one shard: a directory of numbered segment files.
//
// Record format, little-endian: u32 payload size, u32 CRC-32 of the payload, then a payload of
// u64 offset, u32 key size, key bytes, f64 value, i64 timestamp.
class ShardWal {
 public:
  ShardWal(const WalConfig& config, std::size_t shard);
  // Writes and syncs what is buffered.
  ~ShardWal();

  ShardWal(const ShardWal&) = delete;
  ShardWal& operator=(const ShardWal&) = delete;
  ShardWal(ShardWal&&) = delete;
  ShardWal& operator=(ShardWal&&) = delete;

  // Reads the log, removes a torn tail, and returns what survived. Call once, before anything else.
  // Throws std::runtime_error on corruption before the last segment, since that is not a torn
  // write.
  [[nodiscard]] RecoveredShard recover(std::size_t max_events);

  // Buffers one record. Offsets must be consecutive, so callers hold their shard lock.
  void enqueue(std::uint64_t offset, const Event& event);
  // Blocks until the record at `offset` has met the fsync policy. Returns false if the log stopped.
  bool waitDurable(std::uint64_t offset);

  // Consumers have passed `offset`, so sealed segments entirely below it can leave local disk.
  void trimBelow(std::uint64_t offset);

  // Reads events below the local log's start from the archive. Returns the bytes' next offset.
  [[nodiscard]] std::vector<Event> readArchived(std::uint64_t offset, std::size_t max_events,
                                                std::uint64_t& next_offset) const;
  [[nodiscard]] bool hasArchive() const { return config_.archive != nullptr; }

  // Durably records a consumer group's position, by writing a new file and renaming it into place.
  void saveGroupOffset(const std::string& group, std::uint64_t offset);
  [[nodiscard]] std::uint64_t loadGroupOffset(const std::string& group) const;

 private:
  struct Segment {
    std::uint64_t first_offset = 0;
    std::filesystem::path path;
  };

  void start();
  void flushLoop();
  void intervalSyncLoop();
  void writeOut(const std::string& bytes, std::uint64_t next_offset);
  void openSegment(std::uint64_t first_offset);
  void sealCurrent();
  void syncCurrent();
  void applyTrim();
  [[nodiscard]] std::string archiveName(std::uint64_t first_offset) const;

  WalConfig config_;
  std::size_t shard_;
  std::filesystem::path shard_dir_;

  std::mutex mutex_;
  std::condition_variable has_data_;
  std::condition_variable durable_;
  std::string buffer_;
  std::uint64_t buffered_up_to_ = 0;  // Next offset after what is buffered.
  std::uint64_t durable_up_to_ = 0;   // Next offset after what meets the policy.
  std::uint64_t trim_below_ = 0;
  bool stopping_ = false;
  bool failed_ = false;

  // Lets the interval sync thread find the open file while the flush thread may be replacing it.
  mutable std::mutex fd_mutex_;
  int fd_ = -1;
  // Used only by the flush thread, once started.
  std::size_t segment_size_ = 0;
  std::uint64_t current_first_ = 0;
  std::vector<Segment> sealed_;
  std::thread flusher_;
  std::thread interval_syncer_;
};

}  // namespace stream
