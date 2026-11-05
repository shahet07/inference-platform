#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>

#include "stream/sharded_log.h"

namespace stream {

// Counts and sums events per key with each event applied exactly once, across crashes.
//
// A replayed event cannot be counted twice because the state and the read position are written
// together, atomically, in one file. After a crash, both come back from the same moment. Combined
// with idempotent producers, a retried or replayed event has no visible effect. This only covers
// state kept in this file: an external system would need the same treatment.
//
// One instance per shard, used from one thread.
class ExactlyOnceAggregator {
 public:
  struct KeyTotals {
    std::uint64_t count = 0;
    double sum = 0.0;
  };

  // Restores the last checkpoint in `dir`, if there is one. `group` is a registered group of `log`.
  ExactlyOnceAggregator(ShardedLog& log, ShardedLog::GroupId group, std::size_t shard,
                        const std::filesystem::path& dir);

  // Applies up to `max_events`. Returns how many, and checkpoints every `checkpoint_every` events.
  std::size_t poll(std::size_t max_events);
  // Atomically saves the state and position, then lets the log release what is behind them.
  void checkpoint();

  [[nodiscard]] const std::map<std::string, KeyTotals>& totals() const { return totals_; }
  [[nodiscard]] std::uint64_t offset() const { return offset_; }
  [[nodiscard]] std::uint64_t skippedEvents() const { return skipped_; }

  // Parses a checkpoint's text. Returns false if it is incomplete or malformed.
  [[nodiscard]] static bool parseCheckpoint(const std::string& text, std::uint64_t& offset,
                                            std::map<std::string, KeyTotals>& totals);

 private:
  ShardedLog& log_;
  ShardedLog::GroupId group_;
  std::size_t shard_;
  std::filesystem::path file_;
  std::uint64_t offset_ = 0;
  std::uint64_t skipped_ = 0;
  std::map<std::string, KeyTotals> totals_;
};

}  // namespace stream
