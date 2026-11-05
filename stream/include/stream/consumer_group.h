#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#include "stream/sharded_log.h"

namespace stream {

// Shares a log's shards between the members of a group, so each shard has exactly one member.
// Delivery is at-least-once: when a member joins or leaves, every member resumes from the last
// committed offset, so events polled but not committed are delivered again.
class ConsumerGroup {
 public:
  using MemberId = std::uint64_t;

  explicit ConsumerGroup(ShardedLog& log);
  ~ConsumerGroup();

  ConsumerGroup(const ConsumerGroup&) = delete;
  ConsumerGroup& operator=(const ConsumerGroup&) = delete;
  ConsumerGroup(ConsumerGroup&&) = delete;
  ConsumerGroup& operator=(ConsumerGroup&&) = delete;

  [[nodiscard]] MemberId join();
  void leave(MemberId member);
  [[nodiscard]] std::vector<std::size_t> assignment(MemberId member) const;

  // Returns up to `max_events` per assigned shard, without blocking.
  [[nodiscard]] std::vector<Event> poll(MemberId member, std::size_t max_events);
  // Commits everything this member has polled so far.
  void commit(MemberId member);

  [[nodiscard]] std::uint64_t skipped() const;

 private:
  void rewindToCommitted();
  [[nodiscard]] std::vector<std::size_t> assignmentLocked(MemberId member) const;

  ShardedLog& log_;
  ShardedLog::GroupId group_;
  mutable std::mutex mutex_;
  std::vector<MemberId> members_;  // Sorted, since ids only grow.
  std::vector<std::uint64_t> positions_;
  MemberId next_member_ = 0;
  std::uint64_t generation_ = 0;
  std::uint64_t skipped_ = 0;
};

}  // namespace stream
