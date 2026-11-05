#include "stream/consumer_group.h"

#include <algorithm>
#include <iterator>
#include <utility>

namespace stream {

ConsumerGroup::ConsumerGroup(ShardedLog& log)
    : log_(log), group_(log.registerGroup()), positions_(log.shardCount()) {
  rewindToCommitted();
}

ConsumerGroup::~ConsumerGroup() { log_.unregisterGroup(group_); }

ConsumerGroup::MemberId ConsumerGroup::join() {
  const std::scoped_lock lock(mutex_);
  const MemberId member = next_member_++;
  members_.push_back(member);
  ++generation_;
  rewindToCommitted();
  return member;
}

void ConsumerGroup::leave(MemberId member) {
  const std::scoped_lock lock(mutex_);
  std::erase(members_, member);
  ++generation_;
  rewindToCommitted();
}

std::vector<std::size_t> ConsumerGroup::assignment(MemberId member) const {
  const std::scoped_lock lock(mutex_);
  return assignmentLocked(member);
}

std::vector<std::size_t> ConsumerGroup::assignmentLocked(MemberId member) const {
  std::vector<std::size_t> shards;
  const auto it = std::ranges::find(members_, member);
  if (it == members_.end()) {
    return shards;
  }
  const auto index = static_cast<std::size_t>(it - members_.begin());
  for (std::size_t shard = index; shard < positions_.size(); shard += members_.size()) {
    shards.push_back(shard);
  }
  return shards;
}

std::vector<Event> ConsumerGroup::poll(MemberId member, std::size_t max_events) {
  // Each shard has one owner, but a rebalance can move it, so reads are checked against the
  // generation they started from. The log's lock, not this one, protects the events.
  std::vector<std::size_t> shards;
  std::vector<std::uint64_t> starts;
  std::uint64_t generation = 0;
  {
    const std::scoped_lock lock(mutex_);
    shards = assignmentLocked(member);
    starts.reserve(shards.size());
    for (const std::size_t shard : shards) {
      starts.push_back(positions_[shard]);
    }
    generation = generation_;
  }

  std::vector<ReadResult> reads;
  reads.reserve(shards.size());
  for (std::size_t i = 0; i < shards.size(); ++i) {
    reads.push_back(log_.read(shards[i], starts[i], max_events));
  }

  const std::scoped_lock lock(mutex_);
  std::vector<Event> events;
  if (generation != generation_) {
    return events;  // A rebalance happened; these events will be delivered again.
  }
  for (std::size_t i = 0; i < shards.size(); ++i) {
    positions_[shards[i]] = reads[i].next_offset;
    skipped_ += reads[i].skipped;
    std::ranges::move(reads[i].events, std::back_inserter(events));
  }
  return events;
}

void ConsumerGroup::commit(MemberId member) {
  const std::scoped_lock lock(mutex_);
  for (const std::size_t shard : assignmentLocked(member)) {
    log_.commit(group_, shard, positions_[shard]);
  }
}

std::uint64_t ConsumerGroup::skipped() const {
  const std::scoped_lock lock(mutex_);
  return skipped_;
}

void ConsumerGroup::rewindToCommitted() {
  for (std::size_t shard = 0; shard < positions_.size(); ++shard) {
    positions_[shard] = log_.committed(group_, shard);
  }
}

}  // namespace stream
