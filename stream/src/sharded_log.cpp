#include "stream/sharded_log.h"

#include <algorithm>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace stream {

ShardedLog::ShardedLog(const LogConfig& config)
    : capacity_(config.shard_capacity), policy_(config.policy) {
  if (config.shards == 0 || config.shard_capacity == 0) {
    throw std::invalid_argument("shards and shard_capacity must be positive");
  }
  shards_.reserve(config.shards);
  for (std::size_t i = 0; i < config.shards; ++i) {
    auto shard = std::make_unique<Shard>();
    shard->ring.resize(capacity_);
    if (config.wal) {
      shard->wal = std::make_unique<ShardWal>(*config.wal, i);
      RecoveredShard recovered = shard->wal->recover(capacity_);
      shard->head = recovered.first_offset;
      shard->tail = recovered.next_offset;
      shard->stats.recovered = recovered.events.size();
      shard->stats.truncated_bytes = recovered.truncated_bytes;
      for (std::size_t e = 0; e < recovered.events.size(); ++e) {
        shard->ring[(recovered.first_offset + e) % capacity_] = std::move(recovered.events[e]);
      }
    }
    shards_.push_back(std::move(shard));
  }
}

std::size_t ShardedLog::shardFor(std::string_view key) const {
  return std::hash<std::string_view>{}(key) % shards_.size();
}

AppendResult ShardedLog::append(Event event) { return appendImpl(std::move(event), true); }

AppendResult ShardedLog::appendNoWait(Event event) { return appendImpl(std::move(event), false); }

AppendResult ShardedLog::appendImpl(Event event, bool wait_for_wal) {
  const std::size_t index = shardFor(event.key);
  Shard& shard = *shards_[index];
  std::unique_lock lock(shard.mutex);
  if (shard.closed) {
    return {.status = AppendStatus::kClosed, .shard = index, .offset = 0};
  }
  if (shard.tail - shard.head >= capacity_) {
    switch (policy_) {
      case BackpressurePolicy::kBlock:
        ++shard.stats.blocked_appends;
        ++shard.waiting_writers;
        shard.has_room.wait(lock,
                            [&] { return shard.closed || shard.tail - shard.head < capacity_; });
        --shard.waiting_writers;
        if (shard.closed) {
          return {.status = AppendStatus::kClosed, .shard = index, .offset = 0};
        }
        break;
      case BackpressurePolicy::kDropNewest:
        ++shard.stats.dropped_newest;
        return {.status = AppendStatus::kDropped, .shard = index, .offset = 0};
      case BackpressurePolicy::kDropOldest:
        ++shard.head;
        ++shard.stats.evicted_oldest;
        break;
    }
  }
  const std::uint64_t offset = shard.tail;
  if (shard.wal) {
    shard.wal->enqueue(offset, event);
  }
  shard.ring[offset % capacity_] = std::move(event);
  ++shard.tail;
  ++shard.stats.appended;
  const bool wake_reader = shard.tail >= shard.wake_readers_at;
  if (wake_reader) {
    shard.wake_readers_at = kNoWake;
  }
  lock.unlock();
  if (wake_reader) {
    shard.has_events.notify_all();
  }
  if (wait_for_wal && shard.wal && !shard.wal->waitDurable(offset)) {
    return {.status = AppendStatus::kClosed, .shard = index, .offset = offset};
  }
  return {.status = AppendStatus::kAppended, .shard = index, .offset = offset};
}

ReadResult ShardedLog::read(std::size_t shard_index, std::uint64_t offset, std::size_t max_events,
                            std::chrono::milliseconds wait, std::size_t min_events) {
  Shard& shard = *shards_.at(shard_index);
  std::unique_lock lock(shard.mutex);
  if (offset < shard.head && shard.wal && shard.wal->hasArchive()) {
    // Archive reads can be slow, so they do not hold up producers on this shard.
    ShardWal& wal = *shard.wal;
    lock.unlock();
    ReadResult archived;
    archived.events = wal.readArchived(offset, max_events, archived.next_offset);
    if (!archived.events.empty()) {
      return archived;
    }
    lock.lock();
  }
  if (wait.count() > 0) {
    // Never wait for more than a shard can hold, or for more than was asked for.
    const std::uint64_t target = offset + std::clamp<std::size_t>(min_events, 1, capacity_);
    shard.has_events.wait_for(lock, wait, [&] {
      if (shard.closed || shard.tail >= target) {
        return true;
      }
      // Registered on each check: a wake meant for another reader may have cleared this.
      shard.wake_readers_at = std::min(shard.wake_readers_at, target);
      return false;
    });
  }
  ReadResult result;
  const std::uint64_t start = std::max(offset, shard.head);
  result.skipped = start - offset;
  const std::uint64_t end = std::min(shard.tail, start + max_events);
  if (end > start) {
    result.events.reserve(end - start);
    for (std::uint64_t i = start; i < end; ++i) {
      result.events.push_back(shard.ring[i % capacity_]);
    }
  }
  result.next_offset = std::max(start, end);
  result.closed = shard.closed && result.next_offset >= shard.tail;
  return result;
}

ShardedLog::GroupId ShardedLog::registerGroup() {
  std::vector<std::unique_lock<std::mutex>> locks;
  locks.reserve(shards_.size());
  for (const auto& shard : shards_) {
    locks.emplace_back(shard->mutex);
  }
  const GroupId group = shards_.front()->group_offsets.size();
  for (const auto& shard : shards_) {
    shard->group_offsets.push_back(shard->head);
  }
  const std::scoped_lock groups_lock(groups_mutex_);
  durable_groups_.emplace_back();
  return group;
}

ShardedLog::GroupId ShardedLog::registerDurableGroup(const std::string& name) {
  const GroupId group = registerGroup();
  {
    const std::scoped_lock lock(groups_mutex_);
    durable_groups_[group] = name;
  }
  for (const auto& shard : shards_) {
    const std::scoped_lock lock(shard->mutex);
    if (shard->wal) {
      const std::uint64_t saved = shard->wal->loadGroupOffset(name);
      std::uint64_t& position = shard->group_offsets.at(group);
      position = std::max(position, std::min(saved, shard->tail));
      trim(*shard);
    }
  }
  return group;
}

void ShardedLog::unregisterGroup(GroupId group) {
  for (const auto& shard : shards_) {
    {
      const std::scoped_lock lock(shard->mutex);
      shard->group_offsets.at(group) = kNoGroup;
      trim(*shard);
    }
    shard->has_room.notify_all();
  }
}

void ShardedLog::commit(GroupId group, std::size_t shard_index, std::uint64_t next_offset) {
  Shard& shard = *shards_.at(shard_index);
  bool wake_writers = false;
  std::uint64_t position = 0;
  {
    const std::scoped_lock lock(shard.mutex);
    std::uint64_t& committed_offset = shard.group_offsets.at(group);
    if (committed_offset == kNoGroup) {
      return;
    }
    committed_offset = std::max(committed_offset, std::min(next_offset, shard.tail));
    position = committed_offset;
    trim(shard);
    wake_writers = shard.waiting_writers > 0;
    if (shard.wal) {
      shard.wal->trimBelow(shard.head);
    }
  }
  if (shard.wal) {
    std::string name;
    {
      const std::scoped_lock lock(groups_mutex_);
      name = durable_groups_.at(group);
    }
    if (!name.empty()) {
      shard.wal->saveGroupOffset(name, position);
    }
  }
  if (wake_writers) {
    shard.has_room.notify_all();
  }
}

std::uint64_t ShardedLog::committed(GroupId group, std::size_t shard_index) const {
  const Shard& shard = *shards_.at(shard_index);
  const std::scoped_lock lock(shard.mutex);
  return shard.group_offsets.at(group);
}

void ShardedLog::trim(Shard& shard) const {
  std::uint64_t oldest_needed = shard.tail;
  bool any_group = false;
  for (const std::uint64_t offset : shard.group_offsets) {
    if (offset != kNoGroup) {
      any_group = true;
      oldest_needed = std::min(oldest_needed, offset);
    }
  }
  if (any_group) {
    shard.head = std::max(shard.head, oldest_needed);
  }
}

void ShardedLog::close() {
  for (const auto& shard : shards_) {
    {
      const std::scoped_lock lock(shard->mutex);
      shard->closed = true;
    }
    shard->has_events.notify_all();
    shard->has_room.notify_all();
  }
}

LogStats ShardedLog::stats() const {
  LogStats total;
  for (const auto& shard : shards_) {
    const std::scoped_lock lock(shard->mutex);
    total.appended += shard->stats.appended;
    total.dropped_newest += shard->stats.dropped_newest;
    total.evicted_oldest += shard->stats.evicted_oldest;
    total.blocked_appends += shard->stats.blocked_appends;
    total.retained += shard->tail - shard->head;
    total.recovered += shard->stats.recovered;
    total.truncated_bytes += shard->stats.truncated_bytes;
  }
  return total;
}

std::string ShardedLog::renderPrometheus() const {
  const LogStats s = stats();
  std::ostringstream out;
  const auto metric = [&out](const char* name, const char* type, const char* help,
                             std::uint64_t value) {
    out << "# HELP " << name << ' ' << help << "\n# TYPE " << name << ' ' << type << '\n'
        << name << ' ' << value << '\n';
  };
  metric("stream_appended_total", "counter", "Events appended to the log.", s.appended);
  metric("stream_dropped_newest_total", "counter", "Events rejected: shard full.",
         s.dropped_newest);
  metric("stream_evicted_oldest_total", "counter", "Unread events evicted: shard full.",
         s.evicted_oldest);
  metric("stream_blocked_appends_total", "counter", "Appends that waited for room.",
         s.blocked_appends);
  metric("stream_retained_events", "gauge", "Events held across all shards.", s.retained);
  metric("stream_recovered_events", "gauge", "Events restored from disk at startup.", s.recovered);
  metric("stream_truncated_bytes", "gauge", "Torn bytes removed at startup.", s.truncated_bytes);
  return out.str();
}

}  // namespace stream
