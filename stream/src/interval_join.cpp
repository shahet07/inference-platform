#include "stream/interval_join.h"

#include <algorithm>
#include <limits>

namespace stream {
namespace {

constexpr std::int64_t kMin = std::numeric_limits<std::int64_t>::min();
constexpr std::uint64_t kEvictEvery = 1024;

std::int64_t subSaturating(std::int64_t a, std::int64_t b) {
  std::int64_t out = 0;
  return __builtin_sub_overflow(a, b, &out)
             ? (b > 0 ? kMin : std::numeric_limits<std::int64_t>::max())
             : out;
}

// The distance between two timestamps, which can be too large for a signed value.
std::uint64_t distance(std::int64_t a, std::int64_t b) {
  return a > b ? static_cast<std::uint64_t>(a) - static_cast<std::uint64_t>(b)
               : static_cast<std::uint64_t>(b) - static_cast<std::uint64_t>(a);
}

}  // namespace

std::vector<JoinedPair> IntervalJoin::add(JoinSide side, std::string_view key, double value,
                                          std::int64_t timestamp_ms) {
  std::vector<JoinedPair> pairs;
  const bool is_left = side == JoinSide::kLeft;
  // Everything on the other side this could match may already be evicted.
  const std::int64_t other_watermark = is_left ? right_watermark_ : left_watermark_;
  if (timestamp_ms <
      subSaturating(subSaturating(other_watermark, config_.max_gap_ms), config_.max_gap_ms)) {
    ++late_events_;
    return pairs;
  }

  KeyBuffers& buffers = buffers_[std::string(key)];
  const std::deque<Entry>& others = is_left ? buffers.right : buffers.left;
  const auto gap = static_cast<std::uint64_t>(std::max<std::int64_t>(config_.max_gap_ms, 0));
  for (const Entry& other : others) {
    if (distance(timestamp_ms, other.timestamp_ms) <= gap) {
      JoinedPair pair;
      pair.key = std::string(key);
      pair.left_value = is_left ? value : other.value;
      pair.right_value = is_left ? other.value : value;
      pair.left_timestamp_ms = is_left ? timestamp_ms : other.timestamp_ms;
      pair.right_timestamp_ms = is_left ? other.timestamp_ms : timestamp_ms;
      pairs.push_back(std::move(pair));
    }
  }

  std::deque<Entry>& mine = is_left ? buffers.left : buffers.right;
  mine.push_back({.value = value, .timestamp_ms = timestamp_ms});
  ++buffered_;
  if (mine.size() > config_.max_buffered_per_key) {
    mine.pop_front();
    --buffered_;
    ++dropped_events_;
  }
  (is_left ? left_watermark_ : right_watermark_) =
      std::max(is_left ? left_watermark_ : right_watermark_, timestamp_ms);
  if (++events_since_eviction_ >= kEvictEvery) {
    evict();
  }
  return pairs;
}

void IntervalJoin::evict() {
  events_since_eviction_ = 0;
  // A side's event can go once the other side has moved past any possible match.
  const auto evictSide = [this](std::deque<Entry>& entries, std::int64_t other_watermark) {
    const std::int64_t limit = subSaturating(other_watermark, config_.max_gap_ms);
    const auto before = entries.size();
    std::erase_if(entries, [limit](const Entry& entry) { return entry.timestamp_ms < limit; });
    buffered_ -= before - entries.size();
  };
  for (auto it = buffers_.begin(); it != buffers_.end();) {
    evictSide(it->second.left, right_watermark_);
    evictSide(it->second.right, left_watermark_);
    if (it->second.left.empty() && it->second.right.empty()) {
      it = buffers_.erase(it);
    } else {
      ++it;
    }
  }
}

}  // namespace stream
