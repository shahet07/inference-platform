#pragma once

#include <cstdint>
#include <string>

namespace stream {

struct Event {
  std::string key;
  double value = 0.0;
  std::int64_t timestamp_ms = 0;
  // For idempotent appends. Producer 0 means none: the event is always appended.
  std::uint64_t producer_id = 0;
  std::uint64_t sequence = 0;
};

}  // namespace stream
