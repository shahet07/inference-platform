// Arbitrary timestamps and arrival orders must not overflow, and must keep the buffer limit.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#include "stream/interval_join.h"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  if (size < 8) {
    return 0;
  }
  std::int64_t gap = 0;
  std::memcpy(&gap, data, sizeof(gap));
  data += 8;
  size -= 8;
  stream::IntervalJoin join({.max_gap_ms = gap, .max_buffered_per_key = 16});
  while (size >= 9) {
    std::int64_t timestamp = 0;
    std::memcpy(&timestamp, data, sizeof(timestamp));
    const auto side = (data[8] & 1U) != 0 ? stream::JoinSide::kLeft : stream::JoinSide::kRight;
    (void)join.add(side, std::string(1, static_cast<char>(data[8] >> 5U)), 1.0, timestamp);
    if (join.bufferedEvents() > 8U * 2U * 16U) {
      std::abort();
    }
    data += 9;
    size -= 9;
  }
  return 0;
}
