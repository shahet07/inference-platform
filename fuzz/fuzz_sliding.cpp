// Arbitrary window sizes and timestamps must not overflow, or spin through a huge time range.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "stream/sliding_window.h"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  if (size < 10) {
    return 0;
  }
  std::int64_t slide = 0;
  std::memcpy(&slide, data, sizeof(slide));
  const std::int64_t panes = 1 + (data[8] % 8);
  std::int64_t window = 0;
  if (__builtin_mul_overflow(slide, panes, &window)) {
    return 0;
  }
  data += 10;
  size -= 10;
  try {
    stream::SlidingWindowAggregator aggregator(
        {.window_ms = window, .slide_ms = slide, .max_keys_per_pane = 8});
    while (size >= 9) {
      std::int64_t timestamp = 0;
      std::memcpy(&timestamp, data, sizeof(timestamp));
      aggregator.add(std::string(1, static_cast<char>(data[8] % 8)), 1.0, timestamp);
      (void)aggregator.takeClosed();
      data += 9;
      size -= 9;
    }
  } catch (const std::invalid_argument&) {
    return 0;
  }
  return 0;
}
