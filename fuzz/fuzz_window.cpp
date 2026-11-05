// Timestamps in events are arbitrary 64-bit values. UndefinedBehaviorSanitizer catches overflow.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#include "stream/window.h"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  if (size < 16) {
    return 0;
  }
  std::int64_t window_ms = 0;
  std::int64_t lateness_ms = 0;
  std::memcpy(&window_ms, data, sizeof(window_ms));
  std::memcpy(&lateness_ms, data + 8, sizeof(lateness_ms));
  data += 16;
  size -= 16;
  if (window_ms <= 0) {
    return 0;
  }
  stream::WindowAggregator aggregator(
      {.window_ms = window_ms, .allowed_lateness_ms = lateness_ms, .max_keys_per_window = 8});
  while (size >= 9) {
    std::int64_t timestamp = 0;
    std::memcpy(&timestamp, data, sizeof(timestamp));
    aggregator.add(std::string(1, static_cast<char>(data[8] % 16)), 1.0, timestamp);
    data += 9;
    size -= 9;
    (void)aggregator.takeClosed();
  }
  (void)aggregator.flush();
  return 0;
}
