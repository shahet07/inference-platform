#pragma once

#include <cstdint>
#include <string>

namespace stream {

struct Event {
  std::string key;
  double value = 0.0;
  std::int64_t timestamp_ms = 0;
};

}  // namespace stream
