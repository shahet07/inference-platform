// Clients control this header. Anything accepted must survive a round trip.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string_view>

#include "obs/trace.h"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  const std::string_view text(reinterpret_cast<const char*>(data), size);
  const auto parsed = obs::parseTraceparent(text);
  if (parsed) {
    const auto again = obs::parseTraceparent(obs::formatTraceparent(*parsed));
    if (!again || again->trace_id != parsed->trace_id || again->span_id != parsed->span_id ||
        again->sampled != parsed->sampled) {
      std::abort();
    }
  }
  return 0;
}
