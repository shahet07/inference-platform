// Recovery reads whatever is on disk after a crash. It must never read out of bounds, and must
// consume only whole, valid records with consecutive offsets.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string_view>

#include "stream/wal.h"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  const std::string_view bytes(reinterpret_cast<const char*>(data), size);
  std::uint64_t expected = 0;
  const std::size_t used = stream::parseRecords(bytes, 0, [&](std::uint64_t offset, stream::Event) {
    if (offset != expected) {
      std::abort();
    }
    ++expected;
  });
  if (used > size) {
    std::abort();
  }
  return 0;
}
