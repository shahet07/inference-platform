// A node reads this file after a crash, so it must cope with any bytes.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string_view>
#include <vector>

#include "consensus/file_storage.h"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  const std::string_view bytes(reinterpret_cast<const char*>(data), size);
  std::vector<consensus::LogEntry> entries;
  std::vector<std::uint64_t> ends;
  const std::size_t used = consensus::FileStorage::parseLog(bytes, entries, ends);
  if (used > size || entries.size() != ends.size() || (!ends.empty() && ends.back() != used)) {
    std::abort();
  }
  return 0;
}
