// Logged text can come from clients. Escaped text must parse back to the same bytes, unless the
// input was not valid UTF-8, which JSON cannot represent.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>

#include "obs/log.h"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  const std::string_view text(reinterpret_cast<const char*>(data), size);
  const std::string quoted = "\"" + obs::jsonEscape(text) + "\"";
  try {
    if (nlohmann::json::parse(quoted).get<std::string>() != text) {
      std::abort();
    }
  } catch (const nlohmann::json::exception&) {
    for (const char c : text) {
      if (static_cast<unsigned char>(c) >= 0x80) {
        return 0;
      }
    }
    std::abort();  // Pure ASCII must always produce valid JSON.
  }
  return 0;
}
