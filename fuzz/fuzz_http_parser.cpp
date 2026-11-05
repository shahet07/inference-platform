// Clients send anything. Parsing must never read out of bounds, and an accepted request's fields
// must all be views into the input, with a body of exactly the declared size.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string_view>

#include "net/http_parser.h"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  const std::string_view buffer(reinterpret_cast<const char*>(data), size);
  net::HttpRequest request;
  std::size_t consumed = 0;
  if (net::parseRequest(buffer, request, consumed) != net::ParseStatus::kComplete) {
    return 0;
  }
  const auto inside = [&](std::string_view part) {
    return part.data() >= buffer.data() &&
           part.data() + part.size() <= buffer.data() + buffer.size();
  };
  if (consumed > size || !inside(request.method) || !inside(request.path) ||
      (!request.body.empty() &&
       request.body.data() + request.body.size() != buffer.data() + consumed)) {
    std::abort();
  }
  for (const auto& [name, value] : request.headers) {
    if (!inside(name) || name.empty()) {
      std::abort();
    }
  }
  return 0;
}
