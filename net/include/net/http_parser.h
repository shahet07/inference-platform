#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

namespace net {

inline constexpr std::size_t kMaxHeaderBytes = 16 * 1024;
inline constexpr std::size_t kMaxHeaderCount = 100;
inline constexpr std::size_t kMaxBodyBytes = std::size_t{64} * 1024 * 1024;

// Views into the buffer that was parsed, so they last only as long as it does.
struct HttpRequest {
  std::string_view method;
  std::string_view path;
  std::string_view body;
  std::vector<std::pair<std::string_view, std::string_view>> headers;
  bool keep_alive = true;

  // Looks up a header, ignoring case. Empty if it is not there.
  [[nodiscard]] std::string_view header(std::string_view name) const;
};

enum class ParseStatus : std::uint8_t {
  kNeedMore,    // Valid so far, but incomplete.
  kComplete,    // `consumed` bytes hold one whole request.
  kBadRequest,  // Malformed. The connection must close, since its framing can't be trusted.
  kHeaderTooLarge,
  kBodyTooLarge,
  kNotImplemented,  // Valid HTTP this server does not handle, such as chunked uploads.
};

// Parses one HTTP/1.1 request from the start of `buffer`. Requests come from untrusted clients, so
// this is strict: ambiguous framing is rejected, because if this server and a proxy in front of it
// disagree on where a request ends, an attacker can smuggle a second request past the proxy.
[[nodiscard]] ParseStatus parseRequest(std::string_view buffer, HttpRequest& request,
                                       std::size_t& consumed);

}  // namespace net
