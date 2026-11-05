#include "net/http_parser.h"

#include <algorithm>
#include <charconv>
#include <system_error>

namespace net {
namespace {

char lower(char c) { return (c >= 'A' && c <= 'Z') ? static_cast<char>(c + ('a' - 'A')) : c; }

bool equalsIgnoringCase(std::string_view a, std::string_view b) {
  return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(),
                                            [](char x, char y) { return lower(x) == lower(y); });
}

// Token characters from RFC 9110. Header names outside this set can be read differently by other
// servers, which is how request smuggling starts.
bool isToken(std::string_view text) {
  return !text.empty() && std::ranges::all_of(text, [](char c) {
    const bool alphanumeric =
        (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    return alphanumeric || std::string_view("!#$%&'*+-.^_`|~").find(c) != std::string_view::npos;
  });
}

std::string_view trim(std::string_view text) {
  while (!text.empty() && (text.front() == ' ' || text.front() == '\t')) {
    text.remove_prefix(1);
  }
  while (!text.empty() && (text.back() == ' ' || text.back() == '\t')) {
    text.remove_suffix(1);
  }
  return text;
}

bool hasControlCharacter(std::string_view text) {
  return std::ranges::any_of(text, [](char c) {
    const auto byte = static_cast<unsigned char>(c);
    return (byte < 0x20 && c != '\t') || byte == 0x7F;
  });
}

}  // namespace

std::string_view HttpRequest::header(std::string_view name) const {
  for (const auto& [key, value] : headers) {
    if (equalsIgnoringCase(key, name)) {
      return value;
    }
  }
  return {};
}

ParseStatus parseRequest(std::string_view buffer, HttpRequest& request, std::size_t& consumed) {
  const std::size_t head_end = buffer.find("\r\n\r\n");
  if (head_end == std::string_view::npos) {
    return buffer.size() > kMaxHeaderBytes ? ParseStatus::kHeaderTooLarge : ParseStatus::kNeedMore;
  }
  if (head_end > kMaxHeaderBytes) {
    return ParseStatus::kHeaderTooLarge;
  }
  std::string_view head = buffer.substr(0, head_end);

  const std::size_t line_end = head.find("\r\n");
  const std::string_view line = head.substr(0, line_end);
  const std::size_t first_space = line.find(' ');
  const std::size_t last_space = line.rfind(' ');
  if (first_space == std::string_view::npos || first_space == last_space) {
    return ParseStatus::kBadRequest;
  }
  request.method = line.substr(0, first_space);
  request.path = line.substr(first_space + 1, last_space - first_space - 1);
  const std::string_view version = line.substr(last_space + 1);
  if (!isToken(request.method) || request.path.empty() || request.path.front() != '/' ||
      hasControlCharacter(request.path) || request.path.find(' ') != std::string_view::npos) {
    return ParseStatus::kBadRequest;
  }
  if (version != "HTTP/1.1" && version != "HTTP/1.0") {
    return ParseStatus::kBadRequest;
  }
  request.keep_alive = version == "HTTP/1.1";

  request.headers.clear();
  std::string_view rest =
      line_end == std::string_view::npos ? std::string_view{} : head.substr(line_end + 2);
  while (!rest.empty()) {
    const std::size_t end = rest.find("\r\n");
    const std::string_view header_line = rest.substr(0, end);
    rest = end == std::string_view::npos ? std::string_view{} : rest.substr(end + 2);
    const std::size_t colon = header_line.find(':');
    if (colon == std::string_view::npos) {
      return ParseStatus::kBadRequest;
    }
    // A space before the colon, or a line folded onto the previous one, fails the token check.
    const std::string_view name = header_line.substr(0, colon);
    const std::string_view value = trim(header_line.substr(colon + 1));
    if (!isToken(name) || hasControlCharacter(value)) {
      return ParseStatus::kBadRequest;
    }
    if (request.headers.size() >= kMaxHeaderCount) {
      return ParseStatus::kHeaderTooLarge;
    }
    request.headers.emplace_back(name, value);
  }

  // Exactly one way of finding the body's end, or the request is rejected.
  std::size_t body_size = 0;
  bool has_length = false;
  for (const auto& [name, value] : request.headers) {
    if (equalsIgnoringCase(name, "transfer-encoding")) {
      return ParseStatus::kNotImplemented;
    }
    if (equalsIgnoringCase(name, "content-length")) {
      if (has_length) {
        return ParseStatus::kBadRequest;
      }
      has_length = true;
      const auto [end, error] =
          std::from_chars(value.data(), value.data() + value.size(), body_size);
      if (value.empty() || error != std::errc{} || end != value.data() + value.size()) {
        return error == std::errc::result_out_of_range ? ParseStatus::kBodyTooLarge
                                                       : ParseStatus::kBadRequest;
      }
    }
    if (equalsIgnoringCase(name, "connection")) {
      if (equalsIgnoringCase(value, "close")) {
        request.keep_alive = false;
      } else if (equalsIgnoringCase(value, "keep-alive")) {
        request.keep_alive = true;
      }
    }
  }
  if (body_size > kMaxBodyBytes) {
    return ParseStatus::kBodyTooLarge;
  }
  const std::size_t body_start = head_end + 4;
  if (buffer.size() - body_start < body_size) {
    return ParseStatus::kNeedMore;
  }
  request.body = buffer.substr(body_start, body_size);
  consumed = body_start + body_size;
  return ParseStatus::kComplete;
}

}  // namespace net
