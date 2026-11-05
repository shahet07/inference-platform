#include "obs/log.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <sstream>

namespace obs {
namespace {

const char* levelName(LogLevel level) {
  switch (level) {
    case LogLevel::kDebug:
      return "debug";
    case LogLevel::kInfo:
      return "info";
    case LogLevel::kWarn:
      return "warn";
    case LogLevel::kError:
      return "error";
  }
  return "unknown";
}

std::int64_t nowMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

void writeValue(std::ostream& out, const LogValue& value) {
  if (const auto* text = std::get_if<std::string>(&value)) {
    out << '"' << jsonEscape(*text) << '"';
  } else if (const auto* boolean = std::get_if<bool>(&value)) {
    out << (*boolean ? "true" : "false");
  } else if (const auto* integer = std::get_if<std::int64_t>(&value)) {
    out << *integer;
  } else if (const auto* number = std::get_if<double>(&value)) {
    // JSON has no NaN or infinity.
    if (std::isfinite(*number)) {
      out << *number;
    } else {
      out << "null";
    }
  }
}

}  // namespace

std::string jsonEscape(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  for (const char c : text) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          std::array<char, 7> buffer{};
          std::snprintf(buffer.data(), buffer.size(), "\\u%04x",
                        static_cast<unsigned>(static_cast<unsigned char>(c)));
          out += buffer.data();
        } else {
          out += c;
        }
    }
  }
  return out;
}

void Logger::log(LogLevel level, std::string_view message, std::initializer_list<LogField> fields) {
  if (level < min_level_) {
    return;
  }
  std::ostringstream line;
  line << R"({"ts_ms":)" << nowMs() << R"(,"level":")" << levelName(level) << R"(","component":")"
       << jsonEscape(component_) << R"(","msg":")" << jsonEscape(message) << '"';
  for (const auto& [key, value] : fields) {
    line << ",\"" << jsonEscape(key) << "\":";
    writeValue(line, value);
  }
  line << "}\n";
  const std::string text = line.str();
  const std::scoped_lock lock(mutex_);
  out_ << text << std::flush;
}

bool LogLimiter::allow() {
  const std::int64_t now = nowMs();
  std::int64_t next = next_allowed_ms_.load(std::memory_order_relaxed);
  return now >= next && next_allowed_ms_.compare_exchange_strong(next, now + interval_ms_,
                                                                 std::memory_order_relaxed);
}

}  // namespace obs
