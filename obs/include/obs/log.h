#pragma once

#include <atomic>
#include <cstdint>
#include <initializer_list>
#include <mutex>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

namespace obs {

enum class LogLevel : std::uint8_t { kDebug, kInfo, kWarn, kError };

using LogValue = std::variant<std::int64_t, double, bool, std::string>;
using LogField = std::pair<std::string_view, LogValue>;

// Writes one JSON object per line. Thread-safe. Meant for lifecycle events and errors; per-request
// volume belongs in metrics, or in logs only when sampled.
class Logger {
 public:
  Logger(std::ostream& out, std::string component, LogLevel min_level = LogLevel::kInfo)
      : out_(out), component_(std::move(component)), min_level_(min_level) {}

  void log(LogLevel level, std::string_view message, std::initializer_list<LogField> fields = {});

  void info(std::string_view message, std::initializer_list<LogField> fields = {}) {
    log(LogLevel::kInfo, message, fields);
  }
  void warn(std::string_view message, std::initializer_list<LogField> fields = {}) {
    log(LogLevel::kWarn, message, fields);
  }
  void error(std::string_view message, std::initializer_list<LogField> fields = {}) {
    log(LogLevel::kError, message, fields);
  }

 private:
  std::ostream& out_;
  std::string component_;
  LogLevel min_level_;
  std::mutex mutex_;
};

// Allows one event per interval, so a flood of identical errors cannot flood the log. Lock-free.
class LogLimiter {
 public:
  explicit LogLimiter(std::int64_t interval_ms) : interval_ms_(interval_ms) {}

  [[nodiscard]] bool allow();

 private:
  std::int64_t interval_ms_;
  std::atomic<std::int64_t> next_allowed_ms_{0};
};

// Escapes text for use inside a JSON string.
[[nodiscard]] std::string jsonEscape(std::string_view text);

}  // namespace obs
