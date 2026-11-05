#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

namespace security {

// The SHA-256 digest of `data`, as 64 lowercase hex digits.
[[nodiscard]] std::string sha256Hex(std::string_view data);

enum class Role : std::uint8_t {
  kClient,  // Can run inference and read status.
  kAdmin,   // Can also change what is served: models and backends.
};

enum class AuthResult : std::uint8_t {
  kAllowed,
  kUnauthenticated,  // 401: no key, or not a known key.
  kForbidden,        // 403: a known key without the role this needs.
  kRateLimited,      // 429: a known key over its limit.
};

// A token bucket: tokens refill at `rate` per second up to `burst`, and each request spends one.
// Allows short bursts while holding the long-run average to `rate`. Thread-safe.
class TokenBucket {
 public:
  using Clock = std::chrono::steady_clock;

  TokenBucket(double rate, double burst, Clock::time_point now);

  [[nodiscard]] bool tryTake(Clock::time_point now);

 private:
  double rate_;
  double burst_;
  double tokens_;
  Clock::time_point last_;
  std::mutex mutex_;
};

// Who may call the API, loaded from a file with one key per line:
//   <sha256 of the key> <name> <client|admin> <requests per second> <burst>
// A rate of 0 means no limit. Blank lines and lines starting with # are ignored. Only hashes are
// stored, so reading the file does not reveal the keys. tools/make_api_key.py creates entries.
class ApiKeys {
 public:
  // Throws std::runtime_error, naming the line but not its contents, if the file is malformed.
  explicit ApiKeys(const std::filesystem::path& file);
  // For tests: `text` holds the file's contents, and `clock` replaces the real one.
  ApiKeys(std::string_view text, std::function<TokenBucket::Clock::time_point()> clock);

  // `authorization` is the Authorization header's value, expected to be "Bearer <key>".
  [[nodiscard]] AuthResult check(std::string_view authorization, Role needed);

  [[nodiscard]] std::size_t size() const { return keys_.size(); }

 private:
  struct Entry {
    std::string name;
    Role role = Role::kClient;
    std::unique_ptr<TokenBucket> bucket;  // Null when there is no limit.
  };

  void parse(std::string_view text);

  std::function<TokenBucket::Clock::time_point()> clock_;
  std::map<std::string, Entry> keys_;  // Keyed by hash.
};

}  // namespace security
