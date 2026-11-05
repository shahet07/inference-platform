#include "security/api_keys.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace security {
namespace {

constexpr std::string_view kBearer = "Bearer ";

// Takes the same time wherever the strings differ, so response timing cannot be used to guess a
// value one character at a time.
bool constantTimeEquals(std::string_view a, std::string_view b) {
  if (a.size() != b.size()) {
    return false;
  }
  unsigned int difference = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    difference |= static_cast<unsigned int>(static_cast<unsigned char>(a[i])) ^
                  static_cast<unsigned int>(static_cast<unsigned char>(b[i]));
  }
  return difference == 0U;
}

bool isLowerHex(std::string_view text) {
  return std::ranges::all_of(
      text, [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

}  // namespace

TokenBucket::TokenBucket(double rate, double burst, Clock::time_point now)
    : rate_(rate), burst_(burst), tokens_(burst), last_(now) {}

bool TokenBucket::tryTake(Clock::time_point now) {
  const std::scoped_lock lock(mutex_);
  if (now > last_) {
    tokens_ =
        std::min(burst_, tokens_ + (std::chrono::duration<double>(now - last_).count() * rate_));
    last_ = now;
  }
  if (tokens_ < 1.0) {
    return false;
  }
  tokens_ -= 1.0;
  return true;
}

ApiKeys::ApiKeys(const std::filesystem::path& file)
    : clock_([] { return TokenBucket::Clock::now(); }) {
  std::ifstream in(file);
  if (!in) {
    throw std::runtime_error("cannot read API key file: " + file.string());
  }
  std::ostringstream contents;
  contents << in.rdbuf();
  parse(contents.str());
}

ApiKeys::ApiKeys(std::string_view text, std::function<TokenBucket::Clock::time_point()> clock)
    : clock_(std::move(clock)) {
  parse(text);
}

void ApiKeys::parse(std::string_view text) {
  std::istringstream lines{std::string(text)};
  int number = 0;
  for (std::string line; std::getline(lines, line);) {
    ++number;
    if (line.empty() || line.front() == '#') {
      continue;
    }
    std::istringstream fields(line);
    std::string hash;
    std::string name;
    std::string role;
    double rate = -1.0;
    double burst = -1.0;
    std::string extra;
    const bool complete = static_cast<bool>(fields >> hash >> name >> role >> rate >> burst);
    if (!complete || (fields >> extra) || hash.size() != 64 || !isLowerHex(hash) ||
        (role != "client" && role != "admin") || rate < 0.0 || burst < 0.0 ||
        (rate > 0.0 && burst < 1.0)) {
      // The line itself stays out of the message, in case someone pasted a raw key there.
      throw std::runtime_error("malformed API key entry on line " + std::to_string(number));
    }
    Entry entry;
    entry.name = name;
    entry.role = role == "admin" ? Role::kAdmin : Role::kClient;
    if (rate > 0.0) {
      entry.bucket = std::make_unique<TokenBucket>(rate, burst, clock_());
    }
    if (!keys_.emplace(hash, std::move(entry)).second) {
      throw std::runtime_error("duplicate API key on line " + std::to_string(number));
    }
  }
}

AuthResult ApiKeys::check(std::string_view authorization, Role needed) {
  if (!authorization.starts_with(kBearer)) {
    return AuthResult::kUnauthenticated;
  }
  const std::string hash = sha256Hex(authorization.substr(kBearer.size()));
  // Comparing hashes means an attacker cannot steer which bytes match. Every entry is compared, so
  // the time taken does not depend on which one matched.
  Entry* found = nullptr;
  for (auto& [stored, entry] : keys_) {
    if (constantTimeEquals(stored, hash)) {
      found = &entry;
    }
  }
  if (found == nullptr) {
    return AuthResult::kUnauthenticated;
  }
  if (needed == Role::kAdmin && found->role != Role::kAdmin) {
    return AuthResult::kForbidden;
  }
  if (found->bucket && !found->bucket->tryTake(clock_())) {
    return AuthResult::kRateLimited;
  }
  return AuthResult::kAllowed;
}

}  // namespace security
