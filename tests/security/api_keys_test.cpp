#include "security/api_keys.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace std::chrono_literals;
using security::ApiKeys;
using security::AuthResult;
using security::Role;
using security::sha256Hex;
using security::TokenBucket;

TEST(Sha256Test, MatchesPublishedTestVectors) {
  // From FIPS 180-4 and NIST's examples.
  EXPECT_EQ(sha256Hex(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(sha256Hex("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  EXPECT_EQ(sha256Hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  EXPECT_EQ(sha256Hex(std::string(1000000, 'a')),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST(Sha256Test, HandlesEveryLengthAroundBlockBoundaries) {
  // The padding has several cases near 55, 56, and 64 bytes.
  EXPECT_EQ(sha256Hex(std::string(55, 'a')),
            "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318");
  EXPECT_EQ(sha256Hex(std::string(56, 'a')),
            "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a");
  EXPECT_EQ(sha256Hex(std::string(64, 'a')),
            "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb");
}

class ApiKeysTest : public testing::Test {
 protected:
  [[nodiscard]] ApiKeys keys(const std::string& text) {
    return {text, [this] { return now_; }};
  }

  static std::string entry(const std::string& key, const std::string& role, double rate = 0,
                           double burst = 0) {
    return sha256Hex(key) + " someone " + role + " " + std::to_string(rate) + " " +
           std::to_string(burst) + "\n";
  }

  TokenBucket::Clock::time_point now_;
};

TEST_F(ApiKeysTest, AcceptsAKnownKey) {
  ApiKeys store = keys(entry("secret", "client"));
  EXPECT_EQ(store.check("Bearer secret", Role::kClient), AuthResult::kAllowed);
}

TEST_F(ApiKeysTest, RejectsMissingWrongAndMalformedCredentials) {
  ApiKeys store = keys(entry("secret", "client"));
  for (const char* header : {"", "secret", "Bearer", "Bearer ", "Bearer wrong", "bearer secret",
                             "Basic secret", "Bearer secret ", "Bearer  secret"}) {
    EXPECT_EQ(store.check(header, Role::kClient), AuthResult::kUnauthenticated) << header;
  }
}

TEST_F(ApiKeysTest, TheHashItselfIsNotAKey) {
  // Someone who reads the file only has hashes, and they must not work as keys.
  ApiKeys store = keys(entry("secret", "client"));
  EXPECT_EQ(store.check("Bearer " + sha256Hex("secret"), Role::kClient),
            AuthResult::kUnauthenticated);
}

TEST_F(ApiKeysTest, RolesAreEnforced) {
  ApiKeys store = keys(entry("user", "client") + entry("root", "admin"));
  EXPECT_EQ(store.check("Bearer user", Role::kAdmin), AuthResult::kForbidden);
  EXPECT_EQ(store.check("Bearer root", Role::kAdmin), AuthResult::kAllowed);
  EXPECT_EQ(store.check("Bearer root", Role::kClient), AuthResult::kAllowed);
}

TEST_F(ApiKeysTest, RateLimitAllowsABurstThenTheSteadyRate) {
  ApiKeys store = keys(entry("k", "client", 10, 3));
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(store.check("Bearer k", Role::kClient), AuthResult::kAllowed);
  }
  EXPECT_EQ(store.check("Bearer k", Role::kClient), AuthResult::kRateLimited);
  now_ += 100ms;  // At 10 per second, this earns one more.
  EXPECT_EQ(store.check("Bearer k", Role::kClient), AuthResult::kAllowed);
  EXPECT_EQ(store.check("Bearer k", Role::kClient), AuthResult::kRateLimited);
}

TEST_F(ApiKeysTest, TokensDoNotBuildUpBeyondTheBurst) {
  ApiKeys store = keys(entry("k", "client", 10, 2));
  now_ += 1h;
  EXPECT_EQ(store.check("Bearer k", Role::kClient), AuthResult::kAllowed);
  EXPECT_EQ(store.check("Bearer k", Role::kClient), AuthResult::kAllowed);
  EXPECT_EQ(store.check("Bearer k", Role::kClient), AuthResult::kRateLimited);
}

TEST_F(ApiKeysTest, KeysHaveSeparateLimits) {
  ApiKeys store = keys(entry("a", "client", 1, 1) + entry("b", "client", 1, 1));
  EXPECT_EQ(store.check("Bearer a", Role::kClient), AuthResult::kAllowed);
  EXPECT_EQ(store.check("Bearer a", Role::kClient), AuthResult::kRateLimited);
  EXPECT_EQ(store.check("Bearer b", Role::kClient), AuthResult::kAllowed);
}

TEST_F(ApiKeysTest, ZeroRateMeansNoLimit) {
  ApiKeys store = keys(entry("k", "client"));
  for (int i = 0; i < 1000; ++i) {
    ASSERT_EQ(store.check("Bearer k", Role::kClient), AuthResult::kAllowed);
  }
}

TEST_F(ApiKeysTest, ForbiddenRequestsDoNotUseUpTheLimit) {
  ApiKeys store = keys(entry("k", "client", 1, 1));
  EXPECT_EQ(store.check("Bearer k", Role::kAdmin), AuthResult::kForbidden);
  EXPECT_EQ(store.check("Bearer k", Role::kClient), AuthResult::kAllowed);
}

TEST_F(ApiKeysTest, CommentsAndBlankLinesAreIgnored) {
  ApiKeys store = keys("# a comment\n\n" + entry("k", "client"));
  EXPECT_EQ(store.size(), 1U);
}

TEST_F(ApiKeysTest, MalformedEntriesAreRejectedWithoutEchoingThem) {
  const std::string hash = sha256Hex("k");
  for (const std::string& text : {
           std::string("raw-secret-pasted-here\n"),
           hash + " name\n",
           hash + " name superuser 0 0\n",
           hash + " name client -1 0\n",
           hash + " name client 5 0\n",  // A limit with no burst could never allow anything.
           hash + " name client 0 0 extra\n",
           hash.substr(1) + " name client 0 0\n",
           std::string(64, 'Z') + " name client 0 0\n",
           (hash + " a client 0 0\n").append(hash).append(" b client 0 0\n"),
       }) {
    try {
      ApiKeys store = keys(text);
      ADD_FAILURE() << "accepted: " << text;
    } catch (const std::runtime_error& e) {
      EXPECT_EQ(std::string(e.what()).find("raw-secret"), std::string::npos);
    }
  }
}

TEST(ApiKeysFileTest, LoadsFromAFile) {
  const auto path = std::filesystem::temp_directory_path() / "api_keys_test.txt";
  std::ofstream(path) << sha256Hex("k") << " name client 0 0\n";
  ApiKeys store(path);
  EXPECT_EQ(store.check("Bearer k", Role::kClient), AuthResult::kAllowed);
  std::filesystem::remove(path);
  EXPECT_THROW(ApiKeys{path}, std::runtime_error);
}

TEST(TokenBucketTest, ConcurrentTakersGetExactlyTheBurst) {
  const auto now = TokenBucket::Clock::now();
  TokenBucket bucket(0.001, 1000, now);
  std::atomic<int> taken{0};
  std::vector<std::thread> threads;
  threads.reserve(8);
  for (int t = 0; t < 8; ++t) {
    threads.emplace_back([&] {
      for (int i = 0; i < 500; ++i) {
        taken += bucket.tryTake(now) ? 1 : 0;
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(taken.load(), 1000);
}

}  // namespace
