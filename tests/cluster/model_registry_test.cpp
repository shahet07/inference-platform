#include "cluster/model_registry.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "cluster/replicator.h"

namespace {

using namespace std::chrono_literals;
using cluster::ModelMeta;
using cluster::ModelRegistry;
using cluster::Replicator;

ModelMeta meta(const std::string& uri, std::uint64_t version, const std::string& by) {
  return {.name = "m", .uri = uri, .version = version, .updated_by = by};
}

TEST(ModelRegistryTest, PutIncrementsVersion) {
  ModelRegistry registry("a");
  EXPECT_EQ(registry.put("m", "v1").version, 1U);
  EXPECT_EQ(registry.put("m", "v2").version, 2U);
}

TEST(ModelRegistryTest, HigherVersionWins) {
  ModelRegistry registry("a");
  registry.put("m", "local");
  EXPECT_EQ(registry.merge({meta("remote", 5, "b")}), 1U);
  EXPECT_EQ(registry.list().at(0).uri, "remote");
}

TEST(ModelRegistryTest, OlderVersionIsIgnored) {
  ModelRegistry registry("a");
  registry.merge({meta("new", 5, "b")});
  EXPECT_EQ(registry.merge({meta("old", 4, "c")}), 0U);
  EXPECT_EQ(registry.list().at(0).uri, "new");
}

TEST(ModelRegistryTest, PutAfterMergeGoesAboveMergedVersion) {
  ModelRegistry registry("a");
  registry.merge({meta("remote", 5, "b")});
  EXPECT_EQ(registry.put("m", "local").version, 6U);
}

TEST(ModelRegistryTest, ConcurrentWritesConvergeInAnyOrder) {
  const ModelMeta from_a = meta("from-a", 1, "a");
  const ModelMeta from_b = meta("from-b", 1, "b");
  ModelRegistry first("x");
  ModelRegistry second("y");

  first.merge({from_a});
  first.merge({from_b});
  second.merge({from_b});
  second.merge({from_a});

  EXPECT_EQ(first.list().at(0).uri, second.list().at(0).uri);
  EXPECT_EQ(first.list().at(0).uri, "from-b");  // The tie goes to the larger node id.
}

TEST(ModelRegistryTest, MergeIsIdempotent) {
  ModelRegistry registry("a");
  EXPECT_EQ(registry.merge({meta("u", 1, "b")}), 1U);
  EXPECT_EQ(registry.merge({meta("u", 1, "b")}), 0U);
}

TEST(ModelRegistryTest, JsonRoundTrips) {
  ModelRegistry source("a");
  source.put("m", "s3://bucket/model.onnx");
  ModelRegistry destination("b");

  EXPECT_EQ(destination.mergeJson(source.toJson()), 1U);

  const ModelMeta entry = destination.list().at(0);
  EXPECT_EQ(entry.name, "m");
  EXPECT_EQ(entry.uri, "s3://bucket/model.onnx");
  EXPECT_EQ(entry.version, 1U);
  EXPECT_EQ(entry.updated_by, "a");
}

TEST(ModelRegistryTest, MalformedJsonThrows) {
  ModelRegistry registry("a");
  EXPECT_THROW(registry.mergeJson("not json"), std::runtime_error);
  EXPECT_THROW(registry.mergeJson(R"({"models":[{"name":"m"}]})"), std::runtime_error);
}

template <typename Condition>
bool eventually(Condition condition) {
  for (int attempt = 0; attempt < 2000; ++attempt) {
    if (condition()) {
      return true;
    }
    std::this_thread::sleep_for(1ms);
  }
  return false;
}

TEST(ReplicatorTest, PullsFromPeersUntilConverged) {
  ModelRegistry a("a");
  ModelRegistry b("b");
  std::map<std::string, ModelRegistry*> peers{{"a", &a}, {"b", &b}};
  const Replicator::Fetch fetch = [&peers](const std::string& peer) -> std::optional<std::string> {
    return peers.at(peer)->toJson();
  };
  const Replicator from_b(a, {"b"}, fetch, 1ms);
  const Replicator from_a(b, {"a"}, fetch, 1ms);

  a.put("only-a", "1");
  b.put("only-b", "2");

  EXPECT_TRUE(eventually([&] { return a.list().size() == 2 && b.list().size() == 2; }));
}

TEST(ReplicatorTest, UnreachableOrBadPeersDoNotStopOthers) {
  ModelRegistry source("source");
  source.put("m", "u");
  ModelRegistry destination("destination");
  const Replicator replicator(
      destination, {"down", "garbage", "good"},
      [&source](const std::string& peer) -> std::optional<std::string> {
        if (peer == "down") {
          return std::nullopt;
        }
        return peer == "garbage" ? std::string("not json") : source.toJson();
      },
      1ms);
  EXPECT_TRUE(eventually([&] { return destination.list().size() == 1; }));
}

}  // namespace
