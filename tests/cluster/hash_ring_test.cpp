#include "cluster/hash_ring.h"

#include <gtest/gtest.h>

#include <map>
#include <string>

namespace {

using cluster::HashRing;

constexpr int kKeys = 20000;

std::string keyAt(int i) { return "key-" + std::to_string(i); }

HashRing ringOf(int nodes) {
  HashRing ring;
  for (int i = 0; i < nodes; ++i) {
    ring.add("node-" + std::to_string(i));
  }
  return ring;
}

TEST(StableHashTest, SimilarInputsDiffer) {
  EXPECT_NE(cluster::stableHash("node-0#0"), cluster::stableHash("node-0#1"));
}

TEST(HashRingTest, EmptyRingHasNoNodes) {
  const HashRing ring;
  EXPECT_TRUE(ring.nodesFor("key", 2).empty());
}

TEST(HashRingTest, SameKeyMapsToSameNode) {
  const HashRing ring = ringOf(4);
  EXPECT_EQ(ring.nodesFor("key", 1), ring.nodesFor("key", 1));
}

TEST(HashRingTest, KeysSpreadEvenlyAcrossNodes) {
  const HashRing ring = ringOf(4);
  std::map<std::string, int> counts;
  for (int i = 0; i < kKeys; ++i) {
    ++counts[ring.nodesFor(keyAt(i), 1).at(0)];
  }
  ASSERT_EQ(counts.size(), 4U);
  for (const auto& [node, count] : counts) {
    EXPECT_GT(count, kKeys / 4 * 6 / 10) << node;
    EXPECT_LT(count, kKeys / 4 * 14 / 10) << node;
  }
}

TEST(HashRingTest, AddingANodeOnlyMovesKeysToThatNode) {
  HashRing ring = ringOf(4);
  std::map<int, std::string> before;
  for (int i = 0; i < kKeys; ++i) {
    before[i] = ring.nodesFor(keyAt(i), 1).at(0);
  }

  ring.add("node-new");

  int moved = 0;
  for (int i = 0; i < kKeys; ++i) {
    const std::string after = ring.nodesFor(keyAt(i), 1).at(0);
    if (after != before[i]) {
      EXPECT_EQ(after, "node-new");
      ++moved;
    }
  }
  // Ideally 1/5 of the keys.
  EXPECT_GT(moved, kKeys / 10);
  EXPECT_LT(moved, kKeys * 3 / 10);
}

TEST(HashRingTest, RemovingANodeOnlyMovesItsOwnKeys) {
  HashRing ring = ringOf(4);
  std::map<int, std::string> before;
  for (int i = 0; i < kKeys; ++i) {
    before[i] = ring.nodesFor(keyAt(i), 1).at(0);
  }

  ring.remove("node-2");

  for (int i = 0; i < kKeys; ++i) {
    const std::string after = ring.nodesFor(keyAt(i), 1).at(0);
    if (before[i] == "node-2") {
      EXPECT_NE(after, "node-2");
    } else {
      EXPECT_EQ(after, before[i]);
    }
  }
}

TEST(HashRingTest, FailoverOrderIsDistinctNodesStartingWithOwner) {
  const HashRing ring = ringOf(3);
  const auto nodes = ring.nodesFor("key", 3);
  ASSERT_EQ(nodes.size(), 3U);
  EXPECT_EQ(nodes[0], ring.nodesFor("key", 1).at(0));
  EXPECT_NE(nodes[0], nodes[1]);
  EXPECT_NE(nodes[1], nodes[2]);
  EXPECT_NE(nodes[0], nodes[2]);
}

TEST(HashRingTest, AskingForMoreNodesThanExistReturnsAll) {
  const HashRing ring = ringOf(2);
  EXPECT_EQ(ring.nodesFor("key", 5).size(), 2U);
}

TEST(HashRingTest, FailoverOwnerTakesOverWhenOwnerIsRemoved) {
  HashRing ring = ringOf(4);
  const auto order = ring.nodesFor("key", 2);
  ring.remove(order[0]);
  EXPECT_EQ(ring.nodesFor("key", 1).at(0), order[1]);
}

TEST(HashRingTest, AddingTwiceAndRemovingUnknownNodesAreHarmless) {
  HashRing ring = ringOf(2);
  ring.add("node-0");
  ring.remove("unknown");
  EXPECT_EQ(ring.nodeCount(), 2U);
}

}  // namespace
