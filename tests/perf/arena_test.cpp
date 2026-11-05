#include "perf/arena.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace {

using perf::Arena;
using perf::ArenaAllocator;

TEST(ArenaTest, RespectsAlignment) {
  Arena arena(1024);
  for (const std::size_t alignment : {1U, 2U, 4U, 8U, 16U, 32U, 64U, 256U}) {
    (void)arena.allocate(1, 1);  // Knock the position out of alignment.
    const auto address = reinterpret_cast<std::uintptr_t>(arena.allocate(8, alignment));
    EXPECT_EQ(address % alignment, 0U) << alignment;
  }
}

TEST(ArenaTest, AllocationsDoNotOverlap) {
  Arena arena(64);
  std::vector<unsigned char*> blocks;
  for (int i = 0; i < 100; ++i) {
    auto* block = static_cast<unsigned char*>(arena.allocate(16, 1));
    std::memset(block, i, 16);
    blocks.push_back(block);
  }
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(blocks[static_cast<std::size_t>(i)][15], i);
  }
}

TEST(ArenaTest, HandlesAllocationsLargerThanAChunk) {
  Arena arena(64);
  auto* block = static_cast<char*>(arena.allocate(10000, 16));
  std::memset(block, 1, 10000);  // ASan reports this if the chunk is too small.
  EXPECT_GE(arena.capacityBytes(), 10000U);
}

TEST(ArenaTest, ResetReusesMemoryWithoutGrowing) {
  Arena arena(128);
  for (int i = 0; i < 20; ++i) {
    (void)arena.allocate(100, 8);
  }
  const std::size_t capacity = arena.capacityBytes();
  for (int round = 0; round < 50; ++round) {
    arena.reset();
    for (int i = 0; i < 20; ++i) {
      (void)arena.allocate(100, 8);
    }
  }
  EXPECT_EQ(arena.capacityBytes(), capacity);
}

TEST(ArenaTest, ResetSkipsReusedChunksThatAreTooSmall) {
  Arena arena(64);
  (void)arena.allocate(32, 1);
  (void)arena.allocate(60, 1);  // Fills a second chunk.
  arena.reset();
  (void)arena.allocate(32, 1);
  auto* large = static_cast<char*>(arena.allocate(5000, 1));
  std::memset(large, 1, 5000);
}

TEST(ArenaAllocatorTest, WorksWithStandardContainers) {
  Arena arena(1024);
  std::vector<int, ArenaAllocator<int>> values{ArenaAllocator<int>(arena)};
  for (int i = 0; i < 1000; ++i) {
    values.push_back(i);
  }
  for (int i = 0; i < 1000; ++i) {
    EXPECT_EQ(values[static_cast<std::size_t>(i)], i);
  }
}

}  // namespace
