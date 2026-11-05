#include "perf/arena.h"

#include <algorithm>
#include <cstdint>

namespace perf {
namespace {

std::size_t alignUp(std::uintptr_t value, std::size_t alignment) {
  return (value + alignment - 1) & ~(static_cast<std::uintptr_t>(alignment) - 1);
}

}  // namespace

Arena::Arena(std::size_t chunk_bytes) : chunk_bytes_(chunk_bytes) { addChunk(chunk_bytes); }

void Arena::addChunk(std::size_t min_bytes) {
  const std::size_t size = std::max(min_bytes, chunk_bytes_);
  chunks_.emplace_back(size);
}

void* Arena::allocate(std::size_t bytes, std::size_t alignment) {
  while (true) {
    std::vector<std::byte>& chunk = chunks_[current_];
    const auto base = reinterpret_cast<std::uintptr_t>(chunk.data());
    const std::uintptr_t aligned = alignUp(base + used_, alignment);
    const std::size_t end = (aligned - base) + bytes;
    if (end <= chunk.size()) {
      used_ = end;
      return chunk.data() + (aligned - base);
    }
    // Move on to the next chunk, reusing one from before the last reset() if it is large enough.
    if (current_ + 1 == chunks_.size() || chunks_[current_ + 1].size() < bytes + alignment) {
      addChunk(bytes + alignment);
      // A new chunk goes last, so skip any smaller ones in between.
      current_ = chunks_.size() - 1;
    } else {
      ++current_;
    }
    used_ = 0;
  }
}

void Arena::reset() {
  current_ = 0;
  used_ = 0;
}

std::size_t Arena::capacityBytes() const {
  std::size_t total = 0;
  for (const std::vector<std::byte>& chunk : chunks_) {
    total += chunk.size();
  }
  return total;
}

}  // namespace perf
