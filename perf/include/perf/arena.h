#pragma once

#include <cstddef>
#include <vector>

namespace perf {

// A bump allocator: allocation advances a pointer, nothing is freed individually, and reset()
// releases everything at once while keeping the memory for reuse. It suits work with a clear end,
// such as one inference batch. Not thread-safe, so each worker owns one.
class Arena {
 public:
  explicit Arena(std::size_t chunk_bytes = std::size_t{64} * 1024);

  // `alignment` must be a power of two. Throws std::bad_alloc if memory runs out.
  [[nodiscard]] void* allocate(std::size_t bytes, std::size_t alignment);
  // Invalidates everything allocated so far.
  void reset();

  [[nodiscard]] std::size_t capacityBytes() const;

 private:
  void addChunk(std::size_t min_bytes);

  std::size_t chunk_bytes_;
  std::vector<std::vector<std::byte>> chunks_;
  std::size_t current_ = 0;
  std::size_t used_ = 0;
};

// Lets standard containers allocate from an Arena. Deallocation does nothing, so a growing vector
// leaves its old buffers behind until reset().
template <typename T>
class ArenaAllocator {
 public:
  using value_type = T;

  explicit ArenaAllocator(Arena& arena) noexcept : arena_(&arena) {}
  template <typename U>
  // NOLINTNEXTLINE(google-explicit-constructor): the allocator requirements need this implicit.
  ArenaAllocator(const ArenaAllocator<U>& other) noexcept : arena_(&other.arena()) {}

  [[nodiscard]] T* allocate(std::size_t count) {
    return static_cast<T*>(arena_->allocate(count * sizeof(T), alignof(T)));
  }
  void deallocate(T* /*pointer*/, std::size_t /*count*/) noexcept {}

  [[nodiscard]] Arena& arena() const noexcept { return *arena_; }

  template <typename U>
  bool operator==(const ArenaAllocator<U>& other) const noexcept {
    return arena_ == &other.arena();
  }

 private:
  Arena* arena_;
};

}  // namespace perf
