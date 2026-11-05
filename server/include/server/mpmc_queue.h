#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace server {

// A bounded multi-producer, multi-consumer queue with no locks, after Dmitry Vyukov's design. Each
// slot has a sequence number saying whose turn it is, so producers and consumers only contend when
// they want the same slot. It cannot block: callers must spin or back off when it is full or empty.
// That makes it unsuitable where threads should sleep when idle, which is why the batching engine
// does not use it. See docs/benchmarks.md for how it compares.
template <typename T>
// NOLINTNEXTLINE(clang-analyzer-optin.performance.Padding): the padding is the point, see below.
class MpmcQueue {
 public:
  // Capacity is rounded up to a power of two, and is at least 2: with one slot, a full slot's
  // sequence number looks the same as the next free one's, and a push would overwrite the value in
  // it.
  explicit MpmcQueue(std::size_t capacity) {
    if (capacity == 0) {
      throw std::invalid_argument("capacity must be positive");
    }
    std::size_t size = 2;
    while (size < capacity) {
      size *= 2;
    }
    mask_ = size - 1;
    slots_ = std::vector<Slot>(size);
    for (std::size_t i = 0; i < size; ++i) {
      slots_[i].sequence.store(i, std::memory_order_relaxed);
    }
  }

  // Returns false if full, leaving `value` untouched.
  bool tryPush(T& value) {
    std::size_t position = enqueue_.load(std::memory_order_relaxed);
    while (true) {
      Slot& slot = slots_[position & mask_];
      const std::size_t sequence = slot.sequence.load(std::memory_order_acquire);
      const auto difference =
          static_cast<std::intptr_t>(sequence) - static_cast<std::intptr_t>(position);
      if (difference == 0) {
        if (enqueue_.compare_exchange_weak(position, position + 1, std::memory_order_relaxed)) {
          slot.value = std::move(value);
          slot.sequence.store(position + 1, std::memory_order_release);
          return true;
        }
      } else if (difference < 0) {
        return false;
      } else {
        position = enqueue_.load(std::memory_order_relaxed);
      }
    }
  }

  std::optional<T> tryPop() {
    std::size_t position = dequeue_.load(std::memory_order_relaxed);
    while (true) {
      Slot& slot = slots_[position & mask_];
      const std::size_t sequence = slot.sequence.load(std::memory_order_acquire);
      const auto difference =
          static_cast<std::intptr_t>(sequence) - static_cast<std::intptr_t>(position + 1);
      if (difference == 0) {
        if (dequeue_.compare_exchange_weak(position, position + 1, std::memory_order_relaxed)) {
          std::optional<T> value(std::move(slot.value));
          slot.sequence.store(position + mask_ + 1, std::memory_order_release);
          return value;
        }
      } else if (difference < 0) {
        return std::nullopt;
      } else {
        position = dequeue_.load(std::memory_order_relaxed);
      }
    }
  }

 private:
  struct Slot {
    std::atomic<std::size_t> sequence{0};
    T value{};
  };

  std::vector<Slot> slots_;
  std::size_t mask_ = 0;
  alignas(64) std::atomic<std::size_t> enqueue_{
      0};  // Own cache lines, so the two ends do not fight.
  alignas(64) std::atomic<std::size_t> dequeue_{0};
};

}  // namespace server
