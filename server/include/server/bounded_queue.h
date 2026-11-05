#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <utility>

namespace server {

// A fixed-capacity multi-producer, multi-consumer queue. After close(), pushes fail and pops
// drain what is left, then return nullopt.
template <typename T>
class BoundedQueue {
 public:
  explicit BoundedQueue(std::size_t capacity) : capacity_(capacity) {}

  // Returns false if the queue is full or closed. On failure, `value` is left untouched.
  bool tryPush(T& value) {
    {
      const std::scoped_lock lock(mutex_);
      if (closed_ || items_.size() >= capacity_) {
        return false;
      }
      items_.push_back(std::move(value));
    }
    not_empty_.notify_one();
    return true;
  }

  // Blocks until there is room. Returns false if the queue is closed.
  bool push(T value) {
    {
      std::unique_lock lock(mutex_);
      not_full_.wait(lock, [this] { return closed_ || items_.size() < capacity_; });
      if (closed_) {
        return false;
      }
      items_.push_back(std::move(value));
    }
    not_empty_.notify_one();
    return true;
  }

  // Blocks until an item arrives, or returns nullopt once closed and empty.
  std::optional<T> pop() {
    std::unique_lock lock(mutex_);
    not_empty_.wait(lock, [this] { return closed_ || !items_.empty(); });
    return takeFront(lock);
  }

  // Like pop(), but also returns nullopt if `deadline` passes first.
  std::optional<T> popUntil(std::chrono::steady_clock::time_point deadline) {
    std::unique_lock lock(mutex_);
    not_empty_.wait_until(lock, deadline, [this] { return closed_ || !items_.empty(); });
    return takeFront(lock);
  }

  void close() {
    {
      const std::scoped_lock lock(mutex_);
      closed_ = true;
    }
    not_empty_.notify_all();
    not_full_.notify_all();
  }

  [[nodiscard]] std::size_t size() const {
    const std::scoped_lock lock(mutex_);
    return items_.size();
  }

 private:
  std::optional<T> takeFront(std::unique_lock<std::mutex>& lock) {
    if (items_.empty()) {
      return std::nullopt;
    }
    std::optional<T> item(std::move(items_.front()));
    items_.pop_front();
    lock.unlock();
    not_full_.notify_one();
    return item;
  }

  const std::size_t capacity_;
  mutable std::mutex mutex_;
  std::condition_variable not_empty_;
  std::condition_variable not_full_;
  std::deque<T> items_;
  bool closed_ = false;
};

}  // namespace server
