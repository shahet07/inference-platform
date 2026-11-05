#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace net {

struct PollEvent {
  std::uint64_t token = 0;
  bool readable = false;
  bool writable = false;
};

// Tells a thread which sockets are ready, using kqueue on macOS and epoll on Linux. It is level
// triggered: a socket keeps being reported until it is drained. Not thread-safe.
class Poller {
 public:
  Poller() = default;
  virtual ~Poller() = default;
  Poller(const Poller&) = delete;
  Poller& operator=(const Poller&) = delete;
  Poller(Poller&&) = delete;
  Poller& operator=(Poller&&) = delete;

  virtual void add(int fd, std::uint64_t token, bool want_write) = 0;
  virtual void setWantWrite(int fd, std::uint64_t token, bool want_write) = 0;
  // Must be called before the descriptor closes.
  virtual void remove(int fd) = 0;
  virtual void wait(std::vector<PollEvent>& events, int timeout_ms) = 0;

  // The name of this build's backend, "kqueue" or "epoll".
  [[nodiscard]] virtual const char* name() const = 0;
};

[[nodiscard]] std::unique_ptr<Poller> makePoller();

}  // namespace net
