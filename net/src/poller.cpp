#include "net/poller.h"

#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdint>
#include <stdexcept>
#include <system_error>

#if defined(__APPLE__)
#include <sys/event.h>
#elif defined(__linux__)
#include <sys/epoll.h>
#ifdef IP_HAS_IO_URING
#include <liburing.h>
#include <poll.h>

#include <cstdlib>
#include <cstring>
#include <map>
#include <string_view>
#endif
#else
#error "net requires kqueue or epoll"
#endif

namespace net {
namespace {

[[noreturn]] void throwErrno(const char* what) {
  throw std::system_error(errno, std::generic_category(), what);
}

constexpr int kMaxEvents = 256;

#if defined(__APPLE__)

class KqueuePoller final : public Poller {
 public:
  KqueuePoller() : queue_(kqueue()) {
    if (queue_ < 0) {
      throwErrno("kqueue");
    }
  }
  ~KqueuePoller() override { close(queue_); }
  KqueuePoller(const KqueuePoller&) = delete;
  KqueuePoller& operator=(const KqueuePoller&) = delete;
  KqueuePoller(KqueuePoller&&) = delete;
  KqueuePoller& operator=(KqueuePoller&&) = delete;

  void add(int fd, std::uint64_t token, bool want_write) override {
    change(fd, EVFILT_READ, EV_ADD, token);
    change(fd, EVFILT_WRITE, want_write ? EV_ADD : EV_ADD | EV_DISABLE, token);
  }

  void setWantWrite(int fd, std::uint64_t token, bool want_write) override {
    change(fd, EVFILT_WRITE, want_write ? EV_ENABLE : EV_DISABLE, token);
  }

  void remove(int fd) override {
    // Closing the descriptor removes these anyway, so errors here are not interesting.
    std::array<struct kevent, 2> changes{};
    EV_SET(&changes[0], fd, EVFILT_READ, EV_DELETE, 0, 0, nullptr);
    EV_SET(&changes[1], fd, EVFILT_WRITE, EV_DELETE, 0, 0, nullptr);
    (void)kevent(queue_, changes.data(), 2, nullptr, 0, nullptr);
  }

  void wait(std::vector<PollEvent>& events, int timeout_ms) override {
    std::array<struct kevent, kMaxEvents> ready{};
    const timespec timeout{.tv_sec = timeout_ms / 1000, .tv_nsec = (timeout_ms % 1000) * 1000000L};
    const int count = kevent(queue_, nullptr, 0, ready.data(), kMaxEvents, &timeout);
    events.clear();
    if (count < 0) {
      if (errno == EINTR) {
        return;
      }
      throwErrno("kevent");
    }
    for (int i = 0; i < count; ++i) {
      const struct kevent& event = ready[static_cast<std::size_t>(i)];
      events.push_back({.token = reinterpret_cast<std::uint64_t>(event.udata),
                        .readable = event.filter == EVFILT_READ,
                        .writable = event.filter == EVFILT_WRITE});
    }
  }

  [[nodiscard]] const char* name() const override { return "kqueue"; }

 private:
  void change(int fd, short filter, unsigned short flags, std::uint64_t token) {
    struct kevent event{};
    // NOLINTNEXTLINE(performance-no-int-to-ptr): kqueue only offers a pointer to carry this.
    EV_SET(&event, fd, filter, flags, 0, 0, reinterpret_cast<void*>(token));
    if (kevent(queue_, &event, 1, nullptr, 0, nullptr) < 0) {
      throwErrno("kevent change");
    }
  }

  int queue_;
};

#else

class EpollPoller final : public Poller {
 public:
  EpollPoller() : epoll_(epoll_create1(EPOLL_CLOEXEC)) {
    if (epoll_ < 0) {
      throwErrno("epoll_create1");
    }
  }
  ~EpollPoller() override { close(epoll_); }
  EpollPoller(const EpollPoller&) = delete;
  EpollPoller& operator=(const EpollPoller&) = delete;
  EpollPoller(EpollPoller&&) = delete;
  EpollPoller& operator=(EpollPoller&&) = delete;

  void add(int fd, std::uint64_t token, bool want_write) override {
    control(EPOLL_CTL_ADD, fd, token, want_write);
  }

  void setWantWrite(int fd, std::uint64_t token, bool want_write) override {
    control(EPOLL_CTL_MOD, fd, token, want_write);
  }

  void remove(int fd) override { (void)epoll_ctl(epoll_, EPOLL_CTL_DEL, fd, nullptr); }

  void wait(std::vector<PollEvent>& events, int timeout_ms) override {
    std::array<epoll_event, kMaxEvents> ready{};
    const int count = epoll_wait(epoll_, ready.data(), kMaxEvents, timeout_ms);
    events.clear();
    if (count < 0) {
      if (errno == EINTR) {
        return;
      }
      throwErrno("epoll_wait");
    }
    for (int i = 0; i < count; ++i) {
      const epoll_event& event = ready[static_cast<std::size_t>(i)];
      // Errors and hangups count as readable, so that the read reports them and the socket closes.
      events.push_back({.token = event.data.u64,
                        .readable = (event.events & (EPOLLIN | EPOLLERR | EPOLLHUP)) != 0,
                        .writable = (event.events & EPOLLOUT) != 0});
    }
  }

  [[nodiscard]] const char* name() const override { return "epoll"; }

 private:
  void control(int operation, int fd, std::uint64_t token, bool want_write) {
    epoll_event event{};
    event.events = EPOLLIN | (want_write ? static_cast<std::uint32_t>(EPOLLOUT) : 0U);
    event.data.u64 = token;
    if (epoll_ctl(epoll_, operation, fd, &event) < 0) {
      throwErrno("epoll_ctl");
    }
  }

  int epoll_;
};

#ifdef IP_HAS_IO_URING

// Uses io_uring only to learn that sockets are ready, not for the reads and writes themselves, so
// it does not offer io_uring's main advantage of doing the I/O without a system call per operation.
// Polls are one-shot and armed again at the start of each wait, after the caller has drained the
// socket, which gives the same level-triggered behavior as the others.
class UringPoller final : public Poller {
 public:
  UringPoller() {
    const int result = io_uring_queue_init(4096, &ring_, 0);
    if (result < 0) {
      errno = -result;
      throwErrno("io_uring_queue_init");
    }
  }
  ~UringPoller() override { io_uring_queue_exit(&ring_); }
  UringPoller(const UringPoller&) = delete;
  UringPoller& operator=(const UringPoller&) = delete;
  UringPoller(UringPoller&&) = delete;
  UringPoller& operator=(UringPoller&&) = delete;

  void add(int fd, std::uint64_t token, bool want_write) override {
    watched_[token] = {
        .fd = fd, .want_write = want_write, .read_armed = false, .write_armed = false};
    by_fd_[fd] = token;
    arm(token);
  }

  void setWantWrite(int fd, std::uint64_t token, bool want_write) override {
    (void)fd;
    const auto it = watched_.find(token);
    if (it == watched_.end()) {
      return;
    }
    it->second.want_write = want_write;
    arm(token);
  }

  void remove(int fd) override {
    const auto it = by_fd_.find(fd);
    if (it == by_fd_.end()) {
      return;
    }
    const std::uint64_t token = it->second;
    const Watched& watched = watched_.at(token);
    if (watched.read_armed) {
      cancel(encode(token, false));
    }
    if (watched.write_armed) {
      cancel(encode(token, true));
    }
    watched_.erase(token);
    by_fd_.erase(it);
    // Polls must be gone before the descriptor closes, since its number can be reused at once.
    submit();
  }

  void wait(std::vector<PollEvent>& events, int timeout_ms) override {
    events.clear();
    for (const auto& [token, watched] : watched_) {
      arm(token);
    }
    submit();
    __kernel_timespec timeout{.tv_sec = timeout_ms / 1000,
                              .tv_nsec = static_cast<long long>(timeout_ms % 1000) * 1000000};
    io_uring_cqe* completion = nullptr;
    const int result = io_uring_wait_cqe_timeout(&ring_, &completion, &timeout);
    if (result == -ETIME || result == -EINTR) {
      return;
    }
    if (result < 0) {
      errno = -result;
      throwErrno("io_uring_wait_cqe_timeout");
    }
    unsigned head = 0;
    unsigned seen = 0;
    io_uring_for_each_cqe(&ring_, head, completion) {
      ++seen;
      const std::uint64_t data = completion->user_data;
      if (data == kCancelCompletion) {
        continue;
      }
      const std::uint64_t token = data >> 1U;
      const bool is_write = (data & 1U) != 0;
      const auto it = watched_.find(token);
      if (it == watched_.end()) {
        continue;  // Removed since this poll was armed.
      }
      (is_write ? it->second.write_armed : it->second.read_armed) = false;
      if (completion->res == -ECANCELED) {
        continue;
      }
      events.push_back({.token = token, .readable = !is_write, .writable = is_write});
    }
    io_uring_cq_advance(&ring_, seen);
  }

  [[nodiscard]] const char* name() const override { return "io_uring"; }

 private:
  struct Watched {
    int fd = -1;
    bool want_write = false;
    bool read_armed = false;
    bool write_armed = false;
  };

  static constexpr std::uint64_t kCancelCompletion = UINT64_MAX;

  static std::uint64_t encode(std::uint64_t token, bool is_write) {
    return (token << 1U) | (is_write ? 1U : 0U);
  }

  io_uring_sqe* nextEntry() {
    io_uring_sqe* entry = io_uring_get_sqe(&ring_);
    if (entry == nullptr) {
      submit();  // The submission queue is full, so send what is there.
      entry = io_uring_get_sqe(&ring_);
    }
    if (entry == nullptr) {
      throw std::runtime_error("io_uring submission queue is full");
    }
    return entry;
  }

  void arm(std::uint64_t token) {
    Watched& watched = watched_.at(token);
    if (!watched.read_armed) {
      io_uring_sqe* entry = nextEntry();
      io_uring_prep_poll_add(entry, watched.fd, POLLIN);
      io_uring_sqe_set_data64(entry, encode(token, false));
      watched.read_armed = true;
    }
    if (watched.want_write && !watched.write_armed) {
      io_uring_sqe* entry = nextEntry();
      io_uring_prep_poll_add(entry, watched.fd, POLLOUT);
      io_uring_sqe_set_data64(entry, encode(token, true));
      watched.write_armed = true;
    } else if (!watched.want_write && watched.write_armed) {
      cancel(encode(token, true));
      watched.write_armed = false;
    }
  }

  void cancel(std::uint64_t data) {
    io_uring_sqe* entry = nextEntry();
    io_uring_prep_poll_remove(entry, data);
    io_uring_sqe_set_data64(entry, kCancelCompletion);
  }

  void submit() {
    const int result = io_uring_submit(&ring_);
    if (result < 0 && result != -EINTR && result != -EAGAIN && result != -EBUSY) {
      errno = -result;
      throwErrno("io_uring_submit");
    }
  }

  io_uring ring_{};
  std::map<std::uint64_t, Watched> watched_;
  std::map<int, std::uint64_t> by_fd_;
};

#endif

#endif

}  // namespace

std::unique_ptr<Poller> makePoller() {
#if defined(__APPLE__)
  return std::make_unique<KqueuePoller>();
#else
#ifdef IP_HAS_IO_URING
  // NOLINTNEXTLINE(concurrency-mt-unsafe): read at startup, before anything changes the
  // environment.
  const char* requested = std::getenv("IP_POLLER");
  if (requested != nullptr && std::string_view(requested) == "io_uring") {
    return std::make_unique<UringPoller>();
  }
#endif
  return std::make_unique<EpollPoller>();
#endif
}

}  // namespace net
