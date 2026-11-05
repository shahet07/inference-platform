#include "net/poller.h"

#include <unistd.h>

#include <array>
#include <cerrno>
#include <system_error>

#if defined(__APPLE__)
#include <sys/event.h>
#elif defined(__linux__)
#include <sys/epoll.h>
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

#endif

}  // namespace

std::unique_ptr<Poller> makePoller() {
#if defined(__APPLE__)
  return std::make_unique<KqueuePoller>();
#else
  return std::make_unique<EpollPoller>();
#endif
}

}  // namespace net
