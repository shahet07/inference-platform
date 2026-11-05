#include "net/event_server.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <map>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>

#include "net/poller.h"

namespace net {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::uint64_t kWakeToken = 0;
constexpr std::uint64_t kListenToken = 1;
constexpr std::uint64_t kFirstConnectionToken = 2;
constexpr std::size_t kReadChunk = std::size_t{64} * 1024;
constexpr int kPollTimeoutMs = 1000;

[[noreturn]] void throwErrno(const std::string& what) {
  throw std::system_error(errno, std::generic_category(), what);
}

void setNonBlocking(int fd) {
  const int flags = fcntl(fd, F_GETFL, 0);
  // NOLINTNEXTLINE(bugprone-signed-bitwise): fcntl flags are defined as int.
  if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
    throwErrno("fcntl");
  }
}

void setCloseOnExec(int fd) { (void)fcntl(fd, F_SETFD, FD_CLOEXEC); }

const char* reasonPhrase(int status) {
  switch (status) {
    case 200:
      return "OK";
    case 400:
      return "Bad Request";
    case 404:
      return "Not Found";
    case 405:
      return "Method Not Allowed";
    case 413:
      return "Content Too Large";
    case 431:
      return "Request Header Fields Too Large";
    case 500:
      return "Internal Server Error";
    case 501:
      return "Not Implemented";
    case 503:
      return "Service Unavailable";
    case 504:
      return "Gateway Timeout";
    default:
      return "Unknown";
  }
}

std::string buildResponse(int status, std::string_view content_type, std::string_view body,
                          const std::vector<std::pair<std::string, std::string>>& extra_headers,
                          bool keep_alive) {
  std::string out;
  out.reserve(128 + body.size());
  out += "HTTP/1.1 ";
  out += std::to_string(status);
  out += ' ';
  out += reasonPhrase(status);
  out += "\r\nContent-Type: ";
  out += content_type;
  out += "\r\nContent-Length: ";
  out += std::to_string(body.size());
  out += keep_alive ? "\r\nConnection: keep-alive\r\n" : "\r\nConnection: close\r\n";
  for (const auto& [name, value] : extra_headers) {
    out += name;
    out += ": ";
    out += value;
    out += "\r\n";
  }
  out += "\r\n";
  out += body;
  return out;
}

}  // namespace

struct PostedResponse {
  std::uint64_t connection = 0;
  std::string bytes;
  bool keep_alive = true;
};

// What other threads can reach. It outlives the server if a late response still holds it.
struct LoopShared {
  LoopShared() {
    std::array<int, 2> pipe_fds{};
    if (pipe(pipe_fds.data()) != 0) {
      throwErrno("pipe");
    }
    wake_read = pipe_fds[0];
    wake_write = pipe_fds[1];
    setNonBlocking(wake_read);
    setNonBlocking(wake_write);
    setCloseOnExec(wake_read);
    setCloseOnExec(wake_write);
  }
  ~LoopShared() {
    close(wake_read);
    close(wake_write);
  }
  LoopShared(const LoopShared&) = delete;
  LoopShared& operator=(const LoopShared&) = delete;
  LoopShared(LoopShared&&) = delete;
  LoopShared& operator=(LoopShared&&) = delete;

  void wake() {
    if (!wake_pending.exchange(true, std::memory_order_acq_rel)) {
      const char byte = 1;
      (void)!write(wake_write, &byte, 1);
    }
  }

  int wake_read = -1;
  int wake_write = -1;
  std::atomic<bool> wake_pending{false};
  std::mutex mutex;
  std::vector<PostedResponse> responses;
  std::vector<int> new_connections;
};

Responder::Responder(std::shared_ptr<LoopShared> loop, std::uint64_t connection, bool keep_alive)
    : loop_(std::move(loop)), connection_(connection), keep_alive_(keep_alive) {}

void Responder::respond(
    int status, std::string_view content_type, std::string_view body,
    const std::vector<std::pair<std::string, std::string>>& extra_headers) const {
  std::string bytes = buildResponse(status, content_type, body, extra_headers, keep_alive_);
  {
    const std::scoped_lock lock(loop_->mutex);
    loop_->responses.push_back(
        {.connection = connection_, .bytes = std::move(bytes), .keep_alive = keep_alive_});
  }
  loop_->wake();
}

namespace {

struct Connection {
  int fd = -1;
  std::string input;
  std::size_t input_start = 0;
  std::string output;
  std::size_t output_start = 0;
  bool request_in_progress = false;
  bool close_after_write = false;
  bool want_write = false;
  std::uint64_t answered_request = 0;  // Responses are matched to the request they answer.
  std::uint64_t current_request = 0;
  Clock::time_point last_active;
};

// Connection ids carry the request number, so that a second response to one request is ignored.
constexpr unsigned kRequestBits = 24;

class Loop {
 public:
  Loop(const Handler& handler, const EventServerConfig& config, std::atomic<std::uint64_t>& open,
       int listen_fd)
      : handler_(handler),
        config_(config),
        open_(open),
        listen_fd_(listen_fd),
        shared_(std::make_shared<LoopShared>()),
        poller_(makePoller()) {
    poller_->add(shared_->wake_read, kWakeToken, false);
    if (listen_fd_ >= 0) {
      poller_->add(listen_fd_, kListenToken, false);
    }
  }

  ~Loop() {
    for (auto& [token, connection] : connections_) {
      close(connection.fd);
    }
    open_.fetch_sub(connections_.size(), std::memory_order_relaxed);
  }

  Loop(const Loop&) = delete;
  Loop& operator=(const Loop&) = delete;
  Loop(Loop&&) = delete;
  Loop& operator=(Loop&&) = delete;

  void setPeers(std::vector<Loop*> peers) { peers_ = std::move(peers); }
  [[nodiscard]] const std::shared_ptr<LoopShared>& shared() const { return shared_; }

  void run(const std::atomic<bool>& stopping) {
    std::vector<PollEvent> events;
    Clock::time_point next_sweep = Clock::now() + std::chrono::seconds(1);
    while (!stopping.load(std::memory_order_relaxed)) {
      poller_->wait(events, kPollTimeoutMs);
      for (const PollEvent& event : events) {
        if (event.token == kWakeToken) {
          drainWake();
        } else if (event.token == kListenToken) {
          acceptAll();
        } else {
          handleEvent(event);
        }
      }
      drainMailbox();
      const Clock::time_point now = Clock::now();
      if (now >= next_sweep) {
        closeIdle(now);
        next_sweep = now + std::chrono::seconds(1);
      }
    }
  }

  std::atomic<std::uint64_t> accepted{0};

 private:
  void drainWake() {
    std::array<char, 64> buffer{};
    while (read(shared_->wake_read, buffer.data(), buffer.size()) > 0) {
    }
    // Cleared after draining: a wake that arrives from here on will write again, so none is missed.
    shared_->wake_pending.store(false, std::memory_order_release);
  }

  void acceptAll() {
    while (true) {
      const int fd = accept(listen_fd_, nullptr, nullptr);
      if (fd < 0) {
        if (errno == EINTR) {
          continue;
        }
        // Out of descriptors, or the backlog is empty. Level triggering brings us back.
        return;
      }
      if (open_.load(std::memory_order_relaxed) >= config_.max_connections) {
        close(fd);
        continue;
      }
      setCloseOnExec(fd);
      setNonBlocking(fd);
      const int enable = 1;
      (void)setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &enable, sizeof(enable));
#ifdef SO_NOSIGPIPE
      (void)setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &enable, sizeof(enable));
#endif
      accepted.fetch_add(1, std::memory_order_relaxed);
      open_.fetch_add(1, std::memory_order_relaxed);
      Loop* target = peers_[next_peer_++ % peers_.size()];
      if (target == this) {
        addConnection(fd);
      } else {
        {
          const std::scoped_lock lock(target->shared_->mutex);
          target->shared_->new_connections.push_back(fd);
        }
        target->shared_->wake();
      }
    }
  }

  void addConnection(int fd) {
    const std::uint64_t token = next_token_++;
    Connection& connection = connections_[token];
    connection.fd = fd;
    connection.last_active = Clock::now();
    poller_->add(fd, token, false);
  }

  void drainMailbox() {
    std::vector<PostedResponse> responses;
    std::vector<int> new_connections;
    {
      const std::scoped_lock lock(shared_->mutex);
      responses.swap(shared_->responses);
      new_connections.swap(shared_->new_connections);
    }
    for (const int fd : new_connections) {
      addConnection(fd);
    }
    for (PostedResponse& response : responses) {
      const std::uint64_t token = response.connection >> kRequestBits;
      const std::uint64_t request = response.connection & ((std::uint64_t{1} << kRequestBits) - 1);
      const auto it = connections_.find(token);
      // The connection is gone, or this request was already answered.
      if (it == connections_.end() || it->second.current_request != request ||
          !it->second.request_in_progress) {
        continue;
      }
      Connection& connection = it->second;
      connection.request_in_progress = false;
      connection.close_after_write = !response.keep_alive;
      connection.output += response.bytes;
      connection.last_active = Clock::now();
      if (!flush(token, connection)) {
        continue;
      }
      // Pipelined requests were waiting for this one.
      processInput(token, connection);
    }
  }

  void handleEvent(const PollEvent& event) {
    const auto it = connections_.find(event.token);
    if (it == connections_.end()) {
      return;  // Closed earlier in this batch of events.
    }
    Connection& connection = it->second;
    connection.last_active = Clock::now();
    if (event.writable && !flush(event.token, connection)) {
      return;
    }
    if (event.readable) {
      readAvailable(event.token, connection);
    }
  }

  void readAvailable(std::uint64_t token, Connection& connection) {
    while (true) {
      // Reading stops at a bounded backlog, so a client pipelining requests without reading the
      // responses cannot make this buffer grow without limit. The socket's own buffer then fills,
      // and TCP slows the client down.
      if (connection.input.size() - connection.input_start >= kMaxHeaderBytes + kMaxBodyBytes) {
        break;
      }
      // Read into one buffer shared by this thread, so that an idle connection holds no memory for
      // reading. With thousands of connections, that adds up.
      const ssize_t count = read(connection.fd, read_buffer_.data(), read_buffer_.size());
      if (count > 0) {
        connection.input.append(read_buffer_.data(), static_cast<std::size_t>(count));
        if (std::cmp_less(count, read_buffer_.size())) {
          break;
        }
        continue;
      }
      if (count == 0) {
        // The client closed. If it half-closed after sending, a response in progress is still sent.
        closeConnection(token);
        return;
      }
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        break;
      }
      if (errno == EINTR) {
        continue;
      }
      closeConnection(token);
      return;
    }
    processInput(token, connection);
  }

  void processInput(std::uint64_t token, Connection& connection) {
    while (!connection.request_in_progress && !connection.close_after_write) {
      const std::string_view available =
          std::string_view(connection.input).substr(connection.input_start);
      if (available.empty()) {
        break;
      }
      HttpRequest request;
      std::size_t consumed = 0;
      const ParseStatus status = parseRequest(available, request, consumed);
      if (status == ParseStatus::kNeedMore) {
        break;
      }
      if (status != ParseStatus::kComplete) {
        sendErrorAndClose(token, connection, status);
        return;
      }
      connection.request_in_progress = true;
      connection.current_request =
          (connection.current_request + 1) & ((std::uint64_t{1} << kRequestBits) - 1);
      try {
        handler_(request, Responder(shared_, (token << kRequestBits) | connection.current_request,
                                    request.keep_alive));
      } catch (const std::exception&) {
        connection.request_in_progress = false;
        connection.output += buildResponse(500, "text/plain", "internal error\n", {}, false);
        connection.close_after_write = true;
        (void)flush(token, connection);
        return;
      }
      connection.input_start += consumed;
    }
    // Keeps removed bytes from building up, without moving data on every request.
    if (connection.input_start == connection.input.size()) {
      connection.input = std::string();  // Gives the memory back, instead of only emptying it.
      connection.input_start = 0;
    } else if (connection.input_start > kReadChunk) {
      connection.input.erase(0, connection.input_start);
      connection.input_start = 0;
    }
  }

  void sendErrorAndClose(std::uint64_t token, Connection& connection, ParseStatus status) {
    const int code = status == ParseStatus::kHeaderTooLarge   ? 431
                     : status == ParseStatus::kBodyTooLarge   ? 413
                     : status == ParseStatus::kNotImplemented ? 501
                                                              : 400;
    connection.output +=
        buildResponse(code, "text/plain", std::string(reasonPhrase(code)) + "\n", {}, false);
    connection.close_after_write = true;
    connection.input.clear();
    connection.input_start = 0;
    (void)flush(token, connection);
  }

  // Returns false if the connection closed.
  bool flush(std::uint64_t token, Connection& connection) {
    while (connection.output_start < connection.output.size()) {
#ifdef MSG_NOSIGNAL
      constexpr int kFlags = MSG_NOSIGNAL;
#else
      constexpr int kFlags = 0;
#endif
      const ssize_t count = send(connection.fd, connection.output.data() + connection.output_start,
                                 connection.output.size() - connection.output_start, kFlags);
      if (count > 0) {
        connection.output_start += static_cast<std::size_t>(count);
        continue;
      }
      if (count < 0 && errno == EINTR) {
        continue;
      }
      if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        if (!connection.want_write) {
          connection.want_write = true;
          poller_->setWantWrite(connection.fd, token, true);
        }
        return true;
      }
      closeConnection(token);
      return false;
    }
    connection.output = std::string();
    connection.output_start = 0;
    if (connection.want_write) {
      connection.want_write = false;
      poller_->setWantWrite(connection.fd, token, false);
    }
    if (connection.close_after_write) {
      closeConnection(token);
      return false;
    }
    return true;
  }

  void closeConnection(std::uint64_t token) {
    const auto it = connections_.find(token);
    if (it == connections_.end()) {
      return;
    }
    poller_->remove(it->second.fd);
    close(it->second.fd);
    connections_.erase(it);
    open_.fetch_sub(1, std::memory_order_relaxed);
  }

  void closeIdle(Clock::time_point now) {
    std::vector<std::uint64_t> idle;
    for (const auto& [token, connection] : connections_) {
      if (!connection.request_in_progress && now - connection.last_active > config_.idle_timeout) {
        idle.push_back(token);
      }
    }
    for (const std::uint64_t token : idle) {
      closeConnection(token);
    }
  }

  const Handler& handler_;
  const EventServerConfig& config_;
  std::atomic<std::uint64_t>& open_;
  int listen_fd_;
  std::shared_ptr<LoopShared> shared_;
  std::unique_ptr<Poller> poller_;
  std::map<std::uint64_t, Connection> connections_;
  std::uint64_t next_token_ = kFirstConnectionToken;
  std::vector<char> read_buffer_ = std::vector<char>(kReadChunk);
  std::vector<Loop*> peers_;
  std::size_t next_peer_ = 0;
};

}  // namespace

struct EventServer::Impl {
  EventServerConfig config;
  Handler handler;
  int listen_fd = -1;
  std::atomic<bool> stopping{false};
  std::atomic<std::uint64_t> open{0};
  std::vector<std::unique_ptr<Loop>> loops;
  std::mutex mutex;
};

EventServer::EventServer(const EventServerConfig& config, Handler handler)
    : impl_(std::make_unique<Impl>()) {
  impl_->config = config;
  impl_->handler = std::move(handler);
  if (config.threads == 0) {
    throw std::invalid_argument("threads must be positive");
  }
  // Each connection needs a descriptor, and the default limit is often far too low.
  rlimit limit{};
  if (getrlimit(RLIMIT_NOFILE, &limit) == 0 && limit.rlim_cur < limit.rlim_max) {
    limit.rlim_cur = limit.rlim_max == RLIM_INFINITY ? 1048576 : limit.rlim_max;
    (void)setrlimit(RLIMIT_NOFILE, &limit);
  }
}

EventServer::~EventServer() {
  stop();
  if (impl_->listen_fd >= 0) {
    close(impl_->listen_fd);
  }
}

int EventServer::bind(const std::string& host, int port) {
  const int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    throwErrno("socket");
  }
  setCloseOnExec(fd);
  const int enable = 1;
  (void)setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable));
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(static_cast<std::uint16_t>(port));
  if (inet_pton(AF_INET, host.c_str(), &address.sin_addr) != 1) {
    close(fd);
    throw std::invalid_argument("not an IPv4 address: " + host);
  }
  if (::bind(fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0 ||
      listen(fd, 4096) != 0) {
    const int saved = errno;
    close(fd);
    errno = saved;
    throwErrno("bind " + host + ":" + std::to_string(port));
  }
  setNonBlocking(fd);
  socklen_t length = sizeof(address);
  if (getsockname(fd, reinterpret_cast<sockaddr*>(&address), &length) != 0) {
    throwErrno("getsockname");
  }
  impl_->listen_fd = fd;
  return ntohs(address.sin_port);
}

void EventServer::serve() {
  std::vector<Loop*> peers;
  {
    const std::scoped_lock lock(impl_->mutex);
    for (std::size_t i = 0; i < impl_->config.threads; ++i) {
      impl_->loops.push_back(std::make_unique<Loop>(impl_->handler, impl_->config, impl_->open,
                                                    i == 0 ? impl_->listen_fd : -1));
      peers.push_back(impl_->loops.back().get());
    }
    for (const auto& loop : impl_->loops) {
      loop->setPeers(peers);
    }
  }
  std::vector<std::thread> threads;
  threads.reserve(peers.size());
  for (Loop* loop : peers) {
    threads.emplace_back([this, loop] { loop->run(impl_->stopping); });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
}

void EventServer::stop() {
  impl_->stopping.store(true);
  const std::scoped_lock lock(impl_->mutex);
  for (const auto& loop : impl_->loops) {
    loop->shared()->wake();
  }
}

std::uint64_t EventServer::acceptedConnections() const {
  const std::scoped_lock lock(impl_->mutex);
  std::uint64_t total = 0;
  for (const auto& loop : impl_->loops) {
    total += loop->accepted.load(std::memory_order_relaxed);
  }
  return total;
}

std::uint64_t EventServer::openConnections() const {
  return impl_->open.load(std::memory_order_relaxed);
}

}  // namespace net
