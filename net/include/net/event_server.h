#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "net/http_parser.h"

namespace net {

struct LoopShared;

// Sends the response to one request, from any thread, at any time. Copying is fine, but only the
// first call counts, and it is harmless if the client has already gone away.
class Responder {
 public:
  Responder(std::shared_ptr<LoopShared> loop, std::uint64_t connection, bool keep_alive);

  void respond(int status, std::string_view content_type, std::string_view body,
               const std::vector<std::pair<std::string, std::string>>& extra_headers = {}) const;

 private:
  std::shared_ptr<LoopShared> loop_;
  std::uint64_t connection_;
  bool keep_alive_;
};

// Runs on an event loop thread, so it must not block. The request's views last only for this call.
using Handler = std::function<void(const HttpRequest&, const Responder&)>;

struct EventServerConfig {
  std::size_t threads = 2;
  std::size_t max_connections = 100000;
  // Connections with nothing in progress are closed after this long, which stops idle or very slow
  // clients from holding connections forever.
  std::chrono::seconds idle_timeout{60};
};

// An HTTP/1.1 server where a few threads each watch many sockets, instead of one thread per
// connection. A connection handles one request at a time. Further pipelined requests wait until it
// has responded, which keeps responses in order.
class EventServer {
 public:
  EventServer(const EventServerConfig& config, Handler handler);
  ~EventServer();

  EventServer(const EventServer&) = delete;
  EventServer& operator=(const EventServer&) = delete;
  EventServer(EventServer&&) = delete;
  EventServer& operator=(EventServer&&) = delete;

  // Binds to `port`, or any free port if it is 0. Returns the bound port. Throws on failure.
  int bind(const std::string& host, int port);
  // Blocks until stop() is called.
  void serve();
  void stop();

  [[nodiscard]] std::uint64_t acceptedConnections() const;
  [[nodiscard]] std::uint64_t openConnections() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace net
