#include "net/event_server.h"

#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "net/poller.h"

namespace {

using namespace std::chrono_literals;
using net::EventServer;
using net::EventServerConfig;
using net::HttpRequest;
using net::Responder;

// A plain socket, so tests control exactly which bytes arrive and when.
class RawClient {
 public:
  explicit RawClient(int port) : fd_(socket(AF_INET, SOCK_STREAM, 0)) {
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<std::uint16_t>(port));
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    EXPECT_EQ(connect(fd_, reinterpret_cast<const sockaddr*>(&address), sizeof(address)), 0);
    timeval timeout{.tv_sec = 5, .tv_usec = 0};
    setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
#ifdef SO_NOSIGPIPE
    const int enable = 1;
    setsockopt(fd_, SOL_SOCKET, SO_NOSIGPIPE, &enable, sizeof(enable));
#endif
  }
  ~RawClient() { closeNow(); }
  RawClient(const RawClient&) = delete;
  RawClient& operator=(const RawClient&) = delete;
  RawClient(RawClient&&) = delete;
  RawClient& operator=(RawClient&&) = delete;

  void sendText(const std::string& text) const {
    std::size_t sent = 0;
    while (sent < text.size()) {
      const ssize_t count = send(fd_, text.data() + sent, text.size() - sent, 0);
      if (count <= 0) {
        return;
      }
      sent += static_cast<std::size_t>(count);
    }
  }

  // Reads until `text` has been seen, the connection closes, or the timeout passes.
  [[nodiscard]] std::string readUntil(const std::string& text) const {
    std::string all;
    std::string buffer(65536, '\0');
    while (all.find(text) == std::string::npos) {
      const ssize_t count = recv(fd_, buffer.data(), buffer.size(), 0);
      if (count <= 0) {
        break;
      }
      all.append(buffer, 0, static_cast<std::size_t>(count));
    }
    return all;
  }

  [[nodiscard]] bool closedByServer() const {
    char byte = 0;
    return recv(fd_, &byte, 1, 0) == 0;
  }

  void closeNow() {
    if (fd_ >= 0) {
      close(fd_);
      fd_ = -1;
    }
  }

 private:
  int fd_;
};

// Echoes the path and body, on the loop thread or later from another thread.
class EventServerTest : public testing::Test {
 protected:
  void start(const EventServerConfig& config = EventServerConfig{}) {
    server_ = std::make_unique<EventServer>(
        config, [this](const HttpRequest& request, const Responder& responder) {
          handled_.fetch_add(1);
          const std::string reply = std::string(request.path) + ":" + std::string(request.body);
          if (request.path == "/later") {
            // NOLINTNEXTLINE(bugprone-exception-escape)
            std::thread([responder, reply] {
              std::this_thread::sleep_for(20ms);
              responder.respond(200, "text/plain", reply);
            }).detach();
          } else if (request.path == "/throw") {
            throw std::runtime_error("handler failed");
          } else if (request.path == "/twice") {
            responder.respond(200, "text/plain", "first");
            responder.respond(200, "text/plain", "second");
          } else {
            responder.respond(200, "text/plain", reply);
          }
        });
    port_ = server_->bind("127.0.0.1", 0);
    serving_ = std::thread([this] { server_->serve(); });
  }

  ~EventServerTest() override {
    if (server_) {
      server_->stop();
      serving_.join();
      std::this_thread::sleep_for(30ms);  // Lets detached responders finish.
    }
  }

  static std::string post(const std::string& path, const std::string& body) {
    return "POST " + path + " HTTP/1.1\r\nContent-Length: " + std::to_string(body.size()) +
           "\r\n\r\n" + body;
  }

  std::unique_ptr<EventServer> server_;
  std::thread serving_;
  std::atomic<int> handled_{0};
  int port_ = 0;
};

TEST_F(EventServerTest, AnswersARequest) {
  start();
  RawClient client(port_);
  client.sendText(post("/a", "hello"));
  const std::string response = client.readUntil("/a:hello");
  EXPECT_NE(response.find("HTTP/1.1 200 OK"), std::string::npos);
  EXPECT_NE(response.find("Content-Length: 8"), std::string::npos);
  EXPECT_NE(response.find("/a:hello"), std::string::npos);
}

TEST_F(EventServerTest, ConnectionStaysOpenForMoreRequests) {
  start();
  RawClient client(port_);
  for (int i = 0; i < 100; ++i) {
    const std::string body = "n" + std::to_string(i);
    client.sendText(post("/k", body));
    EXPECT_NE(client.readUntil("/k:" + body).find("/k:" + body), std::string::npos);
  }
  EXPECT_EQ(server_->acceptedConnections(), 1U);
}

TEST_F(EventServerTest, RequestArrivingOneByteAtATimeWorks) {
  start();
  RawClient client(port_);
  const std::string request = post("/slow", "abc");
  for (const char c : request) {
    client.sendText(std::string(1, c));
    std::this_thread::sleep_for(100us);
  }
  EXPECT_NE(client.readUntil("/slow:abc").find("/slow:abc"), std::string::npos);
}

TEST_F(EventServerTest, PipelinedRequestsAreAnsweredInOrder) {
  start();
  RawClient client(port_);
  // The first is answered later from another thread, and must still come first.
  client.sendText(post("/later", "1") + post("/b", "2") + post("/later", "3") + post("/c", "4"));
  const std::string response = client.readUntil("/c:4");
  const auto first = response.find("/later:1");
  const auto second = response.find("/b:2");
  const auto third = response.find("/later:3");
  const auto fourth = response.find("/c:4");
  ASSERT_NE(fourth, std::string::npos);
  EXPECT_LT(first, second);
  EXPECT_LT(second, third);
  EXPECT_LT(third, fourth);
}

TEST_F(EventServerTest, LargeBodyAndResponseWork) {
  start();
  RawClient client(port_);
  const std::string body(std::size_t{5} * 1024 * 1024, 'x');
  std::thread sender([&] { client.sendText(post("/big", body)); });
  const std::string response =
      client.readUntil(body.substr(0, 16) + std::string(body.size() - 16, 'x'));
  sender.join();
  EXPECT_GE(response.size(), body.size());
}

TEST_F(EventServerTest, ConnectionCloseIsHonored) {
  start();
  RawClient client(port_);
  client.sendText("GET /x HTTP/1.1\r\nConnection: close\r\n\r\n");
  EXPECT_NE(client.readUntil("/x:").find("Connection: close"), std::string::npos);
  EXPECT_TRUE(client.closedByServer());
}

TEST_F(EventServerTest, MalformedRequestGets400AndCloses) {
  start();
  RawClient client(port_);
  client.sendText("NOT HTTP AT ALL\r\n\r\n");
  EXPECT_NE(client.readUntil("\r\n\r\n").find("400"), std::string::npos);
  EXPECT_TRUE(client.closedByServer());
  EXPECT_EQ(handled_.load(), 0);
}

TEST_F(EventServerTest, SmuggledSecondRequestIsNotProcessed) {
  start();
  RawClient client(port_);
  client.sendText(
      "POST /a HTTP/1.1\r\nContent-Length: 1\r\nContent-Length: 2\r\n\r\nabGET /evil "
      "HTTP/1.1\r\n\r\n");
  EXPECT_NE(client.readUntil("\r\n\r\n").find("400"), std::string::npos);
  EXPECT_TRUE(client.closedByServer());
  EXPECT_EQ(handled_.load(), 0);
}

TEST_F(EventServerTest, OversizedHeaderGets431) {
  start();
  RawClient client(port_);
  client.sendText("GET / HTTP/1.1\r\nX-A: " + std::string(std::size_t{32} * 1024, 'a') +
                  "\r\n\r\n");
  EXPECT_NE(client.readUntil("\r\n\r\n").find("431"), std::string::npos);
}

TEST_F(EventServerTest, ThrowingHandlerGets500AndServerSurvives) {
  start();
  {
    RawClient client(port_);
    client.sendText(post("/throw", ""));
    EXPECT_NE(client.readUntil("\r\n\r\n").find("500"), std::string::npos);
  }
  RawClient next(port_);
  next.sendText(post("/ok", "z"));
  EXPECT_NE(next.readUntil("/ok:z").find("/ok:z"), std::string::npos);
}

TEST_F(EventServerTest, SecondResponseToARequestIsIgnored) {
  start();
  RawClient client(port_);
  client.sendText(post("/twice", ""));
  EXPECT_NE(client.readUntil("first").find("first"), std::string::npos);
  std::this_thread::sleep_for(20ms);
  client.sendText(post("/after", "y"));
  const std::string next = client.readUntil("/after:y");
  EXPECT_EQ(next.find("second"), std::string::npos);
  EXPECT_NE(next.find("/after:y"), std::string::npos);
}

TEST_F(EventServerTest, ClientVanishingMidRequestDoesNotBreakAnything) {
  start();
  for (int i = 0; i < 50; ++i) {
    RawClient client(port_);
    client.sendText(post("/later", "gone"));
    client.closeNow();  // Its response arrives for a connection that no longer exists.
  }
  std::this_thread::sleep_for(50ms);
  RawClient next(port_);
  next.sendText(post("/ok", "z"));
  EXPECT_NE(next.readUntil("/ok:z").find("/ok:z"), std::string::npos);
}

TEST_F(EventServerTest, ManyConnectionsAtOnceAreServed) {
  start({.threads = 2});
  constexpr int kConnections = 500;
  std::vector<std::unique_ptr<RawClient>> clients;
  clients.reserve(kConnections);
  for (int i = 0; i < kConnections; ++i) {
    clients.push_back(std::make_unique<RawClient>(port_));
  }
  for (int i = 0; i < kConnections; ++i) {
    clients[static_cast<std::size_t>(i)]->sendText(post("/m", std::to_string(i)));
  }
  for (int i = 0; i < kConnections; ++i) {
    const std::string want = "/m:" + std::to_string(i);
    ASSERT_NE(clients[static_cast<std::size_t>(i)]->readUntil(want).find(want), std::string::npos);
  }
  EXPECT_EQ(server_->openConnections(), std::uint64_t{kConnections});
}

TEST_F(EventServerTest, IdleConnectionsAreClosed) {
  start({.threads = 1, .max_connections = 100, .idle_timeout = 1s});
  RawClient client(port_);
  EXPECT_TRUE(client.closedByServer());
}

TEST_F(EventServerTest, ConnectionsOverTheLimitAreClosed) {
  start({.threads = 1, .max_connections = 2});
  RawClient a(port_);
  RawClient b(port_);
  a.sendText(post("/a", ""));
  b.sendText(post("/b", ""));
  (void)a.readUntil("/a:");
  (void)b.readUntil("/b:");
  RawClient over(port_);
  EXPECT_TRUE(over.closedByServer());
}

TEST(PollerTest, ReportsARealBackendName) {
  const std::string name = net::makePoller()->name();
  EXPECT_TRUE(name == "kqueue" || name == "epoll" || name == "io_uring") << name;
}

}  // namespace
