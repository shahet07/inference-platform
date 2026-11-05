#include "server/http_server.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <httplib.h>

#include <nlohmann/json.hpp>
#include <span>
#include <thread>
#include <vector>

namespace {

using nlohmann::json;

class HttpServerTest : public testing::Test {
 protected:
  HttpServerTest()
      : engine_(
            [](std::span<const float> input, std::span<const std::int64_t>) {
              std::vector<float> output(input.begin(), input.end());
              for (float& value : output) {
                value += 1.0F;
              }
              return output;
            },
            2, server::EngineConfig{}, registry_),
        http_(engine_, registry_, 4),
        port_(http_.bind("127.0.0.1", 0)),
        client_("127.0.0.1", port_),
        serving_([this] { http_.serve(); }) {
    client_.set_connection_timeout(std::chrono::seconds(5));
    for (int attempt = 0; attempt < 500 && !client_.Get("/healthz"); ++attempt) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  }

  ~HttpServerTest() override {
    http_.stop();
    serving_.join();
  }

  obs::Registry registry_;
  server::BatchingEngine engine_;
  server::HttpServer http_;
  int port_;
  httplib::Client client_;
  std::thread serving_;
};

TEST_F(HttpServerTest, InferReturnsModelOutputs) {
  const auto response = client_.Post("/v1/infer", R"({"inputs":[1.0,2.0]})", "application/json");
  ASSERT_TRUE(response);
  EXPECT_EQ(response->status, 200);
  EXPECT_EQ(json::parse(response->body).at("outputs"), json::parse("[2.0,3.0]"));
}

TEST_F(HttpServerTest, MalformedJsonIsBadRequest) {
  const auto response = client_.Post("/v1/infer", "not json", "application/json");
  ASSERT_TRUE(response);
  EXPECT_EQ(response->status, 400);
}

TEST_F(HttpServerTest, MissingInputsIsBadRequest) {
  const auto response = client_.Post("/v1/infer", R"({"x":1})", "application/json");
  ASSERT_TRUE(response);
  EXPECT_EQ(response->status, 400);
}

TEST_F(HttpServerTest, WrongFeatureCountIsBadRequest) {
  const auto response = client_.Post("/v1/infer", R"({"inputs":[1.0]})", "application/json");
  ASSERT_TRUE(response);
  EXPECT_EQ(response->status, 400);
}

TEST_F(HttpServerTest, StatsAreEmptyWithoutEventStream) {
  const auto response = client_.Get("/v1/stats");
  ASSERT_TRUE(response);
  EXPECT_TRUE(json::parse(response->body).at("windows").empty());
}

TEST_F(HttpServerTest, HealthzReturnsOk) {
  const auto response = client_.Get("/healthz");
  ASSERT_TRUE(response);
  EXPECT_EQ(response->status, 200);
}

TEST_F(HttpServerTest, MetricsCountRequests) {
  ASSERT_TRUE(client_.Post("/v1/infer", R"({"inputs":[1.0,2.0]})", "application/json"));
  const auto response = client_.Get("/metrics");
  ASSERT_TRUE(response);
  EXPECT_THAT(response->body, testing::HasSubstr("inference_requests_total 1\n"));
}

TEST(HttpServerEventsTest, RequestsBecomeWindowedStatsPerClient) {
  obs::Registry registry;
  server::BatchingEngine engine(
      [](std::span<const float> input, std::span<const std::int64_t>) {
        return std::vector<float>(input.begin(), input.end());
      },
      1, server::EngineConfig{}, registry);
  stream::ShardedLog log({.shards = 2, .shard_capacity = 64});
  const auto now_ms = [] {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
  };
  stream::StreamProcessor processor(
      log, stream::ProcessorConfig{.window = {.window_ms = 10}, .clock = now_ms});
  server::HttpServer http(engine, registry, 4);
  http.attachEventStream(log, processor);
  const int port = http.bind("127.0.0.1", 0);
  std::thread serving([&http] { http.serve(); });
  httplib::Client client("127.0.0.1", port);
  for (int attempt = 0; attempt < 500 && !client.Get("/healthz"); ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }

  const httplib::Headers headers{{"X-Client-Id", "client-a"}};
  ASSERT_TRUE(client.Post("/v1/infer", headers, R"({"inputs":[1.0]})", "application/json"));

  json windows = json::array();
  for (int attempt = 0; attempt < 500 && windows.empty(); ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    windows = json::parse(client.Get("/v1/stats")->body).at("windows");
  }
  ASSERT_EQ(windows.size(), 1U);
  EXPECT_EQ(windows[0].at("key"), "client-a");
  EXPECT_EQ(windows[0].at("count"), 1);
  EXPECT_THAT(client.Get("/metrics")->body, testing::HasSubstr("stream_appended_total 1\n"));

  http.stop();
  serving.join();
}

}  // namespace
