#include "server/http_server.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <httplib.h>

#include <cstring>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

using nlohmann::json;

server::LoadedBackend addOneBackend(std::int64_t features) {
  server::LoadedBackend backend;
  backend.row_shape = {features};
  backend.run = [](std::span<const float> input, std::span<const std::int64_t>) {
    std::vector<float> output(input.begin(), input.end());
    for (float& value : output) {
      value += 1.0F;
    }
    return output;
  };
  return backend;
}

server::ModelManager::Loader noLoader() {
  return [](const std::filesystem::path&) -> server::LoadedBackend {
    throw std::runtime_error("no files in this test");
  };
}

class HttpServerTest : public testing::Test {
 protected:
  HttpServerTest()
      : manager_(noLoader(), server::EngineConfig{}, registry_),
        http_(manager_, registry_, 4),
        port_(http_.bind("127.0.0.1", 0)),
        client_("127.0.0.1", port_) {
    manager_.install("default", addOneBackend(2), 1);
    serving_ = std::thread([this] { http_.serve(); });
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
  server::ModelManager manager_;
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

TEST_F(HttpServerTest, NamedModelIsServedSeparately) {
  manager_.install("other", addOneBackend(1), 3);
  const auto response =
      client_.Post("/v1/models/other/infer", R"({"inputs":[5.0]})", "application/json");
  ASSERT_TRUE(response);
  EXPECT_EQ(response->status, 200);
  EXPECT_EQ(json::parse(response->body).at("outputs"), json::parse("[6.0]"));
  EXPECT_EQ(response->get_header_value("X-Model-Version"), "3");
}

TEST_F(HttpServerTest, UnknownModelIsNotFound) {
  const auto response =
      client_.Post("/v1/models/nope/infer", R"({"inputs":[5.0]})", "application/json");
  ASSERT_TRUE(response);
  EXPECT_EQ(response->status, 404);
}

TEST_F(HttpServerTest, BinaryRequestGetsBinaryResponse) {
  const std::vector<float> input{1.5F, 2.5F};
  const std::string body(reinterpret_cast<const char*>(input.data()), input.size() * sizeof(float));
  const auto response = client_.Post("/v1/infer", body, "application/octet-stream");
  ASSERT_TRUE(response);
  ASSERT_EQ(response->status, 200);
  ASSERT_EQ(response->body.size(), 2 * sizeof(float));
  std::vector<float> output(2);
  std::memcpy(output.data(), response->body.data(), response->body.size());
  EXPECT_THAT(output, testing::ElementsAre(2.5F, 3.5F));
}

TEST_F(HttpServerTest, BinaryBodyOfPartialFloatIsBadRequest) {
  const auto response = client_.Post("/v1/infer", std::string(7, 'x'), "application/octet-stream");
  ASSERT_TRUE(response);
  EXPECT_EQ(response->status, 400);
}

TEST_F(HttpServerTest, MalformedDeadlineIsBadRequest) {
  for (const char* value : {"abc", "-1", "99999999999999999999", "5x", ""}) {
    const auto response = client_.Post("/v1/infer", {{"X-Deadline-Ms", value}},
                                       R"({"inputs":[1.0,2.0]})", "application/json");
    ASSERT_TRUE(response);
    EXPECT_EQ(response->status, 400) << value;
  }
}

TEST_F(HttpServerTest, ExpiredDeadlineTimesOut) {
  const auto response = client_.Post("/v1/infer", {{"X-Deadline-Ms", "0"}},
                                     R"({"inputs":[1.0,2.0]})", "application/json");
  ASSERT_TRUE(response);
  EXPECT_EQ(response->status, 504);
}

TEST_F(HttpServerTest, GenerousDeadlineSucceeds) {
  const auto response = client_.Post("/v1/infer", {{"X-Deadline-Ms", "60000"}},
                                     R"({"inputs":[1.0,2.0]})", "application/json");
  ASSERT_TRUE(response);
  EXPECT_EQ(response->status, 200);
}

TEST_F(HttpServerTest, LoadedListsModelsAndVersions) {
  const auto response = client_.Get("/v1/loaded");
  ASSERT_TRUE(response);
  EXPECT_EQ(json::parse(response->body).at("models").at("default"), 1);
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
  server::ModelManager manager(noLoader(), server::EngineConfig{}, registry);
  manager.install("default", addOneBackend(1), 1);
  stream::ShardedLog log({.shards = 2, .shard_capacity = 64});
  const auto now_ms = [] {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
  };
  stream::StreamProcessor processor(
      log, stream::ProcessorConfig{.window = {.window_ms = 10}, .clock = now_ms});
  server::HttpServer http(manager, registry, 4);
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
