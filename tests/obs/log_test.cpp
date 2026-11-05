#include "obs/log.h"

#include <gtest/gtest.h>

#include <limits>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

using nlohmann::json;

TEST(LoggerTest, WritesFieldsAsJson) {
  std::ostringstream out;
  obs::Logger logger(out, "server");
  logger.info("started", {{"port", std::int64_t{8080}},
                          {"ratio", 0.5},
                          {"ready", true},
                          {"host", std::string("localhost")}});

  const json line = json::parse(out.str());
  EXPECT_EQ(line.at("level"), "info");
  EXPECT_EQ(line.at("component"), "server");
  EXPECT_EQ(line.at("msg"), "started");
  EXPECT_EQ(line.at("port"), 8080);
  EXPECT_EQ(line.at("ratio"), 0.5);
  EXPECT_EQ(line.at("ready"), true);
  EXPECT_EQ(line.at("host"), "localhost");
  EXPECT_GT(line.at("ts_ms").get<std::int64_t>(), 0);
}

TEST(LoggerTest, EscapesTextThatWouldBreakJson) {
  std::ostringstream out;
  obs::Logger logger(out, "server");
  const std::string hostile = "quote\" slash\\ newline\n tab\t control\x01";
  logger.error(hostile, {{"detail", hostile}});

  const json line = json::parse(out.str());
  EXPECT_EQ(line.at("msg"), hostile);
  EXPECT_EQ(line.at("detail"), hostile);
}

TEST(LoggerTest, NonFiniteNumbersStayValidJson) {
  std::ostringstream out;
  obs::Logger logger(out, "server");
  logger.info("m", {{"value", std::numeric_limits<double>::quiet_NaN()}});
  EXPECT_TRUE(json::parse(out.str()).at("value").is_null());
}

TEST(LoggerTest, FiltersBelowMinimumLevel) {
  std::ostringstream out;
  obs::Logger logger(out, "server", obs::LogLevel::kWarn);
  logger.info("hidden");
  EXPECT_TRUE(out.str().empty());
  logger.warn("shown");
  EXPECT_FALSE(out.str().empty());
}

TEST(LoggerTest, ConcurrentLinesDoNotInterleave) {
  std::ostringstream out;
  obs::Logger logger(out, "server");
  std::vector<std::thread> threads;
  threads.reserve(4);
  for (int t = 0; t < 4; ++t) {
    threads.emplace_back([&logger] {
      for (int i = 0; i < 500; ++i) {
        logger.info("message", {{"i", std::int64_t{i}}});
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  std::istringstream lines(out.str());
  int count = 0;
  for (std::string line; std::getline(lines, line); ++count) {
    EXPECT_NO_THROW((void)json::parse(line));
  }
  EXPECT_EQ(count, 2000);
}

TEST(LogLimiterTest, AllowsOnePerInterval) {
  obs::LogLimiter limiter(60000);
  EXPECT_TRUE(limiter.allow());
  EXPECT_FALSE(limiter.allow());
}

}  // namespace
