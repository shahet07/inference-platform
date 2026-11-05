#include "obs/trace.h"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using obs::parseTraceparent;
using obs::TraceContext;
using obs::Tracer;

constexpr const char* kValid = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";

TraceContext parseOrThrow(std::string_view header) {
  auto context = parseTraceparent(header);
  if (!context) {
    throw std::runtime_error("expected a valid traceparent");
  }
  return *context;
}

TEST(TraceparentTest, ParsesValidHeader) {
  const TraceContext context = parseOrThrow(kValid);
  EXPECT_EQ(context.trace_id, "0af7651916cd43dd8448eb211c80319c");
  EXPECT_EQ(context.span_id, "b7ad6b7169203331");
  EXPECT_TRUE(context.sampled);
}

TEST(TraceparentTest, ReadsSampledFlagFromTheLowBit) {
  EXPECT_FALSE(parseOrThrow("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-00").sampled);
  EXPECT_TRUE(parseOrThrow("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-03").sampled);
}

TEST(TraceparentTest, RejectsMalformedHeaders) {
  for (const char* header : {
           "",
           "garbage",
           "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331",     // No flags.
           "00-0AF7651916CD43DD8448EB211C80319C-b7ad6b7169203331-01",  // Uppercase.
           "00-00000000000000000000000000000000-b7ad6b7169203331-01",  // Zero trace id.
           "00-0af7651916cd43dd8448eb211c80319c-0000000000000000-01",  // Zero span id.
           "ff-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",  // Forbidden version.
           "00-0af7651916cd43dd8448eb211c80319g-b7ad6b7169203331-01",  // Not hex.
           "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-zz",  // Bad flags.
       }) {
    EXPECT_FALSE(parseTraceparent(header).has_value()) << header;
  }
}

TEST(TraceparentTest, FormatRoundTrips) {
  EXPECT_EQ(obs::formatTraceparent(parseOrThrow(kValid)), kValid);
}

TEST(TraceIdTest, GeneratedIdsAreValidAndUnique) {
  std::set<std::string> seen;
  for (int i = 0; i < 1000; ++i) {
    const TraceContext context{
        .trace_id = obs::newTraceId(), .span_id = obs::newSpanId(), .sampled = true};
    EXPECT_TRUE(parseTraceparent(obs::formatTraceparent(context)).has_value());
    EXPECT_TRUE(seen.insert(context.trace_id).second);
  }
}

class TracerTest : public testing::Test {
 protected:
  TracerTest()
      : path_(std::filesystem::temp_directory_path() /
              ("tracer_test_" +
               std::string(testing::UnitTest::GetInstance()->current_test_info()->name()) +
               ".jsonl")) {
    std::filesystem::remove(path_);
  }
  ~TracerTest() override { std::filesystem::remove(path_); }

  [[nodiscard]] std::vector<nlohmann::json> readSpans() const {
    std::vector<nlohmann::json> spans;
    std::ifstream file(path_);
    for (std::string line; std::getline(file, line);) {
      spans.push_back(nlohmann::json::parse(line));
    }
    return spans;
  }

  std::filesystem::path path_;
};

TEST_F(TracerTest, SamplesOneInN) {
  Tracer tracer(path_.string(), "svc", 4);
  int sampled = 0;
  for (int i = 0; i < 100; ++i) {
    sampled += tracer.startOrContinue(std::nullopt).sampled ? 1 : 0;
  }
  EXPECT_EQ(sampled, 25);
}

TEST_F(TracerTest, ZeroDisablesSamplingOfNewTraces) {
  Tracer tracer(path_.string(), "svc", 0);
  EXPECT_FALSE(tracer.startOrContinue(std::nullopt).sampled);
}

TEST_F(TracerTest, UnsampledTracesCarryIdsSoCalleesCanFollowTheDecision) {
  Tracer tracer(path_.string(), "svc", 0);
  const TraceContext context = tracer.startOrContinue(std::nullopt);
  const TraceContext parsed = parseOrThrow(obs::formatTraceparent(context));
  EXPECT_FALSE(parsed.sampled);
}

TEST_F(TracerTest, ContinuesIncomingTraceAndItsDecision) {
  Tracer tracer(path_.string(), "svc", 0);  // Would not sample on its own.
  const TraceContext incoming = parseOrThrow(kValid);
  const TraceContext context = tracer.startOrContinue(incoming);
  EXPECT_TRUE(context.sampled);
  EXPECT_EQ(context.trace_id, incoming.trace_id);
  EXPECT_NE(context.span_id, incoming.span_id);
}

TEST_F(TracerTest, WritesSpansAsJsonOnDestruction) {
  {
    Tracer tracer(path_.string(), "router", 1);
    tracer.record({.trace_id = "t",
                   .span_id = "s",
                   .parent_span_id = "p",
                   .name = "work",
                   .start = std::chrono::system_clock::time_point(std::chrono::seconds(2)),
                   .duration = std::chrono::microseconds(5),
                   .attributes = {{"backend", "a\"b"}}});
  }
  const auto spans = readSpans();
  ASSERT_EQ(spans.size(), 1U);
  EXPECT_EQ(spans[0].at("service"), "router");
  EXPECT_EQ(spans[0].at("parent_span_id"), "p");
  EXPECT_EQ(spans[0].at("start_us"), 2000000);
  EXPECT_EQ(spans[0].at("duration_us"), 5);
  EXPECT_EQ(spans[0].at("attributes").at("backend"), "a\"b");
}

TEST_F(TracerTest, FullQueueDropsAndCountsInsteadOfBlocking) {
  std::uint64_t dropped = 0;
  {
    Tracer tracer(path_.string(), "svc", 1, 1);
    for (int i = 0; i < 100000; ++i) {
      tracer.record({.trace_id = "t", .span_id = "s", .name = "n"});
    }
    dropped = tracer.droppedSpans();
  }
  EXPECT_EQ(readSpans().size() + dropped, 100000U);
}

TEST(TracerOpenTest, ThrowsWhenFileCannotBeOpened) {
  EXPECT_THROW(Tracer("/no/such/directory/trace.jsonl", "svc", 1), std::runtime_error);
}

}  // namespace
