#include "net/http_parser.h"

#include <gtest/gtest.h>

#include <string>
#include <string_view>

namespace {

using net::HttpRequest;
using net::parseRequest;
using net::ParseStatus;

ParseStatus parse(std::string_view text) {
  HttpRequest request;
  std::size_t consumed = 0;
  return parseRequest(text, request, consumed);
}

TEST(HttpParserTest, ParsesARequestWithABody) {
  const std::string_view text =
      "POST /v1/infer HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\n\r\nhelloEXTRA";
  HttpRequest request;
  std::size_t consumed = 0;
  ASSERT_EQ(parseRequest(text, request, consumed), ParseStatus::kComplete);
  EXPECT_EQ(request.method, "POST");
  EXPECT_EQ(request.path, "/v1/infer");
  EXPECT_EQ(request.body, "hello");
  EXPECT_EQ(consumed, text.size() - 5);
  EXPECT_TRUE(request.keep_alive);
}

TEST(HttpParserTest, HeaderLookupIgnoresCaseAndTrimsValues) {
  HttpRequest request;
  std::size_t consumed = 0;
  ASSERT_EQ(parseRequest("GET / HTTP/1.1\r\nX-Thing:   value \t\r\n\r\n", request, consumed),
            ParseStatus::kComplete);
  EXPECT_EQ(request.header("x-thing"), "value");
  EXPECT_EQ(request.header("missing"), "");
}

TEST(HttpParserTest, EveryPrefixOfAValidRequestNeedsMore) {
  const std::string text = "POST / HTTP/1.1\r\nContent-Length: 3\r\n\r\nabc";
  for (std::size_t i = 0; i < text.size(); ++i) {
    EXPECT_EQ(parse(std::string_view(text).substr(0, i)), ParseStatus::kNeedMore) << i;
  }
  EXPECT_EQ(parse(text), ParseStatus::kComplete);
}

TEST(HttpParserTest, ConnectionHandling) {
  HttpRequest request;
  std::size_t consumed = 0;
  ASSERT_EQ(parseRequest("GET / HTTP/1.1\r\nConnection: CLOSE\r\n\r\n", request, consumed),
            ParseStatus::kComplete);
  EXPECT_FALSE(request.keep_alive);
  ASSERT_EQ(parseRequest("GET / HTTP/1.0\r\n\r\n", request, consumed), ParseStatus::kComplete);
  EXPECT_FALSE(request.keep_alive);
  ASSERT_EQ(parseRequest("GET / HTTP/1.0\r\nConnection: keep-alive\r\n\r\n", request, consumed),
            ParseStatus::kComplete);
  EXPECT_TRUE(request.keep_alive);
}

TEST(HttpParserTest, MalformedRequestsAreRejected) {
  for (const char* text : {
           "GARBAGE\r\n\r\n",
           "GET\r\n\r\n",
           "GET /\r\n\r\n",
           "GET / HTTP/2\r\n\r\n",
           "GET no-slash HTTP/1.1\r\n\r\n",
           "GET /a b HTTP/1.1\r\n\r\n",
           "G@T / HTTP/1.1\r\n\r\n",
           "GET / HTTP/1.1\r\nno colon\r\n\r\n",
           "GET / HTTP/1.1\r\n: no name\r\n\r\n",
       }) {
    EXPECT_EQ(parse(text), ParseStatus::kBadRequest) << text;
  }
}

TEST(HttpParserTest, SmugglingAttemptsAreRejected) {
  for (const char* text : {
           // Two lengths could be read differently by a proxy and this server.
           "POST / HTTP/1.1\r\nContent-Length: 1\r\nContent-Length: 2\r\n\r\nab",
           "POST / HTTP/1.1\r\nContent-Length: 1, 2\r\n\r\nab",
           "POST / HTTP/1.1\r\nContent-Length: +1\r\n\r\na",
           "POST / HTTP/1.1\r\nContent-Length: -1\r\n\r\na",
           "POST / HTTP/1.1\r\nContent-Length: 0x1\r\n\r\na",
           "POST / HTTP/1.1\r\nContent-Length: \r\n\r\n",
           // A space before the colon has been used to hide headers from proxies.
           "POST / HTTP/1.1\r\nContent-Length : 1\r\n\r\na",
           // A line folded onto the previous header.
           "POST / HTTP/1.1\r\nX-A: 1\r\n Content-Length: 1\r\n\r\na",
           // A bare newline inside a header value.
           "GET / HTTP/1.1\r\nX-A: a\nb\r\n\r\n",
       }) {
    const ParseStatus status = parse(text);
    EXPECT_NE(status, ParseStatus::kComplete) << text;
  }
}

TEST(HttpParserTest, NulInAHeaderValueIsRejected) {
  using namespace std::string_view_literals;
  EXPECT_EQ(parse("GET / HTTP/1.1\r\nX-A: a\0b\r\n\r\n"sv), ParseStatus::kBadRequest);
}

TEST(HttpParserTest, ChunkedUploadsAreRefused) {
  EXPECT_EQ(parse("POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"),
            ParseStatus::kNotImplemented);
  EXPECT_EQ(parse("POST / HTTP/1.1\r\nContent-Length: 1\r\nTransfer-Encoding: chunked\r\n\r\na"),
            ParseStatus::kNotImplemented);
}

TEST(HttpParserTest, LimitsAreEnforced) {
  EXPECT_EQ(parse(std::string(net::kMaxHeaderBytes + 1, 'a')), ParseStatus::kHeaderTooLarge);
  std::string many = "GET / HTTP/1.1\r\n";
  for (std::size_t i = 0; i <= net::kMaxHeaderCount; ++i) {
    many += "A: b\r\n";
  }
  EXPECT_EQ(parse(many + "\r\n"), ParseStatus::kHeaderTooLarge);
  EXPECT_EQ(parse("POST / HTTP/1.1\r\nContent-Length: 999999999999\r\n\r\n"),
            ParseStatus::kBodyTooLarge);
  EXPECT_EQ(parse("POST / HTTP/1.1\r\nContent-Length: 99999999999999999999999\r\n\r\n"),
            ParseStatus::kBodyTooLarge);
}

}  // namespace
