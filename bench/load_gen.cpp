// HTTP load generator. Prints one JSON object of results.
//
// Closed loop by default: each connection sends its next request when the last one returns.
// With --rate R (total requests/sec) it runs open loop and measures latency from each request's
// scheduled send time, so a stalled server cannot hide its queueing delay (coordinated omission).
#include <httplib.h>
#include <sys/resource.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "obs/startup.h"

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::string host = "127.0.0.1";
  int port = 8080;
  std::size_t connections = 8;
  std::size_t features = 64;
  double duration_seconds = 10;
  double warmup_seconds = 2;
  double rate = 0;
  bool tcp_nodelay = true;
  bool binary = false;  // Send raw floats instead of JSON.
  std::string path = "/v1/infer";
  std::string raw_body;       // When set, sent as is, as plain text, in place of model inputs.
  bool routing_keys = false;  // Send a per-connection X-Routing-Key header.
};

struct ConnectionResult {
  std::vector<double> latencies_ms;
  std::uint64_t ok = 0;
  std::uint64_t rejected = 0;
  std::uint64_t errors = 0;
};

ConnectionResult runConnection(const Options& options, const std::string& body,
                               Clock::time_point measure_from, Clock::time_point stop_at,
                               std::size_t index) {
  httplib::Client client(options.host, options.port);
  client.set_keep_alive(true);
  client.set_tcp_nodelay(options.tcp_nodelay);
  client.set_read_timeout(std::chrono::seconds(30));

  httplib::Headers headers;
  if (options.routing_keys) {
    headers.emplace("X-Routing-Key", "client-" + std::to_string(index));
  }

  const char* content_type = !options.raw_body.empty() ? "text/plain"
                             : options.binary          ? "application/octet-stream"
                                                       : "application/json";

  ConnectionResult result;
  const std::chrono::duration<double> interval(
      options.rate > 0 ? static_cast<double>(options.connections) / options.rate : 0.0);
  // Stagger connections across one interval so open-loop arrivals do not come in bursts.
  auto scheduled = Clock::now() + std::chrono::duration_cast<Clock::duration>(
                                      interval * static_cast<double>(index) /
                                      static_cast<double>(options.connections));
  while (true) {
    Clock::time_point start = Clock::now();
    if (options.rate > 0) {
      std::this_thread::sleep_until(scheduled);
      start = scheduled;
      scheduled += std::chrono::duration_cast<Clock::duration>(interval);
    }
    if (start >= stop_at) {
      return result;
    }
    const auto response = client.Post(options.path, headers, body, content_type);
    const Clock::time_point end = Clock::now();
    if (start < measure_from) {
      continue;
    }
    if (response && response->status == 200) {
      ++result.ok;
      result.latencies_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    } else if (response && response->status == 503) {
      ++result.rejected;
    } else {
      ++result.errors;
    }
  }
}

double cpuSeconds() {
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  const auto seconds = [](const timeval& t) {
    return static_cast<double>(t.tv_sec) + (static_cast<double>(t.tv_usec) / 1e6);
  };
  return seconds(usage.ru_utime) + seconds(usage.ru_stime);
}

double percentile(const std::vector<double>& sorted, double q) {
  if (sorted.empty()) {
    return 0.0;
  }
  const auto index = static_cast<std::size_t>(q * static_cast<double>(sorted.size() - 1));
  return sorted[index];
}

Options parseOptions(std::span<char*> args) {
  std::map<std::string, std::string> flags;
  for (std::size_t i = 1; i + 1 < args.size(); i += 2) {
    flags[std::string(args[i]).substr(2)] = args[i + 1];
  }
  Options o;
  if (flags.contains("host")) {
    o.host = flags["host"];
  }
  if (flags.contains("port")) {
    o.port = std::stoi(flags["port"]);
  }
  if (flags.contains("connections")) {
    o.connections = std::stoul(flags["connections"]);
  }
  if (flags.contains("features")) {
    o.features = std::stoul(flags["features"]);
  }
  if (flags.contains("duration")) {
    o.duration_seconds = std::stod(flags["duration"]);
  }
  if (flags.contains("warmup")) {
    o.warmup_seconds = std::stod(flags["warmup"]);
  }
  if (flags.contains("rate")) {
    o.rate = std::stod(flags["rate"]);
  }
  if (flags.contains("path")) {
    o.path = flags["path"];
  }
  if (flags.contains("raw-body")) {
    o.raw_body = flags["raw-body"];
  }
  if (flags.contains("binary")) {
    o.binary = flags["binary"] != "0";
  }
  if (flags.contains("routing-keys")) {
    o.routing_keys = flags["routing-keys"] != "0";
  }
  if (flags.contains("tcp-nodelay")) {
    o.tcp_nodelay = flags["tcp-nodelay"] != "0";
  }
  return o;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    obs::primeLocaleCaches();
    // Every connection needs a descriptor, and the default limit can be as low as 256.
    rlimit limit{};
    if (getrlimit(RLIMIT_NOFILE, &limit) == 0 && limit.rlim_cur < limit.rlim_max) {
      limit.rlim_cur = limit.rlim_max == RLIM_INFINITY ? 1048576 : limit.rlim_max;
      (void)setrlimit(RLIMIT_NOFILE, &limit);
    }
    // The server may close a keep-alive connection while a request is being written.
    std::signal(SIGPIPE, SIG_IGN);
    const Options options = parseOptions(std::span(argv, static_cast<std::size_t>(argc)));
    std::vector<float> inputs(options.features);
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      inputs[i] = static_cast<float>(i % 7) * 0.125F;
    }
    const std::string body = options.binary
                                 ? std::string(reinterpret_cast<const char*>(inputs.data()),
                                               inputs.size() * sizeof(float))
                                 : nlohmann::json{{"inputs", inputs}}.dump();

    const auto begin = Clock::now();
    const auto seconds = [](double s) {
      return std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(s));
    };
    const auto measure_from = begin + seconds(options.warmup_seconds);
    const auto stop_at = measure_from + seconds(options.duration_seconds);

    std::vector<ConnectionResult> results(options.connections);
    std::vector<std::thread> threads;
    threads.reserve(options.connections);
    for (std::size_t i = 0; i < options.connections; ++i) {
      threads.emplace_back(
          [&, i] { results[i] = runConnection(options, body, measure_from, stop_at, i); });
    }
    for (std::thread& t : threads) {
      t.join();
    }

    ConnectionResult total;
    for (const ConnectionResult& r : results) {
      total.ok += r.ok;
      total.rejected += r.rejected;
      total.errors += r.errors;
      total.latencies_ms.insert(total.latencies_ms.end(), r.latencies_ms.begin(),
                                r.latencies_ms.end());
    }
    std::ranges::sort(total.latencies_ms);
    std::cout << nlohmann::json{
                     {"connections", options.connections},
                     {"offered_rate", options.rate},
                     {"ok", total.ok},
                     {"rejected", total.rejected},
                     {"errors", total.errors},
                     {"throughput_rps", static_cast<double>(total.ok) / options.duration_seconds},
                     {"p50_ms", percentile(total.latencies_ms, 0.50)},
                     {"p99_ms", percentile(total.latencies_ms, 0.99)},
                     {"p999_ms", percentile(total.latencies_ms, 0.999)},
                     {"client_cpu_seconds", cpuSeconds()},
                 }
                     .dump()
              << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception& e) {
    std::cerr << "fatal: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
}
