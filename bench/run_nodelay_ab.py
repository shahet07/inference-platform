#!/usr/bin/env python3
"""A/B test of TCP_NODELAY and the keep-alive limit, with interleaved repeats."""
import statistics

from harness import Server, load, write_results

# (name, server TCP_NODELAY, client TCP_NODELAY, requests per connection)
CONFIGS = [
    ("baseline", 0, 0, 100),
    ("nodelay_server_only", 1, 0, 100),
    ("nodelay_both", 1, 1, 100),
    ("keepalive_only", 0, 0, 1000000),
    ("both_fixes", 1, 1, 1000000),
]
REPEATS = 3

runs = {name: [] for name, *_ in CONFIGS}
port = 18100
for repeat in range(REPEATS):
    for name, server_flag, client_flag, keep_alive in CONFIGS:
        port += 1
        with Server(port, workers=4, tcp_nodelay=server_flag, keep_alive_max=keep_alive):
            result = load(port, 64, tcp_nodelay=client_flag)
        runs[name].append(result)
        print(repeat, name, round(result["throughput_rps"]), result["p50_ms"], result["p99_ms"],
              result["errors"], flush=True)

summary = {
    name: {
        "median_rps": statistics.median(r["throughput_rps"] for r in rs),
        "median_p50_ms": statistics.median(r["p50_ms"] for r in rs),
        "median_p99_ms": statistics.median(r["p99_ms"] for r in rs),
        "runs": rs,
    }
    for name, rs in runs.items()
}
write_results("nodelay_ab.json", summary)
for name, s in summary.items():
    print(f"{name:20} {s['median_rps']:9.0f} rps  p50 {s['median_p50_ms']:6.2f}  "
          f"p99 {s['median_p99_ms']:7.2f}")
