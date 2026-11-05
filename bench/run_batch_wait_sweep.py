#!/usr/bin/env python3
"""Measures how the batching deadline trades latency for throughput at several load levels."""
import statistics

from harness import Server, fetch_metric, load, write_results

WAITS_US = [0, 200, 1000]
CONNECTIONS = [1, 16, 64]
REPEATS = 3

runs = {}
port = 18200
for repeat in range(REPEATS):
    for connections in CONNECTIONS:
        for wait_us in WAITS_US:
            port += 1
            with Server(port, workers=4, max_wait_us=wait_us):
                result = load(port, connections, duration=5, warmup=1)
                result["mean_batch_size"] = fetch_metric(
                    port, "inference_batch_size_sum") / fetch_metric(
                    port, "inference_batch_size_count")
            runs.setdefault(f"{connections}/{wait_us}", []).append(result)

summary = []
print(f"{'conns':>5} {'wait_us':>7} {'rps':>8} {'p50_ms':>7} {'p99_ms':>7} {'batch':>5}")
for connections in CONNECTIONS:
    for wait_us in WAITS_US:
        rs = runs[f"{connections}/{wait_us}"]
        row = {
            "connections": connections,
            "max_wait_us": wait_us,
            "rps": statistics.median(r["throughput_rps"] for r in rs),
            "p50_ms": statistics.median(r["p50_ms"] for r in rs),
            "p99_ms": statistics.median(r["p99_ms"] for r in rs),
            "mean_batch": statistics.median(r["mean_batch_size"] for r in rs),
        }
        summary.append(row)
        print(f"{connections:>5} {wait_us:>7} {row['rps']:>8.0f} {row['p50_ms']:>7.2f} "
              f"{row['p99_ms']:>7.2f} {row['mean_batch']:>5.1f}")
write_results("batch_wait_sweep.json", summary)
