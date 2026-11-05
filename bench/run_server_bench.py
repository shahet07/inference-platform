#!/usr/bin/env python3
"""Measures throughput, latency, and CPU use at 1, 4, 8, and 16 worker threads."""
import statistics
import time

from harness import Server, fetch_metric, load, write_results

WORKERS = [1, 4, 8, 16]
CONNECTIONS = 64
REPEATS = 3
DURATION, WARMUP = 8, 2

runs = {w: [] for w in WORKERS}
port = 18300
for repeat in range(REPEATS):
    for workers in WORKERS:
        port += 1
        with Server(port, workers=workers):
            cpu_start, started = fetch_metric(port, "process_cpu_seconds_total"), time.time()
            result = load(port, CONNECTIONS, duration=DURATION, warmup=WARMUP)
            elapsed = time.time() - started
            result["server_cpu_cores"] = (
                fetch_metric(port, "process_cpu_seconds_total") - cpu_start) / elapsed
            result["mean_batch"] = fetch_metric(port, "inference_batch_size_sum") / fetch_metric(
                port, "inference_batch_size_count")
        runs[workers].append(result)

summary = []
print(f"{'workers':>7} {'rps':>8} {'p50_ms':>7} {'p99_ms':>7} {'cores':>6} {'batch':>5}")
for workers in WORKERS:
    rs = runs[workers]
    row = {
        "workers": workers,
        "connections": CONNECTIONS,
        "rps": statistics.median(r["throughput_rps"] for r in rs),
        "rps_min": min(r["throughput_rps"] for r in rs),
        "rps_max": max(r["throughput_rps"] for r in rs),
        "p50_ms": statistics.median(r["p50_ms"] for r in rs),
        "p99_ms": statistics.median(r["p99_ms"] for r in rs),
        "p99_ms_min": min(r["p99_ms"] for r in rs),
        "p99_ms_max": max(r["p99_ms"] for r in rs),
        "server_cpu_cores": statistics.median(r["server_cpu_cores"] for r in rs),
        "mean_batch": statistics.median(r["mean_batch"] for r in rs),
        "errors": sum(r["errors"] for r in rs),
    }
    summary.append(row)
    print(f"{workers:>7} {row['rps']:>8.0f} {row['p50_ms']:>7.2f} {row['p99_ms']:>7.2f} "
          f"{row['server_cpu_cores']:>6.2f} {row['mean_batch']:>5.1f}")
write_results("server_scaling.json", summary)
