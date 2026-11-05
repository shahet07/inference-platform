#!/usr/bin/env python3
"""A/B test of the arena on the server's request path, with interleaved repeats."""
import statistics

from harness import Server, load, write_results

REPEATS = 5
runs = {0: [], 1: []}
port = 19300
for _ in range(REPEATS):
    for arena in (0, 1):
        port += 1
        with Server(port, workers=4, events=0, arena=arena):
            runs[arena].append(load(port, 16, duration=6, warmup=1))
summary = {}
for arena, rs in runs.items():
    summary["arena" if arena else "malloc"] = {
        "rps": statistics.median(r["throughput_rps"] for r in rs),
        "rps_min": min(r["throughput_rps"] for r in rs),
        "rps_max": max(r["throughput_rps"] for r in rs),
        "p50_ms": statistics.median(r["p50_ms"] for r in rs),
        "p99_ms": statistics.median(r["p99_ms"] for r in rs),
    }
    print("arena" if arena else "malloc", summary["arena" if arena else "malloc"])
write_results("arena_ab.json", summary)
