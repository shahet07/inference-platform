#!/usr/bin/env python3
"""Measures what tracing costs the request path, with interleaved repeats."""
import statistics
import tempfile
from pathlib import Path

from harness import Server, load, write_results

CONFIGS = {"off": None, "1_in_100": 100, "every_request": 1}
REPEATS = 5
runs = {name: [] for name in CONFIGS}
port = 19400
with tempfile.TemporaryDirectory() as tmp:
    for _ in range(REPEATS):
        for name, sample in CONFIGS.items():
            port += 1
            flags = {} if sample is None else {"trace_file": Path(tmp) / f"{port}.jsonl",
                                               "trace_sample": sample}
            with Server(port, workers=4, **flags):
                runs[name].append(load(port, 16, duration=6, warmup=1))
summary = {}
for name, rs in runs.items():
    summary[name] = {
        "rps": statistics.median(r["throughput_rps"] for r in rs),
        "rps_min": min(r["throughput_rps"] for r in rs),
        "rps_max": max(r["throughput_rps"] for r in rs),
        "p50_ms": statistics.median(r["p50_ms"] for r in rs),
        "p99_ms": statistics.median(r["p99_ms"] for r in rs),
    }
    print(name, summary[name])
write_results("tracing_overhead.json", summary)
