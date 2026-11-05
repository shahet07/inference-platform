#!/usr/bin/env python3
"""Measures what API key checks cost on the request path, with interleaved repeats."""
import hashlib
import statistics
import tempfile
from pathlib import Path

from harness import Server, load, write_results

KEY = "benchmark-key"
REPEATS = 5
runs = {"off": [], "on": []}
port = 24000
with tempfile.TemporaryDirectory() as tmp:
    keys = Path(tmp) / "keys.txt"
    keys.write_text(f"{hashlib.sha256(KEY.encode()).hexdigest()} bench client 0 0\n")
    for _ in range(REPEATS):
        for name in runs:
            port += 1
            flags = {"api_keys": keys} if name == "on" else {}
            load_flags = {"api_key": KEY} if name == "on" else {}
            with Server(port, workers=4, events=0, **flags):
                runs[name].append(load(port, 16, duration=5, warmup=1, **load_flags))
summary = {}
for name, rs in runs.items():
    summary[name] = {
        "rps": statistics.median(r["throughput_rps"] for r in rs),
        "rps_min": min(r["throughput_rps"] for r in rs),
        "rps_max": max(r["throughput_rps"] for r in rs),
        "p50_ms": statistics.median(r["p50_ms"] for r in rs),
        "p99_ms": statistics.median(r["p99_ms"] for r in rs),
        "errors": sum(r["errors"] + r["rejected"] for r in rs),
    }
    print(name, summary[name])
write_results("auth_overhead.json", summary)
