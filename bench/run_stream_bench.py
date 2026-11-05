#!/usr/bin/env python3
"""Runs the streaming benchmarks: C++ vs Python, backpressure policies, and server overhead."""
import json
import statistics
import subprocess
import sys

from harness import BUILD, ROOT, Server, load, write_results

STREAM_BENCH = BUILD / "bench" / "stream_bench"
BASELINE = ROOT / "bench" / "stream_baseline.py"
REPEATS = 3


def run(args):
    return json.loads(subprocess.run(list(map(str, args)), check=True, capture_output=True,
                                     text=True).stdout)


def median_of(runs, *keys):
    return {key: statistics.median(r[key] for r in runs) for key in keys}


FIELDS = ("processed_events_per_sec", "ingest_events_per_sec", "append_p50_us", "append_p99_us",
          "append_p999_us", "peak_rss_mb")
results = {"baseline": [], "consumer_wakeups": [], "policies": [], "server_overhead": {}}

print(f"{'impl':>7} {'prod':>4} {'shards':>6} {'proc ev/s':>11} {'p50us':>7} {'p99us':>9} "
      f"{'p999us':>10} {'rssMB':>6} {'lost':>5}")
for producers, shards in [(1, 1), (4, 8)]:
    common = ["--producers", producers, "--shards", shards]
    # Each implementation gets an event count that takes it a few seconds.
    cpp = [run([STREAM_BENCH, "--events", 20_000_000, *common]) for _ in range(REPEATS)]
    py = [run([sys.executable, BASELINE, "--events", 1_000_000, *common]) for _ in range(REPEATS)]
    for name, runs in (("cpp", cpp), ("python", py)):
        row = {"impl": name, "producers": producers, "shards": shards, "events": runs[0]["events"],
               **median_of(runs, *FIELDS),
               "lost_events": max(r["events"] - r["consumed"] for r in runs)}
        results["baseline"].append(row)
        print(f"{name:>7} {producers:>4} {shards:>6} {row['processed_events_per_sec']:>11.0f} "
              f"{row['append_p50_us']:>7.2f} {row['append_p99_us']:>9.2f} "
              f"{row['append_p999_us']:>10.2f} {row['peak_rss_mb']:>6.0f} {row['lost_events']:>5}")

print("\nwaking consumers per event (min_batch=1) vs in batches (min_batch=256), 4 producers")
for shards in (1, 8, 32):
    for min_batch in (1, 256):
        runs = [run([STREAM_BENCH, "--events", 8_000_000, "--shards", shards,
                     "--min-batch", min_batch]) for _ in range(REPEATS)]
        row = {"shards": shards, "min_batch": min_batch, **median_of(runs, *FIELDS)}
        results["consumer_wakeups"].append(row)
        print(f"shards={shards:<3} min_batch={min_batch:<4} "
              f"{row['processed_events_per_sec']:>11.0f} ev/s  p99 {row['append_p99_us']:.2f} us")

print("\nslow consumer: each shard reads 1024 events, then sleeps 10 ms; capacity 4096 per shard")
print(f"{'policy':>12} {'ingest ev/s':>12} {'consumed':>9} {'dropped':>9} {'evicted':>9} "
      f"{'blocked':>8} {'p99us':>10} {'rssMB':>6}")
for policy in ("block", "drop_newest", "drop_oldest"):
    r = run([STREAM_BENCH, "--events", 4_000_000, "--policy", policy, "--capacity", 4096,
             "--consumer-delay-us", 10000])
    results["policies"].append(r)
    print(f"{policy:>12} {r['ingest_events_per_sec']:>12.0f} {r['consumed']:>9} "
          f"{r['dropped_newest']:>9} {r['evicted_oldest']:>9} {r['blocked_appends']:>8} "
          f"{r['append_p99_us']:>10.2f} {r['peak_rss_mb']:>6.0f}")

print("\nserver with and without event recording, 16 connections")
runs = {0: [], 1: []}
port = 18400
for _ in range(REPEATS):
    for events in (0, 1):
        port += 1
        with Server(port, workers=4, events=events):
            runs[events].append(load(port, 16))
for events, rs in runs.items():
    row = median_of(rs, "throughput_rps", "p50_ms", "p99_ms")
    results["server_overhead"]["on" if events else "off"] = row
    print(f"events={events}  {row['throughput_rps']:.0f} rps  p50 {row['p50_ms']:.3f}  "
          f"p99 {row['p99_ms']:.3f}")
write_results("stream.json", results)
