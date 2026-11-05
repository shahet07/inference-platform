#!/usr/bin/env python3
"""Compares the thread-per-connection frontend with the event-driven one as connections grow.

The load generator uses one thread per connection and shares the machine, which limits what either
frontend can show at high connection counts, so memory and threads may say more there than throughput.
"""
import statistics
import subprocess

from harness import Server, load, write_results

CONNECTIONS = [16, 64, 256, 1000]
REPEATS, DURATION, WARMUP = 3, 5, 2
next_port = 23000


def server_footprint(pid):
    rss_kb = int(subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], check=True, capture_output=True,
                                text=True).stdout.strip())
    lines = subprocess.run(["ps", "-M", "-p", str(pid)], check=True, capture_output=True,
                           text=True).stdout.splitlines()
    return rss_kb / 1024, len(lines) - 1


results = []
print(f"{'frontend':>8} {'conns':>5} {'rps':>7} {'p50 ms':>8} {'p99 ms':>8} {'RSS MB':>7} {'threads':>7} "
      f"{'err':>4}")
for frontend in ("httplib", "event"):
    for connections in CONNECTIONS:
        runs = []
        for _ in range(REPEATS):
            next_port += 1
            # Each thread-per-connection needs a thread available for every connection.
            flags = ({"frontend": "event", "event_threads": 4} if frontend == "event"
                     else {"connections": max(256, connections + 16)})
            with Server(next_port, workers=4, events=0, **flags) as server:
                result = load(next_port, connections, duration=DURATION, warmup=WARMUP, binary=1)
                result["rss_mb"], result["threads"] = server_footprint(server.process.pid)
            runs.append(result)
        row = {"frontend": frontend, "connections": connections,
               "rps": statistics.median(r["throughput_rps"] for r in runs),
               "p50_ms": statistics.median(r["p50_ms"] for r in runs),
               "p99_ms": statistics.median(r["p99_ms"] for r in runs),
               "rss_mb": statistics.median(r["rss_mb"] for r in runs),
               "threads": statistics.median(r["threads"] for r in runs),
               "errors": sum(r["errors"] + r["rejected"] for r in runs)}
        results.append(row)
        print(f"{frontend:>8} {connections:>5} {row['rps']:>7.0f} {row['p50_ms']:>8.2f} "
              f"{row['p99_ms']:>8.2f} {row['rss_mb']:>7.0f} {row['threads']:>7.0f} "
              f"{row['errors']:>4}", flush=True)
write_results("frontend.json", results)
