#!/usr/bin/env python3
"""Measures replicated writes on a 3-node Raft cluster, on one machine, with and without fsync.

All nodes share this machine's single disk and talk over loopback, so a real cluster's numbers would
be different: network latency would add to every commit, and each node would have its own disk.
"""
import json
import statistics
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from harness import RAFT_NODE, load, write_results

REPEATS, DURATION, WARMUP = 3, 5, 1
next_port = 22000


def status(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=2) as response:
        return json.load(response)


def run_once(fsync, connections):
    global next_port
    ports = [next_port + i for i in range(3)]
    next_port += 3
    cluster = ",".join(f"{i + 1}=127.0.0.1:{p}" for i, p in enumerate(ports))
    with tempfile.TemporaryDirectory() as tmp:
        nodes = [subprocess.Popen(
            [str(RAFT_NODE), "--id", str(i + 1), "--cluster", cluster, "--dir",
             str(Path(tmp) / str(i)), "--fsync", str(fsync)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for i in range(3)]
        try:
            leader = None
            deadline = time.time() + 15
            while leader is None and time.time() < deadline:
                for port in ports:
                    try:
                        if status(port)["role"] == "leader":
                            leader = port
                    except OSError:
                        pass
                time.sleep(0.02)
            assert leader, "no leader"
            return load(leader, connections, duration=DURATION, warmup=WARMUP, path="/append",
                        raw_body="x" * 100)
        finally:
            for node in nodes:
                node.kill()
                node.wait()


results = []
print(f"{'fsync':>5} {'conns':>5} {'writes/s':>9} {'p50 ms':>8} {'p99 ms':>8} {'errors':>6}")
for fsync in (0, 1):
    for connections in (1, 16, 64):
        runs = [run_once(fsync, connections) for _ in range(REPEATS)]
        row = {"fsync": bool(fsync), "connections": connections,
               "writes_per_sec": statistics.median(r["throughput_rps"] for r in runs),
               "p50_ms": statistics.median(r["p50_ms"] for r in runs),
               "p99_ms": statistics.median(r["p99_ms"] for r in runs),
               "errors": sum(r["errors"] + r["rejected"] for r in runs)}
        results.append(row)
        print(f"{fsync:>5} {connections:>5} {row['writes_per_sec']:>9.0f} {row['p50_ms']:>8.2f} "
              f"{row['p99_ms']:>8.2f} {row['errors']:>6}", flush=True)
write_results("raft.json", results)
