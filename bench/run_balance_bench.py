#!/usr/bin/env python3
"""Compares consistent hashing with least-loaded routing, on equal nodes and with one much slower.

The heavy model is used, with one worker per node, so a node's speed matters to the client. The slow
node has one worker where the others have four, as if it shared its machine with something else.
"""
import statistics

from harness import ROOT, Router, Server, fetch_metric, load, write_results

MODEL = ROOT / "bench/models/mlp_heavy_quant.onnx"
REPEATS, DURATION, WARMUP = 3, 6, 1
SCENARIOS = {"equal": [4, 4, 4], "one_slow": [1, 4, 4]}
next_port = 21000


def run_once(workers, balance):
    global next_port
    ports = [next_port + i for i in range(3)]
    router_port = next_port + 3
    next_port += 4
    servers = [Server(p, model=MODEL, workers=w, max_batch=1, events=0).__enter__()
               for p, w in zip(ports, workers)]
    router = Router(router_port, ports, balance).__enter__()
    try:
        result = load(router_port, 32, duration=DURATION, warmup=WARMUP, features=512,
                      routing_keys=1)
        counts = [fetch_metric(p, "inference_requests_total") for p in ports]
        result["share"] = [c / sum(counts) for c in counts]
    finally:
        router.__exit__()
        for server in servers:
            server.__exit__()
    return result


results = {}
print(f"{'scenario':>10} {'policy':>13} {'rps':>7} {'p50 ms':>7} {'p99 ms':>7} {'p99.9':>7}  node shares")
for scenario, workers in SCENARIOS.items():
    for balance in ("hash", "least-loaded"):
        runs = [run_once(workers, balance) for _ in range(REPEATS)]
        row = {
            "rps": statistics.median(r["throughput_rps"] for r in runs),
            "p50_ms": statistics.median(r["p50_ms"] for r in runs),
            "p99_ms": statistics.median(r["p99_ms"] for r in runs),
            "p999_ms": statistics.median(r["p999_ms"] for r in runs),
            "errors": sum(r["errors"] + r["rejected"] for r in runs),
            "share_last_run": runs[-1]["share"],
        }
        results[f"{scenario}/{balance}"] = row
        print(f"{scenario:>10} {balance:>13} {row['rps']:>7.0f} {row['p50_ms']:>7.2f} "
              f"{row['p99_ms']:>7.2f} {row['p999_ms']:>7.2f}  "
              f"{[round(s, 2) for s in row['share_last_run']]}  err {row['errors']}", flush=True)
write_results("balance.json", results)
