#!/usr/bin/env python3
"""Tests what limits scaling at 4 nodes by changing one suspect at a time.

If the router limits throughput, adding routers raises it. If uneven key placement limits it,
adding connections (more keys, spread more evenly) raises it. If the machine is out of CPU,
neither does.
"""
import statistics
from concurrent.futures import ThreadPoolExecutor

from harness import ROOT, Router, Server, fetch_metric, load, write_results

WORKLOADS = {
    "light": {"model": ROOT / "bench/models/mlp_quant.onnx", "features": 64, "workers": 4,
              "max_batch": 8},
    "heavy": {"model": ROOT / "bench/models/mlp_heavy_quant.onnx", "features": 512, "workers": 1,
              "max_batch": 1},
}
# (routers, total connections)
CONFIGS = {
    "light": [(1, 64), (2, 64), (3, 63), (1, 192)],
    "heavy": [(1, 32), (2, 32), (3, 33), (1, 128)],
}
NODES, REPEATS, DURATION, WARMUP = 4, 3, 6, 2
next_port = 18800


def run_once(workload, routers, connections):
    global next_port
    node_ports = list(range(next_port, next_port + NODES))
    router_ports = list(range(next_port + NODES, next_port + NODES + routers))
    next_port += NODES + routers
    servers = [Server(p, model=workload["model"], workers=workload["workers"],
                      max_batch=workload["max_batch"], events=0).__enter__() for p in node_ports]
    router_handles = [Router(p, node_ports).__enter__() for p in router_ports]
    try:
        with ThreadPoolExecutor(routers) as pool:
            parts = list(pool.map(
                lambda p: load(p, connections // routers, duration=DURATION, warmup=WARMUP,
                               features=workload["features"], routing_keys=1), router_ports))
        requests = [fetch_metric(p, "inference_requests_total") for p in node_ports]
        return {
            "rps": sum(r["throughput_rps"] for r in parts),
            "p50_ms": statistics.mean(r["p50_ms"] for r in parts),
            "p99_ms": max(r["p99_ms"] for r in parts),
            "busiest_vs_mean": max(requests) / statistics.mean(requests),
        }
    finally:
        for handle in [*router_handles, *servers]:
            handle.__exit__()


results = {}
for name, workload in WORKLOADS.items():
    results[name] = []
    print(f"\n{name}: routers conns     rps    p50    p99  busiest/mean")
    runs = {c: [] for c in CONFIGS[name]}
    for _ in range(REPEATS):
        for config in CONFIGS[name]:
            runs[config].append(run_once(workload, *config))
    for (routers, connections), rs in runs.items():
        row = {"routers": routers, "connections": connections,
               **{k: statistics.median(r[k] for r in rs) for k in rs[0]}}
        results[name].append(row)
        print(f"{'':>{len(name) + 2}}{routers:>7} {connections:>5} {row['rps']:>7.0f} "
              f"{row['p50_ms']:>6.2f} {row['p99_ms']:>6.2f} {row['busiest_vs_mean']:>13.2f}",
              flush=True)
write_results("cluster_bottleneck.json", results)
