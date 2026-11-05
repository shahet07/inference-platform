#!/usr/bin/env python3
"""Scaling from 1 to 4 nodes behind the router, on one machine, for two workloads.

light: a small model, so HTTP handling dominates. heavy and xl: one worker per node on larger
models, so model execution takes a growing share. Every process's CPU use is recorded to locate
the bottleneck. Name workloads as arguments to run only those; other results are kept.
"""
import json
import statistics
import sys

from harness import RESULTS, ROOT, Router, Server, fetch_metric, load, write_results

WORKLOADS = {
    "light": {"model": ROOT / "bench/models/mlp_quant.onnx", "features": 64, "workers": 4,
              "max_batch": 8, "connections": 64},
    "heavy": {"model": ROOT / "bench/models/mlp_heavy_quant.onnx", "features": 512, "workers": 1,
              "max_batch": 1, "connections": 32},
    "xl": {"model": ROOT / "bench/models/mlp_xl_quant.onnx", "features": 1024, "workers": 1,
           "max_batch": 1, "connections": 32},
}
NODE_COUNTS = [1, 2, 3, 4]
REPEATS = 3
DURATION, WARMUP = 6, 2
CPU = "process_cpu_seconds_total"

next_port = 18500


def ports(count):
    global next_port
    first = next_port
    next_port += count
    return list(range(first, first + count))


def run_once(workload, nodes, use_router):
    node_ports = ports(nodes)
    router_port = ports(1)[0]
    servers = [Server(p, model=workload["model"], workers=workload["workers"],
                      max_batch=workload["max_batch"], events=0).__enter__() for p in node_ports]
    router = Router(router_port, node_ports).__enter__() if use_router else None
    try:
        target = router_port if use_router else node_ports[0]
        node_cpu_start = [fetch_metric(p, CPU) for p in node_ports]
        router_cpu_start = fetch_metric(router_port, CPU) if use_router else 0.0
        result = load(target, workload["connections"], duration=DURATION, warmup=WARMUP,
                      features=workload["features"], routing_keys=1)
        elapsed = DURATION + WARMUP
        result["node_cores"] = [(fetch_metric(p, CPU) - s) / elapsed
                                for p, s in zip(node_ports, node_cpu_start)]
        result["router_cores"] = ((fetch_metric(router_port, CPU) - router_cpu_start) / elapsed
                                  if use_router else 0.0)
        result["client_cores"] = result["client_cpu_seconds"] / elapsed
        result["node_requests"] = [fetch_metric(p, "inference_requests_total") for p in node_ports]
    finally:
        if router:
            router.__exit__()
        for server in servers:
            server.__exit__()
    return result


def summarize(runs):
    return {
        "rps": statistics.median(r["throughput_rps"] for r in runs),
        "rps_min": min(r["throughput_rps"] for r in runs),
        "rps_max": max(r["throughput_rps"] for r in runs),
        "p50_ms": statistics.median(r["p50_ms"] for r in runs),
        "p99_ms": statistics.median(r["p99_ms"] for r in runs),
        "errors": sum(r["errors"] for r in runs),
        "rejected": sum(r["rejected"] for r in runs),
        "node_cores": statistics.median(sum(r["node_cores"]) for r in runs),
        "router_cores": statistics.median(r["router_cores"] for r in runs),
        "client_cores": statistics.median(r["client_cores"] for r in runs),
        "node_requests_last_run": runs[-1]["node_requests"],
    }


existing = RESULTS / "cluster_scaling.json"
results = json.loads(existing.read_text()) if existing.exists() else {}
for name in sys.argv[1:] or WORKLOADS:
    workload = WORKLOADS[name]
    configs = [("direct", 1, False)] + [(f"{n}_nodes", n, True) for n in NODE_COUNTS]
    runs = {label: [] for label, _, _ in configs}
    for _ in range(REPEATS):
        for label, nodes, use_router in configs:
            runs[label].append(run_once(workload, nodes, use_router))
    results[name] = {label: summarize(rs) for label, rs in runs.items()}
    print(f"\n{name}: {'config':>8} {'rps':>7} {'p50':>7} {'p99':>7} {'node':>5} {'router':>6} "
          f"{'client':>6} {'err':>3}")
    for label, s in results[name].items():
        print(f"{'':>{len(name) + 1}} {label:>8} {s['rps']:>7.0f} {s['p50_ms']:>7.2f} "
              f"{s['p99_ms']:>7.2f} {s['node_cores']:>5.1f} {s['router_cores']:>6.1f} "
              f"{s['client_cores']:>6.1f} {s['errors'] + s['rejected']:>3}  "
              f"{[round(x) for x in s['node_requests_last_run']]}", flush=True)
write_results("cluster_scaling.json", results)
