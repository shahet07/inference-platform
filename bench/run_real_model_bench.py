#!/usr/bin/env python3
"""Runs MobileNetV2 (real, quantized) to see whether the synthetic-model results carry over.

Asks two questions: does waiting to fill a batch help on a model that takes real time, and what
does JSON cost compared with raw floats for an input of 150,528 values?
"""
import statistics

from harness import ROOT, Server, fetch_metric, load, write_results

MODEL = ROOT / "bench" / "models" / "mobilenetv2-12-int8.onnx"
FEATURES = 3 * 224 * 224
REPEATS, DURATION, WARMUP = 3, 6, 1
next_port = 20000


def measure(connections, workers, max_batch, wait_us, binary=1):
    global next_port
    runs = []
    for _ in range(REPEATS):
        next_port += 1
        with Server(next_port, model=MODEL, workers=workers, max_batch=max_batch,
                    max_wait_us=wait_us, events=0):
            result = load(next_port, connections, duration=DURATION, warmup=WARMUP,
                          features=FEATURES, binary=binary)
            count = fetch_metric(next_port, "inference_batch_size_count")
            run_seconds = fetch_metric(next_port, "inference_model_run_seconds_sum")
            result["mean_batch"] = fetch_metric(next_port, "inference_batch_size_sum") / count
            result["model_ms_per_batch"] = 1000 * run_seconds / count
        runs.append(result)
    return {
        "connections": connections, "workers": workers, "max_batch": max_batch,
        "max_wait_us": wait_us, "binary": bool(binary),
        "rps": statistics.median(r["throughput_rps"] for r in runs),
        "p50_ms": statistics.median(r["p50_ms"] for r in runs),
        "p99_ms": statistics.median(r["p99_ms"] for r in runs),
        "mean_batch": statistics.median(r["mean_batch"] for r in runs),
        "model_ms_per_batch": statistics.median(r["model_ms_per_batch"] for r in runs),
        "errors": sum(r["errors"] + r["rejected"] for r in runs),
    }


def show(row):
    print(f"conns {row['connections']:>3}  workers {row['workers']:>2}  batch<={row['max_batch']:>2}  "
          f"wait {row['max_wait_us']:>6} us  {'binary' if row['binary'] else 'json  '}  "
          f"{row['rps']:>7.1f} rps  p50 {row['p50_ms']:>8.2f}  p99 {row['p99_ms']:>8.2f}  "
          f"batch {row['mean_batch']:>5.2f}  model {row['model_ms_per_batch']:>7.2f} ms  "
          f"err {row['errors']}", flush=True)


results = {"batching": [], "workers": [], "protocol": []}
print("Does a batch wait help? One worker, so batching is the only way to use the queue.")
for connections in (1, 8, 32):
    for max_batch, wait_us in ((1, 0), (32, 0), (32, 1000), (32, 5000), (32, 20000)):
        row = measure(connections, 1, max_batch, wait_us)
        results["batching"].append(row)
        show(row)
print("\nWorkers against batching, 32 connections")
for workers, max_batch in ((1, 1), (4, 1), (8, 1), (4, 8), (8, 8)):
    row = measure(32, workers, max_batch, 0)
    results["workers"].append(row)
    show(row)
print("\nJSON against binary, 8 connections, 4 workers")
for binary in (1, 0):
    row = measure(8, 4, 8, 0, binary)
    results["protocol"].append(row)
    show(row)
write_results("real_model.json", results)
