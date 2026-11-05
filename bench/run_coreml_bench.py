#!/usr/bin/env python3
"""Compares CPU with Core ML on float MobileNetV2, at fixed batch sizes, and through the server.

macOS only. Core ML picks the GPU, the Neural Engine, or the CPU by itself, and this does not show which
it chose. Needs tools/fetch_real_model.py, then tools/make_static_batch_model.py for the batch sizes.
"""
import json
import statistics
import subprocess

from harness import BUILD, ROOT, Server, load, write_results

MODELS = ROOT / "bench" / "models"
BATCHES = (1, 8, 32)
REPEATS, SECONDS = 3, 5
results = {"model_only": [], "server": []}
print(f"{'provider':>8} {'batch':>5} {'images/s':>9} {'ms/image':>9}")
for batch in BATCHES:
    for provider in ("cpu", "coreml"):
        runs = [json.loads(subprocess.run(
            [str(BUILD / "bench" / "model_bench"), str(MODELS / f"mobilenetv2-static-{batch}.onnx"),
             str(SECONDS), provider, str(batch)], check=True, capture_output=True,
            text=True).stdout)["inferences_per_sec"] for _ in range(REPEATS)]
        rate = statistics.median(runs)
        results["model_only"].append({"provider": provider, "batch": batch, "images_per_sec": rate})
        print(f"{provider:>8} {batch:>5} {rate:>9.1f} {1000 / rate:>9.2f}", flush=True)

print(f"\n{'provider':>8} {'workers':>7} {'rps':>8} {'p50 ms':>8} {'p99 ms':>8} {'errors':>6}")
port = 24500
for provider in ("cpu", "coreml"):
    for workers in (1, 4):
        runs = []
        for _ in range(REPEATS):
            port += 1
            with Server(port, model=MODELS / "mobilenetv2-12.onnx", provider=provider, workers=workers,
                        max_batch=1, events=0):
                runs.append(load(port, 16, duration=5, warmup=1, features=3 * 224 * 224, binary=1))
        row = {"provider": provider, "workers": workers,
               "rps": statistics.median(r["throughput_rps"] for r in runs),
               "p50_ms": statistics.median(r["p50_ms"] for r in runs),
               "p99_ms": statistics.median(r["p99_ms"] for r in runs),
               "errors": sum(r["errors"] + r["rejected"] for r in runs)}
        results["server"].append(row)
        print(f"{provider:>8} {workers:>7} {row['rps']:>8.1f} {row['p50_ms']:>8.2f} {row['p99_ms']:>8.2f} "
              f"{row['errors']:>6}", flush=True)
write_results("coreml.json", results)
