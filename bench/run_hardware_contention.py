#!/usr/bin/env python3
"""Runs N independent inference processes with no networking and no shared software.

Any slowdown as N grows comes from sharing the hardware: caches, memory bandwidth, or frequency.
"""
import json
import statistics
import subprocess

from harness import BUILD, ROOT, write_results

MODEL_BENCH = BUILD / "bench" / "model_bench"
MODELS = {"small": "mlp_quant.onnx", "heavy": "mlp_heavy_quant.onnx", "xl": "mlp_xl_quant.onnx"}
COUNTS = [1, 2, 3, 4, 6, 8]
REPEATS, SECONDS = 3, 3

results = {}
print(f"{'model':>6} {'procs':>5} {'total/s':>9} {'per proc/s':>10} {'us each':>8} {'scaling':>7}")
for name, filename in MODELS.items():
    results[name] = []
    single = None
    for count in COUNTS:
        totals = []
        for _ in range(REPEATS):
            processes = [subprocess.Popen([str(MODEL_BENCH), str(ROOT / "bench/models" / filename),
                                           str(SECONDS)], stdout=subprocess.PIPE, text=True)
                         for _ in range(count)]
            totals.append(sum(json.loads(p.communicate()[0])["inferences_per_sec"]
                              for p in processes))
        total = statistics.median(totals)
        single = single or total
        row = {"processes": count, "total_per_sec": total, "per_process_per_sec": total / count,
               "us_per_inference": 1e6 * count / total, "scaling": total / single}
        results[name].append(row)
        print(f"{name:>6} {count:>5} {total:>9.0f} {total / count:>10.0f} "
              f"{row['us_per_inference']:>8.1f} {row['scaling']:>6.2f}x", flush=True)
write_results("hardware_contention.json", results)
