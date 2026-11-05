#!/usr/bin/env python3
"""Runs kernels_bench and fits time = overhead + cost_per_row * batch for each implementation."""
import json
import subprocess

from harness import BUILD, ROOT, write_results

data = json.loads(subprocess.run(
    [str(BUILD / "bench" / "kernels_bench"), str(ROOT / "bench/models/mlp_float.onnx")],
    check=True, capture_output=True, text=True).stdout)

print("256 x 256 x 256 matrix multiplication")
for row in data["matmul_256"]:
    print(f"  {row['impl']:>12} {row['gflops']:>7.1f} GFLOPS  {row['ns'] / 1000:>9.1f} us")

print("\nforward pass, microseconds per call")
impls = [k[:-3] for k in data["forward"][0] if k.endswith("_ns")]
print("  batch " + " ".join(f"{i[:18]:>18}" for i in impls))
for row in data["forward"]:
    print(f"  {row['batch']:>5} " + " ".join(f"{row[i + '_ns'] / 1000:>18.1f}" for i in impls))

print("\nfitted from batches 1 and 256")
data["fit"] = {}
first, last = data["forward"][0], data["forward"][-1]
for impl in impls:
    per_row = (last[impl + "_ns"] - first[impl + "_ns"]) / (last["batch"] - first["batch"])
    overhead = first[impl + "_ns"] - per_row * first["batch"]
    data["fit"][impl] = {"overhead_ns_per_call": overhead, "ns_per_row": per_row}
    print(f"  {impl:>20}  overhead {overhead / 1000:>6.2f} us  per row {per_row / 1000:>6.2f} us")

a = data["allocation"]
print(f"\n64 allocations: malloc {a['malloc_ns']:.0f} ns, arena {a['arena_ns']:.0f} ns "
      f"({a['malloc_ns'] / a['arena_ns']:.1f}x)")
write_results("kernels.json", data)
