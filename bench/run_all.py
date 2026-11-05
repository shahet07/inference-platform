#!/usr/bin/env python3
"""Runs every benchmark in sequence. Needs a release build and an otherwise idle machine.

Takes about 30 minutes. Results go to bench/results/, and bench/plot_results.py draws the charts.
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "run_nodelay_ab.py",
    "run_batch_wait_sweep.py",
    "run_server_bench.py",
    "run_stream_bench.py",
    "run_cluster_bench.py",
    "run_cluster_bottleneck.py",
    "run_hardware_contention.py",
    "run_kernels_bench.py",
    "run_arena_ab.py",
    "run_tracing_overhead.py",
]
if sys.platform == "darwin":
    SCRIPTS.append("run_profiles.py")

for script in SCRIPTS:
    started = time.time()
    print(f"\n##### {script}", flush=True)
    subprocess.run([sys.executable, str(HERE / script)], check=True, cwd=HERE)
    print(f"##### {script} took {time.time() - started:.0f} s", flush=True)
print("\nall benchmarks finished", flush=True)
