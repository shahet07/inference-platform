#!/usr/bin/env python3
"""Runs every benchmark in sequence, and refuses to trust one run on a busy machine.

Before and after each script, a short calibration checks that inference still scales across 4 processes,
which it only does when the cores are free. A busy machine is waited out before a script starts, and a
script disturbed while running is run again. bench/results/run_conditions.json records what happened, and
results with "clean": false should not be quoted.

Takes about an hour on a quiet machine, plus any waiting. Results go to bench/results/.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_BENCH = ROOT / "build" / "release" / "bench" / "model_bench"
MODEL = ROOT / "bench" / "models" / "mlp_quant.onnx"
SCRIPTS = [
    "run_nodelay_ab.py",
    "run_batch_wait_sweep.py",
    "run_server_bench.py",
    "run_frontend_bench.py",
    "run_stream_bench.py",
    "run_cluster_bench.py",
    "run_cluster_bottleneck.py",
    "run_balance_bench.py",
    "run_raft_bench.py",
    "run_llm_bench.py",
    "run_hardware_contention.py",
    "run_wal_bench.py",
    "run_real_model_bench.py",
    "run_kernels_bench.py",
    "run_arena_ab.py",
    "run_queue_bench.py",
    "run_tracing_overhead.py",
    "run_auth_overhead.py",
]
if sys.platform == "darwin":
    SCRIPTS += ["run_coreml_bench.py", "run_profiles.py"]

MIN_SCALING = 3.4  # This model reaches about 3.8x at 4 processes when the machine is quiet.
QUIET_WAIT_LIMIT_S = 1800
MAX_ATTEMPTS = 3


def scaling():
    """Returns how much faster 4 processes are than 1, on a small model that fits in cache."""
    def run(count):
        processes = [subprocess.Popen([str(MODEL_BENCH), str(MODEL), "2"], stdout=subprocess.PIPE,
                                      text=True) for _ in range(count)]
        return sum(json.loads(p.communicate()[0])["inferences_per_sec"] for p in processes)
    single = run(1)
    return run(4) / single


def wait_until_quiet():
    """Returns the scaling once it is good, or the last reading if the wait runs out."""
    deadline = time.time() + QUIET_WAIT_LIMIT_S
    while True:
        value = scaling()
        if value >= MIN_SCALING or time.time() > deadline:
            return value
        print(f"  machine busy: scaling {value:.2f}x, waiting", flush=True)
        time.sleep(20)


selected = [s for s in SCRIPTS if not sys.argv[1:] or s in sys.argv[1:]]
conditions_file = ROOT / "bench" / "results" / "run_conditions.json"
conditions = json.loads(conditions_file.read_text()) if conditions_file.exists() else {}
for script in selected:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        before = wait_until_quiet()
        started = time.time()
        print(f"\n##### {script} attempt {attempt}, scaling before {before:.2f}x", flush=True)
        subprocess.run([sys.executable, str(HERE / script)], check=True, cwd=HERE)
        after = scaling()
        clean = before >= MIN_SCALING and after >= MIN_SCALING
        conditions[script] = {"attempts": attempt, "scaling_before": round(before, 2),
                              "scaling_after": round(after, 2), "clean": clean,
                              "seconds": round(time.time() - started)}
        conditions_file.write_text(json.dumps(conditions, indent=2) + "\n")
        print(f"##### {script} took {time.time() - started:.0f} s, scaling after {after:.2f}x, "
              f"{'clean' if clean else 'DISTURBED'}", flush=True)
        if clean:
            break
unclean = [s for s, c in conditions.items() if not c["clean"]]
print(f"\nall benchmarks finished. Disturbed: {unclean or 'none'}", flush=True)
