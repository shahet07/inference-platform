#!/usr/bin/env python3
"""Records flame graphs of the server before and after its tuning, and of the router, under the
same load. Uses macOS `sample`, which adds overhead: use these to see where time goes, and the
benchmarks for how long things take. Sampling counts wall-clock time, so a thread blocked in a
system call counts as busy there.
"""
import subprocess
import tempfile
import threading
from pathlib import Path

from harness import MODEL, RESULTS, ROOT, Router, Server, load

FLAMEGRAPH = ROOT / "tools" / "flamegraph.py"
CONFIGS = {
    "before": {"max_wait_us": 1000, "keep_alive_max": 100, "tcp_nodelay": 0, "events": 0},
    "after": {"events": 0},
}


def profile(pid, name):
    with tempfile.TemporaryDirectory() as tmp:
        sample_file = Path(tmp) / "sample.txt"
        subprocess.run(["sample", str(pid), "6", "-file", str(sample_file)],
                       check=True, capture_output=True)
        print(f"\n=== {name}")
        print(subprocess.run(
            ["python3", str(FLAMEGRAPH), str(sample_file), "--top", "12",
             "--svg", str(RESULTS / f"flamegraph_{name}.svg")],
            check=True, capture_output=True, text=True).stdout)


def under_load(port, pid, name, **load_flags):
    worker = threading.Thread(target=load, args=(port, 16),
                              kwargs={"duration": 11, "warmup": 1, **load_flags})
    worker.start()
    threading.Event().wait(3)
    profile(pid, name)
    worker.join()


port = 19600
for label, flags in CONFIGS.items():
    port += 1
    with Server(port, model=MODEL, workers=4, **flags) as server:
        under_load(port, server.process.pid, label)

backends = [port + 1, port + 2]
with Server(backends[0], events=0), Server(backends[1], events=0), \
        Router(port + 3, backends) as router:
    under_load(port + 3, router.process.pid, "router", routing_keys=1)
