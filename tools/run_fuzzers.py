#!/usr/bin/env python3
"""Runs each libFuzzer target for a fixed time, starting from the committed seeds.

New corpus entries go to a temporary directory, so a run leaves the repo unchanged. A failure
leaves a crash-* file in the working directory, which can be replayed by passing it to the target.
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGETS = {"fuzz_traceparent": "traceparent", "fuzz_escape": "escape",
           "fuzz_registry": "registry", "fuzz_window": "window"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--build", default=str(ROOT / "build" / "fuzz"))
    args = parser.parse_args()

    failed = []
    for target, corpus in TARGETS.items():
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / corpus
            shutil.copytree(ROOT / "fuzz" / "corpus" / corpus, work)
            result = subprocess.run(
                [str(Path(args.build) / "fuzz" / target), str(work),
                 f"-max_total_time={args.seconds}", "-max_len=4096", "-print_final_stats=1"],
                capture_output=True, text=True)
        runs = [l for l in result.stderr.splitlines() if "stat::number_of_executed_units" in l]
        print(f"{target}: exit {result.returncode}, {runs[-1].split()[-1] if runs else '?'} runs",
              flush=True)
        if result.returncode != 0:
            failed.append(target)
            print("\n".join(result.stderr.splitlines()[-40:]))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
