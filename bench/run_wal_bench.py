#!/usr/bin/env python3
"""Measures what durability costs under each fsync policy, on this machine's disk."""
import json
import subprocess

from harness import BUILD, write_results

lines = subprocess.run([str(BUILD / "bench" / "wal_bench")], check=True, capture_output=True,
                       text=True).stdout.splitlines()
rows = [json.loads(line) for line in lines]
for row in rows:
    print(f"{row['policy']:>14} {row['events_per_sec']:>10.0f} ev/s  p50 {row['p50_us']:>9.1f} us  "
          f"p99 {row['p99_us']:>9.1f} us  p99.9 {row['p999_us']:>9.1f} us")
write_results("wal.json", rows)
