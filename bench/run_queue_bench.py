#!/usr/bin/env python3
"""Compares the mutex queue with the lock-free queue."""
import json
import subprocess

from harness import BUILD, write_results

rows = json.loads(subprocess.run([str(BUILD / "bench" / "queue_bench")], check=True,
                                 capture_output=True, text=True).stdout)
for row in rows:
    print(f"{row['producers_and_consumers']:>2} pairs  mutex {row['mutex_items_per_sec'] / 1e6:>7.2f} M/s  "
          f"lock-free {row['lock_free_items_per_sec'] / 1e6:>7.2f} M/s  "
          f"{row['lock_free_items_per_sec'] / row['mutex_items_per_sec']:.2f}x")
write_results("queue.json", rows)
