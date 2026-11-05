#!/usr/bin/env python3
"""Kills a worker with SIGKILL until it manages to finish, then checks that every total is exact.

The producer retries all events from the beginning after each restart, and the consumer replays
from its last checkpoint, so double counting or loss would show up in the totals. This tests process
crashes, not power loss.

Usage: exactly_once_test.py WORKER_BINARY
"""
import collections
import random
import signal
import subprocess
import sys
import tempfile
import time

WORKER = sys.argv[1]
EVENTS, KEYS = 150000, 50
MAX_KILLS = 40

expected_count, expected_sum = collections.Counter(), collections.Counter()
for i in range(1, EVENTS + 1):
    expected_count[f"key-{i % KEYS}"] += 1
    expected_sum[f"key-{i % KEYS}"] += i % 7

rng = random.Random(5)
with tempfile.TemporaryDirectory() as directory:
    kills = 0
    while True:
        process = subprocess.Popen([WORKER, directory], stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        if kills < MAX_KILLS:
            try:
                out, err = process.communicate(timeout=rng.uniform(0.05, 0.5))
            except subprocess.TimeoutExpired:
                process.send_signal(signal.SIGKILL)
                process.communicate()
                kills += 1
                continue
        else:
            out, err = process.communicate(timeout=120)
        assert process.returncode == 0, f"worker failed: {err[-400:]}"
        break

lines = out.splitlines()
total, total_sum, duplicates, recovered = lines[-1].split()[1:]
actual_count, actual_sum = {}, {}
for line in lines[:-1]:
    key, count, value = line.split()
    assert key not in actual_count, "a key appeared on two shards"
    actual_count[key], actual_sum[key] = int(count), float(value)
assert kills >= 3, f"only {kills} kills, so this proves little"
assert actual_count == dict(expected_count), "counts differ from the exact expected values"
for key, value in expected_sum.items():
    assert actual_sum[key] == value, f"sum differs for {key}"
assert int(total) == EVENTS
print(f"ok: {kills} SIGKILLs, then {EVENTS} events counted exactly once across {KEYS} keys; "
      f"the last run alone recognized {duplicates} retries and recovered {recovered} events")
