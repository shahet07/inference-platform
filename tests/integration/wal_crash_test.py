#!/usr/bin/env python3
"""Kills a writer with SIGKILL at random moments, and checks that no acknowledged event was lost.

A line reaches the writer's stdout only after its append returned. This tests a process crash. It
says nothing about a power failure, which needs fault-injection hardware or a virtual disk.

Usage: wal_crash_test.py WRITER_BINARY
"""
import random
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

WRITER = sys.argv[1]
ROUNDS = 12


def collect_acks(process, acks):
    for line in process.stdout:
        parts = line.split()
        if len(parts) == 3 and line.endswith("\n"):
            acks.append(tuple(int(p) for p in parts))


def dump(directory):
    result = subprocess.run([WRITER, "dump", directory], capture_output=True, text=True, check=True)
    stored, ends, truncated = {}, {}, 0
    for line in result.stdout.splitlines():
        parts = line.split()
        if parts[0] == "truncated":
            truncated = int(parts[1])
        elif parts[0] == "next":
            ends[int(parts[1])] = int(parts[2])
        else:
            stored[(int(parts[0]), int(parts[1]))] = int(float(parts[2]))
    return stored, ends, truncated


with tempfile.TemporaryDirectory() as tmp:
    directory = str(Path(tmp) / "log")
    rng = random.Random(1234)
    total_acked = total_checked = 0
    truncations = 0
    latest = {}
    for round_number in range(ROUNDS):
        process = subprocess.Popen([WRITER, "write", directory], stdout=subprocess.PIPE, text=True)
        acks = []
        reader = threading.Thread(target=collect_acks, args=(process, acks))
        reader.start()
        # Starting can take a while on a slow machine, and killing it before it has done anything would
        # test nothing.
        deadline = time.time() + 60
        while not acks and time.time() < deadline:
            time.sleep(0.005)
        time.sleep(rng.uniform(0.05, 0.4))
        process.send_signal(signal.SIGKILL)
        process.wait()
        reader.join()
        assert process.returncode == -signal.SIGKILL
        assert acks, "the writer acknowledged nothing"
        total_acked += len(acks)
        for shard, offset, value in acks:
            latest[(shard, offset)] = value

        stored, ends, truncated = dump(directory)
        truncations += truncated > 0
        for (shard, offset), value in latest.items():
            assert offset < ends[shard], f"lost acknowledged offset {offset} on shard {shard}"
            # Only the newest events stay in memory, so older ones may no longer be in this dump.
            if (shard, offset) in stored:
                assert stored[(shard, offset)] == value, f"wrong value at {shard}/{offset}"
                total_checked += 1
        # Offsets must be consecutive, with no holes, up to each shard's end.
        for shard, end in ends.items():
            retained = sorted(o for s, o in stored if s == shard)
            assert retained == list(range(end - len(retained), end)), f"hole in shard {shard}"
    print(f"ok: {ROUNDS} SIGKILLs, {total_acked} acknowledged events, none lost, "
          f"{total_checked} values verified, {truncations} recoveries removed a torn tail")
