#!/usr/bin/env python3
"""Runs a real 3-node cluster, and kills the leader with SIGKILL repeatedly under live writes.

Afterwards every node must hold an identical log containing every acknowledged write. Runs without
fsync, so this covers process crashes and not power loss. Clients retry when an outcome is unknown,
as real ones would, so a command may legitimately appear twice. Acknowledged ones must never vanish.

Usage: raft_test.py RAFT_NODE_BINARY FIRST_PORT
"""
import http.client
import json
import random
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

BINARY = sys.argv[1]
FIRST = int(sys.argv[2])
PORTS = {i: FIRST + i for i in (1, 2, 3)}
CLUSTER = ",".join(f"{i}=127.0.0.1:{p}" for i, p in PORTS.items())
KILLS = 6


def get(port, path):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    connection.request("GET", path)
    response = connection.getresponse()
    return json.loads(response.read())


def get_retrying(port, path):
    """For a node that may still be starting."""
    deadline = time.time() + 15
    while True:
        try:
            return get(port, path)
        except (OSError, http.client.HTTPException, ValueError):
            if time.time() > deadline:
                raise
            time.sleep(0.02)


with tempfile.TemporaryDirectory() as tmp:
    processes = {}

    def start(node):
        processes[node] = subprocess.Popen(
            [BINARY, "--id", str(node), "--cluster", CLUSTER, "--dir", str(Path(tmp) / str(node)),
             "--fsync", "0"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def find_leader(timeout=15):
        deadline = time.time() + timeout
        while time.time() < deadline:
            for node, port in PORTS.items():
                if processes[node].poll() is not None:
                    continue
                try:
                    if get(port, "/status")["role"] == "leader":
                        return node
                except (OSError, http.client.HTTPException, ValueError):
                    pass
            time.sleep(0.02)
        raise AssertionError("no leader was elected")

    stop = threading.Event()
    acked, acked_lock = [], threading.Lock()
    unknown = [0]

    def writer(number):
        sequence = 0
        port = PORTS[1 + number % 3]
        connection = None
        while not stop.is_set():
            command = f"w{number}-{sequence}"
            try:
                if connection is None:
                    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=6)
                connection.request("POST", "/append", command)
                response = connection.getresponse()
                body = response.read()
                if response.status == 200:
                    with acked_lock:
                        acked.append((command, json.loads(body)["index"]))
                    sequence += 1
                elif response.status == 409:
                    leader = response.getheader("X-Leader")
                    if leader:
                        port = int(leader.split(":")[1])
                        connection.close()
                        connection = None
                    else:
                        time.sleep(0.02)
                else:
                    unknown[0] += 1
                    sequence += 1
            except (OSError, http.client.HTTPException):
                if connection:
                    connection.close()
                connection = None
                port = random.choice(list(PORTS.values()))
                time.sleep(0.02)
                sequence += 1

    try:
        for node in PORTS:
            start(node)
        first_leader = find_leader()
        threads = [threading.Thread(target=writer, args=(i,)) for i in range(6)]
        for thread in threads:
            thread.start()

        failovers = []
        rng = random.Random(11)
        for _ in range(KILLS):
            time.sleep(rng.uniform(0.3, 0.8))
            leader = find_leader()
            processes[leader].send_signal(signal.SIGKILL)
            processes[leader].wait()
            started = time.time()
            with acked_lock:
                before = len(acked)
            while True:
                with acked_lock:
                    if len(acked) > before + 3:
                        break
                assert time.time() - started < 20, "writes never resumed after the leader died"
                time.sleep(0.005)
            failovers.append(time.time() - started)
            start(leader)
        stop.set()
        for thread in threads:
            thread.join()

        # Every node catches up, once all are back.
        deadline = time.time() + 20
        while time.time() < deadline:
            status = [get_retrying(p, "/status") for p in PORTS.values()]
            if len({s["commit"] for s in status}) == 1 and len({s["last"] for s in status}) == 1:
                break
            time.sleep(0.05)
        logs = {node: get_retrying(port, "/log?from=1")["entries"] for node, port in PORTS.items()}
        assert logs[1] == logs[2] == logs[3], "the nodes' logs differ"
        positions = {tuple(e)[0]: e[2] for e in logs[1]}
        for command, index in acked:
            assert positions.get(index) == command, (
                f"acknowledged {command} at {index}, but the log has {positions.get(index)}")
        print(f"ok: {KILLS} leader kills, {len(acked)} acknowledged writes, none lost, and 3 identical "
              f"logs of {len(logs[1])} entries; {unknown[0]} writes had an unknown outcome")
        print(f"ok: writes resumed after a leader died in {min(failovers) * 1000:.0f} to "
              f"{max(failovers) * 1000:.0f} ms")

        # With only one of three nodes alive, no write can be acknowledged.
        for node in list(PORTS)[1:]:
            processes[node].send_signal(signal.SIGKILL)
            processes[node].wait()
        get_retrying(PORTS[1], "/status")
        connection = http.client.HTTPConnection("127.0.0.1", PORTS[1], timeout=8)
        connection.request("POST", "/append", "must-not-commit")
        response = connection.getresponse()
        response.read()
        assert response.status != 200, "a write was acknowledged without a majority"
        for node in list(PORTS)[1:]:
            start(node)
        find_leader()
        print("ok: a lone node refused a write, and the cluster recovered when the others returned")

        connection = http.client.HTTPConnection("127.0.0.1", PORTS[1], timeout=8)
        connection.request("POST", "/append", b"\xff\xfe")
        response = connection.getresponse()
        response.read()
        assert response.status == 400, response.status
    finally:
        stop.set()
        for process in processes.values():
            process.kill()
            process.wait()
