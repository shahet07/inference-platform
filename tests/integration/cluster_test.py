#!/usr/bin/env python3
"""End-to-end test with real processes: routing, failover, recovery, and metadata replication.

Usage: cluster_test.py SERVER_BINARY ROUTER_BINARY MODEL_PATH FIRST_PORT
"""
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

SERVER, ROUTER, MODEL = sys.argv[1:4]
FIRST_PORT = int(sys.argv[4])
NODE_PORTS = [FIRST_PORT + i for i in range(3)]
ROUTER_PORT = FIRST_PORT + 3
BODY = json.dumps({"inputs": [0.5] * 64}).encode()


def request(port, path, data=None, headers=None, method=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()


def infer(key):
    return request(ROUTER_PORT, "/v1/infer", BODY, {"X-Routing-Key": key})


def wait_for(condition, what, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if condition():
                return
        except OSError:
            pass
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def start_node(index):
    peers = ",".join(f"127.0.0.1:{p}" for i, p in enumerate(NODE_PORTS) if i != index)
    return subprocess.Popen([SERVER, "--model", MODEL, "--port", str(NODE_PORTS[index]),
                             "--node-id", f"node-{index}", "--peers", peers,
                             "--replication-interval-ms", "50"], stdout=subprocess.DEVNULL)


def healthy_count():
    backends = json.loads(request(ROUTER_PORT, "/v1/cluster")[2])["backends"]
    return sum(b["healthy"] for b in backends)


def models(port):
    return json.loads(request(port, "/v1/models")[2])["models"]


processes = [start_node(i) for i in range(3)]
router = subprocess.Popen([ROUTER, "--port", str(ROUTER_PORT), "--backends",
                           ",".join(f"127.0.0.1:{p}" for p in NODE_PORTS)],
                          stdout=subprocess.DEVNULL)
try:
    for port in [*NODE_PORTS, ROUTER_PORT]:
        wait_for(lambda p=port: request(p, "/healthz")[0] == 200, f"port {port}")

    # Routing: keys spread out, and each key stays on one node.
    owners = {}
    for i in range(60):
        key = f"key-{i}"
        status, headers, _ = infer(key)
        assert status == 200, status
        owners[key] = headers["X-Served-By"]
        if infer(key)[1]["X-Served-By"] != owners[key]:
            wait_for(lambda k=key: infer(k)[1]["X-Served-By"] == owners[k],
                     f"{key} to return to {owners[key]}")
    assert len(set(owners.values())) == 3, f"keys did not reach all nodes: {set(owners.values())}"
    print("ok: consistent routing across 3 nodes")

    # Metadata: a write on one node reaches the others.
    status, _, _ = request(NODE_PORTS[0], "/v1/models/resnet", json.dumps(
        {"uri": "s3://models/resnet-v1.onnx"}).encode(), method="PUT")
    assert status == 200, status
    for port in NODE_PORTS[1:]:
        wait_for(lambda p=port: len(models(p)) == 1, f"replication to {port}")
    print("ok: metadata replicated to all nodes")

    # Failure: every request still succeeds, retried on another node when needed.
    victim = 2
    victim_backend = f"127.0.0.1:{NODE_PORTS[victim]}"
    processes[victim].kill()
    processes[victim].wait()
    moved = temporary_moves = 0
    for key, owner in owners.items():
        status, headers, body = infer(key)
        assert status == 200, (status, body)
        assert headers["X-Served-By"] != victim_backend
        if owner != victim_backend:
            if headers["X-Served-By"] != owner:
                # A slow machine can miss a health probe, which rightly moves a key for a moment. A key
                # that stays moved would mean the ring is wrong, so it must come back by itself.
                temporary_moves += 1
                view = request(ROUTER_PORT, "/v1/cluster")[2]
                wait_for(lambda k=key, o=owner: infer(k)[1]["X-Served-By"] == o,
                         f"{key} to return to {owner}, with the router seeing {view}")
        else:
            moved += 1
    assert moved > 0
    wait_for(lambda: healthy_count() == 2, "the failed node to leave the rotation")
    print(f"ok: survived a node kill with no failed requests; {moved} of {len(owners)} keys moved, "
          f"and {temporary_moves} others moved for a moment")

    # A write made while the node is down.
    request(NODE_PORTS[0], "/v1/models/bert", json.dumps({"uri": "s3://models/bert.onnx"}).encode(),
            method="PUT")

    # Recovery: the node returns, gets its keys back, and catches up on metadata.
    processes[victim] = start_node(victim)
    wait_for(lambda: healthy_count() == 3, "the node to rejoin the rotation")
    for key, owner in owners.items():
        assert infer(key)[1]["X-Served-By"] == owner, "a key did not return to its owner"
    wait_for(lambda: len(models(NODE_PORTS[victim])) == 2, "the restarted node to catch up")
    print("ok: node recovered, keys returned, metadata caught up")

    # Total failure: requests fail fast with a clear status rather than hanging.
    for process in processes:
        process.kill()
        process.wait()
    status, _, _ = infer("key-0")
    assert status == 503, status
    wait_for(lambda: request(ROUTER_PORT, "/healthz")[0] == 503, "the router to report unhealthy")
    print("ok: with no nodes, the router returns 503")
finally:
    for process in [*processes, router]:
        process.kill()
        process.wait()
