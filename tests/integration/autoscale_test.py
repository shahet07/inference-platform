#!/usr/bin/env python3
"""Checks the scaling policy's rules, then runs it for real: load arrives and nodes are added, load
stops and they are removed, and no request fails along the way.

Usage: autoscale_test.py SERVER_BINARY ROUTER_BINARY LOAD_GEN_BINARY MODEL_PATH FIRST_PORT
"""
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ops"))
from autoscaler import Policy, State, decide  # noqa: E402

SERVER, ROUTER, LOAD_GEN, MODEL = sys.argv[1:5]
FIRST = int(sys.argv[5])
ROUTER_PORT, NODE_PORT = FIRST, FIRST + 1

# The policy's rules.
policy = Policy(min_nodes=1, max_nodes=3, scale_down_after_ticks=2, cooldown_ticks=1)
state = State()
assert decide(policy, state, 1, 0.01, 100) == 1, "high queue wait should scale up"
assert decide(policy, state, 2, 0.01, 100) == 0, "the cooldown should hold the next reading"
assert decide(policy, state, 2, 0.01, 100) == 1
state = State()
assert decide(policy, state, 3, 0.01, 100) == 0, "must stop at the maximum"
state = State()
assert decide(policy, state, 3, 0.0001, 100) == 0, "one quiet reading must not scale down"
assert decide(policy, state, 3, 0.0001, 100) == -1
state = State()
assert decide(policy, state, 1, None, 0) == 0
assert decide(policy, state, 1, None, 0) == 0, "must stay at the minimum"
state = State()
assert decide(policy, state, 2, 0.0001, 100) == 0
assert decide(policy, state, 2, 0.001, 100) == 0, "a middling reading must reset the quiet count"
assert decide(policy, state, 2, 0.0001, 100) == 0
print("ok: policy rules")


def cluster_size():
    with urllib.request.urlopen(f"http://127.0.0.1:{ROUTER_PORT}/v1/cluster", timeout=2) as r:
        return len(json.load(r)["backends"])


def wait_for(condition, what, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if condition():
                return
        except OSError:
            pass
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


router = subprocess.Popen([ROUTER, "--port", str(ROUTER_PORT), "--backends", ""],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
scaler = subprocess.Popen(
    [sys.executable, str(ROOT / "ops" / "autoscaler.py"), "--server", SERVER, "--model", MODEL,
     "--router-port", str(ROUTER_PORT), "--first-node-port", str(NODE_PORT), "--min-nodes", "1",
     "--max-nodes", "3", "--interval-s", "0.5", "--drain-s", "0.5", "--scale-up-wait-ms", "0.2",
     "--scale-down-wait-ms", "0.05"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
load = None
try:
    wait_for(lambda: cluster_size() == 1, "the first node")
    time.sleep(1)
    assert cluster_size() == 1, "scaled up with no traffic"

    load = subprocess.Popen(
        [LOAD_GEN, "--port", str(ROUTER_PORT), "--connections", "32", "--features", "512",
         "--duration", "25", "--warmup", "0"], stdout=subprocess.PIPE, text=True)
    wait_for(lambda: cluster_size() >= 2, "scaling up under load")
    # Whether it goes on to the maximum depends on how fast this machine is, which makes it a poor thing
    # to require. The policy's own rules above check the limit.
    print("ok: nodes were added under load")

    out, _ = load.communicate(timeout=60)
    result = json.loads(out)
    load = None
    assert result["errors"] == 0, f"{result['errors']} requests failed while scaling up"
    assert result["rejected"] == 0, f"{result['rejected']} requests were rejected"
    wait_for(lambda: cluster_size() == 1, "scaling back down once idle")
    print(f"ok: {result['ok']} requests with none failed, and the cluster shrank when idle")

    # Removal under load must not fail requests either: the node drains after leaving the router.
    load = subprocess.Popen(
        [LOAD_GEN, "--port", str(ROUTER_PORT), "--connections", "32", "--features", "512",
         "--duration", "6", "--warmup", "0"], stdout=subprocess.PIPE, text=True)
    wait_for(lambda: cluster_size() >= 2, "scaling up again")
    out, _ = load.communicate(timeout=60)
    load = None
    assert json.loads(out)["errors"] == 0
    print("ok: a second burst also ran with no failures")
finally:
    if load:
        load.kill()
    scaler.terminate()
    log, _ = scaler.communicate(timeout=60)
    router.terminate()
    router.wait(timeout=30)
print(f"ok: scaling events: {len(re.findall('scaled', log))}")
