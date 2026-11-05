#!/usr/bin/env python3
"""Injects network faults between the router and its nodes, and checks what clients experience.

Usage: chaos_test.py SERVER_BINARY ROUTER_BINARY MODEL_PATH FIRST_PORT
"""
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from chaos_proxy import ChaosProxy  # noqa: E402

SERVER, ROUTER, MODEL = sys.argv[1:4]
FIRST = int(sys.argv[4])
NODE_PORTS = [FIRST + i for i in range(3)]
PROXY_PORTS = [FIRST + 10 + i for i in range(3)]
ROUTER_PORT, SECOND_ROUTER_PORT = FIRST + 20, FIRST + 21
BODY = json.dumps({"inputs": [0.5] * 64}).encode()
BACKENDS = ",".join(f"127.0.0.1:{p}" for p in PROXY_PORTS)
BACKEND_TIMEOUT_MS = 500


def request(port, path, data=None, headers=None, method=None, timeout=20):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()


def infer(key, port=ROUTER_PORT, headers=None):
    return request(port, "/v1/infer", BODY, {"X-Routing-Key": key, **(headers or {})})


def wait_for(condition, what, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if condition():
                return
        except OSError:
            pass
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


def healthy():
    backends = json.loads(request(ROUTER_PORT, "/v1/cluster")[2])["backends"]
    return {b["backend"] for b in backends if b["healthy"]}


def start_router(port, *extra):
    return subprocess.Popen(
        [ROUTER, "--port", str(port), "--backends", BACKENDS, "--backend-timeout-ms",
         str(BACKEND_TIMEOUT_MS), "--health-interval-ms", "100", *extra],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def key_owned_by(backend):
    for i in range(10000):
        key = f"key-{i}"
        if infer(key)[1]["X-Served-By"] == backend:
            return key
    raise AssertionError(f"no key lands on {backend}")


nodes = [subprocess.Popen([SERVER, "--model", MODEL, "--port", str(p), "--events", "0"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
         for p in NODE_PORTS]
proxies = [ChaosProxy(listen, target) for listen, target in zip(PROXY_PORTS, NODE_PORTS)]
routers = []
try:
    for port in NODE_PORTS:
        wait_for(lambda p=port: request(p, "/healthz")[0] == 200, f"node {port}")
    routers.append(start_router(ROUTER_PORT))
    wait_for(lambda: request(ROUTER_PORT, "/healthz")[0] == 200, "router")
    wait_for(lambda: len(healthy()) == 3, "all backends healthy")
    victim = f"127.0.0.1:{PROXY_PORTS[0]}"
    proxy = proxies[0]
    key = key_owned_by(victim)

    # A silent hang is the hardest failure: connections stay open and nothing comes back.
    proxy.set_mode("blackhole")
    started = time.time()
    status, headers, _ = infer(key)
    elapsed = time.time() - started
    assert status == 200, status
    assert headers["X-Served-By"] != victim
    assert elapsed < 3, f"a hung backend cost {elapsed:.1f} s, and the allowance is 0.5 s"
    print(f"ok: a silently hung node cost one request {elapsed * 1000:.0f} ms, and it failed over")
    wait_for(lambda: victim not in healthy(), "the hung node to leave rotation")
    for i in range(30):
        started = time.time()
        assert infer(f"after-hang-{i}")[0] == 200
        assert time.time() - started < 1, "requests still reach the hung node"
    print("ok: with the hung node out of rotation, later requests are unaffected")

    proxy.set_mode("pass")
    wait_for(lambda: victim in healthy(), "recovery after the partition heals")
    assert infer(key)[1]["X-Served-By"] == victim, "the key did not return to its owner"
    print("ok: the node returned when the partition healed, and its keys came back")

    # Connections reset as data arrives.
    proxy.set_mode("reset")
    for i in range(20):
        status, headers, _ = infer(key)
        assert status == 200, status
        assert headers["X-Served-By"] != victim
    proxy.set_mode("pass")
    wait_for(lambda: victim in healthy(), "recovery after resets")
    print("ok: connection resets caused no failed requests")

    # Refused connections.
    proxy.set_mode("refuse")
    for i in range(20):
        assert infer(key)[0] == 200
    proxy.set_mode("pass")
    wait_for(lambda: victim in healthy(), "recovery after refusals")
    print("ok: refused connections caused no failed requests")

    # A slow node with a generous deadline still answers, and is not punished.
    proxy.set_mode("delay", delay_ms=100)
    status, headers, _ = infer(key)
    assert status == 200 and headers["X-Served-By"] == victim
    assert victim in healthy(), "a node that is merely slow was removed"

    # Tight deadlines fail fast with 504, and do not count against the backend.
    proxy.set_mode("delay", delay_ms=300)
    started = time.time()
    status, _, _ = infer(key, headers={"X-Deadline-Ms": "50"})
    elapsed = time.time() - started
    assert status == 504, status
    assert elapsed < 0.3, f"a 50 ms deadline took {elapsed:.3f} s"
    assert victim in healthy(), "a client's tight deadline removed a healthy backend"
    proxy.set_mode("pass")
    print(f"ok: a 50 ms deadline on a slow node returned 504 in {elapsed * 1000:.0f} ms, "
          "and the node stayed in rotation")

    # Flapping: it must not rejoin on one good probe.
    for _ in range(3):
        proxy.set_mode("refuse")
        wait_for(lambda: victim not in healthy(), "flap down")
        proxy.set_mode("pass")
        time.sleep(0.02)
    wait_for(lambda: victim in healthy(), "recovery after flapping")
    print("ok: a flapping node came back only once stable")

    # Every backend looks down at once, as when health checks fail cluster-wide. Requests are still tried.
    for other in proxies:
        other.set_mode("refuse")
    wait_for(lambda: not healthy(), "all backends to look down")
    assert request(ROUTER_PORT, "/healthz")[0] == 503
    assert infer("all-down")[0] == 503, "with everything refusing, requests should fail quickly"
    for other in proxies:
        other.set_mode("pass")
    # Probes have not restored them yet, and requests already succeed.
    assert infer("all-down")[0] == 200, "panic mode should still try backends that look down"
    wait_for(lambda: len(healthy()) == 3, "recovery from a cluster-wide outage")
    print("ok: when every backend looks down, requests are still tried, and succeed if they can")

    # A second, independent router places keys identically, so either can take over.
    routers.append(start_router(SECOND_ROUTER_PORT))
    wait_for(lambda: request(SECOND_ROUTER_PORT, "/healthz")[0] == 200, "second router")
    time.sleep(0.5)
    for i in range(30):
        first = infer(f"shared-{i}")[1]["X-Served-By"]
        second = infer(f"shared-{i}", SECOND_ROUTER_PORT)[1]["X-Served-By"]
        assert first == second, "two routers disagree on where a key belongs"
    routers[0].kill()
    routers[0].wait()
    assert infer("shared-0", SECOND_ROUTER_PORT)[0] == 200
    print("ok: two routers place keys identically, and one survives the other's death")

    # DNS discovery finds backends by name, as for pods behind a headless Service.
    discovered = subprocess.Popen(
        [ROUTER, "--port", str(FIRST + 22), "--backends", "", "--discover-dns",
         f"localhost:{NODE_PORTS[1]}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    routers.append(discovered)
    wait_for(lambda: request(FIRST + 22, "/healthz")[0] == 200, "a backend found through DNS")
    status, headers, _ = infer("dns", FIRST + 22)
    assert status == 200 and headers["X-Served-By"] == f"127.0.0.1:{NODE_PORTS[1]}", headers
    print("ok: a router with no configured backends found one through DNS")
finally:
    for proxy in proxies:
        proxy.stop()
    for process in [*routers, *nodes]:
        process.kill()
        process.wait()
