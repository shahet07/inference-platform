#!/usr/bin/env python3
"""End-to-end test: one request produces a connected trace, JSON logs, and valid metrics.

Usage: observability_test.py SERVER_BINARY ROUTER_BINARY MODEL_PATH FIRST_PORT
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

SERVER, ROUTER, MODEL = sys.argv[1:4]
FIRST_PORT = int(sys.argv[4])
NODE_PORT, ROUTER_PORT = FIRST_PORT, FIRST_PORT + 1
BODY = json.dumps({"inputs": [0.5] * 64}).encode()
TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
CALLER_SPAN = "00f067aa0ba902b7"


def get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as response:
        return response.read().decode()


def wait_healthy(port):
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            get(port, "/healthz")
            return
        except OSError:
            time.sleep(0.05)
    raise AssertionError(f"port {port} never became healthy")


def infer(headers):
    req = urllib.request.Request(f"http://127.0.0.1:{ROUTER_PORT}/v1/infer", data=BODY,
                                 headers=headers)
    with urllib.request.urlopen(req, timeout=10) as response:
        assert response.status == 200


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


with tempfile.TemporaryDirectory() as tmp:
    paths = {name: os.path.join(tmp, name) for name in
             ("node.spans", "router.spans", "node.log", "router.log")}
    with open(paths["node.log"], "w") as node_log, open(paths["router.log"], "w") as router_log:
        node = subprocess.Popen(
            [SERVER, "--model", MODEL, "--port", str(NODE_PORT), "--node-id", "node-a",
             "--trace-file", paths["node.spans"], "--trace-sample", "1"],
            stdout=subprocess.DEVNULL, stderr=node_log)
        router = subprocess.Popen(
            [ROUTER, "--port", str(ROUTER_PORT), "--backends", f"127.0.0.1:{NODE_PORT}",
             "--trace-file", paths["router.spans"], "--trace-sample", "0"],
            stdout=subprocess.DEVNULL, stderr=router_log)
        try:
            wait_healthy(NODE_PORT)
            wait_healthy(ROUTER_PORT)

            infer({"traceparent": f"00-{TRACE_ID}-{CALLER_SPAN}-01", "X-Routing-Key": "k"})
            infer({"traceparent": f"00-{'a' * 32}-{CALLER_SPAN}-00", "X-Routing-Key": "k"})
            infer({"traceparent": "not-a-valid-header", "X-Routing-Key": "k"})
            infer({"X-Routing-Key": "k"})

            metrics = get(NODE_PORT, "/metrics")
            router_metrics = get(ROUTER_PORT, "/metrics")
        finally:
            node.terminate()
            router.terminate()
            assert node.wait(timeout=30) == 0, "node did not shut down cleanly"
            assert router.wait(timeout=30) == 0, "router did not shut down cleanly"

    # Tracing: the sampled trace connects caller -> router -> server -> engine, across processes.
    # The node samples everything on its own, so spans from any other trace would mean it ignored
    # the router's decision not to sample those requests.
    spans = read_jsonl(paths["router.spans"]) + read_jsonl(paths["node.spans"])
    assert all(s["trace_id"] == TRACE_ID for s in spans), "an unsampled request recorded spans"
    by_name = {s["name"]: s for s in spans}
    assert set(by_name) == {"router.request", "server.request", "engine.queue_wait", "model.run"}, \
        set(by_name)
    assert by_name["router.request"]["parent_span_id"] == CALLER_SPAN
    assert by_name["server.request"]["parent_span_id"] == by_name["router.request"]["span_id"]
    for child in ("engine.queue_wait", "model.run"):
        assert by_name[child]["parent_span_id"] == by_name["server.request"]["span_id"]
    assert by_name["router.request"]["duration_us"] >= by_name["server.request"]["duration_us"]
    assert by_name["model.run"]["attributes"]["batch_size"] == "1"
    assert by_name["server.request"]["attributes"]["status"] == "200"
    assert by_name["router.request"]["attributes"]["status"] == "200"
    print("ok: one trace connects router, server, queue wait, and model run")

    # Logs: every line is JSON with the standard fields.
    for name, component in (("node.log", "node-a"), ("router.log", "router")):
        lines = read_jsonl(paths[name])
        assert lines, f"{name} is empty"
        for line in lines:
            assert {"ts_ms", "level", "component", "msg"} <= set(line), line
            assert line["component"] == component, line
        messages = {line["msg"] for line in lines}
        assert "serving" in messages or "routing" in messages, messages
    assert "stopped" in {line["msg"] for line in read_jsonl(paths["node.log"])}
    print("ok: logs are structured JSON, including startup and shutdown")

    # Metrics: every sample line is valid Prometheus text format.
    sample = re.compile(r'^[a-zA-Z_:][a-zA-Z0-9_:]*(\{[^}]*\})? -?[0-9.e+-]+(inf)?$', re.I)
    for text in (metrics, router_metrics):
        for line in text.splitlines():
            assert line.startswith("#") or sample.match(line), f"bad metric line: {line}"
    assert re.search(r"^inference_requests_total 4$", metrics, re.M)
    assert re.search(r"^router_requests_total 4$", router_metrics, re.M)
    print("ok: metrics parse as Prometheus text and count all four requests")
