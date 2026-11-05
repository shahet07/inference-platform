#!/usr/bin/env python3
"""Swaps models repeatedly under steady traffic, over the real metadata API, and checks that no
request fails, that a bad model changes nothing, and that requests run with the new version.

Usage: hot_swap_test.py SERVER_BINARY MODELS_DIR FIRST_PORT
"""
import http.client
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

SERVER, MODELS = sys.argv[1], Path(sys.argv[2])
PORT = int(sys.argv[3])
BODY = json.dumps({"inputs": [0.5] * 64}).encode()
SWAPS = 12


def call(method, path, body=None):
    request = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=body, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, error.read(), dict(error.headers)


def loaded():
    return json.loads(call("GET", "/v1/loaded")[1])["models"]


def wait_for(condition, what):
    deadline = time.time() + 60
    while time.time() < deadline:
        if condition():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


def metric(name):
    for line in call("GET", "/metrics")[1].decode().splitlines():
        if line.startswith(name + " "):
            return float(line.split()[1])
    raise KeyError(name)


with tempfile.TemporaryDirectory() as tmp:
    model_dir = Path(tmp)
    shutil.copy(MODELS / "mlp_quant.onnx", model_dir / "a.onnx")
    shutil.copy(MODELS / "mlp_float.onnx", model_dir / "b.onnx")
    (model_dir / "broken.onnx").write_bytes(b"this is not a model")

    server = subprocess.Popen(
        [SERVER, "--model", str(MODELS / "mlp_quant.onnx"), "--port", str(PORT),
         "--model-dir", str(model_dir), "--model-poll-ms", "5", "--events", "0"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    stop = threading.Event()
    statuses, versions = [], set()

    def traffic():
        # A kept-alive connection, because a new one per request would run out of ephemeral ports.
        connection = http.client.HTTPConnection("127.0.0.1", PORT, timeout=30)
        while not stop.is_set():
            connection.request("POST", "/v1/models/live/infer", BODY)
            response = connection.getresponse()
            response.read()
            statuses.append(response.status)
            if response.status == 200:
                versions.add(int(response.headers["X-Model-Version"]))
        connection.close()

    try:
        deadline = time.time() + 30
        while True:
            try:
                call("GET", "/healthz")
                break
            except OSError:
                assert time.time() < deadline
                time.sleep(0.05)

        call("PUT", "/v1/models/live", json.dumps({"uri": "a.onnx"}).encode())
        wait_for(lambda: loaded().get("live") == 1, "first load")

        threads = [threading.Thread(target=traffic) for _ in range(4)]
        for thread in threads:
            thread.start()
        for i in range(SWAPS):
            uri = "b.onnx" if i % 2 == 0 else "a.onnx"
            status, body, _ = call("PUT", "/v1/models/live", json.dumps({"uri": uri}).encode())
            assert status == 200, body
            want = json.loads(body)["version"]
            wait_for(lambda v=want: loaded()["live"] == v, f"version {want}")
            time.sleep(0.03)

        before = loaded()["live"]
        call("PUT", "/v1/models/live", json.dumps({"uri": "broken.onnx"}).encode())
        wait_for(lambda: metric("model_load_failures_total") >= 1, "the bad model to be rejected")
        time.sleep(0.05)
        assert loaded()["live"] == before, "a broken model replaced the live one"

        status, _, _ = call("PUT", "/v1/models/evil", json.dumps({"uri": "../a.onnx"}).encode())
        wait_for(lambda: metric("model_load_failures_total") >= 2, "the escaping path's rejection")
        assert "evil" not in loaded(), "a path outside the model directory was loaded"

        stop.set()
        for thread in threads:
            thread.join()
        failed = [s for s in statuses if s != 200]
        assert not failed, f"{len(failed)} of {len(statuses)} requests failed: {set(failed)}"
        assert len(versions) >= 3, f"traffic only saw versions {versions}"
        print(f"ok: {SWAPS} swaps under load, {len(statuses)} requests, none failed, "
              f"{len(versions)} versions served; a broken model and an escaping path were refused")
    finally:
        stop.set()
        server.terminate()
        assert server.wait(timeout=30) == 0, "server did not shut down cleanly"
