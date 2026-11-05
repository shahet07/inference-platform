#!/usr/bin/env python3
"""Checks the event-driven frontend against a real server, model, and load generator.

Usage: event_frontend_test.py SERVER_BINARY LOAD_GEN_BINARY MODEL_PATH PORT
"""
import http.client
import json
import socket
import struct
import subprocess
import sys
import time

SERVER, LOAD_GEN, MODEL = sys.argv[1:4]
PORT = int(sys.argv[4])
JSON_BODY = json.dumps({"inputs": [0.5] * 64})


def call(method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", PORT, timeout=10)
    connection.request(method, path, body, headers or {})
    response = connection.getresponse()
    data = response.read()
    connection.close()
    return response.status, data, dict(response.headers)


server = subprocess.Popen([SERVER, "--model", MODEL, "--port", str(PORT), "--frontend", "event",
                           "--event-threads", "2", "--events", "0"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    deadline = time.time() + 30
    while True:
        try:
            assert call("GET", "/healthz")[0] == 200
            break
        except OSError:
            assert time.time() < deadline, "the server never started"
            time.sleep(0.05)

    # Only the thread-per-connection frontend has this, so a 404 shows which one is answering. Without
    # this, a stale binary that ignored --frontend would pass everything below.
    assert call("GET", "/v1/loaded")[0] == 404, "this is not the event frontend"

    status, data, headers = call("POST", "/v1/infer", JSON_BODY, {"Content-Type": "application/json"})
    assert status == 200, status
    json_outputs = json.loads(data)["outputs"]
    assert len(json_outputs) == 10
    assert headers["X-Model-Version"] == "0"

    binary_body = struct.pack("<64f", *([0.5] * 64))
    status, data, _ = call("POST", "/v1/infer", binary_body,
                           {"Content-Type": "application/octet-stream"})
    assert status == 200, status
    binary_outputs = struct.unpack("<10f", data)
    assert all(abs(a - b) < 1e-5 for a, b in zip(json_outputs, binary_outputs)), \
        "JSON and raw floats gave different answers"
    print("ok: JSON and raw floats give the same answers")

    assert call("POST", "/v1/infer", "not json")[0] == 400
    assert call("POST", "/v1/infer", '{"inputs":[1]}')[0] == 400
    assert call("POST", "/v1/infer", b"abc", {"Content-Type": "application/octet-stream"})[0] == 400
    assert call("POST", "/v1/models/nope/infer", JSON_BODY)[0] == 404
    assert call("GET", "/nope")[0] == 404
    assert call("POST", "/v1/infer", JSON_BODY, {"X-Deadline-Ms": "abc"})[0] == 400
    assert call("POST", "/v1/infer", JSON_BODY, {"X-Deadline-Ms": "0"})[0] == 504
    assert call("POST", "/v1/infer", JSON_BODY, {"X-Deadline-Ms": "60000"})[0] == 200
    print("ok: bad requests, unknown models, and deadlines get the right statuses")

    # Connections that vanish with requests in flight must not break anything.
    for _ in range(200):
        # The OS can refuse a connection when this many arrive this fast, and that is not a failure
        # here. The server's clean exit below also shows that sanitizers reported nothing.
        try:
            sock = socket.create_connection(("127.0.0.1", PORT))
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            request = (f"POST /v1/infer HTTP/1.1\r\nContent-Length: {len(JSON_BODY)}\r\n\r\n"
                       f"{JSON_BODY}").encode()
            sock.sendall(request)
            sock.close()
        except OSError:
            pass
    # The burst can fill the OS's accept queue for a moment, so healthy means answering again soon.
    deadline = time.time() + 20
    while True:
        try:
            assert call("GET", "/healthz")[0] == 200
            break
        except OSError:
            assert time.time() < deadline, "the server did not recover from aborted connections"
            time.sleep(0.05)
    print("ok: 200 aborted connections left the server healthy")

    run = subprocess.run(
        [LOAD_GEN, "--port", str(PORT), "--connections", "64", "--duration", "3", "--warmup", "0"],
        capture_output=True, text=True)
    assert run.returncode == 0, f"load generator exited {run.returncode}: {run.stderr[-1500:]}"
    result = json.loads(run.stdout)
    assert result["ok"] > 0
    assert result["errors"] == 0, result
    assert result["rejected"] == 0, result
    print(f"ok: {result['ok']} requests under load, none failed")

    text = call("GET", "/metrics")[1].decode()
    assert "inference_requests_total" in text and "http_request_duration_seconds_count" in text
finally:
    server.terminate()
    assert server.wait(timeout=30) == 0, "the server did not shut down cleanly"
print("ok: clean shutdown")
