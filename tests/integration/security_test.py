#!/usr/bin/env python3
"""Checks API keys, roles, and rate limits on both frontends and the router, with real processes.

Usage: security_test.py SERVER_BINARY ROUTER_BINARY MODEL_PATH FIRST_PORT
"""
import hashlib
import http.client
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SERVER, ROUTER, MODEL = sys.argv[1:4]
FIRST = int(sys.argv[4])
BODY = json.dumps({"inputs": [0.5] * 64})
CLIENT_KEY, ADMIN_KEY, LIMITED_KEY = "client-key", "admin-key", "limited-key"


def call(port, method, path, body=None, key=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    connection.request(method, path, body, headers)
    response = connection.getresponse()
    data = response.read()
    connection.close()
    return response.status, data, dict(response.headers)


def wait_healthy(port):
    deadline = time.time() + 30
    while True:
        try:
            if call(port, "GET", "/healthz")[0] == 200:
                return
        except OSError:
            pass
        assert time.time() < deadline, f"port {port} never became healthy"
        time.sleep(0.05)


def check_auth(port):
    assert call(port, "GET", "/healthz")[0] == 200, "health checks must work with no key"
    status, _, headers = call(port, "POST", "/v1/infer", BODY)
    assert status == 401, status
    assert headers.get("WWW-Authenticate") == "Bearer"
    assert call(port, "POST", "/v1/infer", BODY, "wrong-key")[0] == 401
    assert call(port, "GET", "/metrics")[0] == 401
    assert call(port, "POST", "/v1/infer", BODY, CLIENT_KEY)[0] == 200
    assert call(port, "GET", "/metrics", key=CLIENT_KEY)[0] == 200


def check_rate_limit(port):
    statuses = [call(port, "POST", "/v1/infer", BODY, LIMITED_KEY) for _ in range(30)]
    codes = [s[0] for s in statuses]
    assert 200 in codes and 429 in codes, codes
    limited = next(s for s in statuses if s[0] == 429)
    assert limited[2].get("Retry-After") == "1"
    # Another key is not affected.
    assert call(port, "POST", "/v1/infer", BODY, CLIENT_KEY)[0] == 200
    time.sleep(1.1)
    assert call(port, "POST", "/v1/infer", BODY, LIMITED_KEY)[0] == 200


with tempfile.TemporaryDirectory() as tmp:
    keys = Path(tmp) / "keys.txt"
    entries = [(CLIENT_KEY, "c client 0 0"), (ADMIN_KEY, "a admin 0 0"), (LIMITED_KEY, "l client 5 5")]
    keys.write_text("".join(f"{hashlib.sha256(k.encode()).hexdigest()} {rest}\n" for k, rest in entries))
    assert CLIENT_KEY not in keys.read_text()
    (Path(tmp) / "m.onnx").write_bytes(Path(MODEL).read_bytes())

    plain, event, backend, router, second = (FIRST + i for i in range(5))
    processes = [
        subprocess.Popen([SERVER, "--model", MODEL, "--port", str(plain), "--api-keys", str(keys),
                          "--model-dir", tmp, "--model-poll-ms", "20"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
        subprocess.Popen([SERVER, "--model", MODEL, "--port", str(event), "--api-keys", str(keys),
                          "--frontend", "event", "--events", "0"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
        subprocess.Popen([SERVER, "--model", MODEL, "--port", str(backend), "--api-keys", str(keys)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
        subprocess.Popen([ROUTER, "--port", str(router), "--backends", f"127.0.0.1:{backend}",
                          "--api-keys", str(keys)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
    ]
    try:
        for port in (plain, event, backend, router):
            wait_healthy(port)

        check_auth(plain)
        check_rate_limit(plain)
        print("ok: thread-per-connection frontend requires keys and enforces rate limits")

        check_auth(event)
        check_rate_limit(event)
        print("ok: event-driven frontend requires keys and enforces rate limits")

        # Only an admin key can change what is served.
        model_body = json.dumps({"uri": "m.onnx"})
        assert call(plain, "PUT", "/v1/models/live", model_body)[0] == 401
        assert call(plain, "PUT", "/v1/models/live", model_body, CLIENT_KEY)[0] == 403
        assert call(plain, "GET", "/v1/loaded", key=CLIENT_KEY)[1].find(b"live") < 0, \
            "a client key changed what is served"
        assert call(plain, "PUT", "/v1/models/live", model_body, ADMIN_KEY)[0] == 200
        deadline = time.time() + 20
        while b"live" not in call(plain, "GET", "/v1/loaded", key=CLIENT_KEY)[1]:
            assert time.time() < deadline, "the admin's model never loaded"
            time.sleep(0.02)
        print("ok: only an admin key can replace a model")

        check_auth(router)
        # The router passes the key on, and the backend also checks it.
        backend_address = f"127.0.0.1:{backend}"
        assert call(router, "DELETE", f"/v1/backends/{backend_address}", key=CLIENT_KEY)[0] == 403
        assert call(router, "POST", "/v1/backends", json.dumps({"backend": "127.0.0.1:1"}))[0] == 401
        status, data, _ = call(router, "GET", "/v1/cluster", key=CLIENT_KEY)
        assert status == 200 and b"127.0.0.1:1\"" not in data
        print("ok: the router requires keys, and an admin key to change backends")
    finally:
        for process in processes:
            process.terminate()
        codes = [process.wait(timeout=30) for process in processes]
    assert codes == [0, 0, 0, 0], f"unclean shutdown: {codes}"

    # A bad key file must stop the server starting, rather than leave it open.
    bad = Path(tmp) / "bad.txt"
    bad.write_text("this-is-a-raw-secret\n")
    result = subprocess.run([SERVER, "--model", MODEL, "--port", str(second), "--api-keys", str(bad)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0, "a malformed key file did not stop the server"
    assert "this-is-a-raw-secret" not in result.stderr, "the error message echoed the file's contents"
    print("ok: a malformed key file stops the server, without echoing its contents")
