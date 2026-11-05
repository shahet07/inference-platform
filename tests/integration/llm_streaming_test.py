#!/usr/bin/env python3
"""Checks token streaming over real HTTP, using the fake backend, so that no model download is needed.

Usage: llm_streaming_test.py LLM_SERVER_BINARY PORT
"""
import http.client
import json
import re
import subprocess
import sys
import threading
import time

BINARY = sys.argv[1]
PORT = int(sys.argv[2])
STEP_US = 5000


def metric(name):
    connection = http.client.HTTPConnection("127.0.0.1", PORT, timeout=5)
    connection.request("GET", "/metrics")
    text = connection.getresponse().read().decode()
    match = re.search(rf"^{name} (\S+)$", text, re.M)
    return float(match.group(1))


def post(body):
    connection = http.client.HTTPConnection("127.0.0.1", PORT, timeout=30)
    connection.request("POST", "/v1/generate", json.dumps(body), {"Content-Type": "application/json"})
    return connection, connection.getresponse()


def events(response):
    """Yields each server-sent event as it arrives, with the time it arrived."""
    for line in response:
        if line.startswith(b"data: "):
            yield time.time(), json.loads(line[6:])


server = subprocess.Popen([BINARY, "--fake-step-us", str(STEP_US), "--port", str(PORT), "--slots", "2",
                           "--queue", "4", "--context", "100000"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    deadline = time.time() + 20
    while True:
        try:
            connection = http.client.HTTPConnection("127.0.0.1", PORT, timeout=2)
            connection.request("GET", "/healthz")
            assert connection.getresponse().status == 200
            break
        except OSError:
            assert time.time() < deadline, "the server never started"
            time.sleep(0.05)

    # Without streaming, everything arrives at once.
    _, response = post({"prompt": "a b c", "max_tokens": 3})
    result = json.loads(response.read())
    assert result == {"text": "t3 t4 t5 ", "finish_reason": "max_tokens"}, result

    # With streaming, tokens must arrive over time, and not all together at the end.
    _, response = post({"prompt": "a b c", "max_tokens": 20, "stream": True})
    assert response.getheader("Content-Type").startswith("text/event-stream")
    received = list(events(response))
    tokens = [e for _, e in received if "token" in e]
    assert [t["token"] for t in tokens] == [f"t{i} " for i in range(3, 23)], tokens
    assert received[-1][1] == {"done": True, "finish_reason": "max_tokens"}
    spread = received[-1][0] - received[0][0]
    assert spread > 0.05, f"20 tokens at 5 ms each all arrived within {spread * 1000:.1f} ms"
    print(f"ok: tokens streamed as generated, over {spread * 1000:.0f} ms")

    # Bad requests.
    for body in ({"max_tokens": 1}, {"prompt": "a", "max_tokens": 0}, {"prompt": "a", "max_tokens": -5}):
        _, response = post(body)
        response.read()
        assert response.status == 400, (body, response.status)
    _, response = post({"prompt": "", "max_tokens": 5})
    response.read()
    assert response.status == 400, "an empty prompt should be an error"
    print("ok: bad requests are refused")

    # A client that vanishes must free its slot. With two slots, two vanished clients that kept their
    # slots would block this next request for a very long time.
    before = metric("llm_cancelled_total")
    for _ in range(2):
        connection, response = post({"prompt": "a", "max_tokens": 1000000, "stream": True})
        next(events(response))
        connection.sock.close()
        connection.close()
    deadline = time.time() + 15
    while metric("llm_cancelled_total") < before + 2:
        assert time.time() < deadline, "a vanished client kept generating"
        time.sleep(0.02)
    started = time.time()
    _, response = post({"prompt": "a", "max_tokens": 2})
    assert json.loads(response.read())["finish_reason"] == "max_tokens"
    assert time.time() - started < 5
    print("ok: clients that disconnect free their slots")

    # A short request joins a long one's batch and finishes first.
    connection, long_response = post({"prompt": "a", "max_tokens": 400, "stream": True})
    next(events(long_response))
    started = time.time()
    _, response = post({"prompt": "a", "max_tokens": 3})
    response.read()
    short_took = time.time() - started
    assert short_took < 1.0, f"a short request waited {short_took:.2f} s behind a long one"
    connection.sock.close()
    print(f"ok: a short request finished in {short_took * 1000:.0f} ms alongside a long one")

    # A full queue is refused, not left to wait forever.
    time.sleep(0.2)
    hold = []
    statuses = []
    lock = threading.Lock()

    def flood():
        connection, response = post({"prompt": "a", "max_tokens": 200})
        with lock:
            statuses.append(response.status)
        hold.append(connection)
        response.read()

    threads = [threading.Thread(target=flood) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert 503 in statuses and 200 in statuses, statuses
    print(f"ok: when the queue fills, requests get 503: {sorted(set(statuses))}")
finally:
    server.terminate()
    assert server.wait(timeout=30) == 0, "the server did not shut down cleanly"
print("ok: clean shutdown")
