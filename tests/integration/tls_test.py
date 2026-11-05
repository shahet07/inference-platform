#!/usr/bin/env python3
"""Checks HTTPS with real certificates: the server, the router, and verifying backends.

Usage: tls_test.py SERVER_BINARY ROUTER_BINARY MODEL_PATH FIRST_PORT
"""
import http.client
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SERVER, ROUTER, MODEL = sys.argv[1:4]
FIRST = int(sys.argv[4])
BODY = json.dumps({"inputs": [0.5] * 64})
OPENSSL = shutil.which("openssl")


def run(*args):
    subprocess.run([OPENSSL, *args], check=True, capture_output=True)


def make_ca(directory, name):
    run("req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "2", "-subj", f"/CN={name}",
        "-keyout", str(directory / f"{name}.key"), "-out", str(directory / f"{name}.pem"))


def make_cert(directory, name, ca):
    run("req", "-newkey", "rsa:2048", "-nodes", "-subj", f"/CN={name}", "-keyout",
        str(directory / f"{name}.key"), "-out", str(directory / f"{name}.csr"))
    extensions = directory / f"{name}.ext"
    extensions.write_text("subjectAltName=IP:127.0.0.1\n")
    run("x509", "-req", "-in", str(directory / f"{name}.csr"), "-CA", str(directory / f"{ca}.pem"),
        "-CAkey", str(directory / f"{ca}.key"), "-CAcreateserial", "-days", "2", "-extfile",
        str(extensions), "-out", str(directory / f"{name}.pem"))


class TlsFailure(Exception):
    """The connection failed, for instance because the certificate could not be verified."""


def https(port, method, path, ca, body=None):
    # Uses curl because some systems' Python has a TLS library too old for these servers.
    args = ["curl", "-sS", "--http1.1", "--max-time", "10", "--cacert", str(ca), "-X", method, "-D", "-",
            "-H", "Expect:",
            f"https://127.0.0.1:{port}{path}"]
    if body is not None:
        args += ["--data-binary", body, "-H", "Content-Type: application/json"]
    # Bytes, because text mode would turn the \\r\\n that separates headers into \\n.
    result = subprocess.run(args, capture_output=True, timeout=20)
    if result.returncode != 0:
        raise TlsFailure(f"curl exit {result.returncode}: {result.stderr.decode().strip()}")
    head, _, data = result.stdout.decode().partition("\r\n\r\n")
    if head.startswith("HTTP/1.1 100"):
        head, _, data = data.partition("\r\n\r\n")
    lines = head.split("\r\n")
    headers = dict(line.split(": ", 1) for line in lines[1:] if ": " in line)
    return int(lines[0].split()[1]), data, headers


def wait_for(condition, what):
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            if condition():
                return
        except (OSError, TlsFailure):
            pass
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


with tempfile.TemporaryDirectory() as tmp:
    directory = Path(tmp)
    make_ca(directory, "good-ca")
    make_ca(directory, "other-ca")
    make_cert(directory, "node", "good-ca")
    make_cert(directory, "untrusted", "other-ca")
    good_ca = directory / "good-ca.pem"

    node_port, router_port, bad_router_port, untrusted_port = (FIRST + i for i in range(4))
    tls_flags = ["--tls-cert", str(directory / "node.pem"), "--tls-key", str(directory / "node.key")]
    processes = [
        subprocess.Popen([SERVER, "--model", MODEL, "--port", str(node_port), *tls_flags],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
        subprocess.Popen([SERVER, "--model", MODEL, "--port", str(untrusted_port), "--tls-cert",
                          str(directory / "untrusted.pem"), "--tls-key",
                          str(directory / "untrusted.key")],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
        subprocess.Popen([ROUTER, "--port", str(router_port), "--backends",
                          f"127.0.0.1:{node_port}", "--backend-ca", str(good_ca), *tls_flags],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
        # This router trusts good-ca, but its only backend's certificate comes from another CA.
        subprocess.Popen([ROUTER, "--port", str(bad_router_port), "--backends",
                          f"127.0.0.1:{untrusted_port}", "--backend-ca", str(good_ca), *tls_flags],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
    ]
    try:
        wait_for(lambda: https(node_port, "GET", "/healthz", good_ca)[0] == 200, "the HTTPS server")
        status, data, _ = https(node_port, "POST", "/v1/infer", good_ca, BODY)
        assert status == 200, (status, data[:200])
        assert len(json.loads(data)["outputs"]) == 10, data[:200]
        print("ok: inference works over HTTPS")

        # A client that does not trust the CA must fail to connect.
        try:
            https(node_port, "GET", "/healthz", directory / "other-ca.pem")
            raise AssertionError("a client accepted a certificate from an untrusted CA")
        except TlsFailure as failure:
            assert "exit 60" in str(failure), failure  # curl's code for a failed certificate check.

        # Plain HTTP must get nothing useful from a TLS port.
        plain = http.client.HTTPConnection("127.0.0.1", node_port, timeout=3)
        try:
            plain.request("GET", "/healthz")
            assert plain.getresponse().status != 200, "a TLS port answered plain HTTP"
        except (http.client.HTTPException, socket.timeout, OSError):
            pass
        print("ok: untrusted clients and plain HTTP are refused")

        wait_for(lambda: https(router_port, "GET", "/healthz", good_ca)[0] == 200, "the HTTPS router")
        status, _, headers = https(router_port, "POST", "/v1/infer", good_ca, BODY)
        assert status == 200, status
        assert headers["X-Served-By"] == f"127.0.0.1:{node_port}"
        print("ok: HTTPS works from a client through the router to a verified backend")

        # The router must refuse a backend whose certificate it cannot verify.
        wait_for(lambda: https(bad_router_port, "GET", "/healthz", good_ca)[0] == 503,
                 "the untrusted backend to be marked down")
        status, _, _ = https(bad_router_port, "POST", "/v1/infer", good_ca, BODY)
        assert status == 503, f"a router sent a request to a backend it could not verify: {status}"
        print("ok: the router refuses a backend with an untrusted certificate")
    finally:
        for process in processes:
            process.terminate()
        codes = [process.wait(timeout=30) for process in processes]
    assert codes == [0, 0, 0, 0], f"unclean shutdown: {codes}"
    result = subprocess.run([SERVER, "--model", MODEL, "--port", str(FIRST + 5), "--frontend", "event",
                             *tls_flags], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0, "the event frontend silently ignored a TLS request"
    print("ok: the event frontend refuses TLS rather than serving plain text")
