"""Shared helpers for benchmark scripts: starts a server, waits for it, runs the load generator."""
import json
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build" / "release"
SERVER = BUILD / "server" / "inference_server"
ROUTER = BUILD / "cluster" / "inference_router"
LOAD_GEN = BUILD / "bench" / "load_gen"
MODEL = ROOT / "bench" / "models" / "mlp_quant.onnx"
RESULTS = ROOT / "bench" / "results"


def wait_healthy(port, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"server on port {port} did not become healthy")


def fetch_metric(port, name):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as response:
        for line in response.read().decode().splitlines():
            if line.startswith(name + " "):
                return float(line.split()[1])
    raise KeyError(name)


class Server:
    def __init__(self, port, model=MODEL, **flags):
        args = [SERVER, "--model", model, "--port", port]
        for key, value in flags.items():
            args += [f"--{key.replace('_', '-')}", value]
        self.port = port
        self.process = subprocess.Popen(list(map(str, args)), stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)

    def __enter__(self):
        wait_healthy(self.port)
        return self

    def __exit__(self, *_):
        self.process.terminate()
        self.process.wait(timeout=30)


class Router:
    def __init__(self, port, backends, balance="hash"):
        self.port = port
        self.process = subprocess.Popen(
            [str(ROUTER), "--port", str(port), "--balance", balance, "--backends",
             ",".join(f"127.0.0.1:{b}" for b in backends)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def __enter__(self):
        wait_healthy(self.port)
        return self

    def __exit__(self, *_):
        self.process.terminate()
        self.process.wait(timeout=30)


def load(port, connections, duration=8, warmup=2, **flags):
    args = [LOAD_GEN, "--port", port, "--connections", connections,
            "--duration", duration, "--warmup", warmup]
    for key, value in flags.items():
        args += [f"--{key.replace('_', '-')}", value]
    return json.loads(subprocess.run(list(map(str, args)), check=True, capture_output=True,
                                     text=True).stdout)


def write_results(name, data):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(data, indent=2) + "\n")
