#!/usr/bin/env python3
"""Starts 3 nodes and a router with tracing on, and drives steady load until interrupted.

Ports match ops/prometheus.yml. With --kill-after N, one node is killed N seconds in and restarted
N seconds later, to show a failure on the dashboard. Logs and spans go to the --out directory.
"""
import argparse
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build" / "release"
MODEL = ROOT / "bench" / "models" / "mlp_quant.onnx"
ROUTER_PORT = 18700
NODE_PORTS = [18701, 18702, 18703]


def start_node(index, out):
    port = NODE_PORTS[index]
    peers = ",".join(f"127.0.0.1:{p}" for p in NODE_PORTS if p != port)
    with open(out / f"node-{index}.log", "a") as log:
        return subprocess.Popen(
            [str(BUILD / "server" / "inference_server"), "--model", str(MODEL),
             "--port", str(port), "--node-id", f"node-{index}", "--peers", peers,
             "--trace-file", str(out / f"node-{index}.spans"), "--trace-sample", "1000"],
            stdout=subprocess.DEVNULL, stderr=log)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/tmp/inference-demo")
    parser.add_argument("--rate", type=float, default=3000, help="requests per second")
    parser.add_argument("--duration", type=float, default=3600)
    parser.add_argument("--kill-after", type=float, default=0)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    nodes = [start_node(i, out) for i in range(len(NODE_PORTS))]
    with open(out / "router.log", "a") as log:
        router = subprocess.Popen(
            [str(BUILD / "cluster" / "inference_router"), "--port", str(ROUTER_PORT), "--backends",
             ",".join(f"127.0.0.1:{p}" for p in NODE_PORTS),
             "--trace-file", str(out / "router.spans"), "--trace-sample", "1000"],
            stdout=subprocess.DEVNULL, stderr=log)
    time.sleep(1)
    load = subprocess.Popen(
        [str(BUILD / "bench" / "load_gen"), "--port", str(ROUTER_PORT), "--connections", "32",
         "--rate", str(args.rate), "--duration", str(args.duration), "--routing-keys", "1"],
        stdout=subprocess.DEVNULL)

    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(True))
    signal.signal(signal.SIGINT, lambda *_: stop.append(True))
    print(f"router on :{ROUTER_PORT}, logs and spans in {out}", flush=True)
    started = time.time()
    killed = restarted = False
    while not stop and load.poll() is None:
        elapsed = time.time() - started
        if args.kill_after and not killed and elapsed > args.kill_after:
            nodes[2].kill()
            nodes[2].wait()
            killed = True
            print("killed node-2", flush=True)
        if killed and not restarted and elapsed > 2 * args.kill_after:
            nodes[2] = start_node(2, out)
            restarted = True
            print("restarted node-2", flush=True)
        time.sleep(0.2)

    for process in [load, router, *nodes]:
        process.terminate()
    for process in [load, router, *nodes]:
        process.wait(timeout=30)


if __name__ == "__main__":
    main()
