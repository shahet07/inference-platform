#!/usr/bin/env python3
"""Starts and stops local inference nodes behind a router, following how long requests queue.

This runs processes on one machine, so that the policy can be tested for real. On Kubernetes, the same
signal would drive a HorizontalPodAutoscaler instead (see ops/k8s/).

Usage: autoscaler.py --server BIN --model PATH --router-port N --first-node-port N [options]
"""
import argparse
import dataclasses
import json
import signal
import subprocess
import time
import urllib.request


@dataclasses.dataclass
class Policy:
    """Scales up fast and down slowly, because being short of capacity costs more than spare capacity."""

    min_nodes: int = 1
    max_nodes: int = 4
    scale_up_wait_s: float = 0.002  # Mean queue wait that means more capacity is needed.
    scale_down_wait_s: float = 0.0002  # Mean queue wait that means capacity is going spare.
    scale_down_after_ticks: int = 5  # Consecutive quiet readings before removing a node.
    cooldown_ticks: int = 2  # Readings to ignore after a change, while the system settles.


@dataclasses.dataclass
class State:
    quiet_ticks: int = 0
    cooldown: int = 0


def decide(policy, state, nodes, mean_wait_s, requests_per_s):
    """Returns the change in node count, -1, 0, or 1. mean_wait_s is None when there was no traffic."""
    if state.cooldown > 0:
        state.cooldown -= 1
        return 0
    idle = mean_wait_s is None or requests_per_s == 0
    if not idle and mean_wait_s > policy.scale_up_wait_s:
        state.quiet_ticks = 0
        if nodes < policy.max_nodes:
            state.cooldown = policy.cooldown_ticks
            return 1
        return 0
    if idle or mean_wait_s < policy.scale_down_wait_s:
        state.quiet_ticks += 1
        if state.quiet_ticks >= policy.scale_down_after_ticks and nodes > policy.min_nodes:
            state.quiet_ticks = 0
            state.cooldown = policy.cooldown_ticks
            return -1
        return 0
    state.quiet_ticks = 0
    return 0


def read_metrics(port):
    """Returns (queue wait seconds total, requests total) from one node."""
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=2) as response:
        wait, count = 0.0, 0.0
        for line in response.read().decode().splitlines():
            if line.startswith("inference_queue_wait_seconds_sum "):
                wait = float(line.split()[1])
            elif line.startswith("inference_queue_wait_seconds_count "):
                count = float(line.split()[1])
        return wait, count


class Autoscaler:
    def __init__(self, args, policy):
        self.args, self.policy, self.state = args, policy, State()
        self.nodes = {}  # port -> process
        self.last = {}  # port -> (wait, count)

    def add_node(self):
        port = next(p for p in range(self.args.first_node_port, self.args.first_node_port + 1000)
                    if p not in self.nodes)
        process = subprocess.Popen(
            [self.args.server, "--model", self.args.model, "--port", str(port), "--workers",
             str(self.args.workers), "--max-batch", "1", "--events", "0"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 30
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1)
                break
            except OSError:
                if time.time() > deadline:
                    process.kill()
                    raise
                time.sleep(0.02)
        # Added to the router only once it can serve, so no request reaches a node still starting.
        urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{self.args.router_port}/v1/backends",
            data=json.dumps({"backend": f"127.0.0.1:{port}"}).encode(), method="POST"), timeout=5)
        self.nodes[port] = process
        print(f"scaled up to {len(self.nodes)} nodes (added :{port})", flush=True)

    def remove_node(self):
        port = max(self.nodes)
        # Out of the router first, then a drain, so that requests on the node can finish.
        urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{self.args.router_port}/v1/backends/127.0.0.1:{port}",
            method="DELETE"), timeout=5)
        time.sleep(self.args.drain_s)
        process = self.nodes.pop(port)
        self.last.pop(port, None)
        process.terminate()
        process.wait(timeout=30)
        print(f"scaled down to {len(self.nodes)} nodes (removed :{port})", flush=True)

    def tick(self):
        total_wait = total_count = 0.0
        for port in self.nodes:
            try:
                wait, count = read_metrics(port)
            except OSError:
                continue
            last_wait, last_count = self.last.get(port, (0.0, 0.0))
            total_wait += wait - last_wait
            total_count += count - last_count
            self.last[port] = (wait, count)
        mean_wait = total_wait / total_count if total_count > 0 else None
        rate = total_count / self.args.interval_s
        change = decide(self.policy, self.state, len(self.nodes), mean_wait, rate)
        if change > 0:
            self.add_node()
        elif change < 0:
            self.remove_node()

    def run(self):
        stop = []
        signal.signal(signal.SIGTERM, lambda *_: stop.append(True))
        signal.signal(signal.SIGINT, lambda *_: stop.append(True))
        for _ in range(self.policy.min_nodes):
            self.add_node()
        try:
            while not stop:
                time.sleep(self.args.interval_s)
                self.tick()
        finally:
            for process in self.nodes.values():
                process.terminate()
            for process in self.nodes.values():
                process.wait(timeout=30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--router-port", type=int, required=True)
    parser.add_argument("--first-node-port", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--min-nodes", type=int, default=1)
    parser.add_argument("--max-nodes", type=int, default=4)
    parser.add_argument("--interval-s", type=float, default=1.0)
    parser.add_argument("--drain-s", type=float, default=1.0)
    parser.add_argument("--scale-up-wait-ms", type=float, default=2.0)
    parser.add_argument("--scale-down-wait-ms", type=float, default=0.2)
    args = parser.parse_args()
    Autoscaler(args, Policy(min_nodes=args.min_nodes, max_nodes=args.max_nodes,
                            scale_up_wait_s=args.scale_up_wait_ms / 1000,
                            scale_down_wait_s=args.scale_down_wait_ms / 1000)).run()


if __name__ == "__main__":
    main()
