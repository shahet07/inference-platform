#!/usr/bin/env python3
"""Compares continuous and static batching on a real tiny LLM, with arrivals over time and mixed lengths.

Uses bench/models/stories260K.gguf, a tiny story model, and a build from the llama preset. This shows how
the scheduling behaves. The numbers say little about larger models, whose steps take far longer.
"""
import http.client
import json
import random
import statistics
import subprocess
import threading
import time

from harness import ROOT, write_results

SERVER = ROOT / "build" / "llama" / "llm" / "llm_server"
MODEL = ROOT / "bench" / "models" / "stories260K.gguf"
SLOTS, REQUESTS, REPEATS = 8, 120, 3
PROMPTS = ["Once upon a time", "The little dog", "A girl named Lily", "One day, a big bear"]
RATES = [20, 60, 120, 240]  # Requests per second.
next_port = 25000


def run_client(port, prompt, max_tokens, results, lock):
    started = time.time()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    connection.request("POST", "/v1/generate",
                       json.dumps({"prompt": prompt, "max_tokens": max_tokens, "stream": True}))
    response = connection.getresponse()
    first, tokens = None, 0
    for line in response:
        if line.startswith(b"data: "):
            event = json.loads(line[6:])
            if "token" in event:
                tokens += 1
                if first is None:
                    first = time.time() - started
    with lock:
        results.append({"first_token_s": first, "total_s": time.time() - started, "tokens": tokens,
                        "status": response.status})


def run_once(policy, rate, seed):
    global next_port
    next_port += 1
    port = next_port
    server = subprocess.Popen([str(SERVER), "--model", str(MODEL), "--port", str(port), "--slots", str(SLOTS),
                               "--context", "256", "--threads", "4", "--policy", policy,
                               "--queue", "1000"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 20
        while True:
            try:
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
                connection.request("GET", "/healthz")
                connection.getresponse().read()
                break
            except OSError:
                assert time.time() < deadline, "the server never started"
                time.sleep(0.05)
        rng = random.Random(seed)
        results, lock, threads = [], threading.Lock(), []
        started = time.time()
        for _ in range(REQUESTS):
            # Mostly short answers with some long ones, which is where static batching suffers.
            max_tokens = rng.choice([8, 8, 8, 16, 16, 32, 128])
            thread = threading.Thread(target=run_client, args=(port, rng.choice(PROMPTS), max_tokens,
                                                               results, lock))
            thread.start()
            threads.append(thread)
            time.sleep(rng.expovariate(rate))
        for thread in threads:
            thread.join()
        elapsed = time.time() - started
    finally:
        server.terminate()
        server.wait(timeout=30)
    firsts = sorted(r["first_token_s"] for r in results if r["first_token_s"] is not None)
    totals = sorted(r["total_s"] for r in results)
    return {"tokens_per_sec": sum(r["tokens"] for r in results) / elapsed,
            "ttft_p50_ms": 1000 * firsts[len(firsts) // 2],
            "ttft_p99_ms": 1000 * firsts[int(len(firsts) * 0.99)],
            "total_p50_ms": 1000 * totals[len(totals) // 2],
            "total_p99_ms": 1000 * totals[int(len(totals) * 0.99)],
            "failed": sum(r["status"] != 200 for r in results)}


results = []
print(f"{'policy':>10} {'req/s':>5} {'tokens/s':>9} {'TTFT p50':>9} {'TTFT p99':>9} {'total p50':>10} {'total p99':>10}")
for rate in RATES:
    for policy in ("continuous", "static"):
        runs = [run_once(policy, rate, seed) for seed in range(REPEATS)]
        row = {"policy": policy, "offered_requests_per_sec": rate,
               **{k: statistics.median(r[k] for r in runs) for k in runs[0]}}
        results.append(row)
        print(f"{policy:>10} {rate:>5} {row['tokens_per_sec']:>9.0f} {row['ttft_p50_ms']:>8.1f}m "
              f"{row['ttft_p99_ms']:>8.1f}m {row['total_p50_ms']:>9.1f}m {row['total_p99_ms']:>9.1f}m", flush=True)
write_results("llm.json", results)
