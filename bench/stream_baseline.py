#!/usr/bin/env python3
"""The workload from stream_bench.cpp in pure Python, with the same structure: fixed-capacity
per-shard rings with one lock each, blocking backpressure, and one windowed-aggregation consumer
per shard, which waits for a minimum batch or 5 ms. Prints one JSON object of results. Latency is sampled every 64th append, as in C++.
"""
import argparse
import json
import resource
import sys
import threading
import time

KEYS = [f"key-{i}" for i in range(1000)]
SAMPLE_EVERY = 64
MIN_BATCH = 256


class Shard:
    def __init__(self, capacity):
        self.ring = [None] * capacity
        self.capacity = capacity
        self.head = 0
        self.tail = 0
        self.closed = False
        self.wake_at = None
        self.lock = threading.Lock()
        self.has_room = threading.Condition(self.lock)
        self.has_events = threading.Condition(self.lock)


def append(shard, event):
    with shard.lock:
        while shard.tail - shard.head >= shard.capacity and not shard.closed:
            shard.has_room.wait()
        if shard.closed:
            return
        shard.ring[shard.tail % shard.capacity] = event
        shard.tail += 1
        if shard.wake_at is not None and shard.tail >= shard.wake_at:
            shard.wake_at = None
            shard.has_events.notify()


class Aggregator:
    """Tumbling windows per key, closed once the newest timestamp passes their end."""

    def __init__(self, window_ms):
        self.window_ms = window_ms
        self.open = {}
        self.max_ts = None

    def add(self, key, value, ts):
        start = ts - ts % self.window_ms
        self.max_ts = ts if self.max_ts is None else max(self.max_ts, ts)
        keys = self.open.setdefault(start, {})
        stats = keys.get(key)
        if stats is None:
            keys[key] = [1, value, value, value]
        else:
            stats[0] += 1
            stats[1] += value
            stats[2] = min(stats[2], value)
            stats[3] = max(stats[3], value)

    def take_closed(self):
        if self.max_ts is None:
            return
        for start in [s for s in self.open if s + self.window_ms <= self.max_ts]:
            del self.open[start]


def consume(shard, consumed, index):
    aggregator = Aggregator(1000)
    offset = 0
    while True:
        with shard.lock:
            target = offset + MIN_BATCH
            if shard.tail < target and not shard.closed:
                shard.wake_at = target
                shard.has_events.wait(0.005)
                shard.wake_at = None
            end = min(shard.tail, offset + 1024)
            batch = [shard.ring[i % shard.capacity] for i in range(offset, end)]
            done = shard.closed and end >= shard.tail
        for key, value, ts in batch:
            aggregator.add(key, value, ts)
        aggregator.take_closed()
        consumed[index] += len(batch)
        offset = end
        with shard.lock:
            shard.head = max(shard.head, offset)
            shard.has_room.notify_all()
        if done:
            return


def produce(shards, count, producer_index, latencies):
    nshards = len(shards)
    for i in range(count):
        key = KEYS[(i + producer_index * 37) % len(KEYS)]
        event = (key, float(i % 100), i // 1000)
        shard = shards[hash(key) % nshards]
        if i % SAMPLE_EVERY == 0:
            before = time.perf_counter()
            append(shard, event)
            latencies.append((time.perf_counter() - before) * 1e6)
        else:
            append(shard, event)


def percentile(values, q):
    return values[int(q * (len(values) - 1))] if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=500_000)
    parser.add_argument("--producers", type=int, default=4)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--capacity", type=int, default=65536)
    args = parser.parse_args()

    shards = [Shard(args.capacity) for _ in range(args.shards)]
    consumed = [0] * args.shards
    consumers = [threading.Thread(target=consume, args=(s, consumed, i))
                 for i, s in enumerate(shards)]
    for t in consumers:
        t.start()

    started = time.perf_counter()
    latencies = [[] for _ in range(args.producers)]
    count = args.events // args.producers
    producers = [threading.Thread(target=produce, args=(shards, count, p, latencies[p]))
                 for p in range(args.producers)]
    for t in producers:
        t.start()
    for t in producers:
        t.join()
    ingest_seconds = time.perf_counter() - started
    for shard in shards:
        with shard.lock:
            shard.closed = True
            shard.has_events.notify_all()
            shard.has_room.notify_all()
    for t in consumers:
        t.join()
    total_seconds = time.perf_counter() - started

    merged = sorted(v for l in latencies for v in l)
    offered = count * args.producers
    print(json.dumps({
        "impl": "python",
        "python": sys.version.split()[0],
        "policy": "block",
        "producers": args.producers,
        "shards": args.shards,
        "events": offered,
        "ingest_events_per_sec": offered / ingest_seconds,
        "processed_events_per_sec": sum(consumed) / total_seconds,
        "consumed": sum(consumed),
        "append_p50_us": percentile(merged, 0.50),
        "append_p99_us": percentile(merged, 0.99),
        "append_p999_us": percentile(merged, 0.999),
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024),
    }))


if __name__ == "__main__":
    main()
