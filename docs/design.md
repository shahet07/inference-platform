# Design notes

Why the system is built the way it is, and what each choice gives up. Measurements are in
[benchmarks.md](benchmarks.md).

## Request path

```
client ──► router ──► node: HTTP handler ──► bounded queue ──► worker ──► ONNX Runtime
             │                                                   │
             │ consistent hash of the routing key                └─ fulfils each request's future
             │ retry on the next node if the connection fails
             ▼
         health monitor

node: HTTP handler ──► sharded event log ──► per-shard consumer ──► windowed stats (GET /v1/stats)
```

A handler thread parses the request, submits it to the engine, and blocks on a future. Workers pull
batches from the queue and run the model.

## Inference server

**A bounded queue, and 503 when it is full.** Under overload, an unbounded queue turns into memory
growth and unbounded latency for everyone. Rejecting at the door keeps latency bounded for accepted
requests and tells clients to back off with `Retry-After`. The cost is that some requests fail
during a burst that a larger queue might have absorbed.

**Batching with no wait by default.** Workers take whatever queued while they were busy, and do not
hold requests back to fill a batch. Under light load, requests run alone with no added delay. Under
heavy load, batches form on their own. `--max-wait-us` adds a deliberate wait for models where
batching saves more than the wait costs. The default of 1 ms that this project started with was its
largest measured bottleneck.

**ONNX Runtime uses one thread per inference.** Concurrency comes from the worker pool. Letting
ONNX Runtime also start a thread pool per inference would oversubscribe the cores.

**A blocking thread per connection.** This comes from `cpp-httplib` and is the simplest model that
works, but it costs a thread per connection and system calls per request. It is the first thing to
replace if the HTTP layer needs to go faster. See the profile in benchmarks.md.

**Shapes validated before ONNX Runtime sees them.** It only checks that the buffer is large enough,
so a shape smaller than the input silently ignores the rest.

## Streaming

**One preallocated ring per shard, with one mutex.** Memory is fixed at startup, so no consumer can
make the process grow. Events with the same key go to the same shard in order. A mutex per shard is
simple and uncontended enough: the benchmarks reach tens of millions of events per second. A
lock-free ring would add a lot of subtle code without a measured need.

**The backpressure policy is explicit.** Every choice loses something, so the caller picks:

| Policy | Producer when full | What is lost | Suits |
| --- | --- | --- | --- |
| `kBlock` | Waits | Throughput | Data that must not be lost |
| `kDropNewest` | Continues | New events | Keeping the history intact |
| `kDropOldest` | Continues | Unread old events | Metrics where only recent data matters |

The server uses `kDropNewest`, so analytics can never delay inference.

**Consumers wait for a minimum batch.** Waking a consumer for every event dominated the first
version's runtime, and made throughput fall as shards were added. Readers now wait for 256 events
or 5 ms, whichever comes first, as Kafka does with `fetch.min.bytes` and `fetch.max.wait.ms`. That
adds up to 5 ms of delay to analytics.

**Delivery is at-least-once.** A consumer commits offsets after processing, so a rebalance
redelivers anything uncommitted. Exactly-once would need transactions with whatever the consumer
writes to, and this project has no such store.

**Each shard's consumer owns its aggregator.** A key maps to one shard, so aggregators need no locks.
Windows close by a watermark, late events are counted and dropped, and a per-window key limit means
a flood of unique keys cannot exhaust memory.

**Limitation: the log is in memory.** A restart loses events. Persistence is the natural next step.

## Cluster

**Consistent hashing with 128 virtual nodes per node.** When a node fails, only its keys move, and
they return when it recovers. A stable hash of its own (FNV-1a with a Murmur finalizer) keeps
placement the same across processes and platforms, which `std::hash` does not promise.

**Retrying a failed request on the next node is safe.** Inference has no side effects. At worst, an
event is recorded twice.

**Two failure signals.** A failed live request removes a node immediately, since it is strong
evidence and waiting would fail more requests. Background probes need 3 failures to remove a node
and 2 successes to restore it, so a flapping node stays out.

**Limitation: the router is a single point of failure.** It holds no durable state, so several could
run behind a load balancer, and the benchmarks do try several. There is no leader election.

**Metadata replicates by pull, last writer wins.** Each node periodically pulls every peer's registry
and merges by `(version, node id)`. Merging is commutative, associative, and idempotent, so nodes
converge regardless of order, and a restarted node catches up with no help. The cost is that when two
nodes write the same model at once, one write silently loses. This suits occasional metadata
updates and would not suit anything needing strong consistency. A registry entry is metadata only:
it does not load a model.

## Performance layer

**The arena is available but off by default.** It is about 5 times faster than `malloc` in isolation,
but makes no measurable difference on the request path, because allocation in the batch path is a
small part of the total. The library stays for allocation-heavy work.

**Our kernels are not used in serving.** ONNX Runtime is faster on batches, so replacing it would make
the server slower. The kernels exist to find out where that gap comes from.

**AVX2 was written and compiled but never run.** It builds for x86-64, but the development machine
cannot run x86 binaries. Run `perf_tests` on an x86 machine before relying on it.

## Observability

**Metrics for rates, logs for events, traces for single requests.** Counters and histograms use atomics
and no locks, so they are safe on the request path. Logs are for lifecycle events and errors, with
rate limits on anything overload can trigger. Traces are sampled.

**One sampling decision per trace.** The first service decides, and the W3C `traceparent` header carries
that decision. An early version left the header off unsampled requests, so nodes sampled on their own
and produced traces with no router span.

**Spans go through a bounded queue to a background writer.** A full queue drops spans and counts them.
Tracing must never be able to slow a request.

**Spans are written as JSON lines.** `tools/trace_view.py` merges them across services. This avoids an
exporter dependency, but it means no Jaeger-style UI without adding one.

## Known gaps

- No TLS or authentication. Everything binds to localhost by default and assumes a trusted network.
- A node serves one model, chosen at startup. The registry records metadata but does not load models.
- The event log is not persistent.
- The router is a single instance with no failover of its own.
- All cluster benchmarks ran on one machine, so they show this machine's limits and not those of a
  real cluster with its own network.
