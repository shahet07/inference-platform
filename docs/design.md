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

**Models are replaced by swapping a shared pointer.** A request holds the model it started with until its
response is written, so a swap cannot remove a model mid-request. The old model finishes its queue and goes
away with the last request using it. A new model runs one inference before taking traffic, which rejects
files that load but cannot run, and a failed load leaves the current model serving.

**Each model has its own queue and workers.** A busy model cannot starve another's queue. They do still
share CPU, and nothing limits a model's share of it.

**Metadata can name only plain files in `--model-dir`.** Metadata can arrive over the network from peers, and
loading a model means parsing an untrusted file, so a uri cannot point anywhere else. There is still no
authentication on the write itself, which the known gaps below cover.

**Deadlines drop work nobody is waiting for.** With `X-Deadline-Ms`, an expired request is dropped before it
reaches the model, so overload does not spend compute on answers no client will read.

**Raw floats as well as JSON.** JSON stays as the easy default, and `application/octet-stream` skips text
encoding. On a real image model the difference is 19×.

**Core ML is an option, and batching still stays off.** It was about 18× faster than the CPU on float MobileNetV2 in the
benchmarks here, and batching had no consistent effect on it there. A provider is chosen per server, not per model.

**The engine keeps its mutex queue.** A lock-free queue was built and measured. It is far faster with one producer and
one consumer and far slower under contention, and it cannot let idle workers sleep.

**ONNX Runtime uses one thread per inference.** Concurrency comes from the worker pool. Letting
ONNX Runtime also start a thread pool per inference would oversubscribe the cores.

**Two HTTP frontends.** The default uses `cpp-httplib`, with a blocking thread per connection: simple, and with every
feature. `--frontend event` uses `net/`, where a few threads watch many sockets. It is faster and scales to far more
connections, but serves only inference and metrics, with no tracing, event recording, or model management endpoints.

**An event loop thread must not block.** So the engine has `submitAsync`, whose completion posts the response back to
the connection's loop. One limit remains: JSON is parsed on the loop thread, so a huge JSON body delays that loop's other
connections. Raw floats avoid this.

**A connection handles one request at a time.** Pipelined requests wait their turn, which keeps responses in order
without tagging them, at some cost to clients that pipeline. Reading stops at a bounded backlog, so a client that sends
without reading responses is slowed by TCP, rather than growing a buffer.

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

**Delivery is at-least-once by default, with exactly-once available.** A consumer commits offsets after
processing, so a rebalance redelivers anything uncommitted. Exactly-once takes two parts. Idempotent producers
give each event a producer id and sequence number, so a retry is recognized, and the highest sequence per
producer is snapshotted before old segments are removed, so that survives trimming. `ExactlyOnceAggregator` then
saves its state and read position together in one atomically renamed file, so after a crash both come from the
same moment and a replay cannot double count. This only covers state in that file. Writing to an external
system exactly once would need that system to take part. A producer must also send each shard its sequence numbers
in order, since a lower number counts as a retry.

**Sliding windows count an event once, into a pane.** A window merges its panes as it closes, so adding an event
costs the same however many windows overlap, and long idle gaps skip empty windows instead of stepping through them.

**Interval joins keep events until both sides have moved past any match.** A stalled stream therefore holds
events back, so each key has a buffer limit, with drops counted. Hitting the limit loses matches, which is the
trade for bounded memory.

**Each shard's consumer owns its aggregator.** A key maps to one shard, so aggregators need no locks.
Windows close by a watermark, late events are counted and dropped, and a per-window key limit means
a flood of unique keys cannot exhaust memory.

**Optional write-ahead log, one directory of segments per shard.** Each record has a size, a CRC-32, and
its offset. Recovery keeps everything up to the first bad record in the last segment, which a crash can
leave half-written. Bad data in any earlier segment is an error, because those were synced before the
next began, so that is corruption and silently dropping it would hide lost data.

The fsync policy is explicit, like the backpressure policy:

| Policy | Append returns after | Survives process crash | Survives power loss |
| --- | --- | --- | --- |
| `kEveryAppend` | A sync, shared by concurrent appends | Yes | As far as the disk honors it |
| `kInterval` | The OS has it; synced every 10 ms on another thread | Yes | Up to the last interval lost |
| `kNever` | The OS has it | Yes | Whatever the OS had not written |

The server records events with `appendNoWait`, so requests never wait on the disk. A crash can lose the
most recently buffered analytics events, which is the right trade for analytics.

On macOS a plain `fsync` only reaches the drive's cache, so this uses `F_FULLFSYNC`, which is why
`kEveryAppend` is so slow here.

**Tiered storage behind a small interface.** When consumers pass a sealed segment, it moves to a
`SegmentStore`, and reads of older offsets come from there. Only a directory-backed store exists: a real
S3 store would implement the same three methods, but none has been written or tested.

## Cluster

**Consistent hashing with 128 virtual nodes per node.** When a node fails, only its keys move, and
they return when it recovers. A stable hash of its own (FNV-1a with a Murmur finalizer) keeps
placement the same across processes and platforms, which `std::hash` does not promise.

**Retrying a failed request on the next node is safe.** Inference has no side effects. At worst, an
event is recorded twice.

**Two failure signals, and neither is trusted alone.** Two failed live requests in a row remove a node,
fewer than the 3 failed background probes it otherwise takes, and 2 good probes restore it, so a flapping node
stays out. An early version removed a node on one failed request. A brief burst of refused connections then
took out the only backend, and caused a total outage until probes restored it.

**If every backend looks down, the router tries them all.** Health checks can be wrong, and an overloaded
cluster can fail its checks while still serving. Refusing everything would turn a partial problem into a
total one, and if the backends really are down, requests still fail quickly. `router_panic_mode` shows when this
is happening.

**A request's deadline is not held against a backend.** Running out of a client's own deadline returns 504 and
leaves the backend in rotation, otherwise one impatient client could remove healthy nodes.

**Two balancing policies.** Consistent hashing keeps a key on one node, for per-key state or caches, but load
follows the keys. Least-loaded compares two random backends and picks the one with fewer requests in flight,
which follows load but gives up key affinity.

**Routers are interchangeable.** They hold no durable state and hash keys identically, so several can run
behind a load balancer, and one can die with no coordination. `tests/integration/chaos_test.py` checks both
points. Something still has to send clients to a live router, which is that load balancer's job.

**Metadata replicates by pull, last writer wins.** Each node periodically pulls every peer's registry
and merges by `(version, node id)`. Merging is commutative, associative, and idempotent, so nodes
converge regardless of order, and a restarted node catches up with no help. The cost is that when two
nodes write the same model at once, one write silently loses. This suits occasional metadata
updates and would not suit anything needing strong consistency.

## Performance layer

**The arena is available but off by default.** It is about 5 times faster than `malloc` in isolation,
but makes no measurable difference on the request path, because allocation in the batch path is a
small part of the total. The library stays for allocation-heavy work.

**Our kernels are not used in serving.** ONNX Runtime is faster on batches, so replacing it would make
the server slower. The kernels exist to find out where that gap comes from.

**The packed kernel is not always the best one.** It is fastest on large matrices, and about 3 times slower on a single
row, where copying costs more than it saves. `perf::best()` picks by architecture, not size.

**AVX2 is tested for correctness only.** Its tests run in CI on x86-64, but its speed has not been
benchmarked.

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

## LLM serving

**The scheduler knows nothing about models.** `Backend` is tokenize, one step, and release, so the same scheduler runs
llama.cpp in production and a fake in tests. The fake also checks the contract, so it throws if a scheduler bug sends a
wrong position or reuses a sequence that was not released.

**A failed step ends the whole batch.** The state of every sequence in it is unknown afterwards.

**llama.cpp is an optional build,** with its own preset, because it is a large dependency. CI tests the serving layers with
the fake backend, and does not build the real one. The model is not committed.

## Security

**API keys are stored as SHA-256 hashes.** Reading the key file does not reveal the keys, and a malformed file stops the
server, instead of leaving it open, with an error that never echoes the file's contents, in case someone pasted a raw
key. Keys are random and long, which is why a fast hash is fine here, where a password would need a slow one.

**Comparison takes the same time wherever it fails.** Hashes are compared in constant time, against every entry.

**Two roles.** Running a model needs a client key. Changing what is served needs an admin key. Before this, anyone who
could reach the port could replace a model. `/healthz` stays open, for load balancers and health checks, and `/metrics`
requires a key.

**Rate limits are token buckets per key.** They allow short bursts, and hold the long-run average. A request over the
limit gets 429 with `Retry-After`.

**TLS is a build option**, because it brings in OpenSSL. A router can verify backends against a CA, and refuses one it
cannot verify. The event frontend does not support TLS, and refuses to start if asked, instead of quietly serving plain
text. Certificates are read at startup, so rotating them means a restart. Client certificates are not supported.

## Consensus

**Raft is a deterministic state machine with no threads, clock, or network.** Time and messages are passed in, which
is what makes it possible to simulate a whole cluster under hostile conditions, and replay any failure from a seed.
The networked `raft_node` is a thin layer around it, with all Raft work on one thread.

**All appends that arrive together share one write and sync.** With fsync, that group commit is the difference between
about 173 and 2,558 writes/s.

**A client can see an unknown outcome.** If a leader dies after receiving a write, the write may or may not commit, so
the node returns 503, and a retrying client may add a command twice. Commands would need ids to deduplicate, as the
event log's idempotent producers do.

## Deployment

`ops/k8s/` runs nodes behind a headless Service, which routers discover through DNS, with a CPU-based
HorizontalPodAutoscaler. **The manifests have only passed a client-side dry run, and have not been applied to a
cluster.** What was tested is the image from `ci/Dockerfile.runtime`, read-only and as a non-root user, in Docker:
a router found two nodes through a shared DNS name, served requests from both, and dropped a stopped node.

## Known gaps

- Security is opt-in: without `--api-keys` and TLS flags, anyone who can reach a port has full access. Everything binds
  to localhost by default. Peer replication and Raft traffic are not encrypted, and Raft has no authentication.
- The directory-backed segment store is the only tiered storage backend. There is no real S3 client.
- Kubernetes manifests have had only a client-side dry run. They are plain YAML, with no Helm chart.
- There is no NUMA placement, and core pinning has not been benchmarked.
- Benchmarks have not run on real x86 hardware, so AVX2's speed is unknown.
- The event log persists to one machine's disk. The Raft log is replicated, but is a separate service with no
  compaction or membership changes, and the two are not connected.
- All cluster benchmarks ran on one machine, so they show this machine's limits and not those of a
  real cluster with its own network.
