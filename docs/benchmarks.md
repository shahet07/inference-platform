# Benchmarks

Every number here comes from the committed scripts, run on the final code by `bench/run_all.py`.
Raw results are in [`bench/results/`](../bench/results).

## Read this first

**Machine:** Apple M2 Pro, 8 performance and 4 efficiency cores, 32 GB. macOS, Apple clang 21,
`RelWithDebInfo` build.

These numbers have real limits:

- **Everything runs on one laptop**, including the load generator, which competes for the same cores
  and uses about as much CPU as the server. Traffic is over loopback, so there is no real network.
- **Laptops are noisy.** Core types differ and the machine can throttle. Scripts interleave
  configurations and report the median of 3 to 5 runs, but differences of a few percent are not
  meaningful. Do not compare across machines.
- **The benchmark models are small synthetic MLPs** with random weights: a repeatable compute load,
  with no predictive value. [MobileNetV2](#a-real-model-mobilenetv2) tests whether results carry over.
- **The load is closed loop unless stated**, which understates tail latency under overload.
  `load_gen --rate` runs open loop to avoid this.

| Model | Shape | Inference alone, 1 thread |
| --- | --- | --- |
| Light (`mlp_quant`) | 64 → 256 → 256 → 10, quantized | 5.4 µs |
| Heavy (`mlp_heavy_quant`) | 512 → 2048 → 2048 → 10, quantized | 107 µs |
| XL (`mlp_xl_quant`) | 1024 → 4096 → 4096 → 10, quantized | 426 µs |

## Phase 1: inference server

### Worker threads

Light model, 64 connections, no batching wait.

| Workers | Requests/s | p50 ms | p99 ms | Server cores | Mean batch |
| --- | --- | --- | --- | --- | --- |
| 1 | 28,323 | 2.28 | 4.86 | 5.28 | 7.0 |
| 4 | 32,580 | 1.40 | 6.58 | 5.53 | 3.3 |
| 8 | 34,683 | 1.52 | 6.10 | 5.53 | 1.9 |
| 16 | 35,080 | 1.41 | 5.74 | 5.66 | 1.5 |

![Throughput by worker threads](charts/server_scaling.svg)
![p99 latency by worker threads](charts/server_latency.svg)

Workers barely matter, because the model is not the limit: at this size it takes a small part of each
request. Almost all CPU goes to HTTP, JSON, and sockets. With about 5 cores for the server and about 5
for the load generator, the machine is also close to full.

### The bottleneck fix: removing the batch wait

The first version held each batch open for up to 1 ms. A metrics check showed the model taking about
60 µs per batch, so requests spent most of their time waiting for a batch to fill.

| Connections | Max wait | Requests/s | p50 ms | p99 ms | Mean batch |
| --- | --- | --- | --- | --- | --- |
| 1 | 1000 µs | 543 | 1.85 | 2.44 | 1.0 |
| 1 | 200 µs | 1,635 | 0.56 | 1.05 | 1.0 |
| 1 | **0** | **9,005** | **0.11** | **0.15** | 1.0 |
| 16 | 1000 µs | 10,461 | 1.67 | 2.31 | 4.5 |
| 16 | 200 µs | 29,628 | 0.54 | 0.82 | 3.4 |
| 16 | **0** | **36,741** | **0.41** | **1.01** | 1.3 |
| 64 | 1000 µs | 32,810 | 1.54 | 6.50 | 7.3 |
| 64 | 200 µs | 33,505 | 1.54 | 6.72 | 5.5 |
| 64 | **0** | **33,092** | **1.42** | **7.11** | 3.3 |

**Result at 16 connections: 3.5× the throughput (10,461 to 36,741 requests/s) and p50 down from 1.67
to 0.41 ms.** p99 fell from 2.31 to 1.01 ms. At 64 connections the setting makes no meaningful
difference, since queues fill batches anyway.

Trade-offs: at 16 connections, a 200 µs wait gives a better p99 than no wait (0.82 vs 1.01 ms) with
less throughput. A model where batching saves real compute may also want a wait, so it stays configurable.

### What did not help

Profiling at 64 connections put 68% of busy samples in `__sendto`, with a 42 ms p99 that looked
like the classic stall between Nagle's algorithm and delayed ACKs. **Testing did not support that.** The
42 ms never appeared without the profiler attached, so the profiler most likely caused it.

| Config, 64 connections, no batch wait | Requests/s | p50 ms | p99 ms |
| --- | --- | --- | --- |
| Baseline | 29,958 | 1.97 | 6.01 |
| `TCP_NODELAY` on server | 31,448 | 1.88 | 5.60 |
| `TCP_NODELAY` on both | 31,290 | 1.90 | 5.51 |
| Keep-alive limit removed | 32,267 | 1.44 | 7.37 |
| Both | 33,075 | 1.52 | 6.87 |

Both are kept, for about 10% more throughput and 23% lower p50. p99 got slightly worse with the
keep-alive change, and may be noise.

### Profiles

These come from macOS `sample`, since `perf` is Linux-only, and are drawn by `tools/flamegraph.py`.
Sampling counts wall-clock time, so a thread blocked in a system call counts as busy there.

- [After tuning](../bench/results/flamegraph_after.svg): 712 busy samples. Time is spread across HTTP
  routing (37%) and batch workers (32%), and `__sendto` leads self time at 14%.
- [Before tuning](../bench/results/flamegraph_before.svg): only 11 busy samples, too few to read. The
  server spends its time idle, waiting for batches to fill. **A profile of busy threads barely shows
  this kind of bottleneck**, and it was the latency metrics that exposed it.
- [Router](../bench/results/flamegraph_router.svg): dominated by `__sendto`.

### Sanitizers

All 119 tests pass with no reports under AddressSanitizer with UndefinedBehaviorSanitizer, and under
ThreadSanitizer, including end-to-end tests with real processes. An earlier run of the server under
real load was also clean. ONNX Runtime is a prebuilt binary and is not instrumented.

The same tests pass on Linux arm64 in Docker (`ci/Dockerfile`), with leak detection on under ASan.

Bugs these tools found during development:

- ASan: a range-for over a temporary JSON value.
- TSan on Linux, not on macOS: the router's health thread started before construction finished, racing
  with route setup inside libstdc++'s regex code.
- libFuzzer with UBSan: signed overflow in window arithmetic for timestamps near the 64-bit limits.

### A real model: MobileNetV2

The synthetic models above might mislead, so this repeats the questions on MobileNetV2 from the ONNX Model
Zoo, quantized, with 150,528 input values per image. `tools/fetch_real_model.py` downloads it. Inputs are
fixed synthetic data, so this measures speed and not accuracy. CPU only, one thread per inference.

**Does waiting to fill a batch help?** One worker, up to 32 per batch, raw float requests:

| Connections | Max wait | Requests/s | p50 ms | p99 ms | Mean batch | Model ms per batch |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | None | 240.8 | 4.11 | 4.44 | 1.0 | 3.67 |
| 1 | 5 ms | 80.5 | 12.68 | 13.44 | 1.0 | 4.70 |
| 1 | 20 ms | 30.2 | 33.44 | 37.96 | 1.0 | 6.38 |
| 8 | None | 290.7 | 27.34 | 28.91 | 4.0 | 13.68 |
| 8 | 5 ms | 222.7 | 35.97 | 38.12 | 8.0 | 27.78 |
| 8 | 20 ms | 136.0 | 59.66 | 63.65 | 8.0 | 30.82 |
| 32 | None | 293.3 | 109.88 | 117.87 | 15.8 | 54.06 |
| 32 | 5 ms | 293.3 | 109.49 | 125.01 | 15.9 | 54.36 |
| 32 | 20 ms | 277.3 | 115.24 | 126.10 | 32.0 | 109.53 |

**No. Waiting never helped.** Batching itself barely saves compute here: 54.06 ms for 15.8 images is 3.42 ms
each, against about 3.6 ms alone, about 5% less. That is on this CPU. A GPU, where batching commonly pays
off far more, was not tested.

**Workers against batching**, 32 connections:

| Workers | Max batch | Requests/s | p50 ms | p99 ms |
| --- | --- | --- | --- | --- |
| 1 | 1 | 273.3 | 117.10 | 126.89 |
| 4 | 1 | 1,059.7 | 30.19 | 31.64 |
| 8 | 1 | 1,819.3 | 17.42 | 20.26 |
| 4 | 8 | 1,098.3 | 31.67 | 41.05 |
| 8 | 8 | 1,685.7 | 16.89 | 53.64 |

Unlike the small synthetic model, this one scales with workers, reaching 6.7× at 8, because compute is now the
limit. Batching on top adds little, and at 8 workers it has a worse p99.

**JSON against raw floats**, 8 connections, 4 workers:

| Protocol | Requests/s | p50 ms | p99 ms |
| --- | --- | --- | --- |
| `application/octet-stream` | 1,060.3 | 5.76 | 17.45 |
| JSON | 56.7 | 141.35 | 152.13 |

**Raw floats are 18.7× faster**, since the model takes under 5 ms and encoding 150,528 numbers as text takes
far longer.

### Hot-swapping models

`tests/integration/hot_swap_test.py` swaps a live model 12 times through the metadata API under steady traffic
from 4 connections. In the recorded run, none of 8,963 requests failed, and requests reported 13 different
model versions. A corrupt file and a path outside the model directory were both refused, and the current
model kept serving. This was in a debug build, so it shows correctness and not latency during a swap, which
was not measured.

## Phase 2: streaming

### Against a Python baseline

The same design in pure Python: per-shard rings with locks, blocking backpressure, windowed
aggregation, and minimum-batch consumers. Python is an easy baseline, since its threads are limited
by the GIL.

| Implementation | Producers / shards | Events/s processed | Append p50 µs | Append p99 µs | Append p99.9 µs | Peak MB | Lost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C++ | 1 / 1 | 31,543,436 | 0.04 | 0.08 | 0.08 | 10 | 0 |
| Python | 1 / 1 | 634,109 | 0.62 | 0.83 | 4.04 | 24 | 0 |
| C++ | 4 / 8 | 15,712,094 | 0.04 | 8.21 | 18.04 | 29 | 0 |
| Python | 4 / 8 | 545,096 | 0.62 | 46.33 | 329.04 | 68 | 0 |

**C++ is 50× faster with 1 producer and 29× faster with 4 producers on 8 shards.** Append latency is
sampled on every 64th event, and includes the clock read.

### The bottleneck fix: batching consumer wakeups

The first version woke a consumer for every event once it had caught up. Throughput fell as shards
were added, and the profile put 53% of busy samples in `append`, mostly `notify_all` and mutex
handoff system calls. Consumers now wait for 256 events or 5 ms.

| Shards, 4 producers | Wake per event | Batched | Change | p99 before | p99 after |
| --- | --- | --- | --- | --- | --- |
| 1 | 16,970,215 | 16,971,651 | none | 16.71 µs | 16.71 µs |
| 8 | 2,496,504 | 15,905,910 | 6.4× | 28.08 µs | 8.29 µs |
| 32 | 628,162 | 18,566,489 | 30× | 34.46 µs | 4.92 µs |

### Backpressure with a slow consumer

4 million events, 4096 events per shard, and consumers that sleep 10 ms per 1024 events.

| Policy | Producer events/s | Consumed | Dropped | Evicted | Appends that blocked | Append p99 µs | Peak MB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Block | 586,073 | 4,000,000 | 0 | 0 | 2,262 | 16.29 | 6 |
| Drop newest | 33,556,414 | 95,422 | 3,904,578 | 0 | 0 | 3.08 | 5 |
| Drop oldest | 32,300,187 | 100,508 | 0 | 3,902,462 | 0 | 3.42 | 5 |

Memory stays at 5 to 6 MB under each policy, and every event is accounted for.

### Cost on the request path

Recording one event per request costs nothing measurable at 16 connections: 36,044 requests/s without
it and 36,195 with it, a difference within noise.

### Durability cost

4 producers on 8 shards, every append timed, on this Mac's internal SSD. Disks differ widely, and macOS
`F_FULLFSYNC` is much slower than a typical Linux `fsync`, so treat this as a comparison of policies.

| Policy | Events/s | Append p50 µs | Append p99 µs | Append p99.9 µs |
| --- | --- | --- | --- | --- |
| Sync every append | 459 | 8,448 | 17,975 | 25,067 |
| Sync every 10 ms | 2,857 | 104 | 9,024 | 12,856 |
| OS decides | 166,665 | 23 | 52 | 97 |

Compare about 15.7M events/s in memory. An earlier version synced on the same thread that acknowledged
appends, which held the interval policy to 1,147 events/s.

`tests/integration/wal_crash_test.py` kills a writer with SIGKILL 12 times at random moments, and
checks every event acknowledged before each kill. In the recorded run, none of 1,530 were lost. **This
tests process crashes, not power loss**, and the slow syncs mean few events per run, so the torn-tail
paths are covered by unit tests that cut a segment at each byte, and by fuzzing the record parser (22M
runs, no failures), rather than by this test.

### Stream operations

Single thread, 1000 keys, median of 3 runs of `bench/stream_ops_bench.cpp`.

| Operation | Events/s |
| --- | --- |
| Append | 62,091,393 |
| Idempotent append | 43,672,148 |
| Tumbling window | 44,621,415 |
| Sliding window, 6 overlapping panes | 43,544,551 |
| Interval join | 11,442,806 |

### Exactly-once processing

`tests/integration/exactly_once_test.py` kills a worker with SIGKILL 40 times at random moments, with the producer
retrying every event from the start after each restart, and the consumer replaying from its last checkpoint.
All 150,000 events then count exactly once across 50 keys, and every total matches. The final run alone
recognized 64,219 retries. **This holds for process crashes and for the state in the checkpoint file.** It was
not tested against power loss, since this test uses the fsync policy that survives only a process crash.

## Phase 3: cluster

All nodes, the router, and the load generator share one machine. **This shows where this machine stops
scaling, not what a real cluster would do.** "Direct" means no router.

| Workload | Config | Requests/s | p50 ms | p99 ms | Node cores | Router cores | Client cores |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Light | Direct | 32,870 | 1.50 | 6.70 | 5.6 | 0 | 5.2 |
| Light | 1 node | 20,493 | 2.86 | 7.78 | 2.7 | 5.7 | 2.1 |
| Light | 2 nodes | 21,439 | 2.71 | 7.62 | 2.6 | 6.1 | 2.0 |
| Light | 3 nodes | 21,724 | 2.68 | 8.20 | 2.6 | 6.1 | 1.9 |
| Light | 4 nodes | 21,650 | 2.51 | 9.02 | 2.6 | 6.0 | 2.0 |
| Heavy | Direct | 7,788 | 4.16 | 4.26 | 1.5 | 0 | 0.2 |
| Heavy | 1 node | 8,370 | 3.78 | 4.15 | 1.5 | 0.8 | 0.6 |
| Heavy | 2 nodes | 15,942 | 2.14 | 3.07 | 3.9 | 2.6 | 1.2 |
| Heavy | 3 nodes | 17,620 | 1.44 | 6.53 | 4.8 | 3.6 | 1.5 |
| Heavy | 4 nodes | 18,258 | 1.15 | 12.40 | 4.9 | 3.9 | 1.5 |
| XL | Direct | 2,316 | 13.67 | 14.56 | 1.2 | 0 | 0.1 |
| XL | 1 node | 2,294 | 13.72 | 15.07 | 1.2 | 0.4 | 0.4 |
| XL | 2 nodes | 4,242 | 8.63 | 11.45 | 2.4 | 0.3 | 0.2 |
| XL | 3 nodes | 4,717 | 7.00 | 11.97 | 3.5 | 0.4 | 0.2 |
| XL | 4 nodes | 5,092 | 6.78 | 10.60 | 4.7 | 0.6 | 0.3 |

![Cluster scaling relative to one node](charts/cluster_scaling.svg)

Scaling from 1 to 4 nodes reaches 1.06× for light, 2.18× for heavy, and 2.22× for XL. No requests failed
in any run. Scaling is close to linear from 1 to 2 nodes on the heavy and XL workloads (1.90× and 1.85×),
and flattens after that.

### Where scaling stops, and why

The curves look alike but have different causes. Each suspect was tested on its own:

```
bench/run_cluster_bottleneck.py    changes routers and connections at 4 nodes
bench/run_hardware_contention.py   runs inference alone, with no networking
```

| Test at 4 nodes | Light requests/s | Heavy requests/s |
| --- | --- | --- |
| 1 router, baseline | 21,656 | 18,285 |
| 2 routers | 25,128 (+16%) | 19,247 (+5%) |
| 3 routers | 24,746 | 18,484 |
| 3 or 4× the connections, spreading keys more evenly | 19,979 | 16,694 |

![Inference alone with no networking](charts/hardware_contention.svg)

| Processes, inference only | Light | Heavy | XL |
| --- | --- | --- | --- |
| 1 | 1.00× | 1.00× | 1.00× |
| 2 | 1.99× | 1.96× | 2.01× |
| 3 | 2.91× | 2.97× | 2.28× |
| 4 | 3.85× | 3.91× | 2.48× |
| 6 | 5.81× | 4.23× | 2.44× |
| 8 | 7.67× | 3.47× | 2.50× |

**Light stops at the router and at machine CPU.** Going through the router drops throughput 38%, since
each request gains socket round trips, and it uses more than twice the CPU of all nodes combined. A second
router adds 16%, then a third adds nothing. All processes together use about 10.6 of 12 cores.

**Heavy stops at machine CPU taken up by the HTTP path.** The model alone scales to 3.91× at 4 processes,
so inference is not the limit. A second router adds only 5%, and all processes use about 10.3 of 12 cores.

**XL stops at shared hardware.** Keys are balanced within 0.5% and the router uses about 0.6 cores, and
about 5.7 of 12 cores are busy, so software and CPU are both ruled out. With no networking at all,
inference alone stalls at about 2.5×, while the light model reaches 7.67×. The most likely cause is
memory bandwidth: 21 MB of weights at about 5,800 inferences/s is about 120 GB/s. **That figure is an
estimate. Bandwidth and cache misses were not measured directly.** The heavy model's stall beyond 4
processes may be a cache limit or the efficiency cores, and was not investigated.

**Ruled out: uneven key placement.** Spreading keys more evenly with more connections made throughput
slightly lower, not higher.

### Failure handling

`tests/integration/cluster_test.py` kills real processes. Killing 1 of 3 nodes caused no failed requests
and moved only that node's keys (14 of 60). The node returned, its keys came back, and it caught up on
metadata written while it was down. With all nodes dead, the router returns 503.

### Routing policies

Three nodes with the heavy model, 32 connections, median of 3 runs. "One slow" gives one node 1 worker and the
others 4.

| Scenario | Policy | Requests/s | p50 ms | p99 ms | p99.9 ms |
| --- | --- | --- | --- | --- | --- |
| Equal nodes | Consistent hash | 16,863 | 1.04 | 11.85 | 119.19 |
| Equal nodes | Least loaded | 18,081 | 1.01 | 12.32 | 121.12 |
| One slow | Consistent hash | 17,277 | 1.26 | 13.15 | 55.40 |
| One slow | Least loaded | 17,177 | 1.19 | 16.43 | 53.09 |

**Least-loaded routing made no clear difference here**, and the slow node's share only fell from 26% to 23%. This
model's 107 µs of compute is small next to the router's own cost, so a node with fewer workers is barely slower
as seen by a client. These tails also come from a machine near CPU saturation. This does not show the policy is
useless, only that this setup cannot show a benefit.

### Fault injection

`tests/integration/chaos_test.py` puts a fault-injecting proxy between the router and its nodes, with a 500 ms
backend allowance. Results from a recorded debug run:

| Fault | What clients saw |
| --- | --- |
| Node silently hangs, connections stay open | One request took about 505 ms, then failed over; none failed |
| Connections reset as data arrives | No failed requests |
| Connections refused | No failed requests |
| Node slow, 50 ms client deadline | 504 in about 53 ms; the node stayed in rotation |
| Node flaps up and down | It returned only once stable |
| Every backend looks down, then recovers | Requests were still tried, and succeeded before probes had recovered |
| One of two routers dies | The other serves the same keys on the same nodes |

### Autoscaling

`tests/integration/autoscale_test.py` runs `ops/autoscaler.py` for real on one machine. Nodes went from 1 to the
maximum of 3 under load and back to 1 when idle, with 184,461 requests in the recorded run and none failed.

### Bugs these tests found

- **A connection burst took down the whole cluster.** The HTTP library's listen backlog defaulted to 5, so 32 new
  connections were refused, and one failed request removed a node immediately, even the only one. About 8% of
  requests failed. Fixed with a larger backlog, requiring two failures in a row, and panic mode.
- **Exactly-once could lose events.** The producer snapshot named events not yet on disk, so after a crash their
  retries looked like duplicates. Consumers could also read events a crash could still erase. The test hung when
  this happened, which is how it surfaced. Passing once had been luck. Fixed by snapshotting only synced data, and
  letting readers see only durable events.
- **A deadline could return the wrong status.** A read timeout could fire just before the deadline clock expired,
  which returned 503 and counted against the backend.

**Unexplained:** in one sanitizer run, the exactly-once and autoscale tests timed out. They then passed 8 of 8
repeat runs alone, and the cause was not found.

## Phase 4: performance layer

Single-threaded, `-O3`. Our kernels and Eigen run with random weights in the same shape as
`mlp_float.onnx`, which ONNX Runtime runs.

### Matrix multiplication, 256 × 256 × 256

| Implementation | GFLOPS | Time µs | Gap to Eigen |
| --- | --- | --- | --- |
| Scalar, vectorization disabled | 8.7 | 3,842 | 8.8× |
| Compiler auto-vectorized | 27.9 | 1,203 | 2.8× |
| NEON, 4 rows of `b` per pass | 35.1 | 956 | 2.2× |
| NEON micro-kernel, 4 × 12 in registers | 55.7 | 602 | 1.4× |
| Eigen 3.4 | 76.8 | 437 | none |

### Full forward pass, µs per call

| Batch | Ours, NEON micro | Ours, auto-vectorized | Eigen | ONNX Runtime |
| --- | --- | --- | --- | --- |
| 1 | 5.3 | 7.0 | 5.2 | 7.6 |
| 8 | 27.1 | 54.6 | 27.4 | 16.1 |
| 64 | 213.2 | 432.3 | 156.9 | 114.2 |
| 256 | 856.3 | 1729.3 | 600.9 | 441.4 |

Fitting `time = overhead + cost per row × batch` from batches 1 and 256:

| Implementation | Overhead per call µs | Cost per row µs |
| --- | --- | --- |
| ONNX Runtime | 5.88 | 1.70 |
| Eigen | 2.90 | 2.34 |
| Ours, NEON micro | 1.95 | 3.34 |

The micro-kernel's fit is rough: a single row cannot use the 4-row path, so batch 1 runs different code.

### The gaps and their causes

| Gap | Size | Cause | Evidence |
| --- | --- | --- | --- |
| Scalar to auto-vectorized | 3.2× | SIMD lanes | Same loop, vectorization toggled |
| Auto-vectorized to hand-written NEON | 1.3× | Four rows of `b` per pass over the output | Measured directly |
| NEON to micro-kernel | 1.6× | Sums stay in registers, with no load or store of the output per step | Measured directly |
| Micro-kernel to Eigen | 1.4× | Probably packing and cache blocking, which ours lacks | **Inferred from Eigen's design, not tested** |
| Eigen to ONNX Runtime, per row | 1.4× | Probably its tuned, packed kernels (MLAS) | **Inferred, not tested** |
| ONNX Runtime's overhead at batch 1 | About 6 µs per call | Framework cost: `Run` setup and graph execution | Fitted; matches its being slowest at batch 1 |

**So ONNX Runtime is slowest for one row and fastest for 8 or more.** Serving keeps it, since replacing it
would make batches slower.

### Arena allocator

In isolation, the arena is 5.4× faster: 64 allocations take 453 ns, against 2,437 ns with `malloc`.
**On the server it makes no measurable difference**: 35,587 requests/s without it and 35,691 with it, with
overlapping ranges. Only one buffer per batch came from it, so it is off by default. A profile would have
predicted this, and the plan's condition of optimizing where profiling shows it matters did not hold here.

### AVX2

Its correctness tests pass on x86-64, but **its speed has not been measured**. It also has only the simpler design, not the micro-kernel. The plan's
Phase 4 assumed AVX2, and NEON took its place here.

## Phase 5: observability

### Tracing cost

| Tracing, 16 connections | Requests/s (range) | p50 ms |
| --- | --- | --- |
| Off | 35,615 (35,144 to 36,006) | 0.416 |
| 1 in 100 | 36,057 (35,573 to 36,496) | 0.414 |
| Every request | 35,068 (34,281 to 35,505) | 0.418 |

Sampling 1 in 100 has no measurable cost, and tracing every request costs about 1.5%.

### Dashboard

From `ops/run_demo.py --kill-after 70`, with one node killed and restarted under steady load. The node's
traffic moves to the others, healthy backends drop from 3 to 2 and recover, failovers spike briefly, and
`unavailable` stays at 0.

![Grafana dashboard during a node failure](dashboard.png)

## Reproducing

```sh
python3 -m venv .venv && .venv/bin/pip install onnx onnxruntime numpy
.venv/bin/python tools/make_bench_model.py   # Needed once: the XL model is too large to commit.
cmake --preset release && cmake --build --preset release
python3 bench/run_all.py                     # About 30 minutes, on an otherwise idle machine.
python3 bench/plot_results.py
```
