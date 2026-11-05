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
- **Busy machines are detected.** One full run happened during ordinary desktop use, and its numbers fell by half. It was
  thrown away. `bench/run_all.py` now checks before and after each benchmark that inference still scales across 4
  processes, waits out a busy machine, and reruns anything disturbed. `bench/results/run_conditions.json` records that
  all 20 benchmarks here ran clean, with that check between 3.74× and 4.00×.
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
| 1 | 27,084 | 2.32 | 5.04 | 5.16 | 7.1 |
| 4 | 32,946 | 1.46 | 6.81 | 5.08 | 3.5 |
| 8 | 36,466 | 1.34 | 5.80 | 4.81 | 2.0 |
| 16 | 36,375 | 1.31 | 6.48 | 4.82 | 1.6 |

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
| 1 | 1000 µs | 567 | 1.75 | 2.52 | 1.0 |
| 1 | 200 µs | 1,879 | 0.52 | 0.67 | 1.0 |
| 1 | **0** | **8,343** | **0.12** | **0.18** | 1.0 |
| 16 | 1000 µs | 10,149 | 1.72 | 2.64 | 4.5 |
| 16 | 200 µs | 29,714 | 0.53 | 0.86 | 3.3 |
| 16 | **0** | **35,767** | **0.38** | **1.42** | 1.3 |
| 64 | 1000 µs | 32,101 | 1.39 | 8.23 | 7.4 |
| 64 | 200 µs | 32,261 | 1.40 | 7.62 | 5.8 |
| 64 | **0** | **32,741** | **1.44** | **7.17** | 3.3 |

**Result at 16 connections: 3.5× the throughput (10,149 to 35,767 requests/s) and p50 down from 1.72
to 0.38 ms.** p99 fell from 2.64 to 1.42 ms. At 64 connections the setting makes no meaningful
difference, since queues fill batches anyway.

Trade-offs: at 16 connections, a 200 µs wait gives a better p99 than no wait (0.86 vs 1.42 ms) with
less throughput. A model where batching saves real compute might want a wait, so it stays configurable.

### What did not help

Profiling at 64 connections put 68% of busy samples in `__sendto`, with a 42 ms p99 that looked
like the classic stall between Nagle's algorithm and delayed ACKs. **Testing did not support that.** The
42 ms never appeared without the profiler attached, so the profiler most likely caused it.

| Config, 64 connections, no batch wait | Requests/s | p50 ms | p99 ms |
| --- | --- | --- | --- |
| Baseline | 28,768 | 1.96 | 7.18 |
| `TCP_NODELAY` on server | 29,863 | 1.89 | 6.74 |
| `TCP_NODELAY` on both | 29,761 | 1.89 | 7.16 |
| Keep-alive limit removed | 32,331 | 1.44 | 7.80 |
| Both | 32,617 | 1.44 | 7.23 |

Both are kept, for about 13% more throughput and 27% lower p50. p99 did not improve.

### Profiles

These come from macOS `sample`, since `perf` is Linux-only, and are drawn by `tools/flamegraph.py`.
Sampling counts wall-clock time, so a thread blocked in a system call counts as busy there.

- [After tuning](../bench/results/flamegraph_after.svg): 708 busy samples, spread across HTTP handling and batch
  workers, with `__sendto` leading self time at 14.5%.
- [Before tuning](../bench/results/flamegraph_before.svg): only 53 busy samples, far fewer. The
  server spends its time idle, waiting for batches to fill. **A profile of busy threads barely shows
  this kind of bottleneck**, and it was the latency metrics that exposed it.
- [Router](../bench/results/flamegraph_router.svg): dominated by `__sendto`.

### Sanitizers

All 278 tests pass with no reports under AddressSanitizer with UndefinedBehaviorSanitizer, and under
ThreadSanitizer, including end-to-end tests with real processes. An earlier run of the server under
real load was also clean. ONNX Runtime is a prebuilt binary and is not instrumented.

The same tests pass on Linux arm64 in Docker (`ci/Dockerfile`), with leak detection on under ASan.

Bugs these tools found during development:

- ASan: a range-for over a temporary JSON value.
- TSan on Linux, not on macOS: the router's health thread started before construction finished, racing
  with route setup inside libstdc++'s regex code.
- libFuzzer with UBSan: signed overflow in window arithmetic for timestamps near the 64-bit limits.

### Event-driven frontend

`net/` is an HTTP/1.1 server on kqueue and epoll, where a few threads handle many sockets. `--frontend event` uses it in
place of the thread-per-connection library. Light model, raw floats, 4 workers, median of 3 runs on this Mac.

| Frontend | Connections | Requests/s | p50 ms | p99 ms | RSS MB | Threads |
| --- | --- | --- | --- | --- | --- | --- |
| Thread per connection | 16 | 38,169 | 0.38 | 1.38 | 32 | 262 |
| Thread per connection | 64 | 34,571 | 1.39 | 5.76 | 34 | 262 |
| Thread per connection | 256 | 28,199 | 7.88 | 28.94 | 40 | 278 |
| Thread per connection | 1,000 | 23,594 | 40.19 | 99.64 | 72 | 1,022 |
| Event-driven | 16 | 48,831 | 0.32 | 0.59 | 28 | 10 |
| Event-driven | 64 | 44,457 | 1.01 | 8.00 | 28 | 10 |
| Event-driven | 256 | 39,561 | 6.12 | 15.74 | 29 | 10 |
| Event-driven | 1,000 | 34,781 | 27.19 | 67.91 | 29 | 10 |

**28 to 47% more throughput, with 10 threads instead of up to 1,022, and 29 MB instead of 72 MB at 1,000
connections.** Its p99 is better at every count except 64 connections (8.00 against 5.76 ms). Connections share one read buffer per thread, and release their buffers when idle. Before that, this
frontend used 135 MB at 1,000 connections. The load generator uses a thread per connection on the same machine, which
limits both frontends at high counts.

The parser is strict by design, because framing ambiguity is how request smuggling works: duplicate lengths, spaces
before colons, and folded headers are all rejected, with tests for each. It ran about 12M fuzz iterations with no
failures.

### A real model: MobileNetV2

The synthetic models above might mislead, so this repeats the questions on MobileNetV2 from the ONNX Model
Zoo, quantized, with 150,528 input values per image. `tools/fetch_real_model.py` downloads it. Inputs are
fixed synthetic data, so this measures speed and not accuracy. CPU only, one thread per inference.

**Does waiting to fill a batch help?** One worker, up to 32 per batch, raw float requests:

| Connections | Max wait | Requests/s | p50 ms | p99 ms | Mean batch | Model ms per batch |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | None | 258.5 | 3.84 | 4.35 | 1.0 | 3.43 |
| 1 | 5 ms | 79.2 | 12.90 | 13.47 | 1.0 | 4.72 |
| 1 | 20 ms | 27.7 | 37.77 | 41.98 | 1.0 | 7.98 |
| 8 | None | 290.7 | 27.36 | 29.03 | 4.0 | 13.66 |
| 8 | 5 ms | 225.3 | 35.35 | 36.62 | 8.0 | 27.05 |
| 8 | 20 ms | 130.7 | 62.39 | 64.34 | 8.0 | 31.40 |
| 32 | None | 288.0 | 109.35 | 133.51 | 15.9 | 54.50 |
| 32 | 5 ms | 293.3 | 109.01 | 111.19 | 15.9 | 54.04 |
| 32 | 20 ms | 277.3 | 114.12 | 129.01 | 32.0 | 109.08 |

**No. Waiting never helped.** Batching itself barely saves compute here: 54.50 ms for 15.9 images is 3.44 ms
each, against about 3.43 ms alone, about -0% less, on this CPU.

**Workers against batching**, 32 connections:

| Workers | Max batch | Requests/s | p50 ms | p99 ms |
| --- | --- | --- | --- | --- |
| 1 | 1 | 286.8 | 110.87 | 120.09 |
| 4 | 1 | 1,082.5 | 29.53 | 30.73 |
| 8 | 1 | 1,901.2 | 16.58 | 20.44 |
| 4 | 8 | 1,124.5 | 30.75 | 39.71 |
| 8 | 8 | 1,930.8 | 14.50 | 36.16 |

Unlike the small synthetic model, this one scales with workers, reaching 6.6× at 8, because compute is now the
limit. Batching on top adds little, and at 8 workers it has a worse p99.

**JSON against raw floats**, 8 connections, 4 workers:

| Protocol | Requests/s | p50 ms | p99 ms |
| --- | --- | --- | --- |
| `application/octet-stream` | 1,095.5 | 5.74 | 16.73 |
| JSON | 56.5 | 141.70 | 157.97 |

**Raw floats are 19.4× faster**, since the model takes under 5 ms and encoding 150,528 numbers as text takes
far longer.

### Core ML

`--provider coreml` runs models through Apple's Core ML. Float MobileNetV2, one inference thread, no networking, with
models fixed to each batch size, since accelerators generally want static shapes. Median of 3 runs.

| Provider | Batch | Images/s | ms per image |
| --- | --- | --- | --- |
| cpu | 1 | 79.1 | 12.64 |
| coreml | 1 | 1,463.3 | 0.68 |
| cpu | 8 | 78.3 | 12.76 |
| coreml | 8 | 1,566.2 | 0.64 |
| cpu | 32 | 76.6 | 13.06 |
| coreml | 32 | 1,622.4 | 0.62 |

**Core ML is about 18× faster than the CPU on this float model, and batching has no consistent effect.** Here,
batch 32 is 11% above batch 1, and an earlier run on the same code showed it about 15% below. Core ML chooses between the GPU, Neural Engine, and CPU by itself, and these numbers
do not show which it picked. This is one model on one chip, and NVIDIA GPUs were not tested. The quantized CPU model
above runs about 3.4 ms per image, so for CPU serving, quantizing is the larger win.

Through the server, 16 connections, raw floats, batch size 1:

| Provider | Workers | Requests/s | p50 ms | p99 ms |
| --- | --- | --- | --- | --- |
| cpu | 1 | 79.0 | 202.43 | 209.12 |
| cpu | 4 | 302.8 | 53.03 | 54.17 |
| coreml | 1 | 536.2 | 16.52 | 286.45 |
| coreml | 4 | 575.8 | 24.24 | 96.59 |

### Hot-swapping models

`tests/integration/hot_swap_test.py` swaps a live model 12 times through the metadata API under steady traffic
from 4 connections. In the recorded run, none of 8,963 requests failed, and requests reported 13 different
model versions. A corrupt file and a path outside the model directory were both refused, and the current
model kept serving. This was in a debug build, so it shows correctness and not latency during a swap, which
was not measured.

### LLM streaming

`llm_server` streams tokens as server-sent events, on a scheduler that batches sequences one token at a time. Under
continuous batching, a new request joins the running batch at the next step, and a finished one's slot is reused
immediately. Under static batching, new requests wait until the whole batch has finished. A client that disconnects
frees its slot at the next step.

The model is `stories260K`, a real but tiny LLM through llama.cpp on the CPU, with 8 slots and 4 threads. Requests arrive
randomly, with a mix of mostly short and some long answers. Median of 3 runs, 120 requests each, times in ms.

| Requests/s | Batching | Tokens/s | First token p50 | First token p99 | Total p50 | Total p99 |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | Continuous | 547 | 1.6 | 9.5 | 7.2 | 48.0 |
| 20 | Static | 552 | 1.6 | 38.5 | 8.3 | 56.5 |
| 60 | Continuous | 1,439 | 1.0 | 2.3 | 4.6 | 48.8 |
| 60 | Static | 1,478 | 1.1 | 34.7 | 6.5 | 48.5 |
| 120 | Continuous | 2,708 | 0.8 | 1.6 | 4.4 | 42.2 |
| 120 | Static | 2,703 | 1.1 | 27.9 | 7.1 | 50.0 |
| 240 | Continuous | 5,175 | 0.7 | 1.5 | 4.1 | 43.1 |
| 240 | Static | 5,375 | 6.7 | 28.4 | 13.1 | 48.2 |

**At 240 requests/s, continuous batching's first-token p99 is about 19× better (1.5 against 28.4 ms), with similar
throughput.** This shows how the scheduling behaves. A model this small takes well under a millisecond per step, and a
larger one's steps would take far longer, so absolute numbers would be very different. Its sampling always picks the most
likely token, and there is no prefix caching or per-tenant fairness.

## Phase 2: streaming

### Against a Python baseline

The same design in pure Python: per-shard rings with locks, blocking backpressure, windowed
aggregation, and minimum-batch consumers. Python is an easy baseline, since its threads are limited
by the GIL.

| Implementation | Producers / shards | Events/s processed | Append p50 µs | Append p99 µs | Append p99.9 µs | Peak MB | Lost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C++ | 1 / 1 | 30,003,638 | 0.04 | 0.08 | 0.12 | 11 | 0 |
| Python | 1 / 1 | 629,151 | 0.62 | 3.75 | 4.42 | 24 | 0 |
| C++ | 4 / 8 | 13,475,910 | 0.04 | 9.08 | 19.96 | 38 | 0 |
| Python | 4 / 8 | 532,212 | 0.67 | 59.54 | 377.08 | 68 | 0 |

**C++ is 48× faster with 1 producer and 25× faster with 4 producers on 8 shards.** Append latency is
sampled on every 64th event, and includes the clock read.

### The bottleneck fix: batching consumer wakeups

The first version woke a consumer for every event once it had caught up. Throughput fell as shards
were added, and the profile put 53% of busy samples in `append`, mostly `notify_all` and mutex
handoff system calls. Consumers now wait for 256 events or 5 ms.

| Shards, 4 producers | Wake per event | Batched | Change | p99 before | p99 after |
| --- | --- | --- | --- | --- | --- |
| 1 | 11,809,077 | 12,594,869 | about the same | 17.38 µs | 16.83 µs |
| 8 | 2,567,352 | 14,476,682 | 5.6× | 28.17 µs | 8.58 µs |
| 32 | 669,866 | 17,789,862 | 26.6× | 34.04 µs | 5.00 µs |

### Backpressure with a slow consumer

4 million events, 4096 events per shard, and consumers that sleep 10 ms per 1024 events.

| Policy | Producer events/s | Consumed | Dropped | Evicted | Appends that blocked | Append p99 µs | Peak MB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Block | 592,430 | 4,000,000 | 0 | 0 | 2,085 | 9.71 | 7 |
| Drop newest | 32,392,424 | 107,647 | 3,892,353 | 0 | 0 | 3.08 | 6 |
| Drop oldest | 29,253,051 | 108,662 | 0 | 3,895,428 | 0 | 3.38 | 6 |

Memory stays at 6 to 7 MB under each policy, and every event is accounted for.

### Cost on the request path

Recording one event per request costs nothing measurable at 16 connections: 37,812 requests/s without
it and 37,874 with it, a difference within noise.

### Durability cost

4 producers on 8 shards, every append timed, on this Mac's internal SSD. Disks differ widely, and macOS
`F_FULLFSYNC` is much slower than a typical Linux `fsync`, so treat this as a comparison of policies.

| Policy | Events/s | Append p50 µs | Append p99 µs | Append p99.9 µs |
| --- | --- | --- | --- | --- |
| Sync every append | 550 | 6,984 | 15,981 | 18,014 |
| Sync every 10 ms | 4,807 | 63 | 6,332 | 11,906 |
| OS decides | 167,819 | 23 | 50 | 72 |

Compare about 13.5M events/s in memory. An earlier version synced on the same thread that acknowledged
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
| Append | 60,951,762 |
| Idempotent append | 43,895,320 |
| Tumbling window | 46,184,339 |
| Sliding window, 6 overlapping panes | 44,159,805 |
| Interval join | 11,753,884 |

### Exactly-once processing

`tests/integration/exactly_once_test.py` kills a worker with SIGKILL 20 to 40 times at random moments, with the producer
retrying every event from the start after each restart, and the consumer replaying from its last checkpoint.
All 150,000 events then count exactly once across 50 keys, and every total matches. With the fix below, it passed 80 of
80 runs in a row under ASan on Linux, where it had been failing about 1 run in 8 to 30. **This holds for process crashes and for the state in the checkpoint file.** It was
not tested against power loss, since this test uses the fsync policy that survives only a process crash.

## Phase 3: cluster

All nodes, the router, and the load generator share one machine. **This shows where this machine stops
scaling, not what a real cluster would do.** "Direct" means no router.

| Workload | Config | Requests/s | p50 ms | p99 ms | Node cores | Router cores | Client cores |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Light | Direct | 35,745 | 1.43 | 5.61 | 4.8 | 0 | 4.2 |
| Light | 1 node | 22,750 | 2.78 | 5.51 | 2.6 | 4.6 | 1.8 |
| Light | 2 nodes | 23,772 | 2.63 | 5.20 | 2.5 | 4.7 | 1.8 |
| Light | 3 nodes | 23,548 | 2.61 | 5.58 | 2.5 | 4.8 | 1.9 |
| Light | 4 nodes | 23,316 | 2.66 | 5.96 | 2.4 | 4.4 | 1.8 |
| Heavy | Direct | 7,810 | 4.17 | 4.31 | 1.5 | 0 | 0.2 |
| Heavy | 1 node | 8,337 | 3.80 | 4.16 | 1.6 | 0.6 | 0.3 |
| Heavy | 2 nodes | 15,854 | 2.09 | 3.17 | 3.8 | 2.5 | 1.2 |
| Heavy | 3 nodes | 16,977 | 1.57 | 5.91 | 4.5 | 3.1 | 1.4 |
| Heavy | 4 nodes | 17,392 | 1.08 | 13.28 | 4.4 | 3.0 | 1.3 |
| XL | Direct | 2,331 | 13.58 | 14.90 | 1.2 | 0 | 0.1 |
| XL | 1 node | 2,337 | 13.55 | 15.42 | 1.2 | 0.2 | 0.1 |
| XL | 2 nodes | 4,142 | 8.45 | 11.83 | 2.4 | 0.3 | 0.2 |
| XL | 3 nodes | 4,663 | 7.09 | 10.55 | 3.5 | 0.4 | 0.2 |
| XL | 4 nodes | 4,884 | 6.90 | 12.43 | 4.6 | 0.6 | 0.3 |

![Cluster scaling relative to one node](charts/cluster_scaling.svg)

Scaling from 1 to 4 nodes reaches 1.02× for light, 2.09× for heavy, and 2.09× for XL. No requests failed
in any run. Scaling is close to linear from 1 to 2 nodes on the heavy and XL workloads (1.90× and 1.77×),
and flattens after that.

### Where scaling stops, and why

The curves look alike but have different causes. Each suspect was tested on its own:

```
bench/run_cluster_bottleneck.py    changes routers and connections at 4 nodes
bench/run_hardware_contention.py   runs inference alone, with no networking
```

| Test at 4 nodes | Light requests/s | Heavy requests/s |
| --- | --- | --- |
| 1 router, baseline | 23,693 | 17,690 |
| 2 routers | 27,220 (+15%) | 18,258 (+3%) |
| 3 routers | 27,847 (+18%) | 18,417 (+4%) |
| 3 or 4× the connections, spreading keys more evenly | 22,872 | 15,868 |

![Inference alone with no networking](charts/hardware_contention.svg)

| Processes, inference only | Light | Heavy | XL |
| --- | --- | --- | --- |
| 1 | 1.00× | 1.00× | 1.00× |
| 2 | 1.96× | 1.95× | 1.99× |
| 3 | 2.85× | 2.97× | 2.12× |
| 4 | 3.81× | 3.92× | 2.41× |
| 6 | 5.79× | 4.45× | 2.43× |
| 8 | 7.63× | 4.61× | 2.44× |

**Light stops at the router's per-request cost.** Going through the router drops throughput 36%, since each
request gains socket round trips, and it uses about twice the CPU of all the nodes combined. Further routers add
15% and then 18%, not the doubling that would mean the router was the only limit. All processes together use about
8.7 of 12 cores, which is about what this machine's 8 performance cores can give.

**Heavy stops at the CPU taken by the HTTP path.** The model alone scales to 3.92× at 4 processes, so inference is not the
limit. More routers add at most 4%, and all processes use about 8.8 of 12 cores.

**XL stops at shared hardware.** Keys are balanced within 1% and the router uses about 0.6 cores, and only about
5.5 of 12 cores are busy, so software and CPU are both ruled out. With no networking at all, inference alone stalls at
about 2.4×, while the light model reaches 7.63×. The most likely cause is memory bandwidth: 21 MB of weights at about
5,795 inferences/s is about 122 GB/s. **That figure is an estimate. Bandwidth and cache misses were not measured
directly.** The heavy model's stall beyond 4 processes may be a cache limit or the efficiency cores, and was not
investigated.

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
| Equal nodes | Consistent hash | 18,632 | 1.20 | 7.22 | 107.71 |
| Equal nodes | Least loaded | 18,602 | 1.00 | 10.69 | 125.43 |
| One slow | Consistent hash | 18,270 | 1.17 | 7.54 | 89.48 |
| One slow | Least loaded | 18,395 | 1.04 | 10.92 | 78.38 |

**Least-loaded routing made no clear difference here**, and the slow node's share only fell from 22% to 20%. This
model's 107 µs of compute is small next to the router's own cost, so a node with fewer workers is barely slower
as seen by a client. This does not show the policy is
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
- **Exactly-once could lose events.** Consumers could read events a crash could still erase. Fixed by letting readers
  see only durable events.
- **Exactly-once could count events twice.** The snapshot of each producer's highest sequence was written by a function
  that was never called, because the edit adding the call had silently not applied. Once old segments were trimmed, a
  shard forgot what it had seen, and accepted retries as new. The first version of this fix was therefore committed
  without working. Its unit test passed anyway, because a later event kept the sequence alive. It surfaced only as an
  intermittent hang under ASan on Linux. Rather than raising the timeout, a watchdog and a per-restart history showed a
  shard's highest sequence falling from 62,913 to 0. The test now fails without the fix, which was confirmed.
- **A deadline could return the wrong status.** A read timeout could fire just before the deadline clock expired,
  which returned 503 and counted against the backend.

- **A use-after-free on every shutdown with `--model-dir`.** The model registry was destroyed before the thread that
  reads it. macOS timing hid this, and ThreadSanitizer on Linux found it.
- **Stale connections looked like failing backends.** After an outage, a pooled connection failed at once, so the request
  moved to another node, and a healthy backend was counted as failing. Linux TSan and ASan both exposed this through
  routers disagreeing on where a key belongs. Routers now retry once on a new connection first.
- **A wrong test, caught by testing on a second platform.** Random number libraries differ, so Linux explored different Raft
  scenarios, and one reported a leader missing a committed entry. The check was wrong: it used an entry's creation
  term, and the guarantee is about the term it was committed in. With the check corrected, 6,000 scenarios pass on each
  platform, and the five deliberately injected bugs are all still caught.

**A lesson from this:** one sanitizer run had timed out in the exactly-once test. It passed 8 of 8 repeats on macOS, and
was set aside as unexplained. That timeout was this bug.

### Raft replicated log

`consensus/` implements leader election and log replication. **Testing, in two layers:**

- **Simulation.** A whole cluster on one thread, with a network that drops 15% of messages, repeats and reorders
  them, plus random crashes and partitions. Every step checks election safety, log matching, leader completeness,
  and state machine safety. 600 random scenarios run on each test pass, 300 each on 3 and 5 nodes.
- **Real processes.** `tests/integration/raft_test.py` kills the leader with SIGKILL 6 times under live writes. In a
  recorded run, none of 21,111 acknowledged writes were lost, all 3 logs were identical, and writes resumed
  99 to 169 ms after a leader died. A lone node refused a write. This runs without fsync, so it covers process
  crashes and not power loss.

To check that these tests can fail, five dangerous bugs were put in on purpose, and all five were caught: votes that
ignore log freshness, a vote not saved to disk, committing old-term entries by counting replicas, committing without
a majority, and truncating a log on any append. Chaos runs caught three, and dedicated unit tests the other two.

Writes through the leader, 3 nodes, 100-byte commands, median of 3 runs:

| fsync | Connections | Writes/s | p50 ms | p99 ms |
| --- | --- | --- | --- | --- |
| No | 1 | 3,447 | 0.29 | 0.41 |
| No | 16 | 19,972 | 0.78 | 1.06 |
| No | 64 | 25,753 | 2.41 | 3.67 |
| Yes | 1 | 173 | 5.79 | 10.15 |
| Yes | 16 | 700 | 22.99 | 35.15 |
| Yes | 64 | 2,558 | 25.03 | 37.17 |

No write failed in these runs. All three nodes share one machine and disk, over loopback, so a real cluster would
add network latency to every commit. Concurrent writes share one sync through group commit, which is why synced
throughput rises so much with connections.

The first version resent everything unacknowledged with every write. Correctness tests passed, but under load this
flooded the network until followers started elections, and throughput with 16 connections fell to 13 writes/s.
Pipelining fixed it, at 19,972 writes/s for the same setup.

**Not implemented:** log compaction, so the log grows without bound, and changing cluster membership while running.
This replicated log is also a separate service, and is not wired into the event log or the model registry.

## Phase 4: performance layer

Single-threaded, `-O3`. Our kernels and Eigen run with random weights in the same shape as
`mlp_float.onnx`, which ONNX Runtime runs.

### Matrix multiplication, 256 × 256 × 256

| Implementation | GFLOPS | Time µs | Gap to Eigen |
| --- | --- | --- | --- |
| Scalar, vectorization disabled | 8.7 | 3,843 | 8.8× |
| Compiler auto-vectorized | 27.9 | 1,203 | 2.7× |
| NEON, 4 rows of `b` per pass | 35.2 | 953 | 2.2× |
| NEON micro-kernel, 4 × 12 in registers | 56.0 | 599 | 1.4× |
| NEON micro-kernel with packing and cache blocking | 63.6 | 527 | 1.2× |
| Eigen 3.4 | 76.5 | 439 | none |

### Full forward pass, µs per call

| Batch | Ours, NEON micro | Ours, auto-vectorized | Eigen | ONNX Runtime |
| --- | --- | --- | --- | --- |
| 1 | 5.3 | 7.0 | 5.1 | 7.7 |
| 8 | 27.0 | 54.4 | 28.5 | 16.2 |
| 64 | 219.2 | 430.9 | 157.4 | 113.2 |
| 256 | 853.4 | 1767.1 | 629.5 | 441.0 |

Fitting `time = overhead + cost per row × batch` from batches 1 and 256:

| Implementation | Overhead per call µs | Cost per row µs |
| --- | --- | --- |
| ONNX Runtime | 5.97 | 1.70 |
| Eigen | 2.68 | 2.45 |
| Ours, NEON micro | 1.95 | 3.33 |

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

In isolation, the arena is 5.3× faster: 64 allocations take 455 ns, against 2,412 ns with `malloc`.
**On the server it makes little difference**: 37,429 requests/s without it and 38,122 with it, about 2%, which is
small enough that one run cannot separate it from noise. Only one buffer per batch came from it, so it is off by default. A profile would have
predicted this, and the plan's condition of optimizing where profiling shows it matters did not hold here.

### AVX2

Its correctness tests pass on x86-64, but **its speed has not been measured**. It also has only the simpler design, not the micro-kernel. The plan's
Phase 4 assumed AVX2, and NEON took its place here.

### Packed kernel

Adding packing and cache blocking to the micro-kernel raised 256 × 256 × 256 multiplication from about 56 to about
63.6 GFLOPS, which narrows the gap to Eigen's 76 from 1.4× to 1.2×. **So packing explains part of that gap, and not
all of it.** It also has a cost: a single-row forward pass went from about 5 to 16 µs, since copying costs more than it
saves on small inputs. Run `bench/run_kernels_bench.py` for the full table.

### Lock-free queue

Items per second, handing over small integers, spinning when full or empty.

| Producer and consumer pairs | Mutex queue | Lock-free queue | Lock-free relative |
| --- | --- | --- | --- |
| 1 | 19.16M | 98.08M | 5.12× |
| 2 | 9.74M | 13.58M | 1.39× |
| 4 | 12.29M | 3.77M | 0.31× |
| 8 | 13.29M | 2.39M | 0.18× |

**The lock-free queue is much faster with one pair, and much slower under contention**, on this machine, when threads
outnumber performance cores and spin. That also supports what the engine does: it keeps the mutex queue, which also
lets idle workers sleep. Writing the queue's test found a real bug: with capacity 1, this design cannot tell a full slot
from a free one, so a push overwrote a value. Capacity is now at least 2.

### io_uring

`IP_POLLER=io_uring`, in a build with `IP_ENABLE_IO_URING`, replaces epoll in the event server. The event server's tests
pass with it, and under ASan. In a Linux VM on this Mac, 64 connections, alternating runs:

| Backend | Requests/s | p50 ms | p99 ms |
| --- | --- | --- | --- |
| epoll | 56,316 and 56,322 | 1.00 and 1.01 | 2.53 and 2.51 |
| io_uring | 55,568 and 54,733 | 1.06 and 1.07 | 2.13 and 2.21 |

**No throughput gain, and this was expected.** It only uses io_uring to learn which sockets are ready, and still makes a
system call per read and write, so it skips io_uring's main benefit. Doing the I/O itself through the ring is the real
experiment, and has not been done. Docker's default security profile also blocks io_uring, so this needed
`--security-opt seccomp=unconfined`.

### Core pinning

`--pin-workers 1` gives each worker its own core, on Linux. It has not been benchmarked: macOS has no such control, and
the Linux environment here is a virtual machine. NUMA placement is not implemented, since it needs a multi-socket server
to test.

## Phase 5: observability

### Tracing cost

| Tracing, 16 connections | Requests/s (range) | p50 ms |
| --- | --- | --- |
| Off | 37,614 (36,904 to 38,255) | 0.395 |
| 1 in 100 | 37,276 (34,492 to 37,378) | 0.411 |
| Every request | 37,044 (36,834 to 37,737) | 0.392 |

Both costs are small and within the spread of the runs: 0.9% at 1 in 100, and 1.5% for every request.

### Authentication cost

API key checks, 16 connections, median of 5 interleaved runs:

| API keys | Requests/s (range) | p50 ms | p99 ms |
| --- | --- | --- | --- |
| Off | 37,424 (36,450 to 37,669) | 0.393 | 1.008 |
| On | 37,557 (36,821 to 37,725) | 0.386 | 1.037 |

TLS's cost was not measured, since the load generator does not speak HTTPS.

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
python3 bench/run_all.py                     # About an hour, and it waits if the machine is busy.
python3 bench/plot_results.py
```
