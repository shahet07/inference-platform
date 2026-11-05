# Inference Platform

A C++20 system for serving machine learning models and handling streaming data: an inference server
with dynamic batching, a sharded streaming engine with backpressure, a router with consistent hashing
and failover, and the metrics, logs, and traces to operate them.

Built one subsystem at a time, with each measured before moving on. The most useful parts may be
[what limited performance, and what did not](docs/benchmarks.md), and the
[design trade-offs](docs/design.md).

## Results

On one 12-core Apple Silicon laptop, sharing the machine with the load generator, using a small synthetic
model. [Read the caveats](docs/benchmarks.md#read-this-first) before quoting these.

| What | Result |
| --- | --- |
| Inference server, 16 connections | 36,741 requests/s at p99 1.01 ms |
| Removing the batch wait, found through metrics | 3.5× throughput, p50 from 1.67 to 0.41 ms |
| Streaming engine, 4 producers on 8 shards | 15.7M events/s at append p99 8.2 µs, 29× a Python baseline |
| Batching consumer wakeups, 8 shards | 6.4× throughput, p99 from 28 to 8.3 µs |
| Cluster, 1 to 4 nodes on one machine | 2.2× at best, with the cause identified for each workload |
| Killing 1 of 3 nodes under test | No failed requests; only that node's keys moved |

## Layout

| Directory | Contents |
| --- | --- |
| `runtime/` | Model loading and inference, on ONNX Runtime |
| `server/` | Bounded queue, batching engine, HTTP server |
| `stream/` | Sharded log, consumer groups, windowed aggregation |
| `cluster/` | Hash ring, health checks, router, metadata replication |
| `perf/` | Arena allocator, SIMD kernels |
| `obs/` | Metrics, structured logs, tracing |
| `bench/` | Load generator, benchmarks, results |
| `ops/` | Prometheus and Grafana setup, demo cluster |
| `tests/` | Unit and end-to-end tests |
| `tools/` | Model generation, flame graphs, trace viewer |
| `docs/` | Design notes, benchmark results |

## Build and test

Requires CMake 3.24+, Ninja, Python 3, and a C++20 compiler. CMake downloads its dependencies on first
configure. Pinned ONNX Runtime binaries exist for macOS arm64, Linux arm64, and Linux x86-64. The tests pass
on macOS arm64 and Linux arm64 and x86-64, and CI runs them on every push.

```sh
brew install cmake ninja llvm   # llvm provides clang-tidy and clang-format
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

| Preset | Purpose |
| --- | --- |
| `debug` | clang-tidy and warnings as errors |
| `release` | Optimized, for benchmarks and running |
| `asan` | AddressSanitizer and UndefinedBehaviorSanitizer |
| `tsan` | ThreadSanitizer |

## Run it

```sh
cmake --preset release && cmake --build --preset release

./build/release/server/inference_server --model bench/models/mlp_quant.onnx --port 8080

curl -s localhost:8080/v1/infer -d "{\"inputs\":[$(python3 -c "print(','.join(['0.5']*64))")]}"
curl -s localhost:8080/metrics
curl -s localhost:8080/v1/stats
```

Requests can name a model, set a deadline, and skip JSON:

```sh
curl -s localhost:8080/v1/models/resnet/infer -H 'X-Deadline-Ms: 100' \
  -H 'Content-Type: application/octet-stream' --data-binary @input.bin
```

To serve more models, or replace one with no downtime, start with `--model-dir DIR`, then:

```sh
curl -X PUT localhost:8080/v1/models/resnet -d '{"uri":"resnet-v2.onnx"}'
curl -s localhost:8080/v1/loaded
```

A demo cluster with the dashboard, at http://localhost:3000/d/inference:

```sh
docker compose -f ops/docker-compose.yml up -d
python3 ops/run_demo.py --kill-after 70    # 3 nodes and a router, with a node failure
```

## Runbook

**Endpoints.** Nodes and the router both serve `GET /healthz` and `GET /metrics`. The router's
`GET /v1/cluster` lists backend health. Logs are JSON lines on stderr.

### Requests return 503 "queue full"

The node is shedding load on purpose.

1. Check `inference_queue_depth` and `rate(inference_rejected_total[1m])`.
2. If `inference_model_run_seconds` rose, the model or the machine has slowed down.
3. If not, traffic has outgrown capacity. Add nodes, or `--workers` if CPU is free. A larger `--queue`
   only absorbs short bursts, and it raises latency.

### Requests return 503 "no backend available"

Every backend is down, or the router cannot reach them.

1. `curl -s localhost:8000/v1/cluster`
2. Look for `backend removed from rotation` in the router log.
3. Check each node's `/healthz`, its process, and its log.

The router restores a backend after 2 good probes, about half a second, with no action needed.

### A new model is not taking traffic

Check `curl -s localhost:8080/v1/loaded`, `model_load_failures_total`, and the log for `model load failed` or
`refused model uri`. The previous model keeps serving after a failure. A failed version is not retried, so
fix the file and `PUT` again to make a new version.

### Requests return 504

The request's `X-Deadline-Ms` passed. If this is new, check `inference_queue_wait_seconds` and
`rate(inference_deadline_exceeded_total[1m])`, since queueing is the usual cause.

### Latency is high

1. Compare `router_forward_seconds` with `http_request_duration_seconds`. A large difference means the
   time is in the router or between processes.
2. On a node, compare `inference_queue_wait_seconds` with `inference_model_run_seconds`. Queue wait
   growing means too little capacity, or too long a `--max-wait-us`. Model run growing means the model or
   the machine is slower.
3. Check `process_cpu_seconds_total` for saturation.
4. For one request, start nodes and router with `--trace-file` and merge the files:
   `python3 tools/trace_view.py router.spans node-*.spans`

### Events are missing from `/v1/stats`

Check `rate(stream_dropped_newest_total[1m])`. Above zero, the consumers cannot keep up, and the log is
dropping events by design so that inference is not delayed. Windows also only appear once they close,
1 second by default. Without `--events-dir`, the log is in memory and a restart loses events.

### Model metadata differs between nodes

Each node pulls from `--peers` every `--replication-interval-ms`, 1 second by default. A difference that
lasts means a node cannot reach a peer, so check the `--peers` list and connectivity. If two nodes wrote
the same model at once, the higher `(version, node id)` wins and the other write is lost by design.

### Shutdown

`SIGTERM` or `SIGINT` stops accepting requests and exits cleanly, logging `stopped`. Queued events are
drained. The event log survives the restart only if `--events-dir` is set.

## Limitations

No TLS or authentication, a single router instance. All
benchmarks ran on one machine. See [design notes](docs/design.md#known-gaps).

## License

[MIT](LICENSE)
