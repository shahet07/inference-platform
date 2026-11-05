#!/usr/bin/env python3
"""Generates ops/grafana/dashboards/inference.json, so the panels can be reviewed as code."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "grafana" / "dashboards" / "inference.json"
RATE = "[30s]"


def quantile(q, metric):
    return f"histogram_quantile({q}, sum by (le) (rate({metric}_bucket{RATE})))"


# (title, unit, [(query, legend)])
PANELS = [
    ("Throughput per node", "reqps",
     [(f"sum by (instance) (rate(inference_requests_total{RATE}))", "{{instance}}")]),
    ("Router request latency", "s",
     [(quantile(0.5, "router_forward_seconds"), "p50"),
      (quantile(0.99, "router_forward_seconds"), "p99")]),
    ("Server request latency", "s",
     [(quantile(0.5, "http_request_duration_seconds"), "p50"),
      (quantile(0.99, "http_request_duration_seconds"), "p99")]),
    ("Queue depth", "short", [("sum by (instance) (inference_queue_depth)", "{{instance}}")]),
    ("Mean batch size", "short",
     [(f"sum(rate(inference_batch_size_sum{RATE})) / sum(rate(inference_batch_size_count{RATE}))",
       "requests per batch")]),
    ("CPU cores used", "short",
     [(f"sum by (job) (rate(process_cpu_seconds_total{RATE}))", "{{job}}")]),
    ("Shed load and failovers", "reqps",
     [(f"sum(rate(inference_rejected_total{RATE}))", "rejected: queue full"),
      (f"sum(rate(router_failovers_total{RATE}))", "failovers"),
      (f"sum(rate(router_unavailable_total{RATE}))", "unavailable")]),
    ("Healthy backends", "short", [("router_healthy_backends", "healthy")]),
    ("Event stream", "short",
     [(f"sum(rate(stream_appended_total{RATE}))", "appended/s"),
      (f"sum(rate(stream_dropped_newest_total{RATE}))", "dropped/s")]),
]


def panel(index, title, unit, queries):
    return {
        "id": index + 1,
        "title": title,
        "type": "timeseries",
        "datasource": {"type": "prometheus", "uid": "prometheus"},
        "gridPos": {"h": 8, "w": 8, "x": (index % 3) * 8, "y": (index // 3) * 8},
        "fieldConfig": {"defaults": {"unit": unit, "min": 0,
                                     "custom": {"lineWidth": 2, "fillOpacity": 0}}, "overrides": []},
        "options": {"legend": {"displayMode": "list", "placement": "bottom"},
                    "tooltip": {"mode": "multi"}},
        "targets": [{"refId": chr(65 + i), "expr": q, "legendFormat": legend}
                    for i, (q, legend) in enumerate(queries)],
    }


OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({
    "uid": "inference",
    "title": "Inference Platform",
    "schemaVersion": 39,
    "time": {"from": "now-5m", "to": "now"},
    "refresh": "5s",
    "panels": [panel(i, *p) for i, p in enumerate(PANELS)],
}, indent=2) + "\n")
print(f"Wrote {OUT}")
