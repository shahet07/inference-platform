#!/usr/bin/env python3
"""Merges span files from several services and prints traces as timelines.

Usage: trace_view.py FILE [FILE...] [--last N]
"""
import argparse
import collections
import json


def print_trace(trace_id, spans):
    children = collections.defaultdict(list)
    ids = {s["span_id"] for s in spans}
    for span in spans:
        parent = span["parent_span_id"] if span["parent_span_id"] in ids else ""
        children[parent].append(span)
    start = min(s["start_us"] for s in spans)
    print(f"\ntrace {trace_id}")

    def walk(parent, depth):
        for span in sorted(children[parent], key=lambda s: s["start_us"]):
            attributes = " ".join(f"{k}={v}" for k, v in span["attributes"].items())
            label = "  " * depth + span["name"]
            print(f"  {span['start_us'] - start:>8} us {span['duration_us']:>8} us  "
                  f"{span['service']:<10} {label:<28} {attributes}")
            walk(span["span_id"], depth + 1)

    walk("", 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--last", type=int, default=3)
    args = parser.parse_args()

    traces = collections.defaultdict(list)
    for path in args.files:
        with open(path) as f:
            for line in f:
                span = json.loads(line)
                traces[span["trace_id"]].append(span)
    newest = sorted(traces.items(), key=lambda t: min(s["start_us"] for s in t[1]))[-args.last:]
    for trace_id, spans in newest:
        print_trace(trace_id, spans)


if __name__ == "__main__":
    main()
