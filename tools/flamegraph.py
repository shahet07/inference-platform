#!/usr/bin/env python3
"""Turns the output of macOS `sample` into a flame graph SVG and a table of hot frames.

Idle threads are excluded, so the percentages are shares of busy samples.

Usage: flamegraph.py SAMPLE_FILE [--svg OUT.svg] [--top N]
"""
import argparse
import collections
import html
import re
import zlib

LINE = re.compile(r"^(\s*(?:[+!:|]\s*)*)(\d+)\s+(.*)$")
IDLE_LEAF = re.compile(
    r"__psynch_cvwait|__select|^kevent|^poll|__accept|__recvfrom|__read_nocancel|__semwait"
    r"|__workq_kernreturn|__sigwait|mach_msg|__ulock_wait|__psynch_mutexwait"
)
WIDTH, ROW_HEIGHT, MIN_WIDTH = 1200, 16, 0.3


class Node:
    def __init__(self, name):
        self.name, self.count, self.children = name, 0, {}


def clean(frame):
    """Drops addresses and offsets, keeping the function name."""
    name = re.sub(r"\s+\(in [^)]*\).*$", "", frame)
    return re.sub(r"\s+\[0x[0-9a-f,]+\].*$", "", name).strip()


def parse_stacks(path):
    """Returns {tuple of frames: samples} from the 'Call graph' section."""
    stacks, stack = collections.Counter(), []
    in_graph = False
    for raw in open(path, errors="replace"):
        if raw.startswith("Call graph:"):
            in_graph = True
            continue
        if not in_graph:
            continue
        if raw.startswith("Total number in stack") or raw.startswith("Sort by top of stack"):
            break
        match = LINE.match(raw.rstrip("\n"))
        if not match:
            continue
        depth, count, frame = len(match[1]), int(match[2]), clean(match[3])
        while stack and stack[-1][0] >= depth:
            finish(stack, stacks)
        if stack:
            stack[-1][2] += count
        stack.append([depth, frame, 0, count])
    while stack:
        finish(stack, stacks)
    return stacks


def finish(stack, stacks):
    """Pops a frame, crediting samples not accounted for by its children as self time."""
    _, _, child_total, count = stack[-1]
    frames = tuple(entry[1] for entry in stack)
    if count > child_total:
        stacks[frames] += count - child_total
    stack.pop()


def busy_only(stacks):
    return {s: n for s, n in stacks.items() if not IDLE_LEAF.search(s[-1])}


def render_svg(stacks, path):
    root = Node("all")
    for frames, count in stacks.items():
        root.count += count
        node = root
        for frame in frames:
            node = node.children.setdefault(frame, Node(frame))
            node.count += count

    rects, max_depth = [], 0

    def walk(node, x, depth):
        nonlocal max_depth
        width = WIDTH * node.count / root.count
        if width < MIN_WIDTH:
            return
        max_depth = max(max_depth, depth)
        rects.append((node, x, depth, width))
        child_x = x
        for child in sorted(node.children.values(), key=lambda c: c.name):
            walk(child, child_x, depth + 1)
            child_x += WIDTH * child.count / root.count

    walk(root, 0.0, 0)
    height = (max_depth + 1) * ROW_HEIGHT + 8
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'font-family="monospace" font-size="11"><rect width="100%" height="100%" fill="#fff"/>'
    ]
    for node, x, depth, width in rects:
        y = height - (depth + 1) * ROW_HEIGHT - 4
        hue = zlib.crc32(node.name.encode()) % 50
        name = html.escape(node.name)
        share = 100 * node.count / root.count
        label = html.escape(node.name[: int(width / 6.6)]) if width > 30 else ""
        out.append(
            f'<g><title>{name} ({share:.1f}%)</title>'
            f'<rect x="{x:.1f}" y="{y}" width="{width:.1f}" height="{ROW_HEIGHT - 1}" '
            f'fill="hsl({hue},85%,62%)"/>'
            f'<text x="{x + 3:.1f}" y="{y + 11}" fill="#000">{label}</text></g>'
        )
    out.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(out))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_file")
    parser.add_argument("--svg")
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    stacks = busy_only(parse_stacks(args.sample_file))
    total = sum(stacks.values())
    inclusive, self_time = collections.Counter(), collections.Counter()
    for frames, count in stacks.items():
        self_time[frames[-1]] += count
        for frame in set(frames):
            inclusive[frame] += count

    print(f"busy samples: {total}")
    print("\ninclusive:")
    for frame, count in inclusive.most_common(args.top):
        print(f"{100 * count / total:5.1f}%  {frame[:100]}")
    print("\nself:")
    for frame, count in self_time.most_common(args.top // 2):
        print(f"{100 * count / total:5.1f}%  {frame[:100]}")
    if args.svg:
        render_svg(stacks, args.svg)


if __name__ == "__main__":
    main()
