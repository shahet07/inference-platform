#!/usr/bin/env python3
"""Draws the benchmark charts in docs/charts/ from bench/results/, as standalone SVGs.

No plotting library is needed. Each chart adapts to light and dark mode, each point has a hover
tooltip, and docs/benchmarks.md holds the same numbers as tables.
"""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"
OUT = ROOT / "docs" / "charts"

WIDTH, HEIGHT = 720, 400
LEFT, RIGHT, TOP, BOTTOM = 64, 150, 56, 56
STYLE = """
<style>
  :root { --surface:#fcfcfb; --text:#0b0b0b; --muted:#52514e; --grid:#e4e3df; --ideal:#8a8983;
          --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; }
  @media (prefers-color-scheme: dark) {
    :root { --surface:#1a1a19; --text:#f0efec; --muted:#c3c2b7; --grid:#3a3a37; --ideal:#8a8983;
            --s1:#3987e5; --s2:#d95926; --s3:#199e70; }
  }
  text { font: 13px -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; fill: var(--muted); }
  .title { font-size: 16px; font-weight: 600; fill: var(--text); }
  .label { fill: var(--text); }
  .hit { fill: transparent; }
</style>
"""


def nice_top(value):
    """Rounds up to a tidy axis maximum."""
    magnitude = 10 ** len(str(int(value))) / 10
    for step in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if value <= step * magnitude:
            return step * magnitude
    return 10 * magnitude


def line_chart(path, title, x_label, y_label, xs, series, y_format, ideal=None):
    """Draws series, a list of (name, color var, values). One y-axis, starting at zero."""
    plot_w, plot_h = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM
    x_min, x_max = min(xs), max(xs)
    peak = max([v for _, _, values in series for v in values] + ([max(ideal)] if ideal else []))
    y_max = nice_top(peak)

    def x_at(x):
        return LEFT + (x - x_min) / (x_max - x_min) * plot_w

    def y_at(y):
        return TOP + plot_h * (1 - y / y_max)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
           f'role="img" aria-label="{html.escape(title)}">', STYLE,
           f'<rect width="{WIDTH}" height="{HEIGHT}" fill="var(--surface)"/>',
           f'<text class="title" x="{LEFT}" y="28">{html.escape(title)}</text>']
    for i in range(5):
        value = y_max * i / 4
        y = y_at(value)
        out.append(f'<line x1="{LEFT}" x2="{LEFT + plot_w}" y1="{y:.1f}" y2="{y:.1f}" '
                   f'stroke="var(--grid)"/>')
        out.append(f'<text x="{LEFT - 10}" y="{y + 4:.1f}" text-anchor="end">'
                   f'{y_format(value)}</text>')
    for x in xs:
        out.append(f'<text x="{x_at(x):.1f}" y="{TOP + plot_h + 22}" text-anchor="middle">{x}</text>')
    out.append(f'<text x="{LEFT + plot_w / 2}" y="{HEIGHT - 12}" text-anchor="middle">'
               f'{html.escape(x_label)}</text>')
    out.append(f'<text x="{LEFT}" y="{TOP - 10}">{html.escape(y_label)}</text>')

    if ideal:
        points = " ".join(f"{x_at(x):.1f},{y_at(v):.1f}" for x, v in zip(xs, ideal))
        out.append(f'<polyline points="{points}" fill="none" stroke="var(--ideal)" '
                   f'stroke-width="2" stroke-dasharray="4 5"/>')
        out.append(f'<text x="{x_at(xs[-1]) + 10:.1f}" y="{y_at(ideal[-1]) + 4:.1f}">ideal</text>')

    # Direct labels at the line ends, moved apart where they would collide.
    label_ys = {}
    ends = sorted(((y_at(values[-1]), name) for name, _, values in series))
    previous = -100.0
    for y, name in ends:
        y = max(y, previous + 18)
        label_ys[name] = y
        previous = y

    for name, color, values in series:
        points = " ".join(f"{x_at(x):.1f},{y_at(v):.1f}" for x, v in zip(xs, values))
        out.append(f'<polyline points="{points}" fill="none" stroke="var({color})" '
                   f'stroke-width="2" stroke-linejoin="round"/>')
        for x, v in zip(xs, values):
            cx, cy = x_at(x), y_at(v)
            out.append(f'<g><title>{html.escape(name)}, {x}: {y_format(v)}</title>'
                       f'<circle class="hit" cx="{cx:.1f}" cy="{cy:.1f}" r="14"/>'
                       f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="var({color})" '
                       f'stroke="var(--surface)" stroke-width="2"/></g>')
        out.append(f'<text class="label" x="{x_at(xs[-1]) + 10:.1f}" y="{label_ys[name] + 4:.1f}">'
                   f'{html.escape(name)} {y_format(values[-1])}</text>')
    out.append("</svg>")
    path.write_text("\n".join(out) + "\n")
    print(f"Wrote {path}")


def thousands(value):
    return f"{value / 1000:.0f}k" if value >= 1000 else f"{value:.0f}"


def times(value):
    return f"{value:.1f}x"


def load(name):
    return json.loads((RESULTS / name).read_text())


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    rows = load("server_scaling.json")
    line_chart(OUT / "server_scaling.svg", "Single server: throughput by worker threads",
               "Worker threads", "Requests per second",
               [r["workers"] for r in rows],
               [("Throughput", "--s1", [r["rps"] for r in rows])], thousands)

    rows = load("server_scaling.json")
    line_chart(OUT / "server_latency.svg", "Single server: p99 latency by worker threads",
               "Worker threads", "p99 latency, ms",
               [r["workers"] for r in rows],
               [("p99", "--s1", [r["p99_ms"] for r in rows])], lambda v: f"{v:.1f}")

    cluster = load("cluster_scaling.json")
    nodes = [1, 2, 3, 4]
    series = []
    for name, color, label in (("xl", "--s1", "XL model"), ("heavy", "--s2", "Heavy model"),
                               ("light", "--s3", "Light model")):
        base = cluster[name]["1_nodes"]["rps"]
        series.append((label, color, [cluster[name][f"{n}_nodes"]["rps"] / base for n in nodes]))
    line_chart(OUT / "cluster_scaling.svg", "Cluster: throughput relative to one node",
               "Nodes behind the router, on one machine", "Throughput vs 1 node", nodes, series,
               times, ideal=nodes)

    contention = load("hardware_contention.json")
    counts = [r["processes"] for r in contention["small"]]
    series = [(label, color, [r["scaling"] for r in contention[name]])
              for name, color, label in (("small", "--s3", "Light model"),
                                         ("heavy", "--s2", "Heavy model"),
                                         ("xl", "--s1", "XL model"))]
    line_chart(OUT / "hardware_contention.svg", "Inference alone, with no networking",
               "Independent processes", "Throughput vs 1 process", counts, series, times,
               ideal=counts)


if __name__ == "__main__":
    main()
