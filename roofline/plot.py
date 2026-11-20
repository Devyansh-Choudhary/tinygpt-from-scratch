"""Roofline figure. Owner: Agent G.

``plot_roofline(peak_flops, bandwidth, points, out_png)`` draws a log-log roofline: the
bandwidth slope ``GFLOP/s = GB/s x intensity`` up to the ridge point, then the flat compute roof,
with each op as a labelled point at (arithmetic intensity, achieved GFLOP/s).

A *point* is a dict with keys ``name``, ``intensity`` (FLOP/byte), ``gflops`` (achieved), ``kind``
(``"matmul" | "attention" | "elementwise"``) and optionally ``gflops_alt`` (a second measurement
of the same op, drawn hollow and joined to the first -- used for the page-fault regime).
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Categorical slots 1-3 of the reference palette (validated all-pairs, light mode).
KIND_COLORS = {"matmul": "#2a78d6", "attention": "#1baf7a", "elementwise": "#eb6834"}
KIND_LABELS = {
    "matmul": "Linear-layer matmul",
    "attention": "attention matmul (per head)",
    "elementwise": "elementwise / reduction",
}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"


def _label_layout(points, min_gap: float = 0.085, y_max: float | None = None) -> list[tuple[float, float]]:
    """Label anchor (x, y) per point in data coords: points are clustered by log-intensity, each
    cluster gets a label column just right of its rightmost point, and labels in a column are spread
    vertically (in log10 space) so they are at least ``min_gap`` decades apart."""
    lx = [np.log10(p["intensity"]) for p in points]
    ly = [np.log10(max(p["gflops"], 1e-9)) for p in points]
    order = sorted(range(len(points)), key=lambda i: lx[i])
    clusters, cur = [], [order[0]] if order else []
    for i in order[1:]:
        if lx[i] - lx[cur[-1]] < 0.35:
            cur.append(i)
        else:
            clusters.append(cur)
            cur = [i]
    if cur:
        clusters.append(cur)
    pos = [None] * len(points)
    for cl in clusters:
        col_x = max(lx[i] for i in cl) + 0.12
        idx = sorted(cl, key=lambda i: -ly[i])
        ys = [ly[i] for i in idx]
        for _ in range(200):  # relax overlaps top-down then recentre
            moved = False
            for j in range(1, len(ys)):
                if ys[j - 1] - ys[j] < min_gap:
                    ys[j] = ys[j - 1] - min_gap
                    moved = True
            shift = (np.mean([ly[i] for i in idx]) - np.mean(ys)) * 0.5
            ys = [y + shift for y in ys]
            if not moved and abs(shift) < 1e-4:
                break
        if y_max is not None and ys and ys[0] > np.log10(y_max):
            ys = [y - (ys[0] - np.log10(y_max)) for y in ys]
        for i, y in zip(idx, ys):
            pos[i] = (10**col_x, 10**y)
    return pos


def _slope_label(ax, x, y, x2, y2, text, color):
    """Write ``text`` at (x, y) rotated to follow the segment to (x2, y2) on screen."""
    p1, p2 = ax.transData.transform((x, y)), ax.transData.transform((x2, y2))
    ang = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
    ax.annotate(text, (x, y), xytext=(2, 4), textcoords="offset points", rotation=ang,
                rotation_mode="anchor", color=color, fontsize=8.5)


def draw_roofline(ax, peak_flops, bandwidth, points, cache_bandwidth=None, title=None, xlim=(0.1, 1000)):
    """Draw onto an existing matplotlib ``ax``; returns ``ax``."""
    peak, bw = peak_flops / 1e9, bandwidth / 1e9
    ridge = peak / bw
    xs = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 400)
    ax.plot(xs, np.minimum(peak, bw * xs), color=INK, lw=2, zorder=2)
    ax.annotate(
        f"compute roof {peak:.0f} GFLOP/s",
        (xlim[1], peak),
        xytext=(-4, 5),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color=INK2,
        fontsize=8.5,
    )
    if cache_bandwidth:
        cbw = cache_bandwidth / 1e9
        ax.plot(xs, np.minimum(peak, cbw * xs), color=MUTED, lw=1.2, ls="--", zorder=1)
    ax.axvline(ridge, color=MUTED, lw=1, ls=":", zorder=1)
    ax.annotate(
        f"ridge {ridge:.1f} FLOP/B",
        (ridge, 0),
        xycoords=("data", "axes fraction"),
        xytext=(4, 6),
        textcoords="offset points",
        color=INK2,
        fontsize=8.5,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ymin = min([1.0] + [p["gflops"] for p in points] + [p.get("gflops_alt") or 1e9 for p in points]) / 2
    ax.set_ylim(ymin, peak * 2)
    ax.set_xlabel("arithmetic intensity (FLOPs / byte)", color=INK2)
    ax.set_ylabel("achieved GFLOP/s", color=INK2)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.tick_params(colors=INK2, labelsize=8.5)
    for s in ax.spines.values():
        s.set_color(GRID)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left")

    # Roof labels follow their slope (needs final limits for the on-screen angle).
    x0 = xlim[0] * 1.3
    _slope_label(ax, x0, bw * x0, x0 * 2, bw * x0 * 2, f"DRAM {bw:.0f} GB/s", INK2)
    if cache_bandwidth:
        _slope_label(ax, x0, cbw * x0, x0 * 2, cbw * x0 * 2, f"cache-resident {cbw:.0f} GB/s", MUTED)

    seen = set()
    for p in points:
        col = KIND_COLORS.get(p.get("kind", "matmul"), INK)
        lab = KIND_LABELS.get(p.get("kind")) if p.get("kind") not in seen else None
        seen.add(p.get("kind"))
        if p.get("gflops_alt") is not None:
            ax.plot([p["intensity"]] * 2, [p["gflops"], p["gflops_alt"]], color=col, lw=1, alpha=0.6, zorder=3)
            ax.scatter(
                [p["intensity"]], [p["gflops_alt"]], s=40, facecolors="white", edgecolors=col, linewidths=1.5, zorder=4
            )
        ax.scatter([p["intensity"]], [p["gflops"]], s=56, color=col, edgecolors="white", linewidths=1.5, zorder=5, label=lab)
    for p, (tx, ty) in zip(points, _label_layout(points, y_max=peak * 0.8)):
        ax.annotate(
            p["name"],
            (p["intensity"], p["gflops"]),
            xytext=(tx, ty),
            textcoords="data",
            fontsize=8,
            color=INK,
            va="center",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=0, shrinkB=3),
            zorder=6,
        )
    return ax


def plot_roofline(
    peak_flops: float,
    bandwidth: float,
    points: list[dict],
    out_png: str,
    cache_bandwidth: float | None = None,
    title: str | None = None,
    alt_label: str | None = None,
) -> str:
    """Draw the log-log roofline with each op as a labelled point; save to ``out_png``; return it."""
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
    fig.patch.set_facecolor("white")
    draw_roofline(ax, peak_flops, bandwidth, points, cache_bandwidth, title)
    if alt_label and any(p.get("gflops_alt") is not None for p in points):
        ax.scatter([], [], s=46, facecolors="white", edgecolors=MUTED, linewidths=1.5, label=alt_label)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 0.05), fontsize=8.5, frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def plot_panels(
    peak_flops: float, bandwidth: float, panels: list[tuple[str, list[dict]]], out_png: str, cache_bandwidth=None
) -> str:
    """Small multiples: one roofline per ``(title, points)`` panel, shared axes."""
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5.2), dpi=130, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (title, pts) in zip(axes, panels):
        draw_roofline(ax, peak_flops, bandwidth, pts, cache_bandwidth, title)
    axes[0].legend(loc="lower right", bbox_to_anchor=(1, 0.07), fontsize=8, frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    return out_png
