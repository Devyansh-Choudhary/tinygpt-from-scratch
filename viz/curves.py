"""Training-curve figures: validation BPC vs training time, and final BPC per run. Owner: Agent D.

Reads ``results/*/log.csv`` (header ``step,elapsed_s,train_loss,val_loss,val_bpc``; SPEC.md §3).
"""

from __future__ import annotations

import csv
import glob
import math
import os

from ._style import (GRID, TEXT_PRIMARY, TEXT_SECONDARY, new_figure, run_colors, run_order, save,
                     style_axes)

LOG_FIELDS = ("step", "elapsed_s", "train_loss", "val_loss", "val_bpc")


def _to_float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def read_logs(results_dir: str = "results") -> dict[str, dict[str, list[float]]]:
    """Return ``{run_name: {column: [values...]}}`` for every ``results_dir/*/log.csv``.

    Rows whose ``val_bpc`` is missing/NaN are dropped. Runs with no usable rows are skipped.
    """
    runs: dict[str, dict[str, list[float]]] = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*", "log.csv"))):
        name = os.path.basename(os.path.dirname(path))
        cols: dict[str, list[float]] = {k: [] for k in LOG_FIELDS}
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                vals = {k: _to_float(row.get(k)) for k in LOG_FIELDS}
                if math.isnan(vals["val_bpc"]) or math.isnan(vals["elapsed_s"]):
                    continue
                for k in LOG_FIELDS:
                    cols[k].append(vals[k])
        if cols["val_bpc"]:
            runs[name] = cols
    return runs


def _require_runs(results_dir: str) -> dict[str, dict[str, list[float]]]:
    runs = read_logs(results_dir)
    if not runs:
        raise FileNotFoundError(f"no usable {results_dir}/*/log.csv files found")
    return runs


def _spread_labels(ys: list[float], min_gap: float) -> list[float]:
    """Nudge label y-positions apart (keeping order) so none are closer than ``min_gap``."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    return out


def plot_bpc_curves(results_dir: str, out_png: str, budget_min: float | None = 7.0,
                    clip_start: bool = True) -> str:
    """Plot val BPC vs elapsed training minutes, one line per ``results_dir/*/log.csv``.

    Args:
        budget_min: draw a reference line at the training budget (``None`` to hide).
        clip_start: the step-0 BPC of an untrained model is far above everything else and would
            flatten the interesting part, so the y-axis starts just above the *second* logged point
            of the worst run; clipped step-0 values are listed in a footnote.

    Returns ``out_png``.
    """
    runs = _require_runs(results_dir)
    colors = run_colors(runs)
    fig = new_figure(8.0, 4.8)
    ax = fig.add_subplot(111)
    style_axes(ax)

    all_bpc = [v for r in runs.values() for v in r["val_bpc"]]
    ymin = min(all_bpc)
    ymax = max(all_bpc)
    clipped: list[str] = []
    if clip_start:
        later = [r["val_bpc"][1] if len(r["val_bpc"]) > 1 else r["val_bpc"][0] for r in runs.values()]
        top = max(later)
        cand_top = top + 0.12 * (top - ymin if top > ymin else 1.0)
        if cand_top < ymax:
            ymax = cand_top
            clipped = [f"{n} {r['val_bpc'][0]:.2f}" for n, r in sorted(runs.items())
                       if r["val_bpc"][0] > ymax]
    pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
    ax.set_ylim(ymin - pad, ymax + pad)

    xmax = max(max(r["elapsed_s"]) for r in runs.values()) / 60.0
    ends = []
    for name in run_order(runs):
        r = runs[name]
        x = [s / 60.0 for s in r["elapsed_s"]]
        ax.plot(x, r["val_bpc"], color=colors[name], lw=1.8, solid_joinstyle="round",
                solid_capstyle="round", label=name, zorder=3)
        ax.plot(x[-1], r["val_bpc"][-1], "o", ms=4.5, color=colors[name], mec="white", mew=1.2,
                zorder=4)
        ends.append((name, x[-1], r["val_bpc"][-1]))

    # Direct labels at line ends (text in ink, not series colour); leader lines when nudged.
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    label_y = _spread_labels([e[2] for e in ends], 0.055 * span)
    xr = xmax * 1.03 if xmax > 0 else 0.1
    for (name, x, y), ly in zip(ends, label_y):
        ax.annotate(f"{name}  {y:.3f}", xy=(x, y), xytext=(xr, ly), textcoords="data",
                    va="center", ha="left", fontsize=9, color=TEXT_PRIMARY,
                    arrowprops=dict(arrowstyle="-", color=colors[name], lw=0.8,
                                    shrinkA=2, shrinkB=3),
                    annotation_clip=False)
    ax.set_xlim(0, xr)

    if budget_min is not None and budget_min > 0 and budget_min <= xmax * 1.2:
        ax.axvline(budget_min, color="#b8b7b1", lw=0.8, zorder=1)
        ax.text(budget_min, ax.get_ylim()[1], f" {budget_min:g}-min budget", va="top", ha="left",
                fontsize=8, color=TEXT_SECONDARY)

    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("training time (minutes)")
    ax.set_ylabel("validation bits per character  (lower is better)")
    ax.set_title("Validation BPC vs training time, by tokenizer", loc="left", fontsize=12,
                 color=TEXT_PRIMARY, pad=28)
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=9,
                    ncols=min(4, len(runs)), handlelength=1.6, borderaxespad=0.2)
    for t in leg.get_texts():
        t.set_color(TEXT_PRIMARY)
    if clipped:
        fig.text(0.01, -0.01, "y-axis clipped; step-0 BPC: " + ", ".join(clipped), fontsize=8,
                 color=TEXT_SECONDARY, ha="left", va="top")
    return save(fig, out_png)


def final_bpc(results_dir: str = "results", checkpoint_dir: str | None = "checkpoints"
              ) -> dict[str, tuple[float, str]]:
    """Return ``{run: (bpc, source)}``.

    Prefers ``checkpoints/<run>.pt["final_val_bpc"]`` (full 200k-char eval); falls back to the last
    ``val_bpc`` row of the run's log (which may use a smaller eval during training).
    """
    out: dict[str, tuple[float, str]] = {}
    for name, r in read_logs(results_dir).items():
        out[name] = (r["val_bpc"][-1], "log")
    if checkpoint_dir:
        import torch  # local import: curves are usable without torch

        for path in sorted(glob.glob(os.path.join(checkpoint_dir, "*.pt"))):
            name = os.path.splitext(os.path.basename(path))[0]
            try:
                ck = torch.load(path, map_location="cpu", weights_only=False)
                out[ck.get("run_name", name)] = (float(ck["final_val_bpc"]), "checkpoint")
            except Exception:  # noqa: BLE001 — a bad checkpoint must not break the figure
                continue
    return out


def plot_final_bpc(results_dir: str, out_png: str, checkpoint_dir: str | None = "checkpoints") -> str:
    """Horizontal bar chart of final validation BPC per run (best on top). Returns ``out_png``."""
    vals = final_bpc(results_dir, checkpoint_dir)
    if not vals:
        raise FileNotFoundError(f"no runs found in {results_dir}/*/log.csv or {checkpoint_dir}/*.pt")
    colors = run_colors(vals)
    items = sorted(vals.items(), key=lambda kv: kv[1][0], reverse=True)  # barh draws bottom-up
    names = [k for k, _ in items]
    bpcs = [v[0] for _, v in items]
    fig = new_figure(7.0, 0.55 * len(items) + 1.6)
    ax = fig.add_subplot(111)
    style_axes(ax)
    ax.barh(range(len(items)), bpcs, height=0.55, color=[colors[n] for n in names], zorder=3)
    ax.set_yticks(range(len(items)), names)
    ax.tick_params(axis="y", length=0, labelsize=10, labelcolor=TEXT_PRIMARY)
    for i, b in enumerate(bpcs):
        ax.text(b, i, f"  {b:.3f}", va="center", ha="left", fontsize=9, color=TEXT_PRIMARY)
    ax.set_xlim(0, max(bpcs) * 1.15)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("final validation bits per character  (lower is better)")
    ax.set_title("Final validation BPC per run", loc="left", fontsize=12, color=TEXT_PRIMARY, pad=10)
    sources = {v[1] for v in vals.values()}
    note = {"checkpoint": "full 200k-char eval from checkpoints", "log": "last log.csv row"}
    fig.text(0.01, -0.01, "source: " + "; ".join(note[s] for s in sorted(sources)), fontsize=8,
             color=TEXT_SECONDARY, ha="left", va="top")
    return save(fig, out_png)
