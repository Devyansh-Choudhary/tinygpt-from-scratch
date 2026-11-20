"""Shared colours and figure helpers for viz/ (backend-independent: no pyplot global state)."""

from __future__ import annotations

import os

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
MASKED = "#ecebe7"  # future (causally masked) cells in attention maps

# Categorical slots in fixed order (validated for adjacent pairs; light ones need direct labels).
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER = "#8a8984"

# Colour follows the run, never its rank: the four experiment runs always get the same slot.
KNOWN_RUNS = ["char", "bpe512", "bpe2048", "bpe512_nopos"]

# Single-hue sequential ramp (blue), near-surface -> dark, for attention probabilities.
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "tinygpt_blues",
    [SURFACE, "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)


def run_colors(run_names) -> dict[str, str]:
    """Stable run -> colour mapping: known runs keep their slot; others take the next free slots."""
    colors: dict[str, str] = {}
    free = [c for i, c in enumerate(CATEGORICAL) if i >= len(KNOWN_RUNS)]
    for name in sorted(run_names):
        if name in KNOWN_RUNS:
            colors[name] = CATEGORICAL[KNOWN_RUNS.index(name)]
        else:
            colors[name] = free.pop(0) if free else OTHER
    return colors


def run_order(run_names) -> list[str]:
    """Known experiment runs first (in SPEC order), then the rest alphabetically."""
    return sorted(run_names, key=lambda n: (KNOWN_RUNS.index(n) if n in KNOWN_RUNS else len(KNOWN_RUNS), n))


def new_figure(width: float, height: float, layout: str | None = None) -> Figure:
    fig = Figure(figsize=(width, height), facecolor=SURFACE, layout=layout)
    FigureCanvasAgg(fig)
    return fig


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)


def save(fig: Figure, out_png: str, dpi: int = 150) -> str:
    d = os.path.dirname(os.path.abspath(out_png))
    os.makedirs(d, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return out_png
