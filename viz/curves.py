"""Validation BPC learning curves. Owner: Agent D. See SPEC.md §3."""

from __future__ import annotations


def plot_bpc_curves(results_dir: str, out_png: str) -> str:
    """Plot ``val_bpc`` vs ``elapsed_s`` for every ``{results_dir}/*/log.csv`` (one line per run).

    Saves to ``out_png`` and returns that path.
    """
    raise NotImplementedError
