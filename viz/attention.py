"""Attention heatmaps. Owner: Agent D. See SPEC.md §3."""

from __future__ import annotations


def plot_attention(model, tokenizer, text: str, out_png: str) -> str:
    """Plot attention probabilities for ``text`` as a grid (rows = layers, cols = heads).

    Uses ``model(idx, return_attn=True)``; tick labels are the decoded tokens. ``text`` is
    truncated to ``block_size`` tokens. Saves to ``out_png`` and returns that path.
    """
    raise NotImplementedError
