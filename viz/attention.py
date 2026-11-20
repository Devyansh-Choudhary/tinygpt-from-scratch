"""Attention heatmaps and an "interesting head" finder. Owner: Agent D.

Uses ``model.forward(idx, return_attn=True) -> (logits, loss, attn)`` where ``attn`` is a list (one
per layer) of ``[B, n_head, T, T]`` softmax probabilities (SPEC.md §3).
"""

from __future__ import annotations

import math

import numpy as np

from ._style import (MASKED, SEQ_BLUE, TEXT_PRIMARY, TEXT_SECONDARY, new_figure, save,
                     style_axes)

VISIBLE = {" ": "␣", "\n": "↵", "\t": "→", "\r": "␍"}  # ␣ ↵ → ␍


def visible(s: str) -> str:
    """Make whitespace visible: space -> ␣, newline -> ↵, tab -> →."""
    return "".join(VISIBLE.get(c, c) for c in s)


def token_labels(tokenizer, ids: list[int]) -> list[str]:
    """Decode every token on its own and make whitespace visible (``"␣the"``, ``"↵"``)."""
    out = []
    for i in ids:
        try:
            s = tokenizer.decode([int(i)])
        except Exception:  # noqa: BLE001 — e.g. a lone partial byte; fall back to the id
            s = f"<{int(i)}>"
        out.append(visible(s) if s else f"<{int(i)}>")
    return out


def _block_size(model) -> int | None:
    for attr in ("cfg", "config"):
        cfg = getattr(model, attr, None)
        if cfg is not None and getattr(cfg, "block_size", None):
            return int(cfg.block_size)
    if getattr(model, "block_size", None):
        return int(model.block_size)
    pos = getattr(model, "pos_emb", None)
    if pos is not None and hasattr(pos, "num_embeddings"):
        return int(pos.num_embeddings)
    return None


def get_attention(model, tokenizer, text: str, max_tokens: int = 48):
    """Run ``text`` through ``model`` and return ``(ids, attn)``.

    ``ids`` is the list of the first ``min(max_tokens, block_size)`` token ids of ``text``;
    ``attn`` is a float numpy array ``[n_layer, n_head, T, T]`` (batch element 0).
    """
    import torch

    ids = list(tokenizer.encode(text))
    if not ids:
        raise ValueError("text encodes to zero tokens")
    limit = max_tokens
    bs = _block_size(model)
    if bs is not None:
        limit = min(limit, bs)
    ids = ids[:limit]
    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()
    try:
        with torch.no_grad():
            out = model(torch.tensor([ids], dtype=torch.long), return_attn=True)
    finally:
        if was_training and hasattr(model, "train"):
            model.train()
    attn = out[-1]
    arr = np.stack([a[0].detach().float().cpu().numpy() for a in attn])  # [L, H, T, T]
    return ids, arr


# ----------------------------------------------------------------------------- head statistics

def head_stats(a: np.ndarray) -> dict[str, float]:
    """Summary statistics of one head's attention ``a [T, T]`` (rows = queries).

    Averages over query positions ``t >= 1`` (row 0 can only attend to itself):
      - ``prev``: mean weight on the immediately preceding token ``t-1``.
      - ``first``: mean weight on token 0 (an "attention sink").
      - ``self``: mean weight on the current token ``t``.
      - ``entropy_bits``: mean row entropy in bits.
      - ``norm_entropy``: mean of entropy / log(t+1), i.e. 0 = one-hot, 1 = uniform over the past.
    """
    T = a.shape[-1]
    if T < 2:
        return dict(prev=0.0, first=1.0, self=1.0, entropy_bits=0.0, norm_entropy=0.0)
    rows = np.arange(1, T)
    p = np.clip(a[rows], 0.0, None)
    p = p / np.maximum(p.sum(axis=-1, keepdims=True), 1e-12)
    ent = -(p * np.log(np.where(p > 0, p, 1.0))).sum(axis=-1)  # nats
    return dict(
        prev=float(p[np.arange(T - 1), rows - 1].mean()),
        first=float(p[:, 0].mean()),
        self=float(p[np.arange(T - 1), rows].mean()),
        entropy_bits=float(ent.mean() / math.log(2)),
        norm_entropy=float((ent / np.log(rows + 1)).mean()),
    )


def describe_head(a: np.ndarray, labels: list[str] | None = None) -> tuple[str, float, str]:
    """Classify one head. Returns ``(kind, interest_score in [0,1], one-line description)``.

    Kinds: ``previous-token``, ``current-token``, ``first-token (sink)``, ``focused``, ``diffuse``.
    The first-token pattern is down-weighted: attending to token 0 is usually a "no-op" sink.
    """
    s = head_stats(a)
    focus = 1.0 - s["norm_entropy"]
    cands = {
        "previous-token": s["prev"],
        "current-token": s["self"],
        "first-token (sink)": 0.5 * s["first"],
    }
    kind = max(cands, key=cands.get)
    score = cands[kind]
    ent = s["entropy_bits"]
    if score >= 0.4:
        share = {"previous-token": s["prev"], "current-token": s["self"],
                 "first-token (sink)": s["first"]}[kind]
        where = {"previous-token": "the immediately preceding token",
                 "current-token": "the current token itself",
                 "first-token (sink)": "the very first token"}[kind]
        desc = (f"{kind} head: on average {share:.0%} of each token's attention goes to {where} "
                f"(mean entropy {ent:.2f} bits)")
        return kind, max(score, focus), desc
    if focus >= 0.4:
        T = a.shape[-1]
        t = int(np.argmax(a[1:].max(axis=-1))) + 1 if T > 1 else 0
        j = int(np.argmax(a[t]))
        ex = ""
        if labels is not None and t < len(labels):
            ex = f"; e.g. '{labels[t]}' looks at '{labels[j]}' ({a[t, j]:.0%}, {t - j} back)"
        return "focused", focus, f"focused head: sharp, content-dependent attention (mean entropy {ent:.2f} bits){ex}"
    return "diffuse", focus, (f"diffuse head: spreads attention broadly over the context "
                              f"(mean entropy {ent:.2f} bits), like a running average")


def find_interesting_head(attn: np.ndarray, labels: list[str] | None = None) -> dict:
    """Pick the single most "interesting" head of ``attn [n_layer, n_head, T, T]``.

    Interest = the strongest of (previous-token share, current-token share, ½·first-token share,
    1 − normalised entropy). Returns ``{"layer", "head", "kind", "score", "caption", "stats"}``.
    """
    best = None
    for li in range(attn.shape[0]):
        for hi in range(attn.shape[1]):
            kind, score, desc = describe_head(attn[li, hi], labels)
            if best is None or score > best["score"] + 1e-12:
                best = dict(layer=li, head=hi, kind=kind, score=float(score),
                            caption=f"Layer {li}, head {hi} — {desc}.",
                            stats=head_stats(attn[li, hi]))
    return best


# ----------------------------------------------------------------------------- plotting

def _heatmap(ax, a: np.ndarray, labels, xlabels: bool, ylabels: bool, fontsize: float):
    T = a.shape[-1]
    masked = np.ma.masked_where(np.triu(np.ones((T, T), dtype=bool), k=1), a)
    cmap = SEQ_BLUE.with_extremes(bad=MASKED)
    im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest", aspect="equal")
    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    ax.set_xticklabels(labels if xlabels else [], rotation=90, fontsize=fontsize, family="DejaVu Sans")
    ax.set_yticklabels(labels if ylabels else [], fontsize=fontsize, family="DejaVu Sans")
    ax.tick_params(length=0, pad=1.5, colors=TEXT_SECONDARY)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im


def plot_attention(model, tokenizer, text: str, out_png: str, max_tokens: int = 32,
                   title: str | None = None) -> str:
    """Grid of attention heatmaps (rows = layers, cols = heads) for ``text``. Returns ``out_png``.

    Rows of each map are query tokens, columns are the tokens attended to; tick labels are the
    decoded tokens with visible whitespace (␣ space, ↵ newline). Future positions are grey.
    Only the first ``max_tokens`` tokens (and at most ``block_size``) are shown, for legibility.
    """
    ids, attn = get_attention(model, tokenizer, text, max_tokens=max_tokens)
    labels = token_labels(tokenizer, ids)
    L, H, T, _ = attn.shape
    cell = max(1.8, min(4.2, 0.11 * T + 0.6))
    fs = 7 if T <= 24 else 6
    fig = new_figure(cell * H + 1.4, cell * L + 1.3, layout="constrained")
    axes = fig.subplots(L, H, squeeze=False)
    im = None
    for li in range(L):
        for hi in range(H):
            ax = axes[li, hi]
            im = _heatmap(ax, attn[li, hi], labels, xlabels=(li == L - 1), ylabels=(hi == 0), fontsize=fs)
            if li == 0:
                ax.set_title(f"head {hi}", fontsize=10, color=TEXT_PRIMARY)
            if hi == 0:
                ax.set_ylabel(f"layer {li}\nquery token", fontsize=9, color=TEXT_PRIMARY)
            if li == L - 1:
                ax.set_xlabel("attended-to token", fontsize=8, color=TEXT_SECONDARY)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01, shrink=0.6)
    cb.set_label("attention probability", color=TEXT_SECONDARY, fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    name = getattr(tokenizer, "name", "")
    head = title or f"Attention by layer and head{f'  ·  {name} tokenizer' if name else ''}"
    fig.suptitle(f"{head}\n\"{visible(tokenizer.decode(ids))}\"  ({T} tokens)", fontsize=11,
                 color=TEXT_PRIMARY, x=0.01, ha="left")
    return save(fig, out_png)


def plot_interesting_head(model, tokenizer, text: str, out_png: str, max_tokens: int = 48,
                          ) -> tuple[str, str]:
    """Find the most "interesting" head for ``text`` (see :func:`find_interesting_head`), plot it
    large with a one-line caption, and return ``(out_png, caption)``."""
    ids, attn = get_attention(model, tokenizer, text, max_tokens=max_tokens)
    labels = token_labels(tokenizer, ids)
    info = find_interesting_head(attn, labels)
    a = attn[info["layer"], info["head"]]
    T = a.shape[-1]
    side = max(4.0, min(9.0, 0.17 * T + 1.5))
    fig = new_figure(side + 1.2, side + 0.4)
    ax = fig.add_subplot(111)
    style_axes(ax)
    im = _heatmap(ax, a, labels, True, True, fontsize=9 if T <= 24 else 7)
    ax.set_xlabel("attended-to token")
    ax.set_ylabel("query token")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    cb.set_label("attention probability", color=TEXT_SECONDARY, fontsize=9)
    ax.set_title(f"Most distinctive head: layer {info['layer']}, head {info['head']} ({info['kind']})",
                 loc="left", fontsize=11, color=TEXT_PRIMARY, pad=10)
    fig.text(0.02, -0.005, info["caption"], fontsize=8.5, color=TEXT_SECONDARY, ha="left", va="top",
             wrap=True)
    return save(fig, out_png), info["caption"]
