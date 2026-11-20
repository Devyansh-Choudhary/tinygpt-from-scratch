"""Regenerate every viz figure into ``results/figures/``. Owner: Agent D.

    python -m viz.make_all [--results_dir results] [--checkpoint_dir checkpoints]
                           [--out_dir results/figures] [--text "..."]

Writes ``bpc_curves.png``, ``final_bpc.png`` and, for each ``checkpoints/<run>.pt``,
``attention_<run>.png`` (all layers x heads) and ``attention_<run>_head.png`` (most distinctive head;
its caption also goes to ``attention_captions.txt``). Missing inputs are skipped with a message.
"""

from __future__ import annotations

import argparse
import glob
import os
import random

DEFAULT_TEXT = "Once upon a time, there was a little girl named Lily.\nShe loved to play in the park."


def _load_checkpoint(path: str):
    from tinygpt.evaluate import load_checkpoint

    return load_checkpoint(path)


def _seed() -> None:
    random.seed(1337)
    try:
        import numpy as np
        import torch

        np.random.seed(1337)
        torch.manual_seed(1337)
        torch.set_num_threads(4)
    except ImportError:
        pass


def make_all(results_dir: str = "results", checkpoint_dir: str = "checkpoints",
             out_dir: str | None = None, text: str = DEFAULT_TEXT) -> list[str]:
    """Build every figure; return the list of written paths."""
    from .attention import plot_attention, plot_interesting_head
    from .curves import plot_bpc_curves, plot_final_bpc

    _seed()
    out_dir = out_dir or os.path.join(results_dir, "figures")
    written: list[str] = []

    def attempt(label, fn, *a, **kw):
        try:
            r = fn(*a, **kw)
        except (FileNotFoundError, NotImplementedError) as e:
            print(f"[viz] skip {label}: {type(e).__name__}: {e}")
            return None
        print(f"[viz] wrote {r[0] if isinstance(r, tuple) else r}")
        written.append(r[0] if isinstance(r, tuple) else r)
        return r

    attempt("bpc curves", plot_bpc_curves, results_dir, os.path.join(out_dir, "bpc_curves.png"))
    attempt("final bpc", plot_final_bpc, results_dir, os.path.join(out_dir, "final_bpc.png"),
            checkpoint_dir=checkpoint_dir)

    captions = []
    ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, "*.pt")))
    if not ckpts:
        print(f"[viz] skip attention: no {checkpoint_dir}/*.pt")
    for path in ckpts:
        run = os.path.splitext(os.path.basename(path))[0]
        try:
            model, tok, _ = _load_checkpoint(path)
        except (NotImplementedError, ImportError, FileNotFoundError, KeyError) as e:
            print(f"[viz] skip attention for {run}: cannot load checkpoint ({type(e).__name__}: {e})")
            continue
        attempt(f"attention {run}", plot_attention, model, tok, text,
                os.path.join(out_dir, f"attention_{run}.png"),
                title=f"Attention by layer and head  ·  run {run}")
        r = attempt(f"head {run}", plot_interesting_head, model, tok, text,
                    os.path.join(out_dir, f"attention_{run}_head.png"))
        if r:
            captions.append(f"{run}: {r[1]}")
    if captions:
        cap_path = os.path.join(out_dir, "attention_captions.txt")
        with open(cap_path, "w") as f:
            f.write("\n".join(captions) + "\n")
        written.append(cap_path)
    return written


def main(argv=None) -> list[str]:
    p = argparse.ArgumentParser(description="Regenerate all viz figures into results/figures/.")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--out_dir", default=None, help="default: <results_dir>/figures")
    p.add_argument("--text", default=DEFAULT_TEXT, help="sentence for the attention plots")
    a = p.parse_args(argv)
    return make_all(a.results_dir, a.checkpoint_dir, a.out_dir, a.text)


if __name__ == "__main__":
    main()
