"""Time-budgeted training on CPU. Owner: Agent C. See SPEC.md §3.

CLI:
    python -m tinygpt.train --run_name NAME --tokenizer char|bpe512|bpe2048 --time_budget_min 7 [--no_pos_emb]

Outputs:
    results/NAME/log.csv      step,elapsed_s,train_loss,val_loss,val_bpc
    results/NAME/samples.txt  samples at step 0, ~25%, ~50%, end
    checkpoints/NAME.pt       {"model_config", "model_state", "tokenizer_kind", "tokenizer_path",
                               "run_name", "final_val_bpc"}
"""

from __future__ import annotations

import argparse

from tinygpt.tokenizer import TOKENIZER_DIR

SEED = 1337
NUM_THREADS = 4


def build_arg_parser() -> argparse.ArgumentParser:
    """Argument parser for the CLI described in the module docstring."""
    raise NotImplementedError


def run_training(run_name: str, tokenizer_kind: str, time_budget_min: float = 7.0, use_pos_emb: bool = True,
                 max_steps: int | None = None, text: str | None = None, results_dir: str = "results",
                 checkpoint_dir: str = "checkpoints", tokenizer_dir: str = TOKENIZER_DIR,
                 config_path: str = "configs/default.json") -> dict:
    """Train one run within a wall-clock budget of TRAINING time (eval/sampling/tokenizing excluded).

    Args:
        text: corpus override (tests); defaults to ``load_corpus()``.
        max_steps: optional hard step cap (tests / smoke runs).

    Returns:
        dict with at least ``final_val_bpc`` (float, max_chars=200000) and ``steps`` (int).
    """
    raise NotImplementedError


def main(argv: list[str] | None = None) -> None:
    """Parse ``argv`` and call :func:`run_training`."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
