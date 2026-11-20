"""Corpus loading and train/val splitting. Owner: Agent A. See SPEC.md §3."""

from __future__ import annotations

CORPUS_PATH = "data/tinystories_subset.txt"


def prepare_corpus(out_path: str = CORPUS_PATH, n_chars: int = 5_000_000) -> str:
    """Download TinyStories (plain HTTP, no HF libs), normalise, and write the subset.

    Normalisation: keep only "\\n" and ASCII chars 32..126; replace "<|endoftext|>" with a blank
    line; drop/map every other character. Keep the first ~``n_chars`` characters, ending on a
    story boundary.

    Returns:
        ``out_path``.
    """
    raise NotImplementedError


def load_corpus(path: str = CORPUS_PATH) -> str:
    """Read the (already normalised, ASCII-only) corpus text from ``path``."""
    raise NotImplementedError


def split_text(text: str, val_frac: float = 0.05) -> tuple[str, str]:
    """Deterministically split ``text`` by CHARACTERS into ``(train_text, val_text)``.

    The cut is at ``int(len(text) * (1 - val_frac))``, moved forward to just after the next
    ``"\\n"`` if there is one. Guarantees ``train_text + val_text == text``. No shuffling, so the
    validation text is identical for every tokenizer.
    """
    raise NotImplementedError


if __name__ == "__main__":
    prepare_corpus()
