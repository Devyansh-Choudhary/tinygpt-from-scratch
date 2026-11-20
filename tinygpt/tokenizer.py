"""Char-level and byte-level BPE tokenizers, written from scratch. Owner: Agent A. See SPEC.md §3.

Both tokenizers expose ``name``, ``vocab_size``, ``encode``, ``decode``, ``save``, ``load``, and
satisfy ``decode(encode(s)) == s`` for every ASCII string ``s``. Files are JSON.
"""

from __future__ import annotations

TOKENIZER_DIR = "data/tokenizers"


class CharTokenizer:
    """Fixed 96-symbol vocabulary: ``"\\n"`` followed by chr(32)..chr(126). No training needed."""

    name: str = "char"

    def __init__(self) -> None:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        """Always 96."""
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        """Map each character to its id."""
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        """Inverse of :meth:`encode`."""
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Write the tokenizer to a JSON file at ``path``."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        """Read a tokenizer written by :meth:`save`."""
        raise NotImplementedError


class BPETokenizer:
    """Byte-level BPE: 256 byte tokens + ``vocab_size - 256`` learned merges.

    Text is pre-split with a GPT-2-style regex (stdlib ``re``); merges never cross chunk
    boundaries. ``name`` is ``f"bpe{vocab_size}"``.
    """

    def __init__(self, merges: list[tuple[int, int]] | None = None) -> None:
        """Build from an ordered list of merges; merge ``i`` creates token id ``256 + i``."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        """``256 + len(merges)``."""
        raise NotImplementedError

    @classmethod
    def train(cls, text: str, vocab_size: int) -> "BPETokenizer":
        """Learn ``vocab_size - 256`` merges from ``text`` (train text only; deterministic ties)."""
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        """Encode ``text`` to token ids (should handle ~5M chars in under ~60 s)."""
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        """Inverse of :meth:`encode`."""
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Write merges (and metadata) to a JSON file at ``path``."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Read a tokenizer written by :meth:`save`."""
        raise NotImplementedError


def get_tokenizer(kind: str, path: str | None = None, train_text: str | None = None):
    """Return a tokenizer for ``kind`` in ``{"char", "bpe512", "bpe2048"}``.

    If ``path`` exists, load it. Otherwise build it (BPE requires ``train_text``) and, if ``path``
    is given, save it there. Canonical paths: ``f"{TOKENIZER_DIR}/{kind}.json"``.
    """
    raise NotImplementedError
