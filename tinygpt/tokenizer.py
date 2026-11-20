"""Char-level and byte-level BPE tokenizers, written from scratch. Owner: Agent A. See SPEC.md §3.

Both tokenizers expose ``name``, ``vocab_size``, ``encode``, ``decode``, ``save``, ``load``, and
satisfy ``decode(encode(s)) == s`` for every ASCII string ``s`` (BPE: for every string). Files are JSON.

``python -m tinygpt.tokenizer`` (re)trains the committed tokenizers on the TRAIN split only and
prints compression statistics on the val split.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import re
from collections import Counter, defaultdict

TOKENIZER_DIR = "data/tokenizers"
TOKENIZER_KINDS = ("char", "bpe512", "bpe2048")
BPE_TRAIN_CHARS = 2_000_000  # BPE is trained on this many leading chars of the TRAIN split

# GPT-2-style pre-tokenizer using only stdlib ``re`` (ASCII letter/digit classes instead of \p{..}).
# The alternatives cover every character, so "".join(chunks) == text always holds.
SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
_SPLIT_RE = re.compile(SPLIT_PATTERN)

CHAR_VOCAB = "\n" + "".join(chr(i) for i in range(32, 127))


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)
        f.write("\n")


class CharTokenizer:
    """Fixed 96-symbol vocabulary: ``"\\n"`` followed by chr(32)..chr(126). No training needed.

    Characters outside the vocabulary (the corpus is normalised to ASCII, see ``tinygpt.data``)
    raise ``ValueError`` on encode rather than being silently dropped.
    """

    name: str = "char"

    def __init__(self) -> None:
        self.chars = CHAR_VOCAB
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        """Always 96."""
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        """Map each character to its id."""
        stoi = self.stoi
        try:
            return [stoi[c] for c in text]
        except KeyError as exc:
            raise ValueError(f"CharTokenizer cannot encode {exc.args[0]!r}: only '\\n' and ASCII 32..126 "
                             "are supported (normalise text with tinygpt.data.normalize_text)") from None

    def decode(self, ids: list[int]) -> str:
        """Inverse of :meth:`encode`."""
        chars = self.chars
        return "".join(chars[int(i)] for i in ids)

    def save(self, path: str) -> None:
        """Write the tokenizer to a JSON file at ``path``."""
        _write_json(path, {"type": "char", "name": self.name, "vocab_size": self.vocab_size, "chars": self.chars})

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        """Read a tokenizer written by :meth:`save`."""
        d = _read_json(path)
        if d.get("type") != "char" or d.get("chars") != CHAR_VOCAB:
            raise ValueError(f"{path} is not a CharTokenizer file")
        return cls()

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"


class BPETokenizer:
    """Byte-level BPE: 256 byte tokens + ``vocab_size - 256`` learned merges.

    Text is pre-split with a GPT-2-style regex (stdlib ``re``); merges never cross chunk
    boundaries. ``name`` is ``f"bpe{vocab_size}"``.
    """

    def __init__(self, merges: list[tuple[int, int]] | None = None, metadata: dict | None = None) -> None:
        """Build from an ordered list of merges; merge ``i`` creates token id ``256 + i``."""
        self.merges: list[tuple[int, int]] = [(int(a), int(b)) for a, b in (merges or [])]
        self.metadata: dict = dict(metadata or {})
        self.ranks: dict[tuple[int, int], int] = {}
        self.vocab: list[bytes] = [bytes([i]) for i in range(256)]
        for i, (a, b) in enumerate(self.merges):
            if not (0 <= a < 256 + i and 0 <= b < 256 + i):
                raise ValueError(f"merge {i} ({a}, {b}) refers to a token that does not exist yet")
            self.ranks[(a, b)] = i
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self._cache: dict[str, list[int]] = {}

    @property
    def name(self) -> str:
        return f"bpe{self.vocab_size}"

    @property
    def vocab_size(self) -> int:
        """``256 + len(merges)``."""
        return 256 + len(self.merges)

    @classmethod
    def train(cls, text: str, vocab_size: int, verbose: bool = False) -> "BPETokenizer":
        """Learn ``vocab_size - 256`` merges from ``text`` (train text only; deterministic ties).

        Each step merges the most frequent adjacent pair (ties: smallest ``(a, b)``). Pre-split
        chunks are deduplicated with their counts, and pair counts are updated incrementally only
        for the words that contain the merged pair, with a lazy max-heap to find the best pair.
        Stops early only if no pairs are left at all (tiny texts); ``get_tokenizer`` treats that
        as an error so ``name`` always matches the requested kind.
        """
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be >= 256, got {vocab_size}")
        chunk_counts = Counter(_SPLIT_RE.findall(text))
        words: list[list[int]] = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
        freqs: list[int] = list(chunk_counts.values())

        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, (w, f) in enumerate(zip(words, freqs)):
            for pair in zip(w, w[1:]):
                pair_counts[pair] += f
                where[pair].add(wi)
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        merges: list[tuple[int, int]] = []
        n_merges = vocab_size - 256
        while len(merges) < n_merges and heap:
            neg, pair = heapq.heappop(heap)
            if pair_counts.get(pair, 0) != -neg:
                continue  # stale heap entry
            new_id = 256 + len(merges)
            merges.append(pair)
            changed: dict[tuple[int, int], int] = {}
            a, b = pair
            for wi in where.pop(pair):
                w, f = words[wi], freqs[wi]
                for p in zip(w, w[1:]):  # remove this word's old pair counts
                    pair_counts[p] -= f
                    changed[p] = 0
                out, i = [], 0
                while i < len(w):
                    if i + 1 < len(w) and w[i] == a and w[i + 1] == b:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(w[i])
                        i += 1
                words[wi] = out
                for p in zip(out, out[1:]):  # add the new pair counts
                    pair_counts[p] += f
                    where[p].add(wi)
                    changed[p] = 0
            pair_counts.pop(pair, None)
            for p in changed:
                c = pair_counts.get(p, 0)
                if c > 0:
                    heapq.heappush(heap, (-c, p))
                else:
                    pair_counts.pop(p, None)
                    where.pop(p, None)
            if verbose and len(merges) % 256 == 0:
                print(f"  {len(merges)}/{n_merges} merges")
        meta = {"train_chars": len(text), "train_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
        return cls(merges, metadata=meta)

    def _encode_chunk(self, chunk: str) -> list[int]:
        ids = list(chunk.encode("utf-8"))
        ranks = self.ranks
        while len(ids) >= 2:
            best_rank, best_pair = None, None
            for p in zip(ids, ids[1:]):
                r = ranks.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pair = r, p
            if best_pair is None:
                break
            a, b = best_pair
            new_id = 256 + best_rank
            out, i = [], 0
            while i < len(ids):
                if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        return ids

    def encode(self, text: str) -> list[int]:
        """Encode ``text`` to token ids (should handle ~5M chars in under ~60 s)."""
        cache = self._cache
        out: list[int] = []
        for chunk in _SPLIT_RE.findall(text):
            ids = cache.get(chunk)
            if ids is None:
                ids = self._encode_chunk(chunk)
                if len(cache) < 500_000:
                    cache[chunk] = ids
            out.extend(ids)
        return out

    def decode(self, ids: list[int]) -> str:
        """Inverse of :meth:`encode`. Invalid UTF-8 (e.g. a lone partial token) decodes with U+FFFD."""
        vocab = self.vocab
        return b"".join(vocab[int(i)] for i in ids).decode("utf-8", errors="replace")

    def token_bytes(self, token_id: int) -> bytes:
        """The raw bytes a token stands for."""
        return self.vocab[token_id]

    def save(self, path: str) -> None:
        """Write merges (and metadata) to a JSON file at ``path``."""
        _write_json(path, {"type": "bpe", "name": self.name, "vocab_size": self.vocab_size,
                           "pattern": SPLIT_PATTERN, "metadata": self.metadata,
                           "merges": [list(m) for m in self.merges]})

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Read a tokenizer written by :meth:`save`."""
        d = _read_json(path)
        if d.get("type") != "bpe":
            raise ValueError(f"{path} is not a BPETokenizer file")
        if d.get("pattern", SPLIT_PATTERN) != SPLIT_PATTERN:
            raise ValueError(f"{path} was trained with a different pre-tokenizer pattern")
        tok = cls([tuple(m) for m in d["merges"]], metadata=d.get("metadata"))
        if tok.vocab_size != d["vocab_size"]:
            raise ValueError(f"{path}: vocab_size {d['vocab_size']} != 256 + {len(tok.merges)} merges")
        return tok

    def __repr__(self) -> str:
        return f"BPETokenizer(vocab_size={self.vocab_size})"


def load_tokenizer(path: str):
    """Load either tokenizer type from a JSON file, dispatching on its ``"type"`` field."""
    kind = _read_json(path).get("type")
    if kind == "char":
        return CharTokenizer.load(path)
    if kind == "bpe":
        return BPETokenizer.load(path)
    raise ValueError(f"{path}: unknown tokenizer type {kind!r}")


def _bpe_vocab_size(kind: str) -> int:
    if kind not in TOKENIZER_KINDS or not kind.startswith("bpe"):
        raise ValueError(f"unknown tokenizer kind {kind!r}; expected one of {TOKENIZER_KINDS}")
    return int(kind[3:])


def get_tokenizer(kind: str, path: str | None = None, train_text: str | None = None):
    """Return a tokenizer for ``kind`` in ``{"char", "bpe512", "bpe2048"}``.

    If ``path`` exists, load it. Otherwise build it (BPE requires ``train_text``; only its first
    ``BPE_TRAIN_CHARS`` characters are used) and, if ``path`` is given, save it there.
    Canonical paths: ``f"{TOKENIZER_DIR}/{kind}.json"``. Relative paths that don't exist from the
    current directory are also looked up relative to the repo root.
    """
    if kind not in TOKENIZER_KINDS:
        raise ValueError(f"unknown tokenizer kind {kind!r}; expected one of {TOKENIZER_KINDS}")
    if path is not None:
        from tinygpt.data import resolve_path

        if os.path.exists(resolve_path(path)):
            tok = load_tokenizer(resolve_path(path))
            if tok.name != kind:
                raise ValueError(f"{path} holds a {tok.name!r} tokenizer, but {kind!r} was requested")
            return tok
    if kind == "char":
        tok = CharTokenizer()
    else:
        if train_text is None:
            raise ValueError(f"{kind}: no tokenizer file at {path!r} and no train_text to train one from")
        tok = BPETokenizer.train(train_text[:BPE_TRAIN_CHARS], _bpe_vocab_size(kind))
        if tok.name != kind:
            raise ValueError(f"{kind}: train_text is too small/repetitive, only {tok.vocab_size} tokens learnable")
    if path is not None:
        tok.save(path)
    return tok


def main() -> None:
    """Train all tokenizers on the TRAIN split, save them, and print stats on the VAL split."""
    import time

    from tinygpt.data import load_corpus, resolve_path, split_text

    train_text, val_text = split_text(load_corpus())
    print(f"corpus: train {len(train_text):,} chars, val {len(val_text):,} chars")
    toks = {}
    for kind in TOKENIZER_KINDS:
        path = resolve_path(f"{TOKENIZER_DIR}/{kind}.json")
        if os.path.exists(path):
            os.remove(path)
        t0 = time.time()
        toks[kind] = get_tokenizer(kind, path, train_text=train_text)
        print(f"{kind:8s} trained+saved in {time.time() - t0:6.1f}s -> {path}")
    print("\navg characters per token on the val split:")
    for kind, tok in toks.items():
        t0 = time.time()
        ids = tok.encode(val_text)
        assert tok.decode(ids) == val_text
        print(f"  {kind:8s} vocab {tok.vocab_size:5d}: {len(val_text) / len(ids):.3f} chars/token "
              f"({len(ids):,} tokens, encoded in {time.time() - t0:.2f}s)")
    bpe = toks["bpe2048"]
    longest = sorted(range(256, bpe.vocab_size), key=lambda i: (-len(bpe.vocab[i]), i))[:20]
    print("\n20 longest tokens in bpe2048:")
    for i in longest:
        print(f"  id {i:4d} len {len(bpe.vocab[i]):2d}  {bpe.vocab[i].decode('utf-8', 'replace')!r}")


if __name__ == "__main__":
    main()
