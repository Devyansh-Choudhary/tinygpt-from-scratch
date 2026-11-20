"""Corpus loading and train/val splitting. Owner: Agent A. See SPEC.md §3.

Regenerate the committed corpus with ``python -m tinygpt.data``.
"""

from __future__ import annotations

import os
import re
import sys
import urllib.request

CORPUS_PATH = "data/tinystories_subset.txt"

TINYSTORIES_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
)
FALLBACK_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
STORY_SEP = "<|endoftext|>"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Unicode -> ASCII replacements applied before dropping any remaining non-ASCII characters.
# \x91-\x94 are cp1252 smart quotes that occasionally appear mis-decoded in the source.
_UNICODE_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "\x91": "'", "\x92": "'",
    "“": '"', "”": '"', "„": '"', "\x93": '"', "\x94": '"',
    "–": "-", "—": "-", "‒": "-", "―": "-", "−": "-",
    "…": "...", " ": " ", "\t": " ", "\r": "",
}
_UNICODE_TRANS = str.maketrans(_UNICODE_MAP)
_NON_ASCII = re.compile(r"[^\n\x20-\x7e]")
_MULTI_SPACE = re.compile(r" {2,}")


def resolve_path(path: str) -> str:
    """Return ``path`` as given if it exists, else relative to the repo root (lets notebooks
    and scripts run from any working directory)."""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    return os.path.join(_REPO_ROOT, path)


def normalize_text(text: str) -> str:
    """Normalise to ASCII (``\\n`` + chars 32..126) with tidy whitespace.

    Maps smart quotes/dashes/ellipsis to ASCII, drops other non-ASCII characters, turns tabs into
    spaces, collapses runs of spaces, strips every line and removes empty lines. The result never
    contains a blank line, so blank lines can mark story boundaries.
    """
    text = _NON_ASCII.sub("", text.translate(_UNICODE_TRANS))
    lines = (_MULTI_SPACE.sub(" ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def _download(url: str, max_bytes: int, timeout: float = 60.0) -> str:
    """Download (at most) the first ``max_bytes`` bytes of ``url`` as text (plain HTTP)."""
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{max_bytes - 1}",
                                               "User-Agent": "tinygpt-from-scratch"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read(max_bytes)  # read() caps the size even if the server ignores Range
    return raw.decode("utf-8", errors="ignore")


def stories_to_corpus(raw: str, n_chars: int, drop_first: bool = True) -> str:
    """Turn raw TinyStories text into the normalised corpus.

    Stories are separated by ``<|endoftext|>`` in ``raw`` and by a blank line in the output.
    The first segment is dropped when ``drop_first`` (the source file starts mid-story) and so is
    the last one (a download cut it). Whole stories are kept while the total stays ``<= n_chars``.
    """
    segments = raw.split(STORY_SEP)
    segments = segments[1 if drop_first else 0 : -1]
    out: list[str] = []
    total = 0
    for seg in segments:
        story = normalize_text(seg)
        if not story:
            continue
        add = len(story) + (2 if out else 0) + 1  # "\n\n" separator; +1 for the final "\n"
        if total + add > n_chars:
            break
        out.append(story)
        total += add - 1
    return "\n\n".join(out) + "\n"


def prepare_corpus(out_path: str = CORPUS_PATH, n_chars: int = 5_000_000) -> str:
    """Download TinyStories (plain HTTP, no HF libs), normalise, and write the subset.

    Normalisation: keep only "\\n" and ASCII chars 32..126; replace "<|endoftext|>" with a blank
    line; drop/map every other character. Keep the first ~``n_chars`` characters, ending on a
    story boundary. Falls back to tiny-shakespeare (loudly) if Hugging Face is unreachable.

    Returns:
        ``out_path``.
    """
    try:
        raw = _download(TINYSTORIES_URL, max_bytes=int(n_chars * 1.15) + 1_000_000)
        text = stories_to_corpus(raw, n_chars)
        source = "TinyStoriesV2-GPT4-valid.txt"
    except Exception as exc:  # noqa: BLE001 - any network failure triggers the fallback
        print(f"!!! WARNING: could not download TinyStories ({exc!r}).\n"
              f"!!! FALLING BACK TO TINY SHAKESPEARE: {FALLBACK_URL}", file=sys.stderr)
        raw = _download(FALLBACK_URL, max_bytes=10_000_000)
        text = normalize_text(raw)[:n_chars]
        text = text[: text.rfind("\n") + 1] if "\n" in text else text + "\n"
        source = "tinyshakespeare/input.txt (FALLBACK)"
    out_path = resolve_path(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="ascii", newline="\n") as f:
        f.write(text)
    print(f"wrote {out_path}: {len(text):,} chars, {text.count(chr(10) * 2) + 1:,} stories, source={source}")
    return out_path


def load_corpus(path: str = CORPUS_PATH) -> str:
    """Read the (already normalised, ASCII-only) corpus text from ``path``."""
    with open(resolve_path(path), "r", encoding="ascii", newline="") as f:
        return f.read()


def split_text(text: str, val_frac: float = 0.05) -> tuple[str, str]:
    """Deterministically split ``text`` by CHARACTERS into ``(train_text, val_text)``.

    The cut is at ``int(len(text) * (1 - val_frac))``, moved forward to just after the next
    ``"\\n"`` if there is one. Guarantees ``train_text + val_text == text``. No shuffling, so the
    validation text is identical for every tokenizer.
    """
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")
    cut = int(len(text) * (1 - val_frac))
    nl = text.find("\n", cut)
    if nl != -1 and nl + 1 < len(text):  # never move the cut so far that val becomes empty
        cut = nl + 1
    return text[:cut], text[cut:]


if __name__ == "__main__":
    prepare_corpus()
