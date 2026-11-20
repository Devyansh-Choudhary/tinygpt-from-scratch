"""Tests for tinygpt.tokenizer. Owner: Agent A."""

import hashlib
import os
import random
from collections import Counter

import pytest

from tinygpt.data import CORPUS_PATH, load_corpus, resolve_path, split_text
from tinygpt.tokenizer import (BPE_TRAIN_CHARS, CHAR_VOCAB, TOKENIZER_DIR, TOKENIZER_KINDS, BPETokenizer,
                               CharTokenizer, _SPLIT_RE, get_tokenizer, load_tokenizer)

STORY = (
    "Once upon a time, there was a little girl named Lily. She loved to play outside.\n"
    "One day, Lily saw a big red ball. \"Can I play?\" she asked. Her mom said, \"Yes, you can!\"\n"
    "They played together all day. They were very happy.\n\n"
)
# Deterministic pseudo-random "words": diverse enough to learn a 512-token vocabulary.
_rng = random.Random(1337)
WORDY = " ".join("".join(_rng.choice("etaoinshrdlucmfwyp") for _ in range(_rng.randint(2, 9)))
                 for _ in range(6000)) + "\n"
ASCII_ALL = CHAR_VOCAB + "\n\nend\n"
UNICODE_SAMPLES = ["héllo wörld", "naïve café — “quotes” …", "日本語のテキスト", "emoji 🙂🚀 ok", "tab\tand\r\nCRLF", ""]


@pytest.fixture(scope="module")
def bpe300():
    return BPETokenizer.train(STORY * 20, 300)


# ---------------------------------------------------------------- CharTokenizer
def test_char_vocab_is_fixed_96():
    tok = CharTokenizer()
    assert tok.vocab_size == 96 and tok.name == "char"
    assert tok.encode("\n") == [0] and tok.encode(" ") == [1] and tok.encode("~") == [95]


@pytest.mark.parametrize("text", [ASCII_ALL, STORY, "", "\n\n\n", "a"])
def test_char_roundtrip_exact(text):
    tok = CharTokenizer()
    ids = tok.encode(text)
    assert len(ids) == len(text) and all(0 <= i < 96 for i in ids)
    assert tok.decode(ids) == text


@pytest.mark.parametrize("text", ["héllo", "tab\there", "🙂"])
def test_char_rejects_out_of_vocab(text):
    # SPEC: CharTokenizer covers "\n" + ASCII 32..126 only (the corpus is normalised to ASCII).
    with pytest.raises(ValueError):
        CharTokenizer().encode(text)


def test_char_save_load(tmp_path):
    p = str(tmp_path / "char.json")
    CharTokenizer().save(p)
    tok = CharTokenizer.load(p)
    assert tok.decode(tok.encode(STORY)) == STORY
    assert isinstance(load_tokenizer(p), CharTokenizer)


# ---------------------------------------------------------------- BPETokenizer
@pytest.mark.parametrize("text", [ASCII_ALL, STORY, *UNICODE_SAMPLES, "  leading and trailing  \n\n ", "'s 've 're 'll"])
def test_pretokenizer_covers_everything(text):
    assert "".join(_SPLIT_RE.findall(text)) == text


@pytest.mark.parametrize("text", [ASCII_ALL, STORY * 3, *UNICODE_SAMPLES, "xyzzy unseen-WORDS 12345 !!??"])
def test_bpe_roundtrip_exact(bpe300, text):
    assert bpe300.decode(bpe300.encode(text)) == text


def test_bpe_vocab_size_and_name(bpe300):
    assert bpe300.vocab_size == 300 and bpe300.name == "bpe300" and len(bpe300.merges) == 44
    ids = bpe300.encode(STORY)
    assert max(ids) < 300 and len(ids) < len(STORY)  # merges are used and compress


def test_bpe_training_is_deterministic():
    a = BPETokenizer.train(STORY * 5, 320)
    b = BPETokenizer.train(STORY * 5, 320)
    assert a.merges == b.merges


def _naive_bpe(text, vocab_size):
    """Reference BPE: full pair recount every step, same tie-breaking (max count, then min pair)."""
    words = Counter(tuple(c.encode()) for c in _SPLIT_RE.findall(text))
    merges = []
    for new_id in range(256, vocab_size):
        counts = Counter()
        for w, f in words.items():
            for p in zip(w, w[1:]):
                counts[p] += f
        if not counts:
            break
        best = min(counts, key=lambda p: (-counts[p], p))
        merges.append(best)
        new_words = Counter()
        for w, f in words.items():
            out, i = [], 0
            while i < len(w):
                if i + 1 < len(w) and (w[i], w[i + 1]) == best:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            new_words[tuple(out)] += f
        words = new_words
    return merges


def test_fast_bpe_matches_naive_reference():
    text = (STORY * 3) + "aaaa bbbb aaab abab the then there these 2023 2024 !!! ... ''' " * 4
    assert BPETokenizer.train(text, 400).merges == _naive_bpe(text, 400)


def test_bpe_merges_stay_inside_chunks(bpe300):
    # No learned token may straddle a pre-tokenizer boundary (e.g. "a." or ". T").
    for tok_bytes in bpe300.vocab[256:]:
        s = tok_bytes.decode("utf-8")
        assert len(_SPLIT_RE.findall(s)) == 1 or s.isspace(), s


def test_bpe_save_load(tmp_path, bpe300):
    p = str(tmp_path / "bpe.json")
    bpe300.save(p)
    tok = BPETokenizer.load(p)
    assert tok.merges == bpe300.merges and tok.vocab_size == 300
    assert tok.encode(STORY) == bpe300.encode(STORY)
    assert isinstance(load_tokenizer(p), BPETokenizer)
    with pytest.raises(ValueError):
        CharTokenizer.load(p)


# ---------------------------------------------------------------- get_tokenizer
def test_get_tokenizer_trains_saves_and_reloads(tmp_path):
    p = str(tmp_path / "bpe512.json")
    tok = get_tokenizer("bpe512", p, train_text=WORDY)
    assert os.path.exists(p) and tok.name == "bpe512"
    again = get_tokenizer("bpe512", p)  # loads; no train_text needed
    assert again.merges == tok.merges
    assert get_tokenizer("char").vocab_size == 96


def test_get_tokenizer_errors(tmp_path):
    with pytest.raises(ValueError, match="too small"):
        get_tokenizer("bpe2048", train_text="tiny tiny text")
    with pytest.raises(ValueError):
        get_tokenizer("bpe999")
    with pytest.raises(ValueError):
        get_tokenizer("bpe512", str(tmp_path / "missing.json"))  # no file and no train_text
    p = str(tmp_path / "c.json")
    CharTokenizer().save(p)
    with pytest.raises(ValueError):
        get_tokenizer("bpe512", p)  # file holds the wrong kind


def test_no_val_text_leaks_into_training(tmp_path):
    # The val split is full of a word ("qxqxqx") that never occurs in train. A tokenizer built
    # from the train split must not learn any token containing it.
    text = WORDY + STORY * 20 + "qxqxqx qxqxqx qxqxqx qxqxqx\n" * 100
    train_text, val_text = split_text(text, val_frac=0.1)
    assert "qx" not in train_text and "qx" in val_text
    tok = get_tokenizer("bpe512", str(tmp_path / "t.json"), train_text=train_text)
    assert all(b"qx" not in t and b"xq" not in t for t in tok.vocab[256:])
    assert tok.metadata["train_sha256"] == hashlib.sha256(train_text.encode()).hexdigest()


# ---------------------------------------------------------------- committed artifacts
@pytest.mark.parametrize("kind", TOKENIZER_KINDS)
def test_committed_tokenizers_load(kind):
    path = resolve_path(f"{TOKENIZER_DIR}/{kind}.json")
    if not os.path.exists(path):
        pytest.skip(f"{path} not committed")
    tok = get_tokenizer(kind, path)
    assert tok.name == kind
    assert tok.vocab_size == {"char": 96, "bpe512": 512, "bpe2048": 2048}[kind]
    assert tok.decode(tok.encode(STORY)) == STORY


@pytest.mark.parametrize("kind", ["bpe512", "bpe2048"])
def test_committed_bpe_trained_on_train_split_only(kind):
    path, corpus = resolve_path(f"{TOKENIZER_DIR}/{kind}.json"), resolve_path(CORPUS_PATH)
    if not (os.path.exists(path) and os.path.exists(corpus)):
        pytest.skip("committed tokenizer or corpus missing")
    train_text, val_text = split_text(load_corpus())
    meta = BPETokenizer.load(path).metadata
    n = meta["train_chars"]
    assert n == min(BPE_TRAIN_CHARS, len(train_text))
    assert meta["train_sha256"] == hashlib.sha256(train_text[:n].encode()).hexdigest()
    tok = BPETokenizer.load(path)
    sample = val_text[:20000]
    assert tok.decode(tok.encode(sample)) == sample
