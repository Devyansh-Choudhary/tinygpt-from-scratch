"""Tests for tinygpt.data. Owner: Agent A. No network: downloads are monkeypatched."""

import os

import pytest

import tinygpt.data as data
from tinygpt.data import load_corpus, normalize_text, resolve_path, split_text, stories_to_corpus

TEXT = "".join(f"Story line number {i}. It was a sunny day.\n" for i in range(500))


# ---------------------------------------------------------------- split_text
def test_split_concatenates_back_and_is_deterministic():
    tr, va = split_text(TEXT)
    assert tr + va == TEXT
    assert split_text(TEXT) == (tr, va)


def test_split_cuts_after_newline_near_95_percent():
    tr, va = split_text(TEXT, val_frac=0.05)
    cut = int(len(TEXT) * 0.95)
    assert tr.endswith("\n") and len(tr) >= cut
    assert len(tr) - cut < 60  # moved forward by less than one line
    assert va.startswith("Story line")


def test_split_without_newline_is_exact_char_cut():
    text = "x" * 1000
    tr, va = split_text(text, val_frac=0.1)
    assert (len(tr), len(va)) == (900, 100)


def test_split_never_leaves_val_empty():
    text = "a" * 99 + "\n"  # the only newline is the last char
    tr, va = split_text(text, val_frac=0.05)
    assert va and tr + va == text


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_split_rejects_bad_frac(bad):
    with pytest.raises(ValueError):
        split_text(TEXT, val_frac=bad)


# ---------------------------------------------------------------- normalisation
def test_normalize_maps_unicode_and_whitespace():
    raw = "  “Hi,”   she said — it’s\tfine… café \U0001F642\r\n\n\n  next  line  "
    out = normalize_text(raw)
    assert out == "\"Hi,\" she said - it's fine... caf\nnext line"
    assert all(c == "\n" or 32 <= ord(c) <= 126 for c in out)


def test_stories_to_corpus_drops_fragments_and_respects_size():
    raw = ("tail of a cut story.\n<|endoftext|>\nStory one.\nLine two.\n<|endoftext|>\n\n\n"
           "Story “two”.\n<|endoftext|>\nStory three is long.\n<|endoftext|>\nCut-off sto")
    full = stories_to_corpus(raw, n_chars=10_000)
    assert full == 'Story one.\nLine two.\n\nStory "two".\n\nStory three is long.\n'
    small = stories_to_corpus(raw, n_chars=40)
    assert small == 'Story one.\nLine two.\n\nStory "two".\n' and len(small) <= 40


def test_prepare_corpus_offline(tmp_path, monkeypatch):
    raw = "frag<|endoftext|>" + "Once upon a time.\nThe end.\n<|endoftext|>" * 50 + "partial"
    monkeypatch.setattr(data, "_download", lambda url, max_bytes, timeout=60.0: raw)
    out = data.prepare_corpus(str(tmp_path / "c.txt"), n_chars=200)
    text = load_corpus(out)
    assert text.startswith("Once upon a time.\nThe end.\n\nOnce") and text.endswith("The end.\n")
    assert len(text) <= 200


def test_prepare_corpus_falls_back_when_hf_unreachable(tmp_path, monkeypatch, capsys):
    def fake_download(url, max_bytes, timeout=60.0):
        if url == data.TINYSTORIES_URL:
            raise OSError("unreachable")
        return "First Citizen:\nBefore we proceed any further, hear me speak.\n"

    monkeypatch.setattr(data, "_download", fake_download)
    text = load_corpus(data.prepare_corpus(str(tmp_path / "c.txt"), n_chars=1000))
    assert text.startswith("First Citizen:")
    assert "FALLING BACK" in capsys.readouterr().err


# ---------------------------------------------------------------- committed corpus
def test_committed_corpus_matches_spec():
    path = resolve_path(data.CORPUS_PATH)
    if not os.path.exists(path):
        pytest.skip("corpus not committed")
    text = load_corpus()
    assert 4_000_000 <= len(text) <= 6_000_000
    assert all(c == "\n" or 32 <= ord(c) <= 126 for c in set(text))
    assert text.endswith("\n") and "\n\n\n" not in text and "<|endoftext|>" not in text
    tr, va = split_text(text)
    assert tr + va == text and 0.045 < len(va) / len(text) < 0.055
