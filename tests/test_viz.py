"""Tests for viz/ and demo.ipynb. Owner: Agent D.

All fakes (model, tokenizer, logs, checkpoints) live in this file (SPEC.md §4); a final test uses
the real model/tokenizer and skips while they are still stubs.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from types import SimpleNamespace

import matplotlib.image as mpimg
import numpy as np
import pytest
import torch

from viz import attention, curves, make_all

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------------------- fakes

class FakeCharTokenizer:
    """ASCII char tokenizer (id = ord(c))."""

    name = "fakechar"
    vocab_size = 128

    def encode(self, s):
        return [ord(c) for c in s]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class FakeModel(torch.nn.Module):
    """Returns hand-made attention patterns, one per head:

    layer 0: head 0 previous-token, head 1 uniform, head 2 first-token sink
    layer 1: head 0 current-token, head 1 uniform, head 2 uniform
    """

    def __init__(self, n_layer=2, n_head=3, block_size=64, vocab_size=128):
        super().__init__()
        self.cfg = SimpleNamespace(n_layer=n_layer, n_head=n_head, block_size=block_size,
                                   vocab_size=vocab_size)
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.calls = []

    @staticmethod
    def patterns(T):
        causal = torch.tril(torch.ones(T, T))
        uniform = causal / causal.sum(-1, keepdim=True)
        prev = torch.zeros(T, T)
        prev[0, 0] = 1.0
        prev[torch.arange(1, T), torch.arange(T - 1)] = 1.0
        prev = 0.9 * prev + 0.1 * uniform
        first = torch.zeros(T, T)
        first[:, 0] = 1.0
        first = 0.8 * first + 0.2 * uniform
        cur = 0.7 * torch.eye(T) + 0.3 * uniform
        return [[prev, uniform, first], [cur, uniform, uniform]]

    def forward(self, idx, targets=None, return_attn=False):
        B, T = idx.shape
        assert T <= self.cfg.block_size
        self.calls.append(T)
        logits = torch.zeros(B, T, self.cfg.vocab_size)
        if not return_attn:
            return logits, None
        pats = self.patterns(T)
        attn = [torch.stack(layer)[None].expand(B, -1, -1, -1).clone()
                for layer in pats[: self.cfg.n_layer]]
        attn = [a[:, : self.cfg.n_head] for a in attn]
        return logits, None, attn


def write_log(results_dir, run, rows):
    d = os.path.join(results_dir, run)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "log.csv"), "w") as f:
        f.write("step,elapsed_s,train_loss,val_loss,val_bpc\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


def fake_results(tmp_path):
    rd = str(tmp_path / "results")
    for i, run in enumerate(["char", "bpe512", "bpe2048", "bpe512_nopos"]):
        rows = []
        for k in range(15):
            t = 30.0 * k
            bpc = 6.6 - 0.3 * i if k == 0 else 1.6 + 0.2 * i + 2.0 * math.exp(-k / 3)
            rows.append((100 * k, t, 2.0, 1.9, round(bpc, 4)))
        write_log(rd, run, rows)
    os.makedirs(os.path.join(rd, "figures"), exist_ok=True)  # not a run: no log.csv
    return rd


def assert_png(path):
    assert os.path.isfile(path)
    img = mpimg.imread(path)
    assert img.ndim == 3 and img.shape[0] > 100 and img.shape[1] > 100
    return img


# ----------------------------------------------------------------------------- curves

def test_read_logs_parses_and_skips_bad_rows(tmp_path):
    rd = str(tmp_path / "r")
    write_log(rd, "a", [(0, 0.0, 4.0, 4.0, 6.0), (10, 30.0, 3.0, 3.0, ""), (20, 60.0, 2.0, 2.0, 2.5)])
    write_log(rd, "empty", [])
    os.makedirs(os.path.join(rd, "figures"))
    logs = curves.read_logs(rd)
    assert list(logs) == ["a"]
    assert logs["a"]["val_bpc"] == [6.0, 2.5]
    assert logs["a"]["elapsed_s"] == [0.0, 60.0]


def test_plot_bpc_curves_writes_png(tmp_path):
    rd = fake_results(tmp_path)
    out = str(tmp_path / "figs" / "curves.png")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # e.g. missing glyphs, tight-layout problems
        assert curves.plot_bpc_curves(rd, out) == out
    assert_png(out)


def test_plot_bpc_curves_single_run_and_no_clip(tmp_path):
    rd = str(tmp_path / "r")
    write_log(rd, "only", [(0, 0.0, 4, 4, 5.0)])
    out = curves.plot_bpc_curves(rd, str(tmp_path / "c.png"), budget_min=None, clip_start=False)
    assert_png(out)


def test_plot_bpc_curves_no_runs_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        curves.plot_bpc_curves(str(tmp_path), str(tmp_path / "x.png"))


def test_final_bpc_prefers_checkpoint(tmp_path):
    rd = fake_results(tmp_path)
    cd = tmp_path / "ck"
    cd.mkdir()
    torch.save({"run_name": "char", "final_val_bpc": 1.234, "model_state": {}}, cd / "char.pt")
    (cd / "broken.pt").write_bytes(b"not a checkpoint")
    vals = curves.final_bpc(rd, str(cd))
    assert vals["char"] == (pytest.approx(1.234), "checkpoint")
    assert vals["bpe512"][1] == "log"
    assert "broken" not in vals
    out = curves.plot_final_bpc(rd, str(tmp_path / "bars.png"), checkpoint_dir=str(cd))
    assert_png(out)


def test_plot_final_bpc_no_runs_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        curves.plot_final_bpc(str(tmp_path), str(tmp_path / "x.png"), checkpoint_dir=None)


def test_run_colors_follow_run_not_rank():
    a = curves.run_colors(["bpe2048", "char"])
    b = curves.run_colors(["char", "bpe512", "bpe2048", "bpe512_nopos", "extra"])
    assert a["char"] == b["char"] and a["bpe2048"] == b["bpe2048"]
    assert len(set(b.values())) == 5


# ----------------------------------------------------------------------------- attention

def test_visible_and_token_labels():
    tok = FakeCharTokenizer()
    assert attention.visible("a b\n") == "a␣b↵"
    assert attention.token_labels(tok, tok.encode("hi \n")) == ["h", "i", "␣", "↵"]


def test_get_attention_shapes_and_cropping():
    tok, model = FakeCharTokenizer(), FakeModel(block_size=16)
    ids, attn = attention.get_attention(model, tok, "x" * 40, max_tokens=48)
    assert len(ids) == 16 and attn.shape == (2, 3, 16, 16)  # cropped to block_size
    ids, attn = attention.get_attention(model, tok, "hello world", max_tokens=5)
    assert ids == tok.encode("hello") and attn.shape == (2, 3, 5, 5)
    with pytest.raises(ValueError):
        attention.get_attention(model, tok, "")


def test_head_stats_extremes():
    T = 10
    pats = FakeModel.patterns(T)
    uni = attention.head_stats(pats[0][1].numpy())
    assert uni["norm_entropy"] == pytest.approx(1.0, abs=1e-6)
    onehot = np.zeros((T, T))
    onehot[0, 0] = 1
    onehot[np.arange(1, T), np.arange(T - 1)] = 1
    s = attention.head_stats(onehot)
    assert s["prev"] == pytest.approx(1.0) and s["norm_entropy"] == pytest.approx(0.0, abs=1e-9)
    assert s["entropy_bits"] == pytest.approx(0.0, abs=1e-9)


def test_describe_head_kinds():
    T = 20
    pats = [[p.numpy() for p in layer] for layer in FakeModel.patterns(T)]
    assert attention.describe_head(pats[0][0])[0] == "previous-token"
    assert attention.describe_head(pats[0][1])[0] == "diffuse"
    assert attention.describe_head(pats[0][2])[0] == "first-token (sink)"
    assert attention.describe_head(pats[1][0])[0] == "current-token"
    focused = np.tril(np.ones((T, T))) * 1e-3
    focused[np.arange(T), np.arange(T) // 2] = 1.0  # attends half-way back: not prev/self/first
    focused /= focused.sum(-1, keepdims=True)
    kind, _, desc = attention.describe_head(focused, [str(i) for i in range(T)])
    assert kind == "focused" and "looks at" in desc


def test_find_interesting_head_picks_previous_token_head():
    tok, model = FakeCharTokenizer(), FakeModel()
    ids, attn = attention.get_attention(model, tok, "Once upon a time there was a cat.")
    info = attention.find_interesting_head(attn, attention.token_labels(tok, ids))
    assert (info["layer"], info["head"], info["kind"]) == (0, 0, "previous-token")
    assert "previous" in info["caption"] and "\n" not in info["caption"]


def test_plot_attention_writes_grid_png(tmp_path):
    tok, model = FakeCharTokenizer(), FakeModel()
    out = str(tmp_path / "attn.png")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # fails on missing glyphs for ␣ / ↵
        assert attention.plot_attention(model, tok, "Hi there,\nfriend.", out) == out
    assert_png(out)
    assert model.training  # train/eval mode restored


def test_plot_interesting_head_returns_caption(tmp_path):
    tok, model = FakeCharTokenizer(), FakeModel()
    out, caption = attention.plot_interesting_head(model, tok, "The cat sat on the mat.",
                                                   str(tmp_path / "head.png"))
    assert_png(out)
    assert caption.startswith("Layer 0, head 0")


# ----------------------------------------------------------------------------- make_all

def test_make_all_with_fakes(tmp_path, monkeypatch, capsys):
    rd = fake_results(tmp_path)
    cd = tmp_path / "checkpoints"
    cd.mkdir()
    for run in ("char", "bpe512"):
        torch.save({"run_name": run, "final_val_bpc": 2.0}, cd / f"{run}.pt")
    monkeypatch.setattr(make_all, "_load_checkpoint",
                        lambda path: (FakeModel(), FakeCharTokenizer(), {}))
    written = make_all.main(["--results_dir", rd, "--checkpoint_dir", str(cd)])
    names = sorted(os.path.basename(p) for p in written)
    assert names == sorted(["bpc_curves.png", "final_bpc.png", "attention_char.png",
                            "attention_char_head.png", "attention_bpe512.png",
                            "attention_bpe512_head.png", "attention_captions.txt"])
    assert all(os.path.dirname(p) == os.path.join(rd, "figures") for p in written)


def test_make_all_skips_missing_inputs(tmp_path, capsys):
    written = make_all.make_all(str(tmp_path / "r"), str(tmp_path / "ck"))
    assert written == []
    assert "skip" in capsys.readouterr().out


# ----------------------------------------------------------------------------- notebook

def _notebook():
    nbformat = pytest.importorskip("nbformat")
    nb = nbformat.read(os.path.join(ROOT, "demo.ipynb"), as_version=4)
    nbformat.validate(nb)
    return nb


def test_notebook_valid_and_has_required_parts():
    nb = _notebook()
    src = "\n".join("".join(c.source) for c in nb.cells)
    first_code = next(c for c in nb.cells if c.cell_type == "code").source
    assert "GITHUB_USERNAME = " in first_code and "github.com/{GITHUB_USERNAME}" in first_code
    assert "clone" in first_code and "chdir" in first_code
    for needle in ("checkpoints", "load_checkpoint", "ipywidgets", "Dropdown", "Textarea",
                   "FloatSlider", "IntSlider", "Button", "plot_attention", "plot_bpc_curves",
                   "samples.txt", "generate("):
        assert needle in src, needle
    assert all(not c.get("outputs") for c in nb.cells if c.cell_type == "code")
    json.loads(open(os.path.join(ROOT, "demo.ipynb")).read())


def test_notebook_code_cells_compile():
    nb = _notebook()
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        code = "\n".join(l for l in c.source.splitlines() if not l.lstrip().startswith(("%", "!")))
        compile(code, f"demo.ipynb[cell {i}]", "exec")


# ----------------------------------------------------------------------------- real modules

def test_attention_with_real_model_if_available(tmp_path):
    try:
        from tinygpt.model import GPT, GPTConfig
        from tinygpt.tokenizer import CharTokenizer

        tok = CharTokenizer()
        torch.manual_seed(1337)
        model = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=32, n_layer=2, n_head=2, n_embd=32))
    except NotImplementedError:
        pytest.skip("tinygpt.model / tinygpt.tokenizer are still stubs")
    ids, attn = attention.get_attention(model, tok, "Once upon a time\nthere was")
    assert attn.shape == (2, 2, len(ids), len(ids))
    np.testing.assert_allclose(attn.sum(-1), 1.0, atol=1e-5)
    assert np.allclose(np.triu(attn, k=1), 0.0)
    assert_png(attention.plot_attention(model, tok, "Once upon a time", str(tmp_path / "a.png")))
