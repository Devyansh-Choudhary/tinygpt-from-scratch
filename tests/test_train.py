"""Tests for train + evaluate. Owner: Agent C. See SPEC.md §4.

Fakes of other agents' modules (model, tokenizer, data split) live ONLY in this file and are
monkeypatched into ``tinygpt.train`` / ``tinygpt.evaluate``, so the logic tests are independent of those
modules. ``test_real_*`` and the CLI smoke test use the real model/tokenizers (skip if stubs).
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import re

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import tinygpt.evaluate as ev
import tinygpt.train as tr
from tinygpt.model import GPTConfig

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "configs", "default.json")
ALPHABET = "\n" + "".join(chr(i) for i in range(32, 127))


# ------------------------------------------------------------------------------------------------
# fakes
# ------------------------------------------------------------------------------------------------
class FakeCharTokenizer:
    name = "char"

    def __init__(self):
        self.stoi = {c: i for i, c in enumerate(ALPHABET)}

    @property
    def vocab_size(self):
        return len(ALPHABET)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(ALPHABET[i] for i in ids)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"kind": self.name}, f)


class FakePairTokenizer(FakeCharTokenizer):
    """Every token covers 2 chars (a trailing odd char gets its own token) — a stand-in for BPE."""

    name = "bpe512"

    @property
    def vocab_size(self):
        return 96 + 96 * 96

    def encode(self, s):
        ids = [self.stoi[c] for c in s]
        out = [96 + ids[i] * 96 + ids[i + 1] for i in range(0, len(ids) - 1, 2)]
        if len(ids) % 2:
            out.append(ids[-1])
        return out

    def decode(self, ids):
        out = []
        for i in ids:
            if i < 96:
                out.append(ALPHABET[i])
            else:
                a, b = divmod(i - 96, 96)
                out.append(ALPHABET[a] + ALPHABET[b])
        return "".join(out)


def fake_get_tokenizer(kind, path=None, train_text=None):
    tok = FakeCharTokenizer() if kind == "char" else FakePairTokenizer()
    if path and not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tok.save(path)
    return tok


def fake_split_text(text, val_frac=0.05):
    cut = int(len(text) * (1 - val_frac))
    nl = text.find("\n", cut)
    if nl != -1:
        cut = nl + 1
    return text[:cut], text[cut:]


class FakeGPT(nn.Module):
    """Tiny GPT-interface model: token (+pos) embedding -> causal mean-pool mix -> MLP -> tied head."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = 32
        self.tok_emb = nn.Embedding(cfg.vocab_size, d)
        self.pos_emb = nn.Embedding(cfg.block_size, d) if cfg.use_pos_emb else None
        self.mix = nn.Linear(d, d)
        self.fc = nn.Linear(d, d)
        self.ln_f = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx, targets=None, return_attn=False):
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError("sequence longer than block_size")
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(T))
        prefix_mean = x.cumsum(1) / torch.arange(1, T + 1).view(1, T, 1)
        x = x + self.mix(prefix_mean)
        x = x + self.fc(F.gelu(x))
        logits = self.lm_head(self.ln_f(x))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return (logits, loss, []) if return_attn else (logits, loss)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.cfg.block_size:])[0][:, -1] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], 1)
        return idx

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if self.pos_emb is not None:
                n -= self.pos_emb.weight.numel()
        return n


class UniformModel(nn.Module):
    """Predicts a uniform distribution; records every (input, target) window it sees."""

    def __init__(self, vocab_size, block_size):
        super().__init__()
        self.V, self.block_size = vocab_size, block_size
        self.dummy = nn.Parameter(torch.zeros(1))
        self.seen_inputs = []
        self.modes = []

    def forward(self, idx, targets=None):
        assert idx.shape[1] <= self.block_size
        self.seen_inputs.append(idx.clone())
        self.modes.append((self.training, torch.is_grad_enabled()))
        return torch.zeros(*idx.shape, self.V), None


def make_text(n_chars, seed=0):
    words = ("once upon a time there was a little cat named tom . she liked to play in the park with "
             "her friend lily . one day they found a big red ball and they were very happy .").split()
    rng = random.Random(seed)
    lines, n = [], 0
    while n < n_chars:
        line = " ".join(rng.choice(words) for _ in range(rng.randint(5, 15))).capitalize()
        lines.append(line)
        n += len(line) + 1
    return "\n".join(lines)[:n_chars]


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setattr(tr, "GPT", FakeGPT)
    monkeypatch.setattr(tr, "get_tokenizer", fake_get_tokenizer)
    monkeypatch.setattr(tr, "split_text", fake_split_text)
    monkeypatch.setattr(ev, "GPT", FakeGPT)
    monkeypatch.setattr(ev, "get_tokenizer", fake_get_tokenizer)


def tiny_config(tmp_path, **overrides):
    hp = {"batch_size": 8, "block_size": 32, "eval_interval_s": 0.3, "eval_iters": 2,
          "eval_max_chars_during_training": 2000, "sample_tokens": 20, "warmup_frac": 0.1}
    hp.update(overrides)
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(hp))
    return str(path)


def read_log(path):
    with open(path) as f:
        return list(csv.reader(f))


# ------------------------------------------------------------------------------------------------
# unit tests
# ------------------------------------------------------------------------------------------------
def test_default_config_is_complete_and_matches_brief():
    hp = tr.load_hparams(DEFAULT_CONFIG)
    with open(DEFAULT_CONFIG) as f:
        raw = {k for k in json.load(f) if not k.startswith("_")}
    assert raw == set(tr.DEFAULT_HPARAMS)  # every knob is explicit in the file
    assert hp["batch_size"] == 32 and hp["block_size"] == 128
    assert hp["lr"] == 3e-3 and list(hp["betas"]) == [0.9, 0.95] and hp["weight_decay"] == 0.1
    assert hp["grad_clip"] == 1.0 and hp["eval_iters"] == 20
    assert hp["eval_max_chars_during_training"] == 50000
    assert (hp["n_layer"], hp["n_head"], hp["n_embd"]) == (4, 4, 128)


def test_per_run_configs_hold_no_hyperparameters():
    expected = {"char": ("char", True), "bpe512": ("bpe512", True), "bpe2048": ("bpe2048", True),
                "bpe512_nopos": ("bpe512", False)}
    for name, (kind, pos) in expected.items():
        with open(os.path.join(REPO_ROOT, "configs", f"{name}.json")) as f:
            cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        assert set(cfg) <= tr.RUN_CONFIG_KEYS
        assert cfg["run_name"] == name and cfg["tokenizer"] == kind and cfg["use_pos_emb"] is pos
        assert cfg["time_budget_min"] == 7


def test_unknown_hparam_rejected(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"learning_rate": 1.0}))
    with pytest.raises(ValueError):
        tr.load_hparams(str(p))


def test_lr_schedule():
    w, m = 0.05, 0.1
    assert tr.lr_multiplier(0.0, w, m) == 0.0
    assert tr.lr_multiplier(w / 2, w, m) == pytest.approx(0.5)
    assert tr.lr_multiplier(w, w, m) == pytest.approx(1.0)
    assert tr.lr_multiplier(1.0, w, m) == pytest.approx(m)
    assert tr.lr_multiplier(5.0, w, m) == pytest.approx(m)
    mid = tr.lr_multiplier(w + (1 - w) / 2, w, m)
    assert mid == pytest.approx(m + (1 - m) / 2)
    xs = [w + i * (1 - w) / 50 for i in range(51)]
    vals = [tr.lr_multiplier(x, w, m) for x in xs]
    assert all(a >= b for a, b in zip(vals, vals[1:]))


def test_get_batch_contiguous_shifted():
    data = torch.arange(1000)
    g = torch.Generator().manual_seed(0)
    x, y = tr.get_batch(data, 16, 32, g)
    assert x.shape == y.shape == (16, 32) and x.dtype == torch.long
    assert torch.equal(y[:, :-1], x[:, 1:])
    assert torch.equal(x[:, 1:] - x[:, :-1], torch.ones(16, 31, dtype=torch.long))
    assert int(y.max()) <= 999


def test_optimizer_decay_groups():
    model = FakeGPT(GPTConfig(vocab_size=96, block_size=16))
    opt = tr.configure_optimizer(model, tr.load_hparams(DEFAULT_CONFIG))
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"]) and all(p.dim() < 2 for p in no_decay["params"])
    n = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert n == sum(p.numel() for p in model.parameters())  # tied weight counted once
    assert opt.defaults["betas"] == (0.9, 0.95)


# ------------------------------------------------------------------------------------------------
# bits per char
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("tok_cls", [FakeCharTokenizer, FakePairTokenizer])
@pytest.mark.parametrize("n_chars", [2, 3, 7, 129, 257, 1000, 1001])
def test_bpc_exact_uniform(tok_cls, n_chars):
    tok, bs = tok_cls(), 16
    text = make_text(5000)
    model = UniformModel(tok.vocab_size, bs)
    model.train()
    sub = text[:n_chars]
    ids = tok.encode(sub)
    if len(ids) < 2:
        with pytest.raises(ValueError):
            ev.bits_per_char(model, tok, text, bs, max_chars=n_chars)
        return
    bpc = ev.bits_per_char(model, tok, text, bs, max_chars=n_chars)
    expected = (len(ids) - 1) * math.log2(tok.vocab_size) / (len(sub) - len(tok.decode(ids[:1])))
    assert bpc == pytest.approx(expected, rel=1e-6)  # float32 logits
    assert model.training  # mode restored


@pytest.mark.parametrize("bs", [1, 5, 16])
def test_bpc_every_token_predicted_once(bs):
    tok = FakeCharTokenizer()
    text = make_text(300)
    model = UniformModel(tok.vocab_size, bs)
    ev.bits_per_char(model, tok, text, bs)
    ids = tok.encode(text)
    inputs = torch.cat([x.reshape(-1) for x in model.seen_inputs]).tolist()
    # inputs are exactly ids[:-1] in order (windows of bs, overlapping targets by 1 token)
    assert inputs == ids[:-1]
    assert all(x.shape[1] <= bs for x in model.seen_inputs)
    assert set(model.modes) == {(False, False)}  # eval() + no_grad during scoring
    assert model.training  # train mode restored afterwards


def test_bpc_matches_manual_loop_with_real_logits():
    torch.manual_seed(0)
    tok, bs = FakePairTokenizer(), 8
    model = FakeGPT(GPTConfig(vocab_size=tok.vocab_size, block_size=bs))
    text = make_text(301)
    ids = tok.encode(text)
    nll = 0.0
    with torch.no_grad():
        for k in range(0, len(ids) - 1, bs):
            w = torch.tensor(ids[k: k + bs + 1]).unsqueeze(0)
            logits = model(w[:, :-1])[0]
            nll += F.cross_entropy(logits[0], w[0, 1:], reduction="sum").item()
    expected = nll / math.log(2) / (len(text) - 2)
    assert ev.bits_per_char(model, tok, text, bs) == pytest.approx(expected, rel=1e-6)


def test_bpc_equal_for_equivalent_models_across_tokenizers():
    """A char model and a pair model that assign the same probability to the same text get the same BPC.

    Uniform char model: p(text) = V^-(n-1) over chars. A pair model giving each pair prob V^-2 assigns
    the same probability to the text, so BPC must coincide even though per-token loss doubles.
    """
    text = make_text(1001)
    char = FakeCharTokenizer()

    class PairModelEquivalent(nn.Module):
        def forward(self, idx, targets=None):
            # log-prob of every pair token = -2 ln 96 (spread mass uniformly over the 96*96 pair ids)
            logits = torch.full((*idx.shape, 96 + 96 * 96), -1e9)
            logits[..., 96:] = 0.0
            return logits, None

    bpc_char = ev.bits_per_char(UniformModel(96, 32), char, text, 32)
    pair_text = text[:1000]  # even length => all tokens are pairs
    bpc_char_even = ev.bits_per_char(UniformModel(96, 32), char, pair_text, 32)
    bpc_pair = ev.bits_per_char(PairModelEquivalent(), FakePairTokenizer(), pair_text, 16)
    assert bpc_char == pytest.approx(math.log2(96))
    assert bpc_pair == pytest.approx(bpc_char_even) == pytest.approx(math.log2(96))


# ------------------------------------------------------------------------------------------------
# training (fake model)
# ------------------------------------------------------------------------------------------------
def test_run_training_outputs_and_checkpoint_roundtrip(tmp_path, fakes):
    res_dir, ck_dir, tok_dir = tmp_path / "results", tmp_path / "ckpt", tmp_path / "tok"
    text = make_text(60000)
    out = tr.run_training("t1", "char", time_budget_min=1.0, max_steps=60, text=text, results_dir=str(res_dir),
                          checkpoint_dir=str(ck_dir), tokenizer_dir=str(tok_dir),
                          config_path=tiny_config(tmp_path), cache_dir=str(tmp_path / "cache"), verbose=False)
    assert out["steps"] == 60
    assert math.isfinite(out["final_val_bpc"]) and out["final_val_bpc"] > 0

    rows = read_log(res_dir / "t1" / "log.csv")
    assert rows[0] == ["step", "elapsed_s", "train_loss", "val_loss", "val_bpc"]
    assert rows[1][0] == "0" and float(rows[1][1]) == 0.0
    assert rows[-1][0] == "60"
    assert float(rows[-1][4]) == pytest.approx(out["final_val_bpc"], abs=1e-5)
    # learning happened
    assert float(rows[-1][3]) < float(rows[1][3]) - 0.5
    steps = [int(r[0]) for r in rows[1:]]
    assert steps == sorted(steps) and len(set(steps)) == len(steps)

    samples = (res_dir / "t1" / "samples.txt").read_text()
    headers = re.findall(r"^=== step (\d+) \(elapsed ([\d.]+)s\) ===$", samples, flags=re.M)
    assert len(headers) == 4 and headers[0][0] == "0" and headers[-1][0] == "60"
    assert 10 <= int(headers[1][0]) <= 20 and 25 <= int(headers[2][0]) <= 35  # ~25%, ~50% of 60 steps
    assert samples.count("Once upon a time") == 4

    ckpt = torch.load(ck_dir / "t1.pt", weights_only=True)
    assert set(ckpt) == {"model_config", "model_state", "tokenizer_kind", "tokenizer_path", "run_name",
                         "final_val_bpc"}
    assert ckpt["model_config"] == {"vocab_size": 96, "block_size": 32, "n_layer": 4, "n_head": 4, "n_embd": 128,
                                    "dropout": 0.0, "use_pos_emb": True}
    assert ckpt["tokenizer_kind"] == "char" and ckpt["run_name"] == "t1"
    assert ckpt["final_val_bpc"] == out["final_val_bpc"]

    model, tok, ck = ev.load_checkpoint(str(ck_dir / "t1.pt"))
    assert not model.training and tok.name == "char"
    _, val = fake_split_text(text)
    assert ev.bits_per_char(model, tok, val, 32) == pytest.approx(out["final_val_bpc"], rel=1e-6)

    summary = json.loads((res_dir / "t1" / "summary.json").read_text())
    assert summary["steps"] == 60 and summary["steps_per_sec"] > 0 and summary["tokenize_time_s"] >= 0
    assert os.listdir(tmp_path / "cache")  # encoded tokens cached


def test_run_training_is_deterministic_and_nopos(tmp_path, fakes):
    text = make_text(30000)
    kw = dict(time_budget_min=1.0, max_steps=15, text=text, tokenizer_dir=str(tmp_path / "tok"),
              config_path=tiny_config(tmp_path, eval_interval_s=1e9), cache_dir=None, verbose=False)
    a = tr.run_training("a", "bpe512", results_dir=str(tmp_path / "r"), checkpoint_dir=str(tmp_path / "c"), **kw)
    b = tr.run_training("b", "bpe512", results_dir=str(tmp_path / "r"), checkpoint_dir=str(tmp_path / "c"), **kw)
    assert a["final_val_bpc"] == b["final_val_bpc"]
    c = tr.run_training("c", "bpe512", use_pos_emb=False, results_dir=str(tmp_path / "r"),
                        checkpoint_dir=str(tmp_path / "c"), **kw)
    ck = torch.load(tmp_path / "c" / "c.pt", weights_only=True)
    assert ck["model_config"]["use_pos_emb"] is False
    assert not any(k.startswith("pos_emb") for k in ck["model_state"])
    assert c["final_val_bpc"] != a["final_val_bpc"]


def test_time_budget_counts_training_only(tmp_path, fakes, monkeypatch):
    """Slow eval must not eat into the budget; training time lands just past the budget."""
    real_bpc = ev.bits_per_char

    def slow_bpc(*a, **k):
        import time
        time.sleep(0.3)
        return real_bpc(*a, **k)

    monkeypatch.setattr(tr, "bits_per_char", slow_bpc)
    out = tr.run_training("tb", "char", time_budget_min=1.5 / 60, text=make_text(30000),
                          results_dir=str(tmp_path / "r"), checkpoint_dir=str(tmp_path / "c"),
                          tokenizer_dir=str(tmp_path / "tok"), config_path=tiny_config(tmp_path, eval_interval_s=0.25),
                          cache_dir=None, verbose=False)
    assert 1.5 <= out["train_time_s"] < 1.5 + 0.5
    rows = read_log(tmp_path / "r" / "tb" / "log.csv")[1:]
    assert len(rows) >= 5  # ~6 periodic evals at 0.25 s spacing (+ step 0 + final)
    assert out["wall_time_s"] > out["train_time_s"] + 0.3 * (len(rows) - 2)  # eval time excluded from budget


def test_run_config_rejects_hyperparameters(tmp_path, fakes):
    p = tmp_path / "run.json"
    p.write_text(json.dumps({"run_name": "x", "tokenizer": "char", "lr": 1.0}))
    with pytest.raises(SystemExit):
        tr.main(["--run_config", str(p)])
    with pytest.raises(SystemExit):
        tr.main(["--tokenizer", "char"])  # no run name


# ------------------------------------------------------------------------------------------------
# real modules (model, tokenizers, data split from main); skip if any is still a stub
# ------------------------------------------------------------------------------------------------
REAL_TOKENIZER_DIR = os.path.join(REPO_ROOT, "data", "tokenizers")


def _require_real_modules():
    from tinygpt import data, model, tokenizer

    try:
        tok = tokenizer.CharTokenizer()
        model.GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=8, n_layer=1, n_head=1, n_embd=8))
        data.split_text("a\nb\n")
    except NotImplementedError:
        pytest.skip("model/tokenizer/data still stubs")
    return data


@pytest.mark.parametrize("kind", ["char", "bpe512"])
def test_real_modules_tiny_run(tmp_path, kind):
    data = _require_real_modules()
    text = make_text(30000)
    out = tr.run_training("real", kind, max_steps=3, text=text, results_dir=str(tmp_path / "r"),
                          checkpoint_dir=str(tmp_path / "c"), tokenizer_dir=REAL_TOKENIZER_DIR,
                          config_path=tiny_config(tmp_path, n_layer=1, n_head=2, n_embd=16), cache_dir=None,
                          verbose=False)
    assert out["steps"] == 3 and math.isfinite(out["final_val_bpc"])
    m2, tok2, ck = ev.load_checkpoint(str(tmp_path / "c" / "real.pt"))
    assert tok2.name == kind and ck["model_config"]["vocab_size"] == tok2.vocab_size
    assert m2.lm_head.weight.shape[0] == tok2.vocab_size and not m2.training
    _, val = data.split_text(text)
    # tiny_config trains with block_size=32; BPC must be recomputed with the same block size
    assert ev.bits_per_char(m2, tok2, val, 32) == pytest.approx(out["final_val_bpc"], rel=1e-6)


def test_real_untrained_model_bpc_near_uniform():
    """An N(0, 0.02)-initialised GPT predicts ~uniformly: BPC ~= log2(V) * tokens / chars."""
    _require_real_modules()
    from tinygpt.model import GPT
    from tinygpt.tokenizer import get_tokenizer

    text = make_text(3000)
    for kind in ("char", "bpe512"):
        tok = get_tokenizer(kind, path=os.path.join(REAL_TOKENIZER_DIR, f"{kind}.json"))
        torch.manual_seed(0)
        model = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=64))
        ids = tok.encode(text)
        expected = (len(ids) - 1) * math.log2(tok.vocab_size) / (len(text) - len(tok.decode(ids[:1])))
        assert ev.bits_per_char(model, tok, text, 64) == pytest.approx(expected, rel=0.03)


# ------------------------------------------------------------------------------------------------
# ~20 s smoke test of the full CLI path: REAL model + committed char tokenizer, default config with
# eval made cheaper, 12 s training budget, synthetic corpus (tests must not need the real corpus).
# ------------------------------------------------------------------------------------------------
def test_cli_smoke_20s(tmp_path, monkeypatch, capsys):
    _require_real_modules()
    monkeypatch.setattr(tr, "load_corpus", lambda path=None: make_text(600_000, seed=1))
    hp = tr.load_hparams(DEFAULT_CONFIG)
    hp.update(eval_interval_s=4, eval_iters=4, eval_max_chars_during_training=8000, sample_tokens=50)
    cfg = tmp_path / "default.json"
    cfg.write_text(json.dumps(hp))
    run_cfg = tmp_path / "smoke.json"
    run_cfg.write_text(json.dumps({"run_name": "smoke", "tokenizer": "char", "use_pos_emb": True,
                                   "time_budget_min": 7, "config": str(cfg)}))
    monkeypatch.chdir(tmp_path)
    tr.main(["--run_config", str(run_cfg), "--time_budget_min", "0.2", "--results_dir", "results",
             "--checkpoint_dir", "checkpoints", "--tokenizer_dir", REAL_TOKENIZER_DIR])
    printed = capsys.readouterr().out
    assert "steps/s" in printed and "not counted in budget" in printed

    rows = read_log(tmp_path / "results" / "smoke" / "log.csv")
    assert rows[0] == ["step", "elapsed_s", "train_loss", "val_loss", "val_bpc"]
    assert len(rows) >= 1 + 1 + 2 + 1  # header, step 0, >=2 periodic, final
    elapsed = [float(r[1]) for r in rows[1:]]
    assert elapsed == sorted(elapsed) and 12.0 <= elapsed[-1] < 14.0
    assert all(math.isfinite(float(v)) for r in rows[1:] for v in r[2:])
    assert float(rows[1][3]) == pytest.approx(math.log(96), abs=0.3)  # untrained ~ uniform
    assert float(rows[-1][3]) < float(rows[1][3]) - 1.0  # it learned something
    samples = (tmp_path / "results" / "smoke" / "samples.txt").read_text()
    assert len(re.findall(r"^=== step \d+ \(elapsed [\d.]+s\) ===$", samples, flags=re.M)) == 4
    assert samples.count("Once upon a time") == 4
    model, tok, ck = ev.load_checkpoint(str(tmp_path / "checkpoints" / "smoke.pt"))
    assert ck["model_config"]["block_size"] == 128 and ck["run_name"] == "smoke" and not model.training
    summary = json.loads((tmp_path / "results" / "smoke" / "summary.json").read_text())
    assert summary["steps"] > 10 and summary["final_val_bpc"] == ck["final_val_bpc"]
