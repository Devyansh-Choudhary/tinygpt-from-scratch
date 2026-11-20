# TinyGPT from scratch — SPEC (the contract)

## 1. Goal and experiment
Build a GPT in pure PyTorch (no `nn.Transformer`, no HuggingFace libraries), train it on CPU on a
TinyStories subset, and answer **"Does the tokenizer matter?"**. Four runs, each with the **same
7-minute training wall-clock budget on 4 CPU threads** and identical hyperparameters (only
`vocab_size` / `use_pos_emb` differ):

| run_name       | tokenizer | pos emb |
|----------------|-----------|---------|
| `char`         | char      | yes     |
| `bpe512`       | bpe512    | yes     |
| `bpe2048`      | bpe2048   | yes     |
| `bpe512_nopos` | bpe512    | **no** (ablation) |

Separately, profile the model's ops on a **roofline** of this machine.

**Why bits per character.** Per-token loss is not comparable across tokenizers: a BPE token covers
several characters, so its loss is naturally larger yet "buys" more text. Normalising by the number
of characters of the **identical** validation text gives a tokenizer-independent cost of compressing
that text: `BPC = (total NLL in nats over the val text) / ln(2) / (number of characters predicted)`.

## 2. Repo layout and FILE OWNERSHIP
Edit only files you own. Anything not listed belongs to the planner.

| Owner | Files |
|---|---|
| Agent A (data+tokenizer) | `tinygpt/data.py`, `tinygpt/tokenizer.py`, `data/` (incl. `data/tinystories_subset.txt`, `data/tokenizers/`), `tests/test_data.py`, `tests/test_tokenizer.py` |
| Agent B (model) | `tinygpt/model.py`, `tests/test_model.py` |
| Agent C (train+eval) | `tinygpt/train.py`, `tinygpt/evaluate.py`, `configs/`, `tests/test_train.py` |
| Agent D (viz+demo) | `viz/` (`__init__.py`, `attention.py`, `curves.py`), `demo.ipynb`, `tests/test_viz.py` |
| Agent G (roofline) | `roofline/` (code, `roofline.png`, `ROOFLINE.md`), `tests/test_roofline.py` |
| Integrator only | `results/`, `checkpoints/`, `RESULTS.md`, `README.md` |
| Planner only | `SPEC.md`, `CLAUDE.md`, `requirements.txt`, `pytest.ini`, `.gitignore`, `tinygpt/__init__.py` |

`.cache/` is git-ignored scratch space any code may write to at runtime (e.g. encoded token arrays).

## 3. Interfaces (exact; do not change signatures — extra *optional* kwargs are fine)

**`tinygpt/data.py`** — `CORPUS_PATH = "data/tinystories_subset.txt"`.
- `load_corpus(path=CORPUS_PATH) -> str`. The committed corpus is the first ~5M characters (4–6M) of
  TinyStories (plain-HTTP download of `TinyStoriesV2-GPT4-valid.txt`; no HF libs), normalised to
  **ASCII**: only `\n` and chars 32–126; `<|endoftext|>` → blank line; other chars dropped/mapped.
  Hence #chars == #bytes. `prepare_corpus(out_path=CORPUS_PATH, n_chars=5_000_000)` + `python -m tinygpt.data` regenerate it.
- `split_text(text, val_frac=0.05) -> (train_text, val_text)`: deterministic split **by characters**:
  cut at `int(len(text)*(1-val_frac))`, moved forward to just after the next `"\n"` (if any).
  Guarantees `train_text + val_text == text`. No shuffling.

**`tinygpt/tokenizer.py`** — `TOKENIZER_DIR = "data/tokenizers"`. Both classes expose
`.name` (`"char"`, `"bpe512"`, …), `.vocab_size`, `.encode(str) -> list[int]`, `.decode(list[int]) -> str`,
`.save(path)` (JSON), `@classmethod load(path)`. `decode(encode(s)) == s` for **every** ASCII string.
- `CharTokenizer()`: fixed vocab of the 96 chars `"\n"` + chr(32..126) (no training) → `vocab_size == 96`.
- `BPETokenizer`: byte-level BPE from scratch. `@classmethod train(text, vocab_size) -> BPETokenizer`:
  256 byte tokens + `vocab_size-256` merges; GPT-2-style regex pre-split (stdlib `re`), merges never
  cross chunks; ties broken deterministically. Trained on train text only (may use a prefix, ≥1M chars).
  Encode should handle ~5M chars in under ~60 s (cache per chunk).
- `get_tokenizer(kind, path=None, train_text=None)`: `kind ∈ {"char","bpe512","bpe2048"}`. If `path`
  exists → load it; else build (BPE needs `train_text`) and save to `path` if given.
  Agent A commits `data/tokenizers/{char,bpe512,bpe2048}.json` trained on `split_text(load_corpus())[0]`.

**`tinygpt/model.py`**
- `@dataclass GPTConfig(vocab_size, block_size=128, n_layer=4, n_head=4, n_embd=128, dropout=0.0, use_pos_emb=True)`.
- `GPT(cfg)`: pre-LayerNorm decoder, learned absolute position embeddings (absent if
  `use_pos_emb=False`), causal self-attention written **by hand** (QKᵀ/√d, causal mask, softmax),
  GELU MLP 4×, final LayerNorm, `lm_head` weight tied to `tok_emb`, init N(0, 0.02).
  Module names (used by viz/roofline): `tok_emb`, `pos_emb` (or `None`), `blocks` (ModuleList of
  `Block` with `ln1`, `attn` {`qkv`, `proj`}, `ln2`, `mlp` {`fc`, `proj`}), `ln_f`, `lm_head`.
- `forward(idx, targets=None, return_attn=False)`; `idx` LongTensor `[B,T]`, `T ≤ block_size` (else error).
  Returns `(logits [B,T,V], loss)` — `loss` = mean cross-entropy in nats, `None` if no targets — or
  `(logits, loss, attn)` with `attn`: list (len `n_layer`) of detached softmax probs `[B, n_head, T, T]`.
- `generate(idx, max_new_tokens, temperature=1.0, top_k=None) -> LongTensor` (crops context to block_size; no grad).
- `num_params(non_embedding=True) -> int` (non_embedding excludes `tok_emb` and `pos_emb`).

**`tinygpt/evaluate.py`**
- `bits_per_char(model, tokenizer, val_text, block_size, max_chars=200000) -> float`. Exact recipe:
  `text = val_text[:max_chars]`, `ids = encode(text)`; windows `ids[k*bs : k*bs+bs+1]` (overlap 1) so
  every token except `ids[0]` is predicted exactly once; sum NLL (nats, `eval()` + `no_grad`);
  `BPC = NLL / ln 2 / (len(text) - len(decode(ids[:1])))`.
- `load_checkpoint(path) -> (model, tokenizer, ckpt_dict)` (CPU, eval mode).

**`tinygpt/train.py`**
- CLI: `python -m tinygpt.train --run_name NAME --tokenizer char|bpe512|bpe2048 --time_budget_min 7 [--no_pos_emb]`
  (optional extras: `--config configs/default.json`, `--max_steps N`).
- Programmatic: `run_training(run_name, tokenizer_kind, time_budget_min=7.0, use_pos_emb=True, max_steps=None,
  text=None, results_dir="results", checkpoint_dir="checkpoints", tokenizer_dir=TOKENIZER_DIR,
  config_path="configs/default.json") -> dict` (at least `final_val_bpc`, `steps`); `main(argv=None)`.
- Budget = **training time only** (excludes tokenizing, eval, sampling). LR schedule is a function of
  elapsed fraction of budget (warmup, cosine decay), so it is fair across speeds. Hyperparameters live
  in `configs/default.json` and are identical for every run.
- Writes `results/NAME/log.csv` with header `step,elapsed_s,train_loss,val_loss,val_bpc` (row at step 0,
  then periodically, last row at end; `val_loss` = per-token nats, `val_bpc` may use smaller `max_chars`
  during training); `results/NAME/samples.txt` (samples from prompt `"Once upon a time"` at step 0, ~25%,
  ~50%, end, each headed `=== step S (elapsed Es) ===`); `checkpoints/NAME.pt`:
  `torch.save({"model_config": asdict(cfg), "model_state": model.state_dict(), "tokenizer_kind": kind,
  "tokenizer_path": "data/tokenizers/<kind>.json", "run_name": NAME, "final_val_bpc": float})`
  where `final_val_bpc` uses the full `max_chars=200000`.

**`viz/`** — `viz/attention.py: plot_attention(model, tokenizer, text, out_png) -> str` (grid layers×heads,
tick labels = decoded tokens); `viz/curves.py: plot_bpc_curves(results_dir, out_png) -> str` (val_bpc vs
elapsed_s, one line per `results/*/log.csv`). `demo.ipynb`: Google Colab notebook (clone repo, install,
load checkpoints, BPC table + curves, samples, attention plot); must pass `nbformat.validate`.

**`roofline/`** — measure this machine's peak FLOP/s (large float32 matmul) and memory bandwidth (large
copy) with 4 threads; time each model op (qkv/attn-scores/softmax/attn-out/proj/LayerNorm/GELU/MLP/lm_head,
forward, `B=32, T=128`, default GPTConfig) and compute FLOPs, bytes, arithmetic intensity; place them on
`roofline/roofline.png` with an explanation in `roofline/ROOFLINE.md`. Entry point `python -m roofline.run`.
Ops may be re-created from `GPTConfig` shapes so this works without a trained model.

## 4. Rules
- `seed=1337` (torch, numpy, random); `torch.set_num_threads(4)`; CPU only.
- Dependencies: Python stdlib + torch + numpy + matplotlib (+ tqdm, pytest, nbformat). Nothing else.
- **Fakes only inside your own test files.** Your tests must pass on your branch, where other modules are
  still stubs (use fakes/monkeypatch, or `pytest.skip` when a dependency raises `NotImplementedError`).
  Tests must be fast (< ~30 s per file), need no network and not require `data/tinystories_subset.txt`.
- Run `python -m pytest` from the repo root (`pytest.ini` puts the root on `sys.path`).
- Integrator runs the 4 runs **sequentially** (never concurrently) so timings are fair, e.g.
  `python -m tinygpt.train --run_name bpe512_nopos --tokenizer bpe512 --time_budget_min 7 --no_pos_emb`.
