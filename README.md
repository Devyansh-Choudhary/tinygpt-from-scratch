# tinygpt-from-scratch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Devyansh-Choudhary/tinygpt-from-scratch/blob/main/demo.ipynb)

A small GPT written from scratch in pure PyTorch (no `nn.Transformer`, no HuggingFace) and trained on CPU on a TinyStories subset.
The experiment asks **"Does the tokenizer matter?"**: char-level vs our own byte-level BPE (vocab 512 and 2048), plus a no-positional-embedding ablation.
Every run gets the same budget — 7 minutes of training on 4 CPU threads — and runs are compared by validation **bits per character** on identical text.
A separate roofline study places each of the model's ops against this machine's compute and memory-bandwidth ceilings.
See `SPEC.md` for the full contract, **[`RESULTS.md`](RESULTS.md) for findings**, and `demo.ipynb` to try it in Google Colab.

## Layout

| path | what |
|---|---|
| `tinygpt/data.py`, `tinygpt/tokenizer.py` | corpus download/normalisation, char + byte-level BPE tokenizers |
| `tinygpt/model.py` | the GPT (hand-written causal attention, tied LM head) |
| `tinygpt/train.py`, `tinygpt/evaluate.py` | time-budgeted training, bits-per-character eval, checkpoint loading |
| `configs/` | shared hyperparameters (`default.json`) + one file per run |
| `viz/` | BPC curves, final-BPC bars, attention heatmaps (`python -m viz.make_all`) |
| `roofline/` | machine ceilings, per-op FLOPs/bytes/timings, roofline plot + `ROOFLINE.md` |
| `results/`, `checkpoints/` | committed outputs of the four runs and the figures |
| `demo.ipynb` | Colab demo: loads the committed checkpoints, plots, samples, attention |

## Reproduce

Requires Python 3.10+ and `torch numpy matplotlib tqdm pytest nbformat` (CPU only).

```bash
pip install -r requirements.txt
```

```bash
python -m pytest
```

The corpus (`data/tinystories_subset.txt`) and the three tokenizers (`data/tokenizers/*.json`) are committed.
To regenerate them from scratch (downloads TinyStories over HTTP, then trains the BPE merges):

```bash
python -m tinygpt.data
```

```bash
python -m tinygpt.tokenizer
```

Run the four experiments **one at a time** (they must not share the CPUs); each trains for 7 minutes of
training time (plus tokenisation/eval overhead) and writes `results/<run>/` and `checkpoints/<run>.pt`:

```bash
python -m tinygpt.train --run_name char --tokenizer char --time_budget_min 7
```

```bash
python -m tinygpt.train --run_name bpe512 --tokenizer bpe512 --time_budget_min 7
```

```bash
python -m tinygpt.train --run_name bpe2048 --tokenizer bpe2048 --time_budget_min 7
```

```bash
python -m tinygpt.train --run_name bpe512_nopos --tokenizer bpe512 --time_budget_min 7 --no_pos_emb
```

Then the figures (into `results/figures/`) and the roofline study (~3–4 min, rewrites `roofline/`):

```bash
python -m viz.make_all
```

```bash
python -m roofline.run
```

Results depend on the machine: the budget is wall-clock, so a faster CPU does more steps. The committed
numbers come from a 4-thread Intel Xeon @ 2.3 GHz cloud VM (the original runs used a 2.1 GHz one); see `RESULTS.md`.

## Demo

Click the badge above, or open `demo.ipynb` in Colab and *Runtime → Run all*. It clones this repo, loads the
committed checkpoints and lets you generate stories and inspect attention, no training needed.
