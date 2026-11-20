"""Time-budgeted training on CPU. Owner: Agent C. See SPEC.md §3.

CLI:
    python -m tinygpt.train --run_name NAME --tokenizer char|bpe512|bpe2048 --time_budget_min 7 [--no_pos_emb]
    python -m tinygpt.train --run_config configs/char.json        # same thing, run settings from a file

Outputs:
    results/NAME/log.csv       step,elapsed_s,train_loss,val_loss,val_bpc
    results/NAME/samples.txt   samples at step 0, ~25%, ~50%, end
    results/NAME/summary.json  final numbers + timings (steps/sec, tokenisation time, ...)
    checkpoints/NAME.pt        {"model_config", "model_state", "tokenizer_kind", "tokenizer_path",
                                "run_name", "final_val_bpc"}

Fairness rules implemented here:
  * The budget counts TRAINING time only (batch sampling + forward + backward + optimizer step).
    Tokenisation, evaluation, sampling and checkpointing are excluded (tokenisation time is logged).
  * The learning rate is a function of the elapsed fraction of that budget (linear warmup, then cosine
    decay to ``min_lr_frac * lr``), so every run completes its whole schedule in exactly the budget,
    however fast or slow its steps are.
  * ``elapsed_s`` in log.csv is training seconds, so curves of different runs share an x-axis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict

import numpy as np
import torch

from tinygpt.data import load_corpus, split_text
from tinygpt.evaluate import bits_per_char
from tinygpt.model import GPT, GPTConfig
from tinygpt.tokenizer import TOKENIZER_DIR, get_tokenizer

SEED = 1337
NUM_THREADS = 4
FINAL_EVAL_MAX_CHARS = 200000
TOKENIZER_KINDS = ("char", "bpe512", "bpe2048")

# Fallbacks for keys missing from the config file. configs/default.json is the source of truth.
DEFAULT_HPARAMS = {
    "batch_size": 32,
    "block_size": 128,
    "n_layer": 4,
    "n_head": 4,
    "n_embd": 128,
    "dropout": 0.0,
    "lr": 3e-3,
    "betas": [0.9, 0.95],
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "warmup_frac": 0.05,
    "min_lr_frac": 0.1,
    "eval_interval_s": 30.0,
    "eval_iters": 20,
    "eval_max_chars_during_training": 50000,
    "sample_prompt": "Once upon a time",
    "sample_tokens": 200,
    "sample_temperature": 0.8,
    "sample_top_k": None,
    "sample_fracs": [0.25, 0.5],
}

# Keys allowed in a per-run config (configs/<run>.json). Hyperparameters are NOT allowed there.
RUN_CONFIG_KEYS = {"run_name", "tokenizer", "use_pos_emb", "time_budget_min", "config"}


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------
def load_hparams(config_path: str | None) -> dict:
    """Read hyperparameters from ``config_path`` (keys starting with ``_`` are comments)."""
    hp = dict(DEFAULT_HPARAMS)
    if config_path:
        with open(config_path) as f:
            user = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        unknown = set(user) - set(DEFAULT_HPARAMS)
        if unknown:
            raise ValueError(f"unknown hyperparameter(s) in {config_path}: {sorted(unknown)}")
        hp.update(user)
    return hp


def lr_multiplier(frac: float, warmup_frac: float, min_lr_frac: float) -> float:
    """LR multiplier at elapsed budget fraction ``frac``: linear warmup, then cosine to ``min_lr_frac``."""
    frac = min(max(frac, 0.0), 1.0)
    if warmup_frac > 0 and frac < warmup_frac:
        return frac / warmup_frac
    progress = (frac - warmup_frac) / max(1.0 - warmup_frac, 1e-12)
    return min_lr_frac + (1.0 - min_lr_frac) * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, generator: torch.Generator):
    """Random contiguous windows: ``x = data[i:i+T]``, ``y = data[i+1:i+T+1]``."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    return _windows(data, ix, block_size)


def _windows(data: torch.Tensor, ix: torch.Tensor, block_size: int):
    w = data[ix.unsqueeze(1) + torch.arange(block_size + 1)].long()
    return w[:, :-1].contiguous(), w[:, 1:].contiguous()


def configure_optimizer(model: torch.nn.Module, hp: dict) -> torch.optim.Optimizer:
    """AdamW; weight decay only on >=2-D params (matrices/embeddings), not on biases/LayerNorm."""
    params = [p for p in model.parameters() if p.requires_grad]  # tied weights appear once
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": hp["weight_decay"]},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=hp["lr"], betas=tuple(hp["betas"]))


def encode_cached(tokenizer, text: str, tokenizer_path: str | None, cache_dir: str | None) -> np.ndarray:
    """``tokenizer.encode(text)`` as an int32 array, cached in ``cache_dir`` keyed by text + tokenizer file."""
    key = None
    if cache_dir and tokenizer_path and os.path.exists(tokenizer_path):
        h = hashlib.sha1()
        with open(tokenizer_path, "rb") as f:
            h.update(f.read())
        h.update(b"\0")
        h.update(text.encode("utf-8"))
        key = os.path.join(cache_dir, f"{tokenizer.name}_{h.hexdigest()[:16]}.npy")
        if os.path.exists(key):
            return np.load(key)
    ids = np.asarray(tokenizer.encode(text), dtype=np.int32)
    if key:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = key + f".{os.getpid()}.tmp.npy"
        np.save(tmp, ids)
        os.replace(tmp, key)
    return ids


@torch.no_grad()
def estimate_loss(model, data: torch.Tensor, offsets: torch.Tensor, batch_size: int, block_size: int) -> float:
    """Mean per-token loss (nats) over fixed window ``offsets`` (eval mode, mode restored)."""
    was_training = model.training
    model.eval()
    losses = []
    for s in range(0, len(offsets), batch_size):
        x, y = _windows(data, offsets[s : s + batch_size], block_size)
        losses.append(model(x, y)[1].item())
    if was_training:
        model.train()
    return float(np.mean(losses))


@torch.no_grad()
def sample_text(model, tokenizer, hp: dict, seed: int) -> str:
    was_training = model.training
    model.eval()
    torch.manual_seed(seed)
    idx = torch.tensor([tokenizer.encode(hp["sample_prompt"])], dtype=torch.long)
    out = model.generate(idx, hp["sample_tokens"], temperature=hp["sample_temperature"], top_k=hp["sample_top_k"])
    if was_training:
        model.train()
    return tokenizer.decode(out[0].tolist())


# --------------------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------------------
def run_training(run_name: str, tokenizer_kind: str, time_budget_min: float = 7.0, use_pos_emb: bool = True,
                 max_steps: int | None = None, text: str | None = None, results_dir: str = "results",
                 checkpoint_dir: str = "checkpoints", tokenizer_dir: str = TOKENIZER_DIR,
                 config_path: str = "configs/default.json", cache_dir: str | None = ".cache/tokens",
                 verbose: bool = True) -> dict:
    """Train one run within a wall-clock budget of TRAINING time (eval/sampling/tokenizing excluded).

    Args:
        text: corpus override (tests); defaults to ``load_corpus()``.
        max_steps: optional hard step cap (tests / smoke runs). With a cap, the schedule fraction is
            ``max(train_time / budget, step / max_steps)`` so the schedule still completes.
        cache_dir: where encoded token arrays are cached (``None`` disables caching).
        verbose: print progress.

    Returns:
        dict with ``final_val_bpc`` (float, max_chars=200000), ``steps`` (int) and timing info
        (also written to ``results/NAME/summary.json``).
    """
    if tokenizer_kind not in TOKENIZER_KINDS:
        raise ValueError(f"tokenizer must be one of {TOKENIZER_KINDS}, got {tokenizer_kind!r}")
    log = print if verbose else (lambda *a, **k: None)
    hp = load_hparams(config_path)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(NUM_THREADS)
    bs, T = hp["batch_size"], hp["block_size"]
    budget_s = time_budget_min * 60.0

    # ---- data + tokenisation (not counted in the budget) -----------------------------------------
    if text is None:
        text = load_corpus()
    train_text, val_text = split_text(text)
    tokenizer_path = os.path.join(tokenizer_dir, f"{tokenizer_kind}.json")
    t0 = time.perf_counter()
    tokenizer = get_tokenizer(tokenizer_kind, path=tokenizer_path, train_text=train_text)
    train_ids = encode_cached(tokenizer, train_text, tokenizer_path, cache_dir)
    val_ids = encode_cached(tokenizer, val_text, tokenizer_path, cache_dir)
    tokenize_s = time.perf_counter() - t0
    train_data = torch.from_numpy(train_ids.astype(np.int64))
    val_data = torch.from_numpy(val_ids.astype(np.int64))
    log(f"[{run_name}] tokenizer={tokenizer.name} vocab={tokenizer.vocab_size} | tokenised "
        f"{len(train_text):,}+{len(val_text):,} chars -> {len(train_ids):,}+{len(val_ids):,} tokens "
        f"in {tokenize_s:.1f}s (not counted in budget)")
    if len(train_data) <= T + 1 or len(val_data) <= T + 1:
        raise ValueError("corpus too small for block_size")

    # ---- model + optimizer ------------------------------------------------------------------------
    torch.manual_seed(SEED)
    cfg = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=T, n_layer=hp["n_layer"], n_head=hp["n_head"],
                    n_embd=hp["n_embd"], dropout=hp["dropout"], use_pos_emb=use_pos_emb)
    model = GPT(cfg)
    model.train()
    optimizer = configure_optimizer(model, hp)
    n_params = model.num_params()
    log(f"[{run_name}] params (non-embedding) {n_params:,} | budget {budget_s:.0f}s"
        + (f" | max_steps {max_steps}" if max_steps else ""))

    # Fixed eval windows (separate RNG so eval never perturbs the training batch stream).
    g_eval = torch.Generator().manual_seed(SEED + 1)
    n_eval = hp["eval_iters"] * bs
    val_offsets = torch.randint(len(val_data) - T - 1, (n_eval,), generator=g_eval)
    train_offsets = torch.randint(len(train_data) - T - 1, (n_eval,), generator=g_eval)
    g_train = torch.Generator().manual_seed(SEED)

    run_dir = os.path.join(results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_f = open(os.path.join(run_dir, "log.csv"), "w", newline="")
    writer = csv.writer(log_f)
    writer.writerow(["step", "elapsed_s", "train_loss", "val_loss", "val_bpc"])
    samples_f = open(os.path.join(run_dir, "samples.txt"), "w")

    def progress(step: int, train_time: float) -> float:
        f = train_time / budget_s if budget_s > 0 else 1.0
        if max_steps:
            f = max(f, step / max_steps)
        return f

    def write_row(step, train_time, train_loss, final=False):
        val_loss = estimate_loss(model, val_data, val_offsets, bs, T)
        max_chars = FINAL_EVAL_MAX_CHARS if final else hp["eval_max_chars_during_training"]
        val_bpc = bits_per_char(model, tokenizer, val_text, T, max_chars=max_chars)
        writer.writerow([step, f"{train_time:.2f}", f"{train_loss:.5f}", f"{val_loss:.5f}", f"{val_bpc:.5f}"])
        log_f.flush()
        return val_loss, val_bpc

    def write_sample(step, train_time):
        s = sample_text(model, tokenizer, hp, seed=SEED + step)
        samples_f.write(f"=== step {step} (elapsed {train_time:.1f}s) ===\n{s}\n\n")
        samples_f.flush()

    # ---- step 0 -------------------------------------------------------------------------------------
    step, train_time = 0, 0.0
    train_loss0 = estimate_loss(model, train_data, train_offsets, bs, T)
    vl, vb = write_row(0, 0.0, train_loss0)
    log(f"[{run_name}] step 0 | train {train_loss0:.4f} | val {vl:.4f} | bpc {vb:.4f}")
    write_sample(0, 0.0)

    # ---- training loop ------------------------------------------------------------------------------
    interval = hp["eval_interval_s"]
    next_eval = interval
    pending_fracs = sorted(hp["sample_fracs"])
    recent_losses: list[float] = []
    last_log_step, last_log_time = 0, 0.0
    wall0 = time.perf_counter()
    while True:
        frac = progress(step, train_time)
        if frac >= 1.0:
            break
        t_start = time.perf_counter()
        lr = hp["lr"] * lr_multiplier(frac, hp["warmup_frac"], hp["min_lr_frac"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = get_batch(train_data, bs, T, g_train)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if hp["grad_clip"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
        optimizer.step()
        train_time += time.perf_counter() - t_start
        step += 1
        recent_losses.append(loss.item())

        frac = progress(step, train_time)
        if frac >= 1.0:
            break
        if train_time >= next_eval:
            tl = float(np.mean(recent_losses))
            vl, vb = write_row(step, train_time, tl)
            sps = (step - last_log_step) / max(train_time - last_log_time, 1e-9)
            log(f"[{run_name}] step {step} | t {train_time:6.1f}s ({100 * frac:4.1f}%) | lr {lr:.2e} | "
                f"train {tl:.4f} | val {vl:.4f} | bpc {vb:.4f} | {sps:.2f} steps/s")
            recent_losses, last_log_step, last_log_time = [], step, train_time
            next_eval = (math.floor(train_time / interval) + 1) * interval
        if pending_fracs and frac >= pending_fracs[0]:
            while pending_fracs and frac >= pending_fracs[0]:
                pending_fracs.pop(0)
            write_sample(step, train_time)

    # ---- final eval, sample, checkpoint -------------------------------------------------------------
    tl = float(np.mean(recent_losses)) if recent_losses else train_loss0
    final_val_loss, final_val_bpc = write_row(step, train_time, tl, final=True)
    write_sample(step, train_time)
    log_f.close()
    samples_f.close()

    ckpt_path = os.path.join(checkpoint_dir, f"{run_name}.pt")
    torch.save({"model_config": asdict(cfg), "model_state": model.state_dict(), "tokenizer_kind": tokenizer_kind,
                "tokenizer_path": tokenizer_path, "run_name": run_name, "final_val_bpc": float(final_val_bpc)},
               ckpt_path)

    steps_per_sec = step / train_time if train_time > 0 else 0.0
    val_chars = val_text[:FINAL_EVAL_MAX_CHARS]
    summary = {
        "run_name": run_name,
        "tokenizer": tokenizer_kind,
        "vocab_size": tokenizer.vocab_size,
        "use_pos_emb": use_pos_emb,
        "final_val_bpc": float(final_val_bpc),
        "final_val_loss": float(final_val_loss),
        "final_train_loss": tl,
        "steps": step,
        "train_time_s": train_time,
        "time_budget_s": budget_s,
        "wall_time_s": time.perf_counter() - wall0,
        "steps_per_sec": steps_per_sec,
        "tokens_per_sec": steps_per_sec * bs * T,
        "tokenize_time_s": tokenize_s,
        "n_train_tokens": int(len(train_ids)),
        "n_val_tokens": int(len(val_ids)),
        "chars_per_token_val": len(val_chars) / max(len(tokenizer.encode(val_chars)), 1),
        "num_params_non_embedding": n_params,
        "num_params_total": model.num_params(non_embedding=False),
        "checkpoint": ckpt_path,
        "hparams": hp,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"[{run_name}] done: {step} steps in {train_time:.1f}s training ({steps_per_sec:.2f} steps/s) | "
        f"final val loss {final_val_loss:.4f} | final val BPC {final_val_bpc:.4f} | saved {ckpt_path}")
    return summary


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    """Argument parser for the CLI described in the module docstring."""
    p = argparse.ArgumentParser(prog="python -m tinygpt.train", description="Time-budgeted TinyGPT training (CPU).")
    p.add_argument("--run_name", help="results/<run_name>/, checkpoints/<run_name>.pt")
    p.add_argument("--tokenizer", choices=TOKENIZER_KINDS)
    p.add_argument("--time_budget_min", type=float, default=None, help="training-time budget in minutes (default 7)")
    p.add_argument("--no_pos_emb", action="store_true", default=None, help="disable learned position embeddings")
    p.add_argument("--config", default=None, help="hyperparameter JSON (default configs/default.json)")
    p.add_argument("--run_config", default=None,
                   help="per-run JSON (configs/<run>.json) with run_name/tokenizer/use_pos_emb/time_budget_min; "
                        "explicit flags override it")
    p.add_argument("--max_steps", type=int, default=None, help="optional hard cap on optimizer steps")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--tokenizer_dir", default=TOKENIZER_DIR)
    return p


def main(argv: list[str] | None = None) -> None:
    """Parse ``argv`` and call :func:`run_training`."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_cfg: dict = {}
    if args.run_config:
        with open(args.run_config) as f:
            run_cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        bad = set(run_cfg) - RUN_CONFIG_KEYS
        if bad:
            parser.error(f"{args.run_config}: only {sorted(RUN_CONFIG_KEYS)} allowed in a run config "
                         f"(hyperparameters belong in configs/default.json); got {sorted(bad)}")
    run_name = args.run_name or run_cfg.get("run_name")
    tokenizer = args.tokenizer or run_cfg.get("tokenizer")
    if not run_name or not tokenizer:
        parser.error("--run_name and --tokenizer are required (directly or via --run_config)")
    if tokenizer not in TOKENIZER_KINDS:
        parser.error(f"tokenizer must be one of {TOKENIZER_KINDS}")
    use_pos_emb = False if args.no_pos_emb else bool(run_cfg.get("use_pos_emb", True))
    budget = args.time_budget_min if args.time_budget_min is not None else float(run_cfg.get("time_budget_min", 7.0))
    config_path = args.config or run_cfg.get("config", "configs/default.json")
    run_training(run_name, tokenizer, time_budget_min=budget, use_pos_emb=use_pos_emb, max_steps=args.max_steps,
                 results_dir=args.results_dir, checkpoint_dir=args.checkpoint_dir,
                 tokenizer_dir=args.tokenizer_dir, config_path=config_path)


if __name__ == "__main__":
    main()
