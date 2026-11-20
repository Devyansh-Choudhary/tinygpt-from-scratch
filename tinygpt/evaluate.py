"""Tokenizer-independent evaluation (bits per character). Owner: Agent C. See SPEC.md §3."""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

from tinygpt.model import GPT, GPTConfig
from tinygpt.tokenizer import TOKENIZER_DIR, get_tokenizer

EVAL_BATCH_SIZE = 64  # windows per forward pass in bits_per_char (affects speed only, not the result)


def bits_per_char(model, tokenizer, val_text: str, block_size: int, max_chars: int = 200000) -> float:
    """Validation bits per character on ``val_text[:max_chars]``.

    Recipe (must be followed exactly so all tokenizers are scored on identical text):
      1. ``text = val_text[:max_chars]``; ``ids = tokenizer.encode(text)``.
      2. Windows ``ids[k*block_size : k*block_size + block_size + 1]`` (overlap of 1 token), so every
         token except ``ids[0]`` is predicted exactly once.
      3. Sum token NLL in nats with ``model.eval()`` and ``torch.no_grad()``.
      4. ``BPC = NLL / ln(2) / (len(text) - len(tokenizer.decode(ids[:1])))``.

    Why this makes char and BPE runs comparable
    -------------------------------------------
    The per-token loss of a BPE model is not comparable to that of a char model: one BPE token stands
    for several characters, so it is "harder" to predict (larger loss) but also covers more text.
    What *is* comparable is the total information the model needs to encode one fixed piece of text.
    By the chain rule, ``sum_i -log p(token_i | tokens_<i)`` is ``-log p(text)`` under the model (the
    tokenization of an ASCII string is deterministic and lossless, so the token sequence identifies
    the text). Dividing that total code length (converted from nats to bits by ``/ ln 2``) by the
    number of characters it covers gives bits per character: the same unit for every tokenizer,
    measured on byte-for-byte the same text (we always truncate by *characters*, never by tokens).

    Two details keep it exact:
      * The windows share one token (input ``ids[k*bs : k*bs+bs]``, targets shifted by one), so the
        *predicted* tokens never overlap and none are skipped: every token but ``ids[0]`` is scored
        exactly once. (Context resets at each window boundary — the same for every tokenizer.)
      * ``ids[0]`` is never predicted, so the characters it covers are removed from the denominator.
        For char that is 1 character; for BPE it is the length of the first token's text.

    Returns:
        BPC as a Python float. The model's train/eval mode is restored afterwards.
    """
    text = val_text[:max_chars]
    ids = tokenizer.encode(text)
    if len(ids) < 2:
        raise ValueError("bits_per_char needs a text that encodes to at least 2 tokens")
    n_chars = len(text) - len(tokenizer.decode(ids[:1]))
    data = torch.tensor(ids, dtype=torch.long)
    n_pred = len(ids) - 1
    n_full = n_pred // block_size  # windows with exactly block_size targets
    rem = n_pred - n_full * block_size

    was_training = model.training
    model.eval()
    total_nll = 0.0
    with torch.no_grad():
        if n_full:
            # windows[k] = ids[k*bs : k*bs + bs + 1]
            windows = data[: n_full * block_size + 1].unfold(0, block_size + 1, block_size)
            for s in range(0, n_full, EVAL_BATCH_SIZE):
                w = windows[s : s + EVAL_BATCH_SIZE]
                total_nll += _sum_nll(model, w[:, :-1].contiguous(), w[:, 1:].contiguous())
        if rem:
            w = data[n_full * block_size :].unsqueeze(0)  # rem + 1 tokens
            total_nll += _sum_nll(model, w[:, :-1].contiguous(), w[:, 1:].contiguous())
    if was_training:
        model.train()
    return total_nll / math.log(2) / n_chars


def _sum_nll(model, x: torch.Tensor, y: torch.Tensor) -> float:
    """Summed cross-entropy (nats) of ``model`` predicting ``y`` from ``x``."""
    logits = model(x)[0]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), reduction="sum").item()


def load_checkpoint(path: str):
    """Load a checkpoint written by ``tinygpt.train``.

    The tokenizer is loaded from ``ckpt["tokenizer_path"]``; if that file does not exist (e.g. the
    checkpoint was written from another working directory) it falls back to
    ``f"{TOKENIZER_DIR}/{kind}.json"``.

    Returns:
        ``(model, tokenizer, ckpt_dict)`` with the model on CPU in eval mode.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    kind = ckpt["tokenizer_kind"]
    tok_path = ckpt.get("tokenizer_path")
    if not tok_path or not os.path.exists(tok_path):
        tok_path = os.path.join(TOKENIZER_DIR, f"{kind}.json")
    tokenizer = get_tokenizer(kind, path=tok_path)
    return model, tokenizer, ckpt
