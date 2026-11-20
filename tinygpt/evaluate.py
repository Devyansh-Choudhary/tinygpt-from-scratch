"""Tokenizer-independent evaluation (bits per character). Owner: Agent C. See SPEC.md §3."""

from __future__ import annotations


def bits_per_char(model, tokenizer, val_text: str, block_size: int, max_chars: int = 200000) -> float:
    """Validation bits per character on ``val_text[:max_chars]``.

    Recipe (must be followed exactly so all tokenizers are scored on identical text):
      1. ``text = val_text[:max_chars]``; ``ids = tokenizer.encode(text)``.
      2. Windows ``ids[k*block_size : k*block_size + block_size + 1]`` (overlap of 1 token), so every
         token except ``ids[0]`` is predicted exactly once.
      3. Sum token NLL in nats with ``model.eval()`` and ``torch.no_grad()``.
      4. ``BPC = NLL / ln(2) / (len(text) - len(tokenizer.decode(ids[:1])))``.
    """
    raise NotImplementedError


def load_checkpoint(path: str):
    """Load a checkpoint written by ``tinygpt.train``.

    Returns:
        ``(model, tokenizer, ckpt_dict)`` with the model on CPU in eval mode.
    """
    raise NotImplementedError
