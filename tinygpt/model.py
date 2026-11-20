"""GPT model written from scratch (no nn.Transformer). Owner: Agent B. See SPEC.md §3.

Required attribute names (used by viz/ and roofline/):
    GPT: tok_emb, pos_emb (None if use_pos_emb=False), blocks (nn.ModuleList[Block]), ln_f, lm_head
    Block: ln1, attn (CausalSelfAttention: qkv, proj), ln2, mlp (MLP: fc, proj)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    use_pos_emb: bool = True


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention computed by hand: softmax(QK^T / sqrt(d) + causal mask) V."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(y [B,T,C], att_probs [B,n_head,T,T])``."""
        raise NotImplementedError


class MLP(nn.Module):
    """``fc`` (C -> 4C), GELU, ``proj`` (4C -> C), dropout."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Block(nn.Module):
    """Pre-LayerNorm transformer block: x + attn(ln1(x)); x + mlp(ln2(x))."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(x, att_probs)``."""
        raise NotImplementedError


class GPT(nn.Module):
    """Decoder-only transformer; ``lm_head`` weight tied to ``tok_emb``; init N(0, 0.02)."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, return_attn: bool = False):
        """Run the model.

        Args:
            idx: LongTensor ``[B, T]`` with ``T <= block_size`` (raise otherwise).
            targets: optional LongTensor ``[B, T]``.
            return_attn: also return attention probabilities.

        Returns:
            ``(logits [B,T,vocab], loss)`` where loss is mean cross-entropy in nats (``None`` without
            targets), or ``(logits, loss, attn)`` where ``attn`` is a list (one per layer) of
            detached tensors ``[B, n_head, T, T]``.
        """
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None) -> torch.Tensor:
        """Autoregressively sample ``max_new_tokens`` tokens, cropping context to ``block_size``.

        Returns the LongTensor ``[B, T + max_new_tokens]``.
        """
        raise NotImplementedError

    def num_params(self, non_embedding: bool = True) -> int:
        """Parameter count; ``non_embedding`` excludes ``tok_emb`` and ``pos_emb``."""
        raise NotImplementedError
