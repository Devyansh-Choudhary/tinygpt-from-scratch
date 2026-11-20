"""GPT model written from scratch (no nn.Transformer). Owner: Agent B. See SPEC.md §3.

Required attribute names (used by viz/ and roofline/):
    GPT: tok_emb, pos_emb (None if use_pos_emb=False), blocks (nn.ModuleList[Block]), ln_f, lm_head
    Block: ln1, attn (CausalSelfAttention: qkv, proj), ln2, mlp (MLP: fc, proj)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        if cfg.n_embd % cfg.n_head != 0:
            raise ValueError(f"n_embd ({cfg.n_embd}) must be divisible by n_head ({cfg.n_head})")
        self.n_head = cfg.n_head
        self.d_head = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        # True where attention is allowed (j <= i). Not persistent: not part of the checkpoint.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.block_size, cfg.block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(y [B,T,C], att_probs [B,n_head,T,T])``."""
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # [B, T, C] -> [B, n_head, T, d_head]
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B, nh, T, T]
        scores = scores.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))
        att = F.softmax(scores, dim=-1)
        y = self.attn_drop(att) @ v  # [B, nh, T, d_head]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y)), att


class MLP(nn.Module):
    """``fc`` (C -> 4C), GELU, ``proj`` (4C -> C), dropout."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-LayerNorm transformer block: x + attn(ln1(x)); x + mlp(ln2(x))."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(x, att_probs)``."""
        a, att = self.attn(self.ln1(x))
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, att


class GPT(nn.Module):
    """Decoder-only transformer; ``lm_head`` weight tied to ``tok_emb``; init N(0, 0.02)."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd) if cfg.use_pos_emb else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init_weights)
        # GPT-2: scale residual-output projections by 1/sqrt(2 * n_layer).
        for name, p in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("mlp.proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

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
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(x)
        attn = []
        for block in self.blocks:
            x, att = block(x)
            if return_attn:
                attn.append(att.detach())
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.reshape(B * T))
        if return_attn:
            return logits, loss, attn
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None) -> torch.Tensor:
        """Autoregressively sample ``max_new_tokens`` tokens, cropping context to ``block_size``.

        ``temperature <= 0`` means greedy decoding. Runs in eval mode and restores the previous mode.

        Returns the LongTensor ``[B, T + max_new_tokens]``.
        """
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                logits, _ = self(idx[:, -self.cfg.block_size:])
                logits = logits[:, -1, :]
                if temperature <= 0:
                    nxt = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_k is not None:
                        k = min(int(top_k), logits.size(-1))
                        kth = torch.topk(logits, k, dim=-1).values[:, -1:]
                        logits = logits.masked_fill(logits < kth, float("-inf"))
                    nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                idx = torch.cat([idx, nxt], dim=1)
        finally:
            self.train(was_training)
        return idx

    def num_params(self, non_embedding: bool = True) -> int:
        """Parameter count; ``non_embedding`` excludes ``tok_emb`` and ``pos_emb``.

        Tied weights are counted once (``lm_head`` shares ``tok_emb``'s matrix).
        """
        n = sum(p.numel() for p in self.parameters())  # parameters() de-duplicates tied tensors
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if self.pos_emb is not None:
                n -= self.pos_emb.weight.numel()
        return n
