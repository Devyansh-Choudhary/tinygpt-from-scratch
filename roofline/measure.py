"""Timing helpers: machine ceilings and per-op timings with plain torch ops. Owner: Agent G.

Deliberately does **not** import ``tinygpt.model``: every op is re-created from the SPEC shapes with
the same torch calls the model is specified to use (``F.linear``, ``matmul``, ``masked_fill``,
``softmax``, ``F.gelu``, ``F.layer_norm``, ``F.cross_entropy``).
"""

from __future__ import annotations

import math
import resource
import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from roofline import flops as fl

SEED = 1337
THREADS = 4


def setup(threads: int = THREADS, seed: int = SEED) -> None:
    """Pin threads and seed, as SPEC §4 requires."""
    torch.set_num_threads(threads)
    torch.manual_seed(seed)


def best_time(
    fn: Callable[[], object],
    repeats: int = 5,
    warmup: int = 2,
    min_time: float = 0.05,
    clock: Callable[[], float] = time.perf_counter,
) -> float:
    """Best (minimum) seconds per call of ``fn``.

    After ``warmup`` calls, the number of inner calls per repeat is chosen so one repeat lasts at
    least ``min_time`` seconds (amortises timer and Python overhead); the best of ``repeats``
    repeats is returned -- the least-disturbed run is the closest to what the hardware can do.
    """
    for _ in range(warmup):
        fn()
    t0 = clock()
    fn()
    one = max(clock() - t0, 1e-9)
    inner = max(1, int(math.ceil(min_time / one)))
    best = float("inf")
    for _ in range(repeats):
        t0 = clock()
        for _ in range(inner):
            fn()
        best = min(best, (clock() - t0) / inner)
    return best


# --------------------------------------------------------------------------- ceilings


def matmul_gflops(n: int, repeats: int = 5) -> float:
    """Best GFLOP/s of an ``n x n`` fp32 matmul (``2*n^3`` FLOPs)."""
    a, b = torch.randn(n, n), torch.randn(n, n)
    c = torch.empty(n, n)
    t = best_time(lambda: torch.mm(a, b, out=c), repeats=repeats, min_time=0.2)
    return 2 * n**3 / t / 1e9


def measure_peak_flops(sizes=(1024, 2048, 3072, 4096), repeats: int = 5) -> tuple[float, dict[int, float]]:
    """Peak fp32 FLOP/s = best over large square matmuls. Returns ``(flop_per_s, {n: GFLOP/s})``."""
    per = {n: matmul_gflops(n, repeats) for n in sizes}
    return max(per.values()) * 1e9, per


def stream_gbps(n_bytes: int, kind: str = "copy", repeats: int = 5) -> float:
    """Best GB/s for a STREAM-like kernel over fp32 arrays of ``n_bytes`` each.

    ``copy``: ``b = a``   -> counts 2 arrays of traffic (read a, write b).
    ``add``:  ``c = a + b`` -> counts 3 arrays (read a, read b, write c).
    (Write-allocate reads of the destination are not counted, as in STREAM.)
    """
    n = max(1, n_bytes // fl.FP32)
    a, b, c = torch.randn(n), torch.randn(n), torch.empty(n)
    if kind == "copy":
        fn, arrays = (lambda: c.copy_(a)), 2
    elif kind == "add":
        fn, arrays = (lambda: torch.add(a, b, out=c)), 3
    else:
        raise ValueError(kind)
    t = best_time(fn, repeats=repeats, min_time=0.1)
    return arrays * n * fl.FP32 / t / 1e9


def measure_bandwidth(n_bytes: int = 256 * 2**20, repeats: int = 5) -> tuple[float, dict[str, float]]:
    """DRAM bandwidth = best of copy and a+b over ``n_bytes`` arrays (far bigger than the cache
    share this VM effectively gets). Returns ``(bytes_per_s, {kind: GB/s})``."""
    per = {k: stream_gbps(n_bytes, k, repeats) for k in ("copy", "add")}
    return max(per.values()) * 1e9, per


def bandwidth_sweep(sizes_mb=(1, 4, 16, 64, 256, 1024), repeats: int = 3) -> dict[int, float]:
    """a+b GB/s vs per-array size in MB: shows where the data falls out of L2 / L3 into DRAM."""
    return {mb: stream_gbps(mb * 2**20, "add", repeats) for mb in sizes_mb}


# --------------------------------------------------------------------------- per-op timing


@dataclass
class Shape:
    """Mirror of the ``GPTConfig`` fields the roofline needs (no import of tinygpt.model)."""

    vocab_size: int = 512
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128


def op_callables(B: int, T: int, C: int, H: int, V: int) -> dict[str, Callable[[], object]]:
    """One zero-arg callable per op, on freshly allocated tensors of the exact forward shapes.

    Keys match ``OpCost.name`` from :func:`roofline.flops.forward_op_costs`. Each op allocates its
    output like eager PyTorch does inside the model. q/k/v are contiguous ``[B,H,T,hd]`` tensors.
    """
    M, hd = B * T, C // H
    x = torch.randn(M, C)
    ln_w, ln_b = torch.ones(C), torch.zeros(C)
    w_qkv, b_qkv = torch.randn(3 * C, C) * 0.02, torch.zeros(3 * C)
    w_o, b_o = torch.randn(C, C) * 0.02, torch.zeros(C)
    w_fc, b_fc = torch.randn(4 * C, C) * 0.02, torch.zeros(4 * C)
    w_pr, b_pr = torch.randn(C, 4 * C) * 0.02, torch.zeros(C)
    w_emb = torch.randn(V, C) * 0.02
    q, k, v = (torch.randn(B, H, T, hd) for _ in range(3))
    scale = 1.0 / math.sqrt(hd)
    scores = torch.randn(B, H, T, T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    probs = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
    h = torch.randn(M, 4 * C)
    logits = torch.randn(M, V)
    targets = torch.randint(0, V, (M,))
    return {
        "LayerNorm": lambda: F.layer_norm(x, (C,), ln_w, ln_b),
        "QKV proj": lambda: F.linear(x, w_qkv, b_qkv),
        "QK^T": lambda: (q @ k.transpose(-2, -1)) * scale,
        "softmax": lambda: torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1),
        "attn@V": lambda: probs @ v,
        "out proj": lambda: F.linear(x, w_o, b_o),
        "MLP up": lambda: F.linear(x, w_fc, b_fc),
        "GELU": lambda: F.gelu(h),
        "MLP down": lambda: F.linear(h, w_pr, b_pr),
        f"LM head V={V}": lambda: F.linear(x, w_emb),
        f"cross-ent V={V}": lambda: F.cross_entropy(logits, targets),
    }


def minor_faults_per_call(fn: Callable[[], object], n: int = 20) -> float:
    """Minor page faults per call of ``fn`` (from ``getrusage``): each one is a fresh 4 KiB page the
    kernel had to zero and map because the allocator handed back memory the OS had reclaimed."""
    fn()
    r0 = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
    for _ in range(n):
        fn()
    return (resource.getrusage(resource.RUSAGE_SELF).ru_minflt - r0) / n


@dataclass
class OpResult:
    """An :class:`OpCost` plus its measured best time (and page faults per call)."""

    cost: fl.OpCost
    seconds: float
    faults: float = 0.0

    @property
    def name(self) -> str:
        return self.cost.name

    @property
    def achieved(self) -> float:
        """Achieved FLOP/s."""
        return self.cost.flops / self.seconds


def profile_shapes(
    B: int = 32, T: int = 128, C: int = 128, H: int = 4, L: int = 4, V: int = 512, repeats: int = 5
) -> list[OpResult]:
    """Time every op of :func:`forward_op_costs` at these shapes (forward, no grad)."""
    costs = fl.forward_op_costs(B, T, C, H, L, V)
    fns = op_callables(B, T, C, H, V)
    out = []
    with torch.no_grad():
        for c in costs:
            fn = fns[c.name]
            out.append(OpResult(c, best_time(fn, repeats=repeats), minor_faults_per_call(fn)))
    return out


# --------------------------------------------------------------------------- whole step


def _replica_params(C: int, L: int, V: int, T: int) -> dict[str, torch.Tensor]:
    p = {"tok_emb": torch.randn(V, C) * 0.02, "pos_emb": torch.randn(T, C) * 0.02}
    for i in range(L):
        p[f"{i}.ln1.w"], p[f"{i}.ln1.b"] = torch.ones(C), torch.zeros(C)
        p[f"{i}.qkv.w"], p[f"{i}.qkv.b"] = torch.randn(3 * C, C) * 0.02, torch.zeros(3 * C)
        p[f"{i}.proj.w"], p[f"{i}.proj.b"] = torch.randn(C, C) * 0.02, torch.zeros(C)
        p[f"{i}.ln2.w"], p[f"{i}.ln2.b"] = torch.ones(C), torch.zeros(C)
        p[f"{i}.fc.w"], p[f"{i}.fc.b"] = torch.randn(4 * C, C) * 0.02, torch.zeros(4 * C)
        p[f"{i}.mproj.w"], p[f"{i}.mproj.b"] = torch.randn(C, 4 * C) * 0.02, torch.zeros(C)
    p["ln_f.w"], p["ln_f.b"] = torch.ones(C), torch.zeros(C)
    for t in p.values():
        t.requires_grad_(True)
    return p


def replica_forward(p: dict, idx: torch.Tensor, targets: torch.Tensor, H: int, L: int) -> torch.Tensor:
    """Functional pre-LN GPT forward matching SPEC §3 (tied head), built from plain torch ops."""
    B, T = idx.shape
    C = p["tok_emb"].shape[1]
    hd = C // H
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    x = p["tok_emb"][idx] + p["pos_emb"][:T]
    for i in range(L):
        a = F.layer_norm(x, (C,), p[f"{i}.ln1.w"], p[f"{i}.ln1.b"])
        q, k, v = F.linear(a, p[f"{i}.qkv.w"], p[f"{i}.qkv.b"]).split(C, dim=2)
        q, k, v = (t.view(B, T, H, hd).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hd))
        att = torch.softmax(att.masked_fill(~mask, float("-inf")), dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        x = x + F.linear(y, p[f"{i}.proj.w"], p[f"{i}.proj.b"])
        m = F.layer_norm(x, (C,), p[f"{i}.ln2.w"], p[f"{i}.ln2.b"])
        x = x + F.linear(F.gelu(F.linear(m, p[f"{i}.fc.w"], p[f"{i}.fc.b"])), p[f"{i}.mproj.w"], p[f"{i}.mproj.b"])
    x = F.layer_norm(x, (C,), p["ln_f.w"], p["ln_f.b"])
    logits = F.linear(x, p["tok_emb"])
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


def time_training_step(
    B: int = 32, T: int = 128, C: int = 128, H: int = 4, L: int = 4, V: int = 512, repeats: int = 3
) -> dict[str, float]:
    """Best seconds for the replica's forward (no grad), forward+backward, and a full AdamW step."""
    p = _replica_params(C, L, V, T)
    idx = torch.randint(0, V, (B, T))
    tgt = torch.randint(0, V, (B, T))
    opt = torch.optim.AdamW(p.values(), lr=1e-3, weight_decay=0.1)

    def fwd():
        with torch.no_grad():
            replica_forward(p, idx, tgt, H, L)

    def fwd_bwd():
        replica_forward(p, idx, tgt, H, L).backward()

    def step():
        opt.zero_grad(set_to_none=True)
        replica_forward(p, idx, tgt, H, L).backward()
        torch.nn.utils.clip_grad_norm_(p.values(), 1.0)
        opt.step()

    return {
        "forward": best_time(fwd, repeats=repeats, min_time=0.2),
        "forward+backward": best_time(fwd_bwd, repeats=repeats, min_time=0.2),
        "train step (fwd+bwd+AdamW)": best_time(step, repeats=repeats, min_time=0.2),
    }
