"""First-principles FLOP and byte counts for one TinyGPT forward pass. Owner: Agent G.

Notation (SPEC defaults in brackets)::

    B  batch size            [32]         T  sequence length   [128]
    C  n_embd                [128]        H  n_head            [4]
    hd head dim = C / H      [32]         L  n_layer           [4]
    V  vocab size            [96 | 512 | 2048]
    M  = B * T  rows of the activation matrix [4096]
    s  = 4 bytes per fp32 element

Counting conventions
--------------------
* **FLOPs**: one multiply-add = 2 FLOPs, so ``(m x k) @ (k x n)`` costs ``2*m*k*n``.
  Every elementwise add/sub/mul/div/max and every transcendental (``exp``, ``erf``) counts as
  **1 FLOP**. (Real hardware spends ~10-20 instructions on an ``exp``/``erf``; the roofline therefore
  *understates* how hard elementwise ops make the CPU work, which only strengthens the conclusion
  that they are cheap in FLOPs and dominated by memory/overhead.)
* **Bytes**: the *compulsory* traffic of a perfectly fused kernel -- every input (activations,
  weights, biases) read exactly once and every output written exactly once, in fp32. Real eager
  PyTorch moves at least this much (unfused ops add extra round trips; write-allocate adds a read of
  the output), so the intensity computed here is an *upper bound* on the real one.

Per-op formulas (one occurrence; ``count`` says how many times it occurs per forward)
---------------------------------------------------------------------------------------
=================  =========================  ======================================  ===========
op                 FLOPs                      bytes / s                               count
=================  =========================  ======================================  ===========
LayerNorm          7*M*C                      M*C (in) + M*C (out) + 2*C (gamma,beta)  2L + 1
QKV projection     2*M*C*3C + 3*M*C (bias)    M*C + 3C*C + 3C + 3*M*C                 L
QK^T (scaled)      2*B*T*T*C + B*H*T*T        2*B*T*C (q,k) + B*H*T*T (scores)        L
softmax (+mask)    5*B*H*T*T                  2*B*H*T*T (+ T*T/4 for a bool mask)     L
attn @ V           2*B*T*T*C                  B*H*T*T + B*T*C (v) + B*T*C (out)       L
output proj        2*M*C*C + M*C              M*C + C*C + C + M*C                     L
MLP up (fc)        2*M*C*4C + 4*M*C           M*C + 4C*C + 4C + 4*M*C                 L
GELU               5*M*4C                     4*M*C + 4*M*C                           L
MLP down (proj)    2*M*4C*C + M*C             4*M*C + 4C*C + C + M*C                  L
LM head            2*M*C*V (tied, no bias)    M*C + C*V + M*V                         1
cross-entropy*     5*M*V                      M*V (+ M targets as int64)              1
=================  =========================  ======================================  ===========

``*`` cross-entropy is not part of the forward network but is computed every training step on the
``[M, V]`` logits, so it is included as an extra row: it is the other op that grows with ``V``.

Where the elementwise constants come from:
  * LayerNorm per element: mean (1 add), centre (1 sub), square (1 mul), variance sum (1 add),
    scale by rstd (1 mul), gamma (1 mul), beta (1 add) = 7 (per-row rsqrt is O(M), ignored).
  * softmax per element: max (1), subtract max (1), exp (1), sum (1), divide (1) = 5. The causal mask
    is a select, 0 FLOPs.
  * GELU (exact, ``0.5*x*(1+erf(x/sqrt 2))``): divide, erf, add, mul, mul = 5.
  * cross-entropy per logit: the log-softmax (max, sub, exp, sum ~ 4) + the gather/sum ~ 1 = 5.

Sanity check built into the tests: the matmul FLOPs of all Linear layers sum to
``2 * M * (12*L*C*C + C*V)`` = 2 x (weights touched) x (tokens) -- the classic "2N FLOPs per token
per parameter" rule -- and the attention matmuls add ``4*L*B*T*T*C``.
"""

from __future__ import annotations

from dataclasses import dataclass

FP32 = 4  # bytes per element
INT64 = 8


@dataclass(frozen=True)
class OpCost:
    """FLOPs and compulsory bytes for one occurrence of an op; ``count`` occurrences per forward."""

    name: str
    flops: float
    bytes: float
    count: int = 1
    kind: str = "matmul"  # "matmul" | "attention" | "elementwise"
    shape: str = ""

    @property
    def intensity(self) -> float:
        """Arithmetic intensity in FLOPs per byte."""
        return self.flops / self.bytes


# --------------------------------------------------------------------------- building blocks


def matmul_flops(m: int, k: int, n: int, batch: int = 1, bias: bool = False) -> int:
    """FLOPs of ``batch`` products ``(m x k) @ (k x n)``: ``2*m*k*n`` per product (+ ``m*n`` bias adds)."""
    return batch * (2 * m * k * n + (m * n if bias else 0))


def matmul_bytes(
    m: int,
    k: int,
    n: int,
    batch: int = 1,
    bias: bool = False,
    shared_rhs: bool = False,
    dtype_bytes: int = FP32,
) -> int:
    """Compulsory bytes of ``(m x k) @ (k x n)`` (x ``batch``): read A and B once, write C once.

    ``shared_rhs=True`` means the right operand is one weight matrix shared by every batch item
    (a Linear layer), so it is read once rather than ``batch`` times.
    """
    a = batch * m * k
    b = k * n * (1 if shared_rhs else batch)
    c = batch * m * n
    bias_elems = n if bias else 0
    return dtype_bytes * (a + b + c + bias_elems)


def linear_cost(name: str, M: int, c_in: int, c_out: int, bias: bool = True, count: int = 1) -> OpCost:
    """``y[M, c_out] = x[M, c_in] @ W[c_in, c_out] (+ b)``.

    FLOPs = ``2*M*c_in*c_out (+ M*c_out)``;
    bytes = ``s*(M*c_in + c_in*c_out (+ c_out) + M*c_out)``.
    """
    return OpCost(
        name,
        matmul_flops(M, c_in, c_out, bias=bias),
        matmul_bytes(M, c_in, c_out, bias=bias, shared_rhs=True),
        count,
        "matmul",
        f"[{M},{c_in}]@[{c_in},{c_out}]",
    )


# --------------------------------------------------------------------------- per-op formulas


def layernorm_cost(M: int, C: int, count: int = 1) -> OpCost:
    """LayerNorm over the last dim of ``[M, C]``. FLOPs = ``7*M*C``; bytes = ``s*(2*M*C + 2*C)``."""
    return OpCost("LayerNorm", 7 * M * C, FP32 * (2 * M * C + 2 * C), count, "elementwise", f"[{M},{C}]")


def qkv_cost(M: int, C: int, count: int = 1) -> OpCost:
    """Fused QKV projection ``[M,C] @ [C,3C] + b``. FLOPs = ``6*M*C^2 + 3*M*C``."""
    return linear_cost("QKV proj", M, C, 3 * C, bias=True, count=count)


def attn_scores_cost(B: int, T: int, C: int, H: int, count: int = 1) -> OpCost:
    """Scaled scores ``(q @ k^T) / sqrt(hd)`` for B*H heads of ``[T,hd] @ [hd,T]``.

    FLOPs = ``2*B*H*T*T*hd + B*H*T*T = 2*B*T*T*C + B*H*T*T``;
    bytes = ``s*(2*B*T*C + B*H*T*T)`` (read q and k, write the score matrix).
    """
    hd = C // H
    flops = matmul_flops(T, hd, T, batch=B * H) + B * H * T * T
    nbytes = FP32 * (2 * B * T * C + B * H * T * T)
    return OpCost("QK^T", flops, nbytes, count, "attention", f"{B * H}x[{T},{hd}]@[{hd},{T}]")


def softmax_cost(B: int, T: int, H: int, count: int = 1) -> OpCost:
    """Causal mask + row softmax over ``[B,H,T,T]``.

    FLOPs = ``5*B*H*T*T``; bytes = ``s*2*B*H*T*T + T*T`` (scores in, probs out, 1-byte bool mask).
    """
    n = B * H * T * T
    return OpCost("softmax", 5 * n, FP32 * 2 * n + T * T, count, "elementwise", f"[{B},{H},{T},{T}]")


def attn_values_cost(B: int, T: int, C: int, H: int, count: int = 1) -> OpCost:
    """``att @ v`` for B*H heads of ``[T,T] @ [T,hd]``.

    FLOPs = ``2*B*H*T*T*hd = 2*B*T*T*C``; bytes = ``s*(B*H*T*T + 2*B*T*C)``.
    """
    hd = C // H
    flops = matmul_flops(T, T, hd, batch=B * H)
    nbytes = FP32 * (B * H * T * T + 2 * B * T * C)
    return OpCost("attn@V", flops, nbytes, count, "attention", f"{B * H}x[{T},{T}]@[{T},{hd}]")


def out_proj_cost(M: int, C: int, count: int = 1) -> OpCost:
    """Attention output projection ``[M,C] @ [C,C] + b``. FLOPs = ``2*M*C^2 + M*C``."""
    return linear_cost("out proj", M, C, C, bias=True, count=count)


def mlp_up_cost(M: int, C: int, count: int = 1) -> OpCost:
    """MLP expansion ``[M,C] @ [C,4C] + b``. FLOPs = ``8*M*C^2 + 4*M*C``."""
    return linear_cost("MLP up", M, C, 4 * C, bias=True, count=count)


def gelu_cost(M: int, C: int, count: int = 1) -> OpCost:
    """Exact GELU on ``[M, 4C]``. FLOPs = ``5*4*M*C``; bytes = ``s*2*4*M*C``."""
    n = M * 4 * C
    return OpCost("GELU", 5 * n, FP32 * 2 * n, count, "elementwise", f"[{M},{4 * C}]")


def mlp_down_cost(M: int, C: int, count: int = 1) -> OpCost:
    """MLP contraction ``[M,4C] @ [4C,C] + b``. FLOPs = ``8*M*C^2 + M*C``."""
    return linear_cost("MLP down", M, 4 * C, C, bias=True, count=count)


def lm_head_cost(M: int, C: int, V: int, count: int = 1) -> OpCost:
    """Tied LM head ``[M,C] @ [C,V]`` (no bias). FLOPs = ``2*M*C*V``;
    bytes = ``s*(M*C + C*V + M*V)``."""
    op = linear_cost(f"LM head V={V}", M, C, V, bias=False, count=count)
    return op


def cross_entropy_cost(M: int, V: int, count: int = 1) -> OpCost:
    """Mean cross-entropy over ``[M, V]`` logits. FLOPs = ``5*M*V``; bytes = ``s*M*V + 8*M``."""
    return OpCost(f"cross-ent V={V}", 5 * M * V, FP32 * M * V + INT64 * M, count, "elementwise", f"[{M},{V}]")


# --------------------------------------------------------------------------- whole forward


def forward_op_costs(
    B: int = 32,
    T: int = 128,
    C: int = 128,
    H: int = 4,
    L: int = 4,
    V: int = 512,
    include_loss: bool = True,
) -> list[OpCost]:
    """All ops of one forward pass with their per-forward ``count`` (embeddings/residuals omitted:
    they are a few ``M*C`` adds/gathers, < 1% of time)."""
    if C % H:
        raise ValueError("n_embd must be divisible by n_head")
    M = B * T
    ops = [
        layernorm_cost(M, C, count=2 * L + 1),
        qkv_cost(M, C, count=L),
        attn_scores_cost(B, T, C, H, count=L),
        softmax_cost(B, T, H, count=L),
        attn_values_cost(B, T, C, H, count=L),
        out_proj_cost(M, C, count=L),
        mlp_up_cost(M, C, count=L),
        gelu_cost(M, C, count=L),
        mlp_down_cost(M, C, count=L),
        lm_head_cost(M, C, V, count=1),
    ]
    if include_loss:
        ops.append(cross_entropy_cost(M, V, count=1))
    return ops


def total_flops(ops: list[OpCost]) -> float:
    """Sum of ``flops * count`` over ``ops``."""
    return float(sum(o.flops * o.count for o in ops))


def total_bytes(ops: list[OpCost]) -> float:
    """Sum of ``bytes * count`` over ``ops``."""
    return float(sum(o.bytes * o.count for o in ops))


def attainable(intensity: float, peak_flops: float, bandwidth: float) -> float:
    """Roofline: attainable FLOP/s = ``min(peak, bandwidth * intensity)``."""
    return min(peak_flops, bandwidth * intensity)


def ridge_point(peak_flops: float, bandwidth: float) -> float:
    """Intensity (FLOPs/byte) where the bandwidth slope meets the compute roof."""
    return peak_flops / bandwidth
