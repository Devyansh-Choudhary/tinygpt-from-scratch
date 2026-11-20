"""Roofline analysis entry point: ``python -m roofline.run``. Owner: Agent G. See SPEC.md §3.

Agent G owns all of ``roofline/`` and may add/split modules freely. Outputs:
``roofline/roofline.png`` and ``roofline/ROOFLINE.md``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OpProfile:
    name: str
    flops: float
    bytes: float
    seconds: float

    @property
    def intensity(self) -> float:
        """Arithmetic intensity, FLOPs per byte."""
        return self.flops / self.bytes

    @property
    def achieved_flops(self) -> float:
        """Achieved FLOP/s."""
        return self.flops / self.seconds


def measure_peak_flops(n: int = 2048, repeats: int = 10) -> float:
    """Peak float32 FLOP/s from timed ``n x n`` matmuls with 4 threads."""
    raise NotImplementedError


def measure_bandwidth(n_bytes: int = 256 * 2**20, repeats: int = 10) -> float:
    """Memory bandwidth in bytes/s from timed large-array copies with 4 threads."""
    raise NotImplementedError


def profile_ops(cfg=None, batch_size: int = 32, seq_len: int = 128) -> list[OpProfile]:
    """Time each forward op of a GPT with ``cfg`` (default ``GPTConfig(vocab_size=512)``) shapes."""
    raise NotImplementedError


def plot_roofline(peak_flops: float, bandwidth: float, ops: list[OpProfile], out_png: str) -> str:
    """Draw the log-log roofline with each op as a point; save to ``out_png`` and return it."""
    raise NotImplementedError


def main() -> None:
    """Measure ceilings, profile ops, write roofline.png and ROOFLINE.md."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
