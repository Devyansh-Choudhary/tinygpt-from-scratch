"""Roofline analysis entry point: ``python -m roofline.run``. Owner: Agent G. See SPEC.md §3.

Steps
  1. Measure this machine's ceilings with 4 threads: peak fp32 FLOP/s (best large square matmul,
     n = 1024..4096) and DRAM bandwidth (best of STREAM copy / a+b over 256 MiB arrays), plus a
     bandwidth-vs-size sweep that shows the cache levels.
  2. Count FLOPs and compulsory bytes of every forward op from first principles
     (:mod:`roofline.flops`).
  3. Time each op with standalone torch ops at the SPEC shapes (:mod:`roofline.measure`), for the
     base config and the what-if configs (vocab 96/2048, n_embd 512, batch 128), and time a full
     training step of a plain-torch replica of the model.
  4. Write ``roofline/roofline.png``, ``roofline/roofline_scenarios.png``, ``roofline/results.json``
     and ``roofline/ROOFLINE.md``.

Allocator regimes. Op timings depend strongly on whether PyTorch's output buffers are *fresh*
pages from the OS (every 4 KiB page costs a minor page fault: the kernel zeroes and maps it) or
*reused* heap memory. glibc decides this with its mmap/trim thresholds, which it adjusts
dynamically, making in-process timings depend on what ran before. So each regime is measured in its
own subprocess with the thresholds pinned by environment variables:

* ``reuse``   -- ``MALLOC_MMAP_THRESHOLD_=1GiB MALLOC_TRIM_THRESHOLD_=16GiB``: freed buffers stay in
  the heap and are reused; ~0 page faults. This is the op's intrinsic speed (plotted filled).
* ``fresh``   -- ``MALLOC_MMAP_THRESHOLD_=128KiB`` (glibc's initial value, pinned): every buffer above
  128 KiB is a fresh mmap and is page-faulted in (plotted hollow).
* ``default`` -- untouched glibc (what ``python -m tinygpt.train`` gets); used for the training-step
  timing only.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass

import numpy as np
import torch

from roofline import flops as fl
from roofline import measure as ms
from roofline.plot import plot_panels
from roofline.plot import plot_roofline as _plot_points

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(HERE, "roofline.png")
OUT_SCEN_PNG = os.path.join(HERE, "roofline_scenarios.png")
OUT_MD = os.path.join(HERE, "ROOFLINE.md")
OUT_JSON = os.path.join(HERE, "results.json")

BASE = dict(B=32, T=128, C=128, H=4, L=4, V=512)
SCENARIOS = {
    "base": {},
    "V=96": dict(V=96),
    "V=2048": dict(V=2048),
    "n_embd=512": dict(C=512),
    "batch=128": dict(B=128),
}
MALLOC_VARS = ("MALLOC_MMAP_THRESHOLD_", "MALLOC_TRIM_THRESHOLD_", "MALLOC_TOP_PAD_")
REGIMES = {
    "reuse": {"MALLOC_MMAP_THRESHOLD_": str(2**30), "MALLOC_TRIM_THRESHOLD_": str(2**34)},
    "fresh": {"MALLOC_MMAP_THRESHOLD_": str(128 * 1024)},
    "default": {},
}


# --------------------------------------------------------------------------- legacy stub API


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
    ms.setup()
    return ms.matmul_gflops(n, repeats) * 1e9


def measure_bandwidth(n_bytes: int = 256 * 2**20, repeats: int = 10) -> float:
    """Memory bandwidth in bytes/s (best of copy and a+b) over ``n_bytes`` arrays with 4 threads."""
    ms.setup()
    return ms.measure_bandwidth(n_bytes, repeats)[0]


def profile_ops(cfg=None, batch_size: int = 32, seq_len: int = 128) -> list[OpProfile]:
    """Time each forward op of a GPT with ``cfg`` shapes (default ``vocab_size=512``, SPEC sizes).

    ``cfg`` may be a ``GPTConfig`` or anything with ``n_embd, n_head, n_layer, vocab_size``.
    """
    ms.setup()
    cfg = cfg or ms.Shape()
    res = ms.profile_shapes(batch_size, seq_len, cfg.n_embd, cfg.n_head, cfg.n_layer, cfg.vocab_size)
    return [OpProfile(r.name, r.cost.flops, r.cost.bytes, r.seconds) for r in res]


def plot_roofline(peak_flops: float, bandwidth: float, ops: list, out_png: str) -> str:
    """Draw the log-log roofline with each op (``OpProfile``/``OpResult``/dict) as a point."""
    pts = []
    for o in ops:
        if isinstance(o, dict):
            pts.append(o)
        elif isinstance(o, ms.OpResult):
            pts.append(dict(name=o.name, intensity=o.cost.intensity, gflops=o.achieved / 1e9, kind=o.cost.kind))
        else:
            pts.append(dict(name=o.name, intensity=o.intensity, gflops=o.achieved_flops / 1e9, kind="matmul"))
    return _plot_points(peak_flops, bandwidth, pts, out_png)


# --------------------------------------------------------------------------- worker (subprocess)


def shape_of(key: str) -> dict:
    return {**BASE, **SCENARIOS[key]}


def run_worker(regime: str, scenarios=None, repeats: int = 5, step_repeats: int = 5) -> dict:
    """Measurements done inside one allocator regime. Returns a JSON-able dict."""
    ms.setup()
    out = {"regime": regime, "env": {k: os.environ.get(k) for k in MALLOC_VARS}, "scenarios": {}}
    # Warm the thread pool / CPU clocks before anything is timed.
    ms.best_time(lambda: torch.mm(torch.randn(1024, 1024), torch.randn(1024, 1024)), repeats=3, min_time=0.3)
    for key in scenarios or SCENARIOS:
        s = shape_of(key)
        entry = {"shape": s}
        if regime != "default":
            entry["ops"] = [
                dict(
                    name=r.name,
                    kind=r.cost.kind,
                    shape=r.cost.shape,
                    flops=r.cost.flops,
                    bytes=r.cost.bytes,
                    count=r.cost.count,
                    seconds=r.seconds,
                    faults=r.faults,
                )
                for r in ms.profile_shapes(**s, repeats=repeats)
            ]
        entry["step"] = ms.time_training_step(**s, repeats=step_repeats)
        out["scenarios"][key] = entry
    return out


def run_regime_subprocess(regime: str, json_path: str, scenarios=None) -> dict:
    """Run :func:`run_worker` in a fresh interpreter with the regime's malloc env vars."""
    env = {k: v for k, v in os.environ.items() if k not in MALLOC_VARS}
    env.update(REGIMES[regime])
    env["PYTHONPATH"] = os.pathsep.join([os.path.dirname(HERE), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    cmd = [sys.executable, "-m", "roofline.run", "--worker", regime, "--json", json_path]
    if scenarios:
        cmd += ["--scenarios", *scenarios]
    subprocess.run(cmd, check=True, env=env)
    with open(json_path) as f:
        return json.load(f)


def run_regime(regime: str, cache_dir: str) -> dict:
    """Pinned regimes run all scenarios in one subprocess. Untouched glibc adapts its thresholds to
    what ran before, so ``default`` gets one fresh subprocess per scenario -- like a real training run."""
    if REGIMES[regime]:
        return run_regime_subprocess(regime, os.path.join(cache_dir, f"roofline_{regime}.json"))
    merged = None
    for i, key in enumerate(SCENARIOS):
        part = run_regime_subprocess(regime, os.path.join(cache_dir, f"roofline_{regime}_{i}.json"), [key])
        if merged is None:
            merged = part
        else:
            merged["scenarios"].update(part["scenarios"])
    return merged


# --------------------------------------------------------------------------- analysis helpers


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def cache_resident_bytes(sweep: dict, cache_bw: float | None, frac: float = 0.9, arrays: int = 3) -> float:
    """Largest working set (bytes) that the a+b bandwidth sweep shows still running at cache speed.

    ``sweep`` maps per-array MiB -> GB/s for ``c = a + b`` (three arrays touched). A size counts as cache-
    resident while its bandwidth is >= ``frac * cache_bw``; the working set is ``arrays x`` that size.
    Returns 0 when there is no cache measurement.
    """
    if not cache_bw or not sweep:
        return 0.0
    ok = [int(mb) for mb, g in sweep.items() if g * 1e9 >= frac * cache_bw]
    return float(arrays * max(ok) * 2**20) if ok else 0.0


def op_rows(ops: list[dict], peak: float, bw: float, cache_bw: float | None = None,
            cache_bytes: float = 0.0) -> list[dict]:
    """Add intensity, achieved, roof, % of roof and bound to measured op dicts.

    The memory ceiling for an op is ``cache_bw`` when its compulsory traffic fits in ``cache_bytes`` (the
    working set stays cache-resident, as the bandwidth sweep measures), else DRAM ``bw``. Classifying a
    cache-resident op against the DRAM slope overstates "% of roof" and calls ops memory-bound that are
    compute-bound at the bandwidth they actually see. Without ``cache_bw`` this is the plain DRAM roofline.
    """
    rows = []
    for o in ops:
        in_cache = bool(cache_bw) and o["bytes"] <= cache_bytes
        mem_bw = cache_bw if in_cache else bw
        ridge = fl.ridge_point(peak, mem_bw)
        i = o["flops"] / o["bytes"]
        ach = o["flops"] / o["seconds"]
        roof = fl.attainable(i, peak, mem_bw)
        rows.append(
            {
                **o,
                "intensity": i,
                "achieved": ach,
                "roof": roof,
                "pct_roof": 100 * ach / roof,
                "pct_dram_roof": 100 * ach / fl.attainable(i, peak, bw),
                "mem_roof": "cache" if in_cache else "DRAM",
                "bound": "memory" if i < ridge else "compute",
                "roof_time": o["flops"] / roof,
            }
        )
    return rows


def forward_summary(rows: list[dict]) -> dict:
    """Per-forward totals (ops x count): FLOPs, bytes, measured time, roofline lower bound."""
    t = sum(r["seconds"] * r["count"] for r in rows)
    return {
        "flops": sum(r["flops"] * r["count"] for r in rows),
        "bytes": sum(r["bytes"] * r["count"] for r in rows),
        "seconds": t,
        "roof_seconds": sum(r["roof_time"] * r["count"] for r in rows),
        "time_share": {r["name"]: r["seconds"] * r["count"] / t for r in rows},
    }


def _fmt_flops(x: float) -> str:
    return f"{x / 1e9:.2f} G" if x >= 1e9 else f"{x / 1e6:.1f} M"


def _fmt_bytes(x: float) -> str:
    return f"{x / 2**20:.2f} MiB"


# --------------------------------------------------------------------------- report


def write_report(res: dict, out_md: str = OUT_MD) -> str:
    """Render ``ROOFLINE.md`` from the results dict produced by :func:`main`."""
    peak = res["peak"]["flops"]
    bw = res["bandwidth"]["bytes_per_s"]
    cbw = res["bandwidth"].get("cache_bytes_per_s")
    ridge = fl.ridge_point(peak, bw)
    cbytes = cache_resident_bytes(res["bandwidth"].get("sweep", {}), cbw)
    cridge = fl.ridge_point(peak, cbw) if cbw else ridge
    reuse = res["regimes"]["reuse"]["scenarios"]
    fresh = res["regimes"]["fresh"]["scenarios"]
    default = res["regimes"]["default"]["scenarios"]
    budget = 7 * 60

    rows = {k: op_rows(v["ops"], peak, bw, cbw, cbytes) for k, v in reuse.items()}
    frows = {k: op_rows(v["ops"], peak, bw, cbw, cbytes) for k, v in fresh.items()}
    summ = {k: forward_summary(v) for k, v in rows.items()}
    fsumm = {k: forward_summary(v) for k, v in frows.items()}
    base = rows["base"]
    fbase = {r["name"]: r for r in frows["base"]}

    L = []
    w = L.append
    w("# Roofline: why TinyGPT trains as fast as it does on this machine\n")
    w(
        "_Generated by `python -m roofline.run` (Agent G). All numbers below are measured on this machine "
        "unless marked as formulas; re-running overwrites this file._\n"
    )
    w("## 1. Machine ceilings\n")
    w(f"- CPU: {res['machine']['cpu']}, {res['machine']['threads']} threads (`torch.set_num_threads(4)`), "
      f"fp32, torch {res['machine']['torch']}.")
    per = ", ".join(f"n={n}: {g:.0f}" for n, g in res["peak"]["per_size"].items())
    w(f"- **Peak compute: {peak / 1e9:.0f} GFLOP/s** = best large square `torch.mm` ({per} GFLOP/s; "
      f"best of 5 repeats after warmup). From first principles a core retires 2 AVX-512 FMAs x 16 fp32 lanes x "
      f"2 FLOPs = 64 FLOPs/cycle, i.e. 4 x 2.1 GHz x 64 = 538 GFLOP/s at the nominal clock; the measured "
      f"peak implies an effective {peak / (4 * 64) / 1e9:.2f} GHz (turbo), so MKL's big GEMM runs at the "
      f"hardware limit.")
    bk = res["bandwidth"]["per_kind"]
    w(f"- **Memory bandwidth: {bw / 1e9:.1f} GB/s** = best of STREAM-style copy ({bk['copy']:.1f} GB/s) and "
      f"a+b ({bk['add']:.1f} GB/s) over {res['bandwidth']['n_bytes'] // 2**20} MiB fp32 arrays "
      f"(traffic counted as bytes read + bytes written).")
    sweep = res["bandwidth"]["sweep"]
    w("- Bandwidth vs array size (a+b, per-array MiB -> GB/s): "
      + ", ".join(f"{mb}: {g:.0f}" for mb, g in sweep.items())
      + ". 1 MiB arrays (3 MiB working set) live in the four 2 MiB L2s. The OS reports a 260 MB L3, but this "
      "VM effectively gets a slice of it: 4-16 MiB arrays -- the size of our activations -- run at "
      f"{cbw / 1e9:.0f} GB/s ({cbw / bw:.1f}x DRAM; dashed line in the plots), and bandwidth falls to the "
      "DRAM level once the working set passes ~200 MiB." if cbw else "")
    w(f"- **Ridge point = {peak / 1e9:.0f} / {bw / 1e9:.1f} = {ridge:.1f} FLOPs per byte.** An op needs more than "
      f"~{ridge:.0f} FLOPs per byte moved to be compute-bound; below that, DRAM bandwidth caps it." + (
      f" **For working sets up to {cbytes / 2**20:.0f} MiB** (the sweep's cache-speed range, x3 arrays for a+b) "
      f"the memory ceiling is the cache bandwidth, so the ridge is **{peak / 1e9:.0f} / {cbw / 1e9:.0f} = "
      f"{cridge:.1f} FLOPs per byte**. Every per-op working set of the base config is in that range, so the op "
      f"table below classifies ops against the cache roof (DRAM-roof percentages are kept in their own "
      f"column).\n" if cbytes else "\n"))

    w("## 2. FLOPs and bytes per op (formulas)\n")
    w("`B=32, T=128, C=n_embd=128, H=4, hd=C/H=32, L=4, M=B*T=4096`, fp32 (s=4 bytes). FLOPs count a "
      "multiply-add as 2 and each elementwise op or `exp`/`erf` as 1. Bytes are the *compulsory* traffic of a "
      "perfectly fused kernel: inputs + weights read once, output written once. Full derivations are the "
      "docstrings of [`roofline/flops.py`](flops.py).\n")
    w("| op | FLOPs (one call) | bytes ÷ 4 = fp32 elements moved (one call) | calls per forward |")
    w("|---|---|---|---|")
    formulas = [
        ("LayerNorm", "7·M·C", "2·M·C + 2·C", "2L+1 = 9"),
        ("QKV proj", "2·M·C·3C + 3·M·C", "M·C + 3C² + 3C + 3·M·C", "L"),
        ("QK^T (× 1/√hd)", "2·B·T²·C + B·H·T²", "2·B·T·C + B·H·T²", "L"),
        ("softmax (+ causal mask)", "5·B·H·T²", "2·B·H·T² (+ T²/4 for the bool mask)", "L"),
        ("attn@V", "2·B·T²·C", "B·H·T² + 2·B·T·C", "L"),
        ("out proj", "2·M·C² + M·C", "2·M·C + C² + C", "L"),
        ("MLP up", "8·M·C² + 4·M·C", "5·M·C + 4C² + 4C", "L"),
        ("GELU (exact, erf)", "5·4·M·C", "2·4·M·C", "L"),
        ("MLP down", "8·M·C² + M·C", "5·M·C + 4C² + C", "L"),
        ("LM head (tied, no bias)", "2·M·C·V", "M·C + C·V + M·V", "1"),
        ("cross-entropy (training extra)", "5·M·V", "M·V (+ 2·M for int64 targets)", "1"),
    ]
    for f in formulas:
        w(f"| {f[0]} | {f[1]} | {f[2]} | {f[3]} |")
    w("\nA Linear layer `[M,k]@[k,n]` has intensity `2Mkn / 4(Mk + kn + Mn)`, which for M ≫ k, n tends to "
      "`kn / 2(k+n)` -- **set by the layer width, not the batch**. Elementwise ops do O(1) FLOPs per 8 bytes, "
      "so their intensity is < 1.3 FLOP/B whatever the shape. Note QK^T / softmax / attn@V compute the full "
      "T×T matrix although the causal mask discards half of it.\n")

    def op_table(rs, frs=None, title=None):
        if title:
            w(title)
        hdr = ("| op | shape | FLOPs | bytes | FLOP/B | achieved GFLOP/s | memory roof | % of roof | % of DRAM roof "
               "| bound | µs / call | × per fwd | share of fwd time |")
        sep = "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
        if frs:
            hdr += " fresh-pages µs (faults/call) |"
            sep += "---|"
        w(hdr)
        w(sep)
        sm = forward_summary(rs)
        for r in rs:
            line = (
                f"| {r['name']} | `{r['shape']}` | {_fmt_flops(r['flops'])} | {_fmt_bytes(r['bytes'])} | "
                f"{r['intensity']:.2f} | {r['achieved'] / 1e9:.1f} | {r['mem_roof']} | {r['pct_roof']:.0f}% | "
                f"{r['pct_dram_roof']:.0f}% | {r['bound']} | "
                f"{r['seconds'] * 1e6:.0f} | {r['count']} | {100 * sm['time_share'][r['name']]:.1f}% |"
            )
            if frs:
                fr = frs[r["name"]]
                line += f" {fr['seconds'] * 1e6:.0f} ({fr['faults']:.0f}) |"
            w(line)
        w("")

    w("## 3. Measured ops at the SPEC config (vocab 512)\n")
    w("Each op is a standalone torch call at the exact forward shape (`F.linear`, `@`, `masked_fill`+`softmax`, "
      "`F.gelu`, `F.layer_norm`, `F.cross_entropy`), forward only, no grad, best of 5 repeats after warmup. "
      "**% of roof** = achieved / min(peak, BW × intensity), where BW is the bandwidth of the memory level the "
      f"op's working set lives in (**memory roof**: cache for ≤ {cbytes / 2**20:.0f} MiB, else DRAM). **bound** "
      f"compares the intensity with that level's ridge (cache {cridge:.1f}, DRAM {ridge:.1f}). Main columns use the *reuse* allocator regime (no page faults); the last column is "
      "the same op when every output buffer is fresh memory from the OS (see §4, takeaway 2).\n")
    op_table(base, fbase)
    w("Vocabulary only changes the last two ops:\n")
    w("| vocab | LM head FLOP/B | LM head GFLOP/s | LM head µs | cross-ent µs | LM head + CE share of fwd |")
    w("|---|---|---|---|---|---|")
    for key, V in (("V=96", 96), ("base", 512), ("V=2048", 2048)):
        rr = {r["name"]: r for r in rows[key]}
        lm, ce = rr[f"LM head V={V}"], rr[f"cross-ent V={V}"]
        share = summ[key]["time_share"][lm["name"]] + summ[key]["time_share"][ce["name"]]
        w(f"| {V} | {lm['intensity']:.1f} | {lm['achieved'] / 1e9:.0f} | {lm['seconds'] * 1e6:.0f} | "
          f"{ce['seconds'] * 1e6:.0f} | {100 * share:.1f}% |")
    w("")
    w("![roofline](roofline.png)\n")
    w("Filled markers: reuse regime. Hollow markers joined by a line: the same op paying page faults for fresh "
      "output buffers. Dashed line: bandwidth when the working set stays in cache.\n")

    # ---------------------------------------------------------------- forward / step summary
    sb, fsb = summ["base"], fsumm["base"]
    st_def, st_reu, st_fr = default["base"]["step"], reuse["base"]["step"], fresh["base"]["step"]
    key_step = "train step (fwd+bwd+AdamW)"
    train_flops = 3 * sb["flops"]
    w("### Whole forward and training step (base config, vocab 512)\n")
    w("| quantity | value |")
    w("|---|---|")
    w(f"| forward FLOPs (formula, incl. CE) | {sb['flops'] / 1e9:.2f} GFLOP |")
    w(f"| forward compulsory bytes (formula) | {sb['bytes'] / 2**20:.0f} MiB |")
    w(f"| whole-forward intensity | {sb['flops'] / sb['bytes']:.1f} FLOP/B |")
    w(f"| roofline lower bound for the forward (Σ max(FLOPs/peak, bytes/BW)) | {sb['roof_seconds'] * 1e3:.1f} ms |")
    w(f"| Σ measured op times (reuse / fresh pages) | {sb['seconds'] * 1e3:.1f} ms / {fsb['seconds'] * 1e3:.1f} ms |")
    w(f"| replica forward, no grad (default glibc / reuse / fresh) | "
      f"{st_def['forward'] * 1e3:.1f} / {st_reu['forward'] * 1e3:.1f} / {st_fr['forward'] * 1e3:.1f} ms |")
    w(f"| replica fwd+bwd+clip+AdamW step (default glibc / reuse / fresh) | "
      f"{st_def[key_step] * 1e3:.0f} / {st_reu[key_step] * 1e3:.0f} / {st_fr[key_step] * 1e3:.0f} ms |")
    w(f"| training FLOPs per step (≈ 3 × forward: backward does 2 matmuls per forward matmul) | "
      f"{train_flops / 1e9:.1f} GFLOP |")
    mfu = train_flops / st_def[key_step] / peak
    w(f"| model FLOP utilisation (default glibc) | {100 * mfu:.0f}% of peak "
      f"({train_flops / st_def[key_step] / 1e9:.0f} GFLOP/s) |")
    w(f"| steps in the 7-min budget (default glibc) | ≈ {budget / st_def[key_step]:.0f} "
      f"(≈ {budget / st_def[key_step] * 4096 / 1e6:.1f} M tokens) |")
    w(f"| upper bound if every op hit its roof (3 × forward lower bound) | ≈ {budget / (3 * sb['roof_seconds']):.0f} steps |")
    w("\nThe replica is a functional, plain-torch copy of the SPEC architecture (`roofline/measure.py:"
      "replica_forward`), not `tinygpt.model`, so the integrator's real numbers may differ by a few percent.\n")

    # ---------------------------------------------------------------- takeaways
    w("## 4. Takeaways\n")
    by = {r["name"]: r for r in base}
    lin = [r for r in base if r["kind"] == "matmul"]
    att = [r for r in base if r["kind"] == "attention"]
    ew = [r for r in base if r["kind"] == "elementwise"]

    def share(group, s=sb):
        return sum(s["time_share"][r["name"]] for r in group)

    def fshare(group, s=sb):
        return sum(r["flops"] * r["count"] for r in group) / s["flops"]

    lin_eff = sum(r["flops"] * r["count"] for r in lin) / sum(r["seconds"] * r["count"] for r in lin)
    mem_ops = [r["name"] for r in base if r["bound"] == "memory"]
    pr = lambda n: f"{by[n]['pct_roof']:.0f}%"  # noqa: E731
    w(f"1. **The Linear layers are compute-bound and fast; the elementwise ops are memory-bound and eat a "
      f"disproportionate share of the time.** Linear matmuls (QKV, out proj, MLP up/down, LM head) do "
      f"{100 * fshare(lin):.0f}% of the forward FLOPs at 30-50 FLOP/B (right of the {ridge:.1f} ridge) and run "
      f"at {lin_eff / 1e9:.0f} GFLOP/s ≈ {100 * lin_eff / peak:.0f}% of peak, taking {100 * share(lin):.0f}% of "
      f"the forward time. The elementwise/reduction ops (LayerNorm, softmax, GELU, cross-entropy) do "
      f"{100 * fshare(ew):.1f}% of the FLOPs but take {100 * share(ew):.0f}% of the time. At 0.6-1.3 FLOP/B "
      f"they are capped at ~{bw * 1.3 / 1e9:.0f} GFLOP/s from DRAM (~{cbw * 1.3 / 1e9:.0f} from cache). "
      f"Their 4-16 MiB working sets sit in cache, so the cache roof is the right ceiling: LayerNorm and GELU "
      f"reach {pr('LayerNorm')} / {pr('GELU')} of it (against the DRAM roof they would look like "
      f"{by['LayerNorm']['pct_dram_roof']:.0f}% / {by['GELU']['pct_dram_roof']:.0f}%, which overstates how "
      f"close they are to the hardware limit). **softmax is the single most expensive "
      f"op in the forward** ({100 * sb['time_share']['softmax']:.0f}% of the time for "
      f"{100 * by['softmax']['flops'] * by['softmax']['count'] / sb['flops']:.1f}% of the FLOPs, "
      f"{pr('softmax')} of its roof). Cross-entropy reaches {pr('cross-ent V=512')}. Both are slow because eager "
      f"PyTorch makes several passes over the T×T or M×V tensor (masked_fill, then max, exp-sum and normalise "
      f"inside softmax), and an `exp` costs ~10-20 instructions, not 1. Memory-bound ops (intensity below the "
      f"ridge of their memory level): **{', '.join(mem_ops)}**. The per-head attention matmuls "
      f"(hd = 32 ⇒ {by['QK^T']['intensity']:.1f} FLOP/B) are below the DRAM ridge ({ridge:.1f}) but above the "
      f"cache ridge ({cridge:.1f}); their 12 MiB working sets stay in cache, so they are "
      f"**{by['QK^T']['bound']}-bound** and run at {pr('QK^T')} (QK^T) / {pr('attn@V')} (attn@V) of the "
      f"compute roof. They take {100 * share(att):.0f}% of the time.\n")
    fr_pen = fsb["seconds"] / sb["seconds"]
    rB_def, rB_reu = default["batch=128"]["step"][key_step], reuse["batch=128"]["step"][key_step]
    w(f"2. **Page faults are a hidden, shape-dependent tax; the rest of the gap is only partly attributed.** "
      f"Every `[B,H,T,T]` fp32 tensor is {32 * 4 * 128 * 128 * 4 / 2**20:.0f} MiB. If glibc has returned such a "
      f"buffer to the OS, the next allocation is a fresh mmap and the kernel must zero-fill and map 2048 new "
      f"4 KiB pages. QK^T then costs {fbase['QK^T']['seconds'] * 1e6:.0f} µs instead of "
      f"{by['QK^T']['seconds'] * 1e6:.0f} µs ({fbase['QK^T']['faults']:.0f} page faults per call) and softmax "
      f"{fbase['softmax']['seconds'] * 1e6:.0f} µs instead of {by['softmax']['seconds'] * 1e6:.0f} µs. Summed "
      f"over the forward, fresh pages make it {fr_pen:.1f}x slower ({fsb['seconds'] * 1e3:.0f} vs "
      f"{sb['seconds'] * 1e3:.0f} ms). Untouched glibc raises its thresholds after the first frees, so at "
      f"B=32 the real training step pays little: {st_def[key_step] * 1e3:.0f} ms vs {st_reu[key_step] * 1e3:.0f} ms "
      f"with buffers pinned in the heap ({st_def[key_step] / st_reu[key_step]:.2f}x). At B=128 it pays "
      f"{rB_def / rB_reu:.2f}x ({rB_def * 1e3:.0f} vs {rB_reu * 1e3:.0f} ms), because the 32 MiB score tensors "
      f"hit glibc's 32 MiB ceiling for that adaptive threshold, so they stay mmapped and fault in every call. (Pinning `MALLOC_MMAP_THRESHOLD_` / "
      f"`MALLOC_TRIM_THRESHOLD_` in the environment is a free fix.) Overall the base step runs at "
      f"{100 * mfu:.0f}% of peak, and the roofline lower bound for fwd+bwd is ~{3 * sb['roof_seconds'] * 1e3:.0f} ms "
      f"vs {st_def[key_step] * 1e3:.0f} ms measured. What is measured about that gap: of the summed forward op "
      f"time, Linear layers take {100 * share(lin):.0f}%, attention matmuls {100 * share(att):.0f}% and "
      f"elementwise ops {100 * share(ew):.0f}%; and the replica forward ({st_reu['forward'] * 1e3:.1f} ms, reuse) "
      f"is {100 * (st_reu['forward'] / sb['seconds'] - 1):.0f}% longer than the sum of its profiled ops "
      f"({sb['seconds'] * 1e3:.1f} ms) -- time in unprofiled work (residual adds, transposes/contiguous copies, "
      f"embedding gather) and per-call dispatch. The backward pass and AdamW were not profiled per op, so how "
      f"the remaining ~{100 * (1 - st_reu['forward'] / st_reu[key_step]):.0f}% of the step splits is not "
      f"measured here. The matmuls still take the largest share of the forward; elementwise ops take a share "
      f"far above their FLOPs. All four runs share the same shapes except V, so this affects them about "
      f"equally.\n")

    # scenario takeaway (recomputed, not guessed)
    def scen_line(key):
        rs = {r["name"]: r for r in rows[key]}
        s = summ[key]
        stp = default[key]["step"][key_step]
        return rs, s, stp

    rC, sC, stC = scen_line("n_embd=512")
    rB, sB, stB = scen_line("batch=128")
    rb = {r["name"]: r for r in base}
    w("3. **What changes with a wider model or a bigger batch (recomputed from the formulas and re-measured, "
      "not guessed):**\n")
    w("   | quantity | base (C=128, B=32) | n_embd=512 | batch=128 |")
    w("   |---|---|---|---|")
    for name in ("QKV proj", "MLP up", "QK^T", "attn@V", "softmax", "GELU", "LM head V=512"):
        w(f"   | {name} intensity (FLOP/B) | {rb[name]['intensity']:.1f} | {rC[name]['intensity']:.1f} | "
          f"{rB[name]['intensity']:.1f} |")
    for name in ("QKV proj", "MLP up", "QK^T", "softmax"):
        w(f"   | {name} achieved GFLOP/s | {rb[name]['achieved'] / 1e9:.0f} | {rC[name]['achieved'] / 1e9:.0f} | "
          f"{rB[name]['achieved'] / 1e9:.0f} |")
    w(f"   | forward FLOPs | {sb['flops'] / 1e9:.2f} G | {sC['flops'] / 1e9:.2f} G "
      f"({sC['flops'] / sb['flops']:.1f}x) | {sB['flops'] / 1e9:.2f} G ({sB['flops'] / sb['flops']:.1f}x) |")
    w(f"   | memory-bound ops | {sum(r['bound'] == 'memory' for r in base)}/{len(base)} | "
      f"{sum(r['bound'] == 'memory' for r in rows['n_embd=512'])}/{len(base)} | "
      f"{sum(r['bound'] == 'memory' for r in rows['batch=128'])}/{len(base)} |")
    w(f"   | elementwise share of forward time | {100 * share(ew):.0f}% | "
      f"{100 * sum(sC['time_share'][r['name']] for r in rows['n_embd=512'] if r['kind'] == 'elementwise'):.0f}% | "
      f"{100 * sum(sB['time_share'][r['name']] for r in rows['batch=128'] if r['kind'] == 'elementwise'):.0f}% |")
    w(f"   | train step (default glibc) | {st_def[key_step] * 1e3:.0f} ms | {stC * 1e3:.0f} ms | {stB * 1e3:.0f} ms |")
    w(f"   | training throughput | {4096 / st_def[key_step] / 1e3:.1f} k tok/s | "
      f"{4096 / stC / 1e3:.1f} k tok/s | {4 * 4096 / stB / 1e3:.1f} k tok/s |")
    w(f"   | model FLOP utilisation | {100 * mfu:.0f}% | {100 * 3 * sC['flops'] / stC / peak:.0f}% | "
      f"{100 * 3 * sB['flops'] / stB / peak:.0f}% |")
    w(f"   | steps in 7 min | {budget / st_def[key_step]:.0f} | {budget / stC:.0f} | {budget / stB:.0f} |\n")
    w("   ![scenarios](roofline_scenarios.png)\n")
    w(f"   - **n_embd = 512**: Linear intensity scales with width (≈ kn/2(k+n) → "
      f"{rC['QKV proj']['intensity']:.0f}-{rC['MLP up']['intensity']:.0f} FLOP/B), and hd = 128 lifts the "
      f"attention matmuls to {rC['QK^T']['intensity']:.0f} FLOP/B, past the ridge -- the model becomes more "
      f"compute-bound and uses the CPU better (MFU {100 * mfu:.0f}% → {100 * 3 * sC['flops'] / stC / peak:.0f}%). "
      f"But the Linear FLOPs grow 16x (12·L·C² per token), so a step costs {stC / st_def[key_step]:.1f}x more "
      f"and the 7-minute budget buys only ~{budget / stC:.0f} steps: a 16x bigger model that sees "
      f"{st_def[key_step] / stC:.2f}x as many tokens.\n")
    rB_reuse = reuse["batch=128"]["step"][key_step]
    w(f"   - **batch = 128**: intensities barely move. The weights are tiny next to the activations, so a "
      f"Linear layer's intensity is set by its width, and the attention and elementwise intensities do not "
      f"depend on B at all. On the roofline the points stay put. Each step does {sB['flops'] / sb['flops']:.1f}x "
      f"the work. With reused buffers it takes {rB_reuse / st_reu[key_step]:.1f}x the time "
      f"({4 * st_reu[key_step] / rB_reuse:.2f}x the tokens/s: larger GEMMs amortise per-call overhead). With "
      f"untouched glibc it takes {stB / st_def[key_step]:.1f}x ({st_def[key_step] * 4 / stB:.2f}x the tokens/s), "
      f"because the 32 MiB score tensors are mmapped and page-faulted every step "
      f"({stB / rB_reuse:.2f}x slower than with reused buffers). Bigger batches also mean fewer, larger "
      f"optimizer steps in the same 7 minutes ({budget / stB:.0f} vs {budget / st_def[key_step]:.0f}).\n")
    lm96 = {r['name']: r for r in rows['V=96']}
    lm2k = {r['name']: r for r in rows['V=2048']}
    w(f"   - **vocab** (the experiment's variable): LM head intensity goes {lm96['LM head V=96']['intensity']:.0f} → "
      f"{rb['LM head V=512']['intensity']:.0f} → {lm2k['LM head V=2048']['intensity']:.0f} FLOP/B (≈ CV/2(C+V), "
      f"saturating at C/2 = 64) so it stays compute-bound, but its FLOPs and the memory-bound cross-entropy "
      f"scale linearly in V. Measured train step: char {default['V=96']['step'][key_step] * 1e3:.0f} ms, "
      f"bpe512 {st_def[key_step] * 1e3:.0f} ms, bpe2048 {default['V=2048']['step'][key_step] * 1e3:.0f} ms. "
      f"Since a BPE token covers several characters, bpe tokenizers see more text per step for a modest "
      f"per-step cost -- the trade-off the 7-minute experiment measures.\n")

    w("## 5. Reproduce\n")
    w("```bash\npython -m roofline.run\n```\n")
    w("Takes ~3-4 minutes. Raw numbers go to `roofline/results.json`. Each allocator regime runs in its own "
      "subprocess so glibc's dynamic thresholds cannot leak between measurements. This is a shared cloud VM: "
      "repeat runs vary by roughly ±10-30% (Linear layers measured anywhere from ~60% to ~85% of peak across runs); the conclusions above do not depend on that.\n")
    text = "\n".join(L)
    with open(out_md, "w") as f:
        f.write(text)
    return out_md


# --------------------------------------------------------------------------- plotting from results


def points_for(rows: list[dict], frows: list[dict] | None = None) -> list[dict]:
    fr = {r["name"]: r for r in frows or []}
    pts = []
    for r in rows:
        p = dict(name=r["name"], intensity=r["intensity"], gflops=r["achieved"] / 1e9, kind=r["kind"])
        if r["name"] in fr:
            alt = fr[r["name"]]["achieved"] / 1e9
            if abs(np.log10(alt / p["gflops"])) > 0.05:  # only draw a visible difference
                p["gflops_alt"] = alt
        pts.append(p)
    return pts


def make_plots(res: dict) -> tuple[str, str]:
    peak = res["peak"]["flops"]
    bw = res["bandwidth"]["bytes_per_s"]
    cbw = res["bandwidth"].get("cache_bytes_per_s")
    reuse = res["regimes"]["reuse"]["scenarios"]
    fresh = res["regimes"]["fresh"]["scenarios"]
    cbytes = cache_resident_bytes(res["bandwidth"].get("sweep", {}), cbw)
    rows = {k: op_rows(v["ops"], peak, bw, cbw, cbytes) for k, v in reuse.items()}
    frows = {k: op_rows(v["ops"], peak, bw, cbw, cbytes) for k, v in fresh.items()}
    # Main plot: base ops (V=512) + LM head / CE for the other vocabularies.
    main_rows = list(rows["base"])
    main_frows = list(frows["base"])
    for k in ("V=96", "V=2048"):
        main_rows += [r for r in rows[k] if r["name"].startswith(("LM head", "cross-ent"))]
        main_frows += [r for r in frows[k] if r["name"].startswith(("LM head", "cross-ent"))]
    _plot_points(
        peak,
        bw,
        points_for(main_rows, main_frows),
        OUT_PNG,
        cache_bandwidth=cbw,
        title="TinyGPT forward ops on a 4-thread Xeon (fp32, B=32, T=128, n_embd=128)",
        alt_label="same op, fresh pages (page faults)",
    )
    panels = [
        ("base: n_embd=128, B=32", points_for(rows["base"])),
        ("n_embd=512, B=32", points_for(rows["n_embd=512"])),
        ("n_embd=128, B=128", points_for(rows["batch=128"])),
    ]
    plot_panels(peak, bw, panels, OUT_SCEN_PNG, cache_bandwidth=cbw)
    return OUT_PNG, OUT_SCEN_PNG


# --------------------------------------------------------------------------- main


def measure_ceilings(repeats: int = 5) -> dict:
    ms.setup()
    ms.best_time(lambda: torch.mm(torch.randn(2048, 2048), torch.randn(2048, 2048)), repeats=3, min_time=0.5)
    peak, per = ms.measure_peak_flops(repeats=repeats)
    n_bytes = 256 * 2**20
    bwv, per_kind = ms.measure_bandwidth(n_bytes, repeats)
    sweep = ms.bandwidth_sweep()
    # Op working sets are 4-16 MiB, so the relevant cache ceiling is the 4-16 MiB plateau (L3 slice).
    cache = max(g for mb, g in sweep.items() if 4 <= mb <= 16) * 1e9
    return {
        "peak": {"flops": peak, "per_size": per},
        "bandwidth": {
            "bytes_per_s": bwv,
            "per_kind": per_kind,
            "n_bytes": n_bytes,
            "sweep": sweep,
            "cache_bytes_per_s": cache,
        },
    }


def main(argv=None) -> None:
    """Measure ceilings, profile ops, write roofline.png and ROOFLINE.md."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worker", choices=list(REGIMES), help=argparse.SUPPRESS)
    ap.add_argument("--json", help=argparse.SUPPRESS)
    ap.add_argument("--scenarios", nargs="*", help=argparse.SUPPRESS)
    ap.add_argument("--from-json", help="re-render plots and ROOFLINE.md from an existing results.json")
    args = ap.parse_args(argv)

    if args.worker:
        out = run_worker(args.worker, args.scenarios)
        with open(args.json, "w") as f:
            json.dump(out, f)
        return

    if args.from_json:
        with open(args.from_json) as f:
            res = json.load(f)
    else:
        res = {"machine": {"cpu": cpu_name(), "threads": 4, "torch": torch.__version__}}
        print("measuring ceilings ...", flush=True)
        res.update(measure_ceilings())
        print(
            f"  peak {res['peak']['flops'] / 1e9:.0f} GFLOP/s, bandwidth {res['bandwidth']['bytes_per_s'] / 1e9:.1f} GB/s",
            flush=True,
        )
        res["regimes"] = {}
        cache_dir = os.path.join(os.path.dirname(HERE), ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        for regime in REGIMES:
            print(f"profiling ops, allocator regime '{regime}' ...", flush=True)
            res["regimes"][regime] = run_regime(regime, cache_dir)
        with open(OUT_JSON, "w") as f:
            json.dump(res, f, indent=1)
    # JSON turns int keys into strings; normalise for printing.
    res["peak"]["per_size"] = {int(k): v for k, v in res["peak"]["per_size"].items()}
    res["bandwidth"]["sweep"] = {int(k): v for k, v in res["bandwidth"]["sweep"].items()}
    make_plots(res)
    write_report(res)
    ridge = fl.ridge_point(res["peak"]["flops"], res["bandwidth"]["bytes_per_s"])
    print(f"ridge point {ridge:.1f} FLOP/B; wrote {OUT_PNG}, {OUT_SCEN_PNG}, {OUT_MD}, {OUT_JSON}")


if __name__ == "__main__":
    main()
