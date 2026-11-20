"""Tests for roofline. Owner: Agent G. Fast (< ~15 s), no network, no data files, no tinygpt.model."""

import math
import os

import pytest
import torch

from roofline import flops as fl
from roofline import measure as ms
from roofline import plot as rplot
from roofline import run as rrun

B, T, C, H, L, V = 32, 128, 128, 4, 4, 512
M = B * T


# --------------------------------------------------------------------------- formulas


@pytest.mark.parametrize("m,k,n", [(1, 1, 1), (2, 3, 4), (4096, 128, 384), (7, 5, 3)])
def test_matmul_flops_is_2mkn(m, k, n):
    assert fl.matmul_flops(m, k, n) == 2 * m * k * n
    assert fl.matmul_flops(m, k, n, bias=True) == 2 * m * k * n + m * n
    assert fl.matmul_flops(m, k, n, batch=5) == 5 * 2 * m * k * n


def test_matmul_flops_counts_multiply_adds_by_brute_force():
    """Count the multiply-adds of a naive triple loop: 1 mul + 1 add per (i, j, p) = 2mkn."""
    m, k, n = 3, 4, 5
    ops = 0
    for _i in range(m):
        for _j in range(n):
            for _p in range(k):
                ops += 2  # acc += a * b
    assert ops == fl.matmul_flops(m, k, n)


def test_matmul_bytes_reads_inputs_and_writes_output_once():
    assert fl.matmul_bytes(2, 3, 4) == 4 * (2 * 3 + 3 * 4 + 2 * 4)
    assert fl.matmul_bytes(2, 3, 4, bias=True) == 4 * (6 + 12 + 8 + 4)
    # Batched: right operand per item unless shared (a weight).
    assert fl.matmul_bytes(2, 3, 4, batch=10) == 4 * 10 * (6 + 12 + 8)
    assert fl.matmul_bytes(2, 3, 4, batch=10, shared_rhs=True) == 4 * (60 + 12 + 80)


def test_linear_layer_formulas():
    qkv = fl.qkv_cost(M, C)
    assert qkv.flops == 2 * M * C * 3 * C + 3 * M * C
    assert qkv.bytes == 4 * (M * C + 3 * C * C + 3 * C + 3 * M * C)
    assert fl.out_proj_cost(M, C).flops == 2 * M * C * C + M * C
    assert fl.mlp_up_cost(M, C).flops == 8 * M * C * C + 4 * M * C
    assert fl.mlp_down_cost(M, C).flops == 8 * M * C * C + M * C
    lm = fl.lm_head_cost(M, C, V)
    assert lm.flops == 2 * M * C * V
    assert lm.bytes == 4 * (M * C + C * V + M * V)


def test_attention_formulas():
    hd = C // H
    s = fl.attn_scores_cost(B, T, C, H)
    assert s.flops == 2 * B * H * T * T * hd + B * H * T * T
    assert s.flops == 2 * B * T * T * C + B * H * T * T  # same thing, via H * hd = C
    assert s.bytes == 4 * (2 * B * T * C + B * H * T * T)
    a = fl.attn_values_cost(B, T, C, H)
    assert a.flops == 2 * B * T * T * C
    assert a.bytes == 4 * (B * H * T * T + 2 * B * T * C)
    sm = fl.softmax_cost(B, T, H)
    assert sm.flops == 5 * B * H * T * T
    assert sm.bytes == 8 * B * H * T * T + T * T


def test_elementwise_formulas_and_intensity():
    ln = fl.layernorm_cost(M, C)
    assert ln.flops == 7 * M * C and ln.bytes == 4 * (2 * M * C + 2 * C)
    g = fl.gelu_cost(M, C)
    assert g.flops == 20 * M * C and g.bytes == 32 * M * C
    assert g.intensity == pytest.approx(5 / 8)
    ce = fl.cross_entropy_cost(M, V)
    assert ce.flops == 5 * M * V
    # Elementwise ops can never be compute-bound on any realistic machine.
    for op in (ln, g, ce, fl.softmax_cost(B, T, H)):
        assert op.intensity < 1.5


def test_linear_intensity_limit_is_independent_of_batch():
    """For M >> k, n the intensity of [M,k]@[k,n] tends to kn / (2(k+n)) FLOP/B (fp32)."""
    k, n = 128, 384
    limit = k * n / (2 * (k + n))
    i_small = fl.linear_cost("x", 4096, k, n, bias=False).intensity
    i_big = fl.linear_cost("x", 4096 * 1000, k, n, bias=False).intensity
    assert i_small < i_big < limit
    assert i_big == pytest.approx(limit, rel=1e-3)


def test_forward_counts_and_total_matmul_flops():
    ops = fl.forward_op_costs(B, T, C, H, L, V, include_loss=False)
    by = {o.name: o for o in ops}
    assert by["LayerNorm"].count == 2 * L + 1
    assert all(by[n].count == L for n in ("QKV proj", "QK^T", "softmax", "attn@V", "MLP up", "GELU", "MLP down"))
    assert by[f"LM head V={V}"].count == 1
    # "2 FLOPs per weight per token": Linear weights are 12*L*C^2 (+ C*V tied head).
    linear_mm = sum(2 * M * k * n for k, n in [(C, 3 * C), (C, C), (C, 4 * C), (4 * C, C)]) * L + 2 * M * C * V
    assert linear_mm == 2 * M * (12 * L * C * C + C * V)
    attn_mm = 4 * L * B * T * T * C
    biases_scale = L * (3 * M * C + M * C + 4 * M * C + M * C + B * H * T * T)
    elementwise = 7 * M * C * (2 * L + 1) + L * (5 * B * H * T * T + 20 * M * C)
    assert fl.total_flops(ops) == linear_mm + attn_mm + biases_scale + elementwise
    assert fl.total_bytes(ops) == sum(o.bytes * o.count for o in ops)
    with_loss = fl.forward_op_costs(B, T, C, H, L, V)
    assert fl.total_flops(with_loss) == fl.total_flops(ops) + 5 * M * V


def test_forward_rejects_bad_heads():
    with pytest.raises(ValueError):
        fl.forward_op_costs(C=130, H=4)


def test_roofline_helpers():
    assert fl.ridge_point(600e9, 50e9) == pytest.approx(12.0)
    assert fl.attainable(1.0, 600e9, 50e9) == pytest.approx(50e9)
    assert fl.attainable(100.0, 600e9, 50e9) == pytest.approx(600e9)


def test_wider_model_raises_matmul_intensity_not_elementwise():
    base = {o.name: o for o in fl.forward_op_costs(C=128)}
    wide = {o.name: o for o in fl.forward_op_costs(C=512)}
    for n in ("QKV proj", "MLP up", "MLP down", "QK^T", "attn@V"):
        assert wide[n].intensity > base[n].intensity
    for n in ("GELU", "softmax"):
        assert wide[n].intensity == pytest.approx(base[n].intensity, rel=0.01)


# --------------------------------------------------------------------------- measurement


def test_best_time_uses_minimum_with_fake_clock():
    times = iter([0.0, 1.0] + [0.0, 3.0, 10.0, 12.0, 20.0, 21.5])
    calls = []
    t = ms.best_time(lambda: calls.append(1), repeats=3, warmup=1, min_time=0.5, clock=lambda: next(times))
    assert t == pytest.approx(1.5)  # repeats took 3.0, 2.0, 1.5 s with 1 inner call each
    assert len(calls) == 1 + 1 + 3


def test_op_callables_shapes_match_formulas():
    b, t, c, h, v = 2, 8, 16, 2, 11
    ms.setup()
    fns = ms.op_callables(b, t, c, h, v)
    names = [o.name for o in fl.forward_op_costs(b, t, c, h, 1, v)]
    assert set(names) == set(fns)
    with torch.no_grad():
        assert fns["QKV proj"]().shape == (b * t, 3 * c)
        assert fns["QK^T"]().shape == (b, h, t, t)
        p = fns["softmax"]()
        assert torch.allclose(p.sum(-1), torch.ones(b, h, t))
        assert torch.all(p.triu(1) == 0)  # causal
        assert fns["attn@V"]().shape == (b, h, t, c // h)
        assert fns["MLP down"]().shape == (b * t, c)
        assert fns[f"LM head V={v}"]().shape == (b * t, v)
        assert fns[f"cross-ent V={v}"]().dim() == 0


def test_profile_shapes_tiny():
    res = ms.profile_shapes(2, 8, 16, 2, 1, 11, repeats=1)
    assert len(res) == 11
    for r in res:
        assert r.seconds > 0 and r.achieved > 0 and r.faults >= 0


def test_ceiling_measurements_tiny():
    ms.setup()
    assert ms.matmul_gflops(64, repeats=1) > 0
    gbps, per = ms.measure_bandwidth(1 << 20, repeats=1)
    assert gbps > 0 and set(per) == {"copy", "add"}


def test_replica_forward_is_a_causal_lm():
    b, t, c, h, l, v = 2, 6, 16, 2, 1, 13
    torch.manual_seed(0)
    p = ms._replica_params(c, l, v, t)
    idx = torch.randint(0, v, (b, t))
    loss = ms.replica_forward(p, idx, idx, h, l)
    assert loss.dim() == 0 and math.isfinite(loss.item())
    assert loss.item() == pytest.approx(math.log(v), rel=0.1)  # init N(0, 0.02) -> ~uniform
    loss.backward()
    assert p["0.qkv.w"].grad is not None
    times = ms.time_training_step(b, t, c, h, l, v, repeats=1)
    assert all(x > 0 for x in times.values())


def test_legacy_stub_api_still_works():
    ops = rrun.profile_ops(ms.Shape(vocab_size=11, n_embd=16, n_head=2, n_layer=1), batch_size=2, seq_len=8)
    assert ops and all(o.intensity > 0 and o.achieved_flops > 0 for o in ops)


# --------------------------------------------------------------------------- plot + report


def _fake_results():
    """A results dict shaped like ``roofline/results.json`` with made-up but self-consistent timings."""

    def ops_for(s, slow=1.0):
        out = []
        for o in fl.forward_op_costs(s["B"], s["T"], s["C"], s["H"], s["L"], s["V"]):
            gf = 400e9 if o.kind != "elementwise" else 20e9
            out.append(
                dict(name=o.name, kind=o.kind, shape=o.shape, flops=o.flops, bytes=o.bytes, count=o.count,
                     seconds=slow * o.flops / gf, faults=0.0 if slow == 1 else 1000.0)
            )
        return out

    def regime(slow, with_ops=True):
        sc = {}
        for k in rrun.SCENARIOS:
            s = rrun.shape_of(k)
            e = {"shape": s, "step": {"forward": 0.03 * slow, "forward+backward": 0.09 * slow,
                                      "train step (fwd+bwd+AdamW)": 0.1 * slow * s["B"] / 32 * (s["C"] / 128) ** 2}}
            if with_ops:
                e["ops"] = ops_for(s, slow)
            sc[k] = e
        return {"scenarios": sc}

    return {
        "machine": {"cpu": "fake", "threads": 4, "torch": torch.__version__},
        "peak": {"flops": 600e9, "per_size": {1024: 600.0}},
        "bandwidth": {"bytes_per_s": 50e9, "per_kind": {"copy": 45.0, "add": 50.0}, "n_bytes": 256 * 2**20,
                      "sweep": {1: 280.0, 4: 100.0, 256: 50.0}, "cache_bytes_per_s": 100e9},
        "regimes": {"reuse": regime(1.0), "fresh": regime(2.0), "default": regime(1.2, with_ops=False)},
    }


def test_plot_roofline_writes_png(tmp_path):
    pts = [
        dict(name="mm", intensity=50.0, gflops=500.0, kind="matmul"),
        dict(name="gelu", intensity=0.6, gflops=30.0, kind="elementwise", gflops_alt=8.0),
        dict(name="qk", intensity=10.0, gflops=250.0, kind="attention"),
    ]
    out = rplot.plot_roofline(600e9, 50e9, pts, str(tmp_path / "r.png"), cache_bandwidth=100e9, alt_label="alt")
    assert os.path.getsize(out) > 10_000
    with open(out, "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"


def test_report_and_plots_from_fake_results(tmp_path, monkeypatch):
    res = _fake_results()
    monkeypatch.setattr(rrun, "OUT_PNG", str(tmp_path / "roofline.png"))
    monkeypatch.setattr(rrun, "OUT_SCEN_PNG", str(tmp_path / "scen.png"))
    rrun.make_plots(res)
    assert (tmp_path / "roofline.png").exists() and (tmp_path / "scen.png").exists()
    md = rrun.write_report(res, str(tmp_path / "R.md"))
    text = open(md).read()
    assert "Ridge point = 600 / 50.0 = 12.0" in text
    assert "## 4. Takeaways" in text and "n_embd = 512" in text and "batch = 128" in text
    # Elementwise ops at 20 GFLOP/s are memory-bound; Linear ops are compute-bound.
    rows = {r["name"]: r for r in rrun.op_rows(res["regimes"]["reuse"]["scenarios"]["base"]["ops"], 600e9, 50e9)}
    assert rows["softmax"]["bound"] == "memory" and rows["MLP up"]["bound"] == "compute"
    assert rows["MLP up"]["pct_roof"] == pytest.approx(100 * 400 / 600)


def test_cache_resident_ops_are_classified_against_the_cache_roof():
    """Review fix: an op whose working set fits in cache must be judged against the cache bandwidth.
    With peak 600 GFLOP/s, DRAM 50 GB/s (ridge 12) and cache 100 GB/s (ridge 6), a 12 MiB op at 10 FLOP/B
    is compute-bound; the same intensity with a 200 MiB working set is DRAM/memory-bound."""
    sweep = {1: 280.0, 4: 100.0, 16: 98.0, 64: 70.0, 256: 50.0}
    cbytes = rrun.cache_resident_bytes(sweep, 100e9)
    assert cbytes == 3 * 16 * 2**20  # 16 MiB arrays still at >= 90% of cache speed, a+b touches 3 arrays
    assert rrun.cache_resident_bytes(sweep, None) == 0.0
    small = {"name": "qk", "flops": 10 * 12 * 2**20, "bytes": 12 * 2**20, "seconds": 1e-3, "count": 1}
    big = dict(small, name="big", flops=10 * 200 * 2**20, bytes=200 * 2**20)
    ew = {"name": "gelu", "flops": 0.6 * 16 * 2**20, "bytes": 16 * 2**20, "seconds": 16 * 2**20 / 40e9, "count": 1}
    r = {x["name"]: x for x in rrun.op_rows([small, big, ew], 600e9, 50e9, 100e9, cbytes)}
    assert r["qk"]["mem_roof"] == "cache" and r["qk"]["bound"] == "compute"
    assert r["big"]["mem_roof"] == "DRAM" and r["big"]["bound"] == "memory"
    # 40 GB/s on a cache-resident elementwise op is 80% of the DRAM roof but only 40% of the real roof.
    assert r["gelu"]["pct_dram_roof"] == pytest.approx(80) and r["gelu"]["pct_roof"] == pytest.approx(40)
    # Without a cache measurement the classification is the plain DRAM roofline (old behaviour).
    plain = {x["name"]: x for x in rrun.op_rows([small], 600e9, 50e9)}
    assert plain["qk"]["bound"] == "memory" and plain["qk"]["mem_roof"] == "DRAM"


def test_report_uses_cache_roof_and_does_not_claim_unmeasured_causes(tmp_path):
    res = _fake_results()
    res["bandwidth"]["sweep"] = {1: 280.0, 4: 100.0, 16: 98.0, 256: 50.0}
    text = open(rrun.write_report(res, str(tmp_path / "R.md"))).read()
    assert "| memory roof |" in text and "| cache |" in text
    assert "= 6.0 FLOPs per byte" in text  # cache ridge 600 / 100
    assert "not a shortage of FLOP/s" not in text and "overhead, not FLOPs" not in text
    assert "were not profiled per op" in text
