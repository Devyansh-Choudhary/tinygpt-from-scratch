"""Tests for tinygpt.model. Owner: Agent B. Run with ``-s`` to see the printed numbers."""

import math

import pytest
import torch

from tinygpt.model import GPT, GPTConfig

torch.set_num_threads(4)


def small_cfg(**kw):
    base = dict(vocab_size=96, block_size=32, n_layer=2, n_head=4, n_embd=64)
    base.update(kw)
    return GPTConfig(**base)


def make(cfg, seed=1337):
    torch.manual_seed(seed)
    return GPT(cfg).eval()


def rand_idx(cfg, B=3, T=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, cfg.vocab_size, (B, T or cfg.block_size), generator=g)


# ---------------------------------------------------------------- shapes / loss
@pytest.mark.parametrize("use_pos_emb", [True, False])
def test_output_shapes(use_pos_emb):
    cfg = small_cfg(use_pos_emb=use_pos_emb)
    m = make(cfg)
    idx = rand_idx(cfg, B=3, T=17)
    logits, loss = m(idx)
    assert logits.shape == (3, 17, cfg.vocab_size) and loss is None
    logits, loss, attn = m(idx, targets=idx, return_attn=True)
    assert logits.shape == (3, 17, cfg.vocab_size) and loss.ndim == 0
    assert len(attn) == cfg.n_layer
    assert all(a.shape == (3, cfg.n_head, 17, 17) and not a.requires_grad for a in attn)


def test_too_long_sequence_raises():
    cfg = small_cfg()
    with pytest.raises(ValueError):
        make(cfg)(rand_idx(cfg, T=cfg.block_size + 1))


@pytest.mark.parametrize("vocab", [96, 512, 2048])
def test_initial_loss_is_ln_vocab(vocab):
    cfg = GPTConfig(vocab_size=vocab)  # default architecture from SPEC
    m = make(cfg)
    idx = rand_idx(cfg, B=8)
    _, loss = m(idx, targets=rand_idx(cfg, B=8, seed=1))
    print(f"\nvocab {vocab}: init loss {loss.item():.4f}  vs ln(V) {math.log(vocab):.4f}")
    assert abs(loss.item() - math.log(vocab)) < 0.1


# ---------------------------------------------------------------- causality / attention
@pytest.mark.parametrize("use_pos_emb", [True, False])
def test_causality_future_tokens_do_not_change_past_logits(use_pos_emb):
    cfg = small_cfg(use_pos_emb=use_pos_emb)
    m = make(cfg)
    idx = rand_idx(cfg, B=2)
    base, _ = m(idx)
    for t in [0, 5, cfg.block_size // 2, cfg.block_size - 2]:
        changed = idx.clone()
        changed[:, t + 1:] = (changed[:, t + 1:] + 1 + torch.arange(cfg.block_size - t - 1)) % cfg.vocab_size
        out, _ = m(changed)
        assert torch.allclose(out[:, : t + 1], base[:, : t + 1], atol=1e-5), f"leak at t={t}"
        assert not torch.allclose(out[:, t + 1:], base[:, t + 1:], atol=1e-5)  # sanity: future did change


def test_causality_via_gradients():
    cfg = small_cfg()
    m = make(cfg)
    x = m.tok_emb(rand_idx(cfg, B=1)).detach().requires_grad_(True)
    h = x + m.pos_emb(torch.arange(cfg.block_size))
    for block in m.blocks:
        h, _ = block(h)
    t = 10
    h[0, t].sum().backward()
    assert x.grad[0, t + 1:].abs().max() == 0  # output at t has no gradient path to inputs after t
    assert x.grad[0, : t + 1].abs().sum() > 0


def test_attention_rows_sum_to_one_and_are_causal():
    cfg = small_cfg()
    m = make(cfg)
    _, _, attn = m(rand_idx(cfg), return_attn=True)
    upper = torch.triu(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool), diagonal=1)
    for a in attn:
        assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-5)
        assert (a >= 0).all()
        assert a[..., upper].abs().max() == 0
        assert torch.allclose(a[..., 0, 0], torch.ones_like(a[..., 0, 0]))  # first token attends to itself


def test_no_pos_emb_has_no_position_parameters():
    m_pos, m_nopos = make(small_cfg()), make(small_cfg(use_pos_emb=False))
    assert m_nopos.pos_emb is None and m_pos.pos_emb is not None
    assert m_pos.num_params(non_embedding=False) - m_nopos.num_params(non_embedding=False) == 32 * 64
    assert m_pos.num_params() == m_nopos.num_params()
    assert not any("pos_emb" in k for k in m_nopos.state_dict())


# ---------------------------------------------------------------- params / tying / state_dict
def test_weight_tying_and_param_counts():
    cfg = GPTConfig(vocab_size=96)
    m = make(cfg)
    assert m.lm_head.weight is m.tok_emb.weight
    C, L, V, T = cfg.n_embd, cfg.n_layer, cfg.vocab_size, cfg.block_size
    per_block = 2 * 2 * C + (3 * C * C + 3 * C) + (C * C + C) + (4 * C * C + 4 * C) + (4 * C * C + C)
    expected_non_emb = L * per_block + 2 * C  # + final LayerNorm
    assert m.num_params() == expected_non_emb
    assert m.num_params(non_embedding=False) == expected_non_emb + V * C + T * C


def test_param_counts_report():
    print("\nparam counts (default SPEC architecture, tied lm_head):")
    rows = []
    for vocab in (96, 512, 2048):
        for pos in (True, False):
            m = GPT(GPTConfig(vocab_size=vocab, use_pos_emb=pos))
            rows.append((vocab, pos, m.num_params(), m.num_params(non_embedding=False)))
            print(f"  vocab {vocab:5d} pos_emb={pos!s:5}: non-embedding {rows[-1][2]:,}  total {rows[-1][3]:,}")
    assert len({r[2] for r in rows}) == 1  # non-embedding count is the same for every run


def test_state_dict_roundtrip_like_checkpoint():
    from dataclasses import asdict

    cfg = small_cfg()
    m = make(cfg)
    ckpt = {"model_config": asdict(cfg), "model_state": m.state_dict()}
    m2 = GPT(GPTConfig(**ckpt["model_config"])).eval()
    m2.load_state_dict(ckpt["model_state"])
    idx = rand_idx(cfg)
    assert torch.equal(m(idx)[0], m2(idx)[0])
    assert m2.lm_head.weight is m2.tok_emb.weight  # tying survives load_state_dict


# ---------------------------------------------------------------- generate
def test_generate_shapes_crop_and_topk():
    cfg = small_cfg(block_size=8)
    m = make(cfg)
    idx = rand_idx(cfg, B=2, T=5)
    torch.manual_seed(0)
    out = m.generate(idx, max_new_tokens=20, temperature=0.8, top_k=5)  # 25 > block_size: must crop
    assert out.shape == (2, 25) and torch.equal(out[:, :5], idx)
    assert out.min() >= 0 and out.max() < cfg.vocab_size
    # top_k=1 equals greedy decoding
    torch.manual_seed(0)
    a = m.generate(idx, 10, top_k=1)
    b = m.generate(idx, 10, temperature=0)
    assert torch.equal(a, b)


def test_generate_restores_training_mode():
    m = make(small_cfg(dropout=0.1)).train()
    m.generate(torch.zeros(1, 1, dtype=torch.long), 3)
    assert m.training


def test_top_k_only_samples_from_top_k():
    cfg = small_cfg()
    m = make(cfg)
    idx = rand_idx(cfg, B=1, T=4)
    logits, _ = m(idx)
    allowed = set(torch.topk(logits[0, -1], 3).indices.tolist())
    torch.manual_seed(0)
    samples = {m.generate(idx, 1, temperature=2.0, top_k=3)[0, -1].item() for _ in range(50)}
    assert samples <= allowed


# ---------------------------------------------------------------- learning
def test_overfit_one_tiny_batch():
    torch.manual_seed(1337)
    cfg = GPTConfig(vocab_size=96, block_size=16, n_layer=2, n_head=4, n_embd=64)
    m = GPT(cfg).train()
    data = torch.randint(0, cfg.vocab_size, (4, cfg.block_size + 1))
    x, y = data[:, :-1], data[:, 1:]
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for step in range(200):
        _, loss = m(x, y)
        first = first if first is not None else loss.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    print(f"\noverfit: loss {first:.3f} -> {loss.item():.4f} after 200 steps")
    assert loss.item() < 0.05
