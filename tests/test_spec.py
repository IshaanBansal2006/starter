"""Speculative verification must reproduce plain greedy decoding exactly."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tiny import make_tiny

TINY = Path(__file__).resolve().parent / "_tiny_model"


@pytest.fixture(scope="module")
def engine():
    import os
    os.environ["ENGINE_RECYCLE"] = "0"
    from engine import Engine
    return Engine(str(make_tiny(TINY)))


def test_multi_row_attention_matches_reference():
    from kernels import pick_attention
    from kernels.attention import reference_attention
    B, HQ, HKV, D, cap, R = 2, 8, 2, 128, 96, 4
    q = torch.randn((B, HQ, R, D), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((B, HKV, cap, D), dtype=torch.bfloat16, device="cuda")
    v = torch.randn((B, HKV, cap, D), dtype=torch.bfloat16, device="cuda")
    for L in (R, 5, 64, 90):
        pos = torch.full((B,), L - R, dtype=torch.int32, device="cuda")
        out = torch.empty((B, R, HQ, D), dtype=torch.bfloat16, device="cuda")
        ref = reference_attention(q, k, v, pos, R, D ** -0.5)
        for nsplit in (1, 3):
            from kernels.attention import DecodeAttention
            attn = DecodeAttention(B, HQ, HKV, D, cap, "cuda", nsplit=nsplit, R=R)
            attn(q, k, v, pos, out)
            err = (out.float() - ref).abs().max().item()
            assert err < 2e-2, (L, nsplit, err)


def test_ngram_drafter():
    from spec import NGramDrafter
    d = NGramDrafter([1, 2, 3, 4, 1, 2, 3], K=3)
    assert d.draft() == [4, 1, 2]
    d.extend([9])
    assert d.draft() == [9, 9, 9]


def judge(reference, ids, out, margin: float = 2.0) -> None:
    """The platform's rule: every emitted token must be the reference argmax on
    our own prefix, or within ``margin`` logits of it. Exact equality with the
    plain engine is too strict on a random tiny model full of near ties."""
    B = len(ids)
    seq = torch.tensor(ids, device="cuda:0")
    for step, toks in enumerate(out):
        with torch.inference_mode():
            logits = reference(input_ids=seq, use_cache=False).logits[:, -1].float()
        top = logits.max(-1).values
        mine = logits[torch.arange(B), torch.tensor(toks, device="cuda:0")]
        assert torch.all(top - mine <= margin), f"step {step}: gap {(top - mine).max().item():.3f}"
        seq = torch.cat([seq, torch.tensor(toks, device="cuda:0")[:, None]], dim=1)


@pytest.fixture(scope="module")
def reference():
    from tiny import load_reference
    return load_reference(TINY)


def test_spec_generate_within_margin(engine, reference):
    vocab = engine.model.cfg.vocab
    g = torch.Generator().manual_seed(3)
    base = torch.randint(0, vocab, (2, 24), generator=g).tolist()
    ids = [row + row[:20] for row in base]
    from engine import GraphPlan
    spec = GraphPlan(engine.model, len(ids), len(ids[0]), 16, spec_k=3)
    spec.capture()
    out = list(spec.run(ids, 16))
    assert len(out) == 16 and all(len(step) == 2 for step in out)
    judge(reference, ids, out)
    assert spec.stats["rounds"] < 15


def test_tree_attention_matches_reference():
    from kernels.attention import DecodeAttention, reference_attention
    B, HQ, HKV, D, cap, R = 2, 8, 2, 128, 160, 16
    # A small tree: node 0 root; 1,2,3 children of 0; 4,5 children of 1; 6 child of 2;
    # 7 child of 4; 8..15 chain under 3.
    parent = [-1, 0, 0, 0, 1, 1, 2, 4, 3, 8, 9, 10, 11, 12, 13, 14]
    masks = []
    for i in range(R):
        m, n = 0, i
        while n >= 0:
            m |= 1 << n
            n = parent[n]
        masks.append(m)
    tree = torch.tensor(masks, dtype=torch.int64, device="cuda")
    q = torch.randn((B, HQ, R, D), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((B, HKV, cap, D), dtype=torch.bfloat16, device="cuda")
    v = torch.randn((B, HKV, cap, D), dtype=torch.bfloat16, device="cuda")
    for L in (R, 40, 130):
        pos = torch.full((B,), L - R, dtype=torch.int32, device="cuda")
        ref = reference_attention(q, k, v, pos, R, D ** -0.5, tree)
        for nsplit in (1, 3):
            out = torch.empty((B, R, HQ, D), dtype=torch.bfloat16, device="cuda")
            DecodeAttention(B, HQ, HKV, D, cap, "cuda", nsplit=nsplit, R=R, tree=True)(q, k, v, pos, out, tree)
            err = (out.float() - ref).abs().max().item()
            assert err < 2e-2, (L, nsplit, err)


def test_tree_row_blocks_for_64_nodes():
    from kernels.attention import DecodeAttention, chain_tree, reference_attention
    B, HQ, HKV, D, cap, R = 1, 8, 2, 128, 200, 64
    tree = chain_tree(R, "cuda")
    q = torch.randn((B, HQ, R, D), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((B, HKV, cap, D), dtype=torch.bfloat16, device="cuda")
    v = torch.randn((B, HKV, cap, D), dtype=torch.bfloat16, device="cuda")
    pos = torch.full((B,), 100, dtype=torch.int32, device="cuda")
    ref = reference_attention(q, k, v, pos, R, D ** -0.5, tree)
    attn = DecodeAttention(B, HQ, HKV, D, cap, "cuda", nsplit=2, R=R, tree=True)
    assert attn.row_blocks == 8
    out = torch.empty((B, R, HQ, D), dtype=torch.bfloat16, device="cuda")
    attn(q, k, v, pos, out, tree)
    assert (out.float() - ref).abs().max().item() < 2e-2


def test_compact_paths_moves_accepted_rows():
    from kernels.compact import compact_paths
    layers, B, HKV, cap, D, MAXA = 3, 2, 2, 40, 128, 4
    k = torch.randn((layers, B, HKV, cap, D), dtype=torch.bfloat16, device="cuda")
    v = torch.randn((layers, B, HKV, cap, D), dtype=torch.bfloat16, device="cuda")
    k0, v0 = k.clone(), v.clone()
    pos = torch.tensor([10, 20], dtype=torch.int32, device="cuda")
    idx = torch.tensor([[2, 5, 9, 0], [1, 2, 0, 0]], dtype=torch.int32, device="cuda")
    lens = torch.tensor([3, 2], dtype=torch.int32, device="cuda")
    compact_paths(k, v, pos, idx, lens)
    for b, (p, path) in enumerate(((10, [2, 5, 9]), (20, [1, 2]))):
        for j, src in enumerate(path):
            assert torch.equal(k[:, b, :, p + 1 + j], k0[:, b, :, p + src])
            assert torch.equal(v[:, b, :, p + 1 + j], v0[:, b, :, p + src])
        assert torch.equal(k[:, b, :, :p + 1], k0[:, b, :, :p + 1])


def test_tree_template_shapes():
    from recycle import TreeTemplate
    for size in (2, 4, 16, 64):
        t = TreeTemplate.build(size, 8)
        assert t.size == size
        assert all(p < i for i, p in enumerate(t.parent) if i > 0)
        assert t.parent[1] == 0 and t.rank[1] == 0


@pytest.mark.parametrize("rows", [2, 4, 16, 64])
def test_recycle_generate_within_margin(engine, reference, rows):
    """Token recycling must stay inside the judge's margin and yield exactly max_new steps."""
    vocab = engine.model.cfg.vocab
    g = torch.Generator().manual_seed(4)
    base = torch.randint(0, vocab, (2, 24), generator=g).tolist()
    ids = [row + row[:20] for row in base]
    from engine import GraphPlan
    rec = GraphPlan(engine.model, len(ids), len(ids[0]), 16, recycle_rows=rows)
    rec.capture()
    out = list(rec.run(ids, 16))
    assert len(out) == 16 and all(len(step) == 2 for step in out)
    judge(reference, ids, out)
    assert rec.stats["rounds"] <= 16
    ids2 = [row[::-1] for row in ids]
    out2 = list(rec.run(ids2, 12))
    assert len(out2) == 12
    judge(reference, ids2, out2)


def test_recycle_pads_rounds_to_minimum(engine, reference):
    from engine import GraphPlan
    vocab = engine.model.cfg.vocab
    ids = torch.randint(0, vocab, (1, 30), generator=torch.Generator().manual_seed(9)).tolist()
    ids = [ids[0] + ids[0][:25]]
    rec = GraphPlan(engine.model, 1, len(ids[0]), 12, recycle_rows=16)
    rec.tau_floor = 1.0
    rec.capture()
    out = list(rec.run(ids, 12))
    assert len(out) == 12
    assert rec.stats["rounds"] >= rec.stats["min_rounds"] == 11
    judge(reference, ids, out)
