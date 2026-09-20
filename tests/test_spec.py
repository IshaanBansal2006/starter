"""Speculative verification must reproduce plain greedy decoding exactly."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tiny import make_tiny

TINY = Path(__file__).resolve().parent / "_tiny_model"


@pytest.fixture(scope="module")
def engine():
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


def test_spec_generate_equals_plain(engine, monkeypatch):
    vocab = engine.model.cfg.vocab
    g = torch.Generator().manual_seed(3)
    base = torch.randint(0, vocab, (2, 24), generator=g).tolist()
    ids = [row + row[:20] for row in base]
    plain = list(engine.generate(ids, 16))
    from engine import GraphPlan
    spec = GraphPlan(engine.model, len(ids), len(ids[0]), 16, spec_k=3)
    spec.capture()
    out = list(spec.run(ids, 16))
    assert out == plain
    assert spec.stats["rounds"] < 15
