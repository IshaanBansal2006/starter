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


def test_spine_nodes_follow_rank_zero_path():
    from recycle import TreeTemplate
    t = TreeTemplate.build(64, 8)
    spine = t.spine
    assert spine and t.parent[spine[0]] == 0 and all(t.rank[n] == 0 for n in spine)
    for a, b in zip(spine, spine[1:]):
        assert t.parent[b] == a


def test_accept_kernel_matches_host_walk():
    """The device accept must reproduce ``Recycler.accept`` node for node, and
    carry the round bookkeeping the host used to keep: the new root, the token
    counter, and the frozen flag for the next round."""
    from kernels.accept import accept_paths
    from recycle import Recycler

    torch.manual_seed(0)
    B, vocab, cap = 5, 64, 40
    for rows in (2, 4, 16, 64):
        rec = Recycler(vocab, B, rows, 8, "cuda")
        maxa, guard = rec.maxa, 2 * rows
        for trial in range(20):
            cand = torch.randint(0, 6, (B, rows), dtype=torch.int64, device="cuda")
            blk = torch.randint(0, 6, (B, rows), dtype=torch.int64, device="cuda")
            for b in range(B):  # plant a real path so the walk has something to find
                node = 0
                while rec.template.children[node] and torch.rand(()).item() > 0.25:
                    kids = rec.template.children[node]
                    node = kids[int(torch.randint(0, len(kids), ()).item())]
                    blk[b, node] = cand[b, rec.template.parent[node]]
            pos = torch.randint(10, 20, (B,), dtype=torch.int32, device="cuda")
            nseen = torch.randint(1, 4, (B,), dtype=torch.int32, device="cuda")
            done = (torch.rand(B, device="cuda") < 0.3).to(torch.int32)
            limit_v = int(torch.randint(1, 8, ()).item())
            limit = torch.tensor([limit_v], dtype=torch.int32, device="cuda")
            root = torch.randint(0, vocab, (B,), dtype=torch.int64, device="cuda")
            path_idx = torch.full((B, maxa), 99, dtype=torch.int32, device="cuda")
            path_len = torch.zeros((B,), dtype=torch.int32, device="cuda")
            acc_tok = torch.zeros((B, maxa + 1), dtype=torch.int64, device="cuda")
            acc_cnt = torch.zeros((B,), dtype=torch.int32, device="cuda")
            was_frozen, seen0, pos0, root0 = done.tolist(), nseen.tolist(), pos.tolist(), root.tolist()
            accept_paths(blk, cand, rec.child_start, rec.child_list, rec.child_par, done, nseen,
                         pos, limit, root, path_idx, path_len, acc_tok, acc_cnt, cap, guard)
            blk_l, cand_l = blk.tolist(), cand.tolist()
            for b in range(B):
                if was_frozen[b]:
                    assert (int(path_len[b]), int(acc_cnt[b])) == (-1, 0)
                    assert int(root[b]) == root0[b] and int(nseen[b]) == seen0[b]
                    assert int(done[b]) == 1
                    continue
                toks, path = rec.accept(blk_l[b], cand_l[b])
                assert int(path_len[b]) == len(path)
                assert path_idx[b].tolist()[:len(path)] == path
                assert int(acc_cnt[b]) == len(toks)
                assert acc_tok[b].tolist()[:len(toks)] == toks
                assert int(root[b]) == toks[-1]
                assert int(nseen[b]) == seen0[b] + len(toks)
                end = pos0[b] + len(path) + 1
                assert int(done[b]) == int(int(nseen[b]) >= limit_v or end + guard >= cap)


@pytest.mark.parametrize("rows", [4, 16, 64])
def test_device_accept_loop_matches_judge(engine, reference, rows):
    """The whole round — accept, compact, draft, verify — is one graph replay and
    the host reads the accepted tokens a round behind, so the loop must still
    yield exactly max_new steps, all inside the judge's margin."""
    vocab = engine.model.cfg.vocab
    g = torch.Generator().manual_seed(11)
    base = torch.randint(0, vocab, (3, 20), generator=g).tolist()
    ids = [row + row[:16] for row in base]
    from engine import GraphPlan
    rec = GraphPlan(engine.model, len(ids), len(ids[0]), 20, recycle_rows=rows)
    rec.capture()
    out = list(rec.run(ids, 20))
    assert len(out) == 20 and all(len(step) == 3 for step in out)
    judge(reference, ids, out)
    assert rec.stats["rounds"] >= 1
    assert rec.stats["rounds"] >= rec.stats["min_rounds"]
    # Rounds are only queued when one more is certain, so none is ever wasted.
    assert rec.stats["rounds"] == rec.stats["launched"]
    out2 = list(rec.run([row[::-1] for row in ids], 9))
    assert len(out2) == 9
    judge(reference, [row[::-1] for row in ids], out2)
