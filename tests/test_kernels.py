"""Kernel-level checks against the reference formulas at the real head width."""

from __future__ import annotations

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False


def ref_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return w * h.to(x.dtype)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def test_swiglu_matches_reference():
    from kernels import swiglu
    gu = torch.randn(37, 2 * 9728, device="cuda", dtype=torch.bfloat16) * 3
    g, u = gu[:, :9728], gu[:, 9728:]
    ref = F.silu(g) * u
    assert torch.equal(swiglu(gu), ref)


def test_rope_cache_matches_reference():
    from kernels import qk_norm_rope_cache
    from model import Config, rope_tables
    B, T, HQ, HKV, D, CAP = 2, 5, 8, 2, 128, 9
    cfg = Config(hidden=0, intermediate=0, layers=0, heads=HQ, kv_heads=HKV, head_dim=D, vocab=0, eps=1e-6, rope_theta=5e6, tie_embeddings=True)
    cos, sin = rope_tables(cfg, CAP, "cuda")
    qkv = torch.randn(B * T, (HQ + 2 * HKV) * D, device="cuda", dtype=torch.bfloat16)
    qw = 1 + 0.1 * torch.randn(D, device="cuda", dtype=torch.bfloat16)
    kw = 1 + 0.1 * torch.randn(D, device="cuda", dtype=torch.bfloat16)
    start = 3
    pos = torch.full((B,), start, dtype=torch.int32, device="cuda")
    q_out = torch.empty(B, HQ, T, D, device="cuda", dtype=torch.bfloat16)
    kc = torch.zeros(B, HKV, CAP, D, device="cuda", dtype=torch.bfloat16)
    vc = torch.zeros_like(kc)
    qk_norm_rope_cache(qkv, qw, kw, cos, sin, pos, q_out, kc, vc, T, 1e-6)

    q = qkv[:, : HQ * D].view(B, T, HQ, D)
    k = qkv[:, HQ * D:(HQ + HKV) * D].view(B, T, HKV, D)
    v = qkv[:, (HQ + HKV) * D:].view(B, T, HKV, D)
    c = cos[start:start + T][None, :, None, :]
    s = sin[start:start + T][None, :, None, :]
    qn, kn = ref_rmsnorm(q, qw, 1e-6), ref_rmsnorm(k, kw, 1e-6)
    q_ref = ((qn * c) + (rotate_half(qn) * s)).transpose(1, 2)
    k_ref = ((kn * c) + (rotate_half(kn) * s)).transpose(1, 2)
    assert torch.equal(q_out, q_ref)
    assert torch.equal(kc[:, :, start:start + T], k_ref)
    assert torch.equal(vc[:, :, start:start + T], v.transpose(1, 2))
    assert torch.all(kc[:, :, :start] == 0) and torch.all(kc[:, :, start + T:] == 0)


def test_decode_attention_matches_sdpa():
    from kernels import DecodeAttention
    B, HQ, HKV, D, CAP = 3, 8, 2, 128, 200
    lengths = torch.tensor([200, 1, 77], dtype=torch.int32, device="cuda")
    q = torch.randn(B, HQ, D, device="cuda", dtype=torch.bfloat16)
    kc = torch.randn(B, HKV, CAP, D, device="cuda", dtype=torch.bfloat16)
    vc = torch.randn(B, HKV, CAP, D, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(q)
    for nsplit in (1, 4):
        attn = DecodeAttention(B, HQ, HKV, D, CAP, "cuda", nsplit=nsplit)
        attn(q, kc, vc, lengths - 1, out)
        for b in range(B):
            L = int(lengths[b])
            k = kc[b, :, :L].repeat_interleave(HQ // HKV, dim=0)
            v = vc[b, :, :L].repeat_interleave(HQ // HKV, dim=0)
            ref = F.scaled_dot_product_attention(q[b, :, None], k, v, scale=D ** -0.5)[:, 0]
            err = (out[b].float() - ref.float()).abs().max().item()
            assert err < 2e-2, f"nsplit={nsplit} b={b}: {err}"


def test_sdpa_gqa_bitwise_equals_repeat_kv():
    B, HQ, HKV, T, D = 2, 8, 2, 64, 128
    q = torch.randn(B, HQ, T, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, HKV, T, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, HKV, T, D, device="cuda", dtype=torch.bfloat16)
    a = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=D ** -0.5, enable_gqa=True)
    b = F.scaled_dot_product_attention(q, k.repeat_interleave(4, 1), v.repeat_interleave(4, 1), is_causal=True, scale=D ** -0.5)
    assert torch.equal(a, b)


def test_add_rms_norm_matches_reference():
    from kernels import add_rms_norm
    x = torch.randn(9, 2560, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(9, 2560, device="cuda", dtype=torch.bfloat16)
    w = 1 + 0.1 * torch.randn(2560, device="cuda", dtype=torch.bfloat16)
    x_ref = x + y
    h_ref = ref_rmsnorm(x_ref, w, 1e-6)
    h = add_rms_norm(x, y, w, 1e-6)
    assert torch.equal(x, x_ref)
    assert torch.equal(h, h_ref)


def test_skinny_matmul_configs_match_cublas():
    from kernels.gemm import CONFIGS, SkinnyMatmul
    for M in (1, 4, 16):
        for N, K in ((6144, 2560), (2560, 9728), (777, 2560)):
            a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
            ref = (a @ w.t()).float()
            for cfg in CONFIGS:
                out = SkinnyMatmul(M, N, K, a.device, **cfg)(a, w).float()
                err = (out - ref).abs().max().item()
                assert err <= 0.02 * ref.abs().max().item() + 1e-3, (M, N, K, cfg, err)


def test_fused_rope_launch_equals_split_launch():
    from kernels import qk_norm_rope_cache
    from model import Config, rope_tables
    B, T, HQ, HKV, D, CAP = 2, 3, 8, 2, 128, 7
    cfg = Config(hidden=0, intermediate=0, layers=0, heads=HQ, kv_heads=HKV, head_dim=D, vocab=0, eps=1e-6, rope_theta=5e6, tie_embeddings=True)
    cos, sin = rope_tables(cfg, CAP, "cuda")
    qkv = torch.randn(B * T, (HQ + 2 * HKV) * D, device="cuda", dtype=torch.bfloat16)
    qw = 1 + 0.1 * torch.randn(D, device="cuda", dtype=torch.bfloat16)
    kw = 1 + 0.1 * torch.randn(D, device="cuda", dtype=torch.bfloat16)
    pos = torch.full((B,), 2, dtype=torch.int32, device="cuda")
    outs = []
    for fused in (False, True):
        q_out = torch.empty(B, HQ, T, D, device="cuda", dtype=torch.bfloat16)
        kc = torch.zeros(B, HKV, CAP, D, device="cuda", dtype=torch.bfloat16)
        vc = torch.zeros_like(kc)
        qk_norm_rope_cache(qkv, qw, kw, cos, sin, pos, q_out, kc, vc, T, 1e-6, fused=fused)
        outs.append((q_out, kc, vc))
    for a, b in zip(*outs):
        assert torch.equal(a, b)


def test_gateup_configs_match_cublas_swiglu():
    from kernels import swiglu
    from kernels.gemm import CONFIGS, SkinnyGateUp
    for M in (1, 16):
        I, K = 9728, 2560
        a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        wgu = torch.randn(2 * I, K, device="cuda", dtype=torch.bfloat16) * 0.02
        ref = swiglu(a @ wgu.t()).float()
        for cfg in CONFIGS:
            out = SkinnyGateUp(M, I, K, a.device, **cfg)(a, wgu[:I], wgu[I:]).float()
            err = (out - ref).abs().max().item()
            assert err <= 0.02 * ref.abs().max().item() + 1e-3, (M, cfg, err)


def test_pick_attention_returns_matching_kernel():
    from kernels import pick_attention
    attn = pick_attention(2, 8, 2, 128, 300, 250, "cuda")
    assert attn.NSPLIT >= 1


def test_norm_prologue_matches_add_norm_then_gemm():
    from kernels import add_rms_norm
    from kernels.gemm import CONFIGS, SkinnyGateUp, SkinnyMatmul
    torch.manual_seed(0)
    for M in (1, 5, 16):
        K, N, I = 2560, 1536, 1024
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        y = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        wn = 1 + 0.1 * torch.randn(K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
        wgu = torch.randn(2 * I, K, device="cuda", dtype=torch.bfloat16) * 0.02
        xout_ref = torch.empty_like(x)
        h = add_rms_norm(x, y, wn, 1e-6, xout_ref)
        ref = (h @ w.t()).float()
        ref_gu = torch.nn.functional.silu(h @ wgu[:I].t()) * (h @ wgu[I:].t())
        for cfg in CONFIGS:
            xout = torch.empty_like(x)
            out = SkinnyMatmul(M, N, K, x.device, **cfg)(x, w, norm=(y, wn, xout, 1e-6)).float()
            assert torch.equal(xout, xout_ref), (M, cfg)
            assert (out - ref).abs().max().item() <= 0.02 * ref.abs().max().item() + 1e-3, (M, cfg)
            xout2 = torch.empty_like(x)
            out2 = SkinnyGateUp(M, I, K, x.device, **cfg)(x, wgu[:I], wgu[I:], norm=(y, wn, xout2, 1e-6)).float()
            assert torch.equal(xout2, xout_ref), (M, cfg)
            assert (out2 - ref_gu.float()).abs().max().item() <= 0.02 * ref_gu.abs().max().item() + 1e-3, (M, cfg)
