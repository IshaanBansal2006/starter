"""Split-K flash-decoding for grouped-query attention with a device-side length.

One query token per sequence, all ``G = HQ // HKV`` query heads of a KV head
share one program so K/V are read once per group. Keys are split into NSPLIT
ranges so short batches still fill the GPU; partial (m, l, acc) triples are
merged by ``_reduce_kernel``. The valid key count is ``pos[b] + 1``, read from
device memory so a CUDA graph can replay the kernel as the sequence grows.

Numerics follow FlashAttention-2, which is what SDPA runs for the reference:
scores and softmax statistics in fp32, probabilities rounded to bf16 before the
PV product, output accumulated in fp32 and normalised once at the end.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _split_kernel(
    q_ptr, k_ptr, v_ptr, pos_ptr, o_part_ptr, m_part_ptr, l_part_ptr, o_ptr,
    CAP, scale,
    HQ: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, GP: tl.constexpr,
    D: tl.constexpr, BLOCK_N: tl.constexpr, NSPLIT: tl.constexpr, SPLIT_LEN: tl.constexpr,
    FINAL: tl.constexpr,
):
    b = tl.program_id(0)
    kh = tl.program_id(1)
    s = tl.program_id(2)
    L = tl.load(pos_ptr + b) + 1
    start = s * SPLIT_LEN
    end = tl.minimum(start + SPLIT_LEN, L)

    rows = tl.arange(0, GP)
    d = tl.arange(0, D)
    row_mask = rows < G
    q = tl.load(
        q_ptr + ((b * HQ + kh * G + rows[:, None]) * D + d[None, :]),
        mask=row_mask[:, None], other=0.0,
    )
    kv_base = (b * HKV + kh) * CAP

    m = tl.full([GP], float("-inf"), tl.float32)
    l = tl.zeros([GP], tl.float32)
    acc = tl.zeros([GP, D], tl.float32)
    for n0 in range(start, end, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)
        kmask = n < end
        k = tl.load(k_ptr + (kv_base + n[:, None]) * D + d[None, :], mask=kmask[:, None], other=0.0)
        sc = tl.dot(q, tl.trans(k)) * scale
        sc = tl.where(kmask[None, :], sc, float("-inf"))
        m_new = tl.maximum(m, tl.max(sc, axis=1))
        alpha = tl.exp(m - m_new)
        p = tl.exp(sc - m_new[:, None])
        l = l * alpha + tl.sum(p, axis=1)
        v = tl.load(v_ptr + (kv_base + n[:, None]) * D + d[None, :], mask=kmask[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m = m_new

    head = kh * G + rows
    if FINAL:
        out = acc / l[:, None]
        tl.store(o_ptr + (b * HQ + head[:, None]) * D + d[None, :], out.to(tl.bfloat16), mask=row_mask[:, None])
    else:
        part = (b * HQ + head) * NSPLIT + s
        tl.store(o_part_ptr + part[:, None] * D + d[None, :], acc, mask=row_mask[:, None])
        tl.store(m_part_ptr + part, m, mask=row_mask)
        tl.store(l_part_ptr + part, l, mask=row_mask)


@triton.jit
def _reduce_kernel(
    o_part_ptr, m_part_ptr, l_part_ptr, o_ptr,
    HQ: tl.constexpr, D: tl.constexpr, NSPLIT: tl.constexpr, NSP: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.arange(0, NSP)
    smask = s < NSPLIT
    base = (b * HQ + h) * NSPLIT
    m = tl.load(m_part_ptr + base + s, mask=smask, other=float("-inf"))
    l = tl.load(l_part_ptr + base + s, mask=smask, other=0.0)
    M = tl.max(m, axis=0)
    w = tl.exp(m - M)
    L = tl.sum(w * l, axis=0)
    d = tl.arange(0, D)
    o = tl.load(o_part_ptr + (base + s[:, None]) * D + d[None, :], mask=smask[:, None], other=0.0)
    out = tl.sum(o * w[:, None], axis=0) / L
    tl.store(o_ptr + (b * HQ + h) * D + d, out.to(tl.bfloat16))


class DecodeAttention:
    """Workspace-owning wrapper; one instance per (B, HQ, cap) plan."""

    def __init__(self, B: int, HQ: int, HKV: int, D: int, cap: int, device, nsplit: int | None = None):
        self.B, self.HQ, self.HKV, self.D, self.cap = B, HQ, HKV, D, cap
        self.G = HQ // HKV
        self.GP = max(16, triton.next_power_of_2(self.G))
        self.BLOCK_N = 64
        if nsplit is None:
            nsplit = max(1, min(16, (256 + B * HKV - 1) // (B * HKV)))
        blocks = triton.cdiv(cap, self.BLOCK_N)
        nsplit = max(1, min(nsplit, blocks))
        self.SPLIT_LEN = triton.cdiv(blocks, nsplit) * self.BLOCK_N
        self.NSPLIT = triton.cdiv(cap, self.SPLIT_LEN)
        self.NSP = triton.next_power_of_2(self.NSPLIT)
        self.scale = 1.0 / math.sqrt(D)
        self.o_part = torch.empty((B, HQ, self.NSPLIT, D), dtype=torch.float32, device=device)
        self.m_part = torch.empty((B, HQ, self.NSPLIT), dtype=torch.float32, device=device)
        self.l_part = torch.empty((B, HQ, self.NSPLIT), dtype=torch.float32, device=device)

    def __call__(self, q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, pos: torch.Tensor, out: torch.Tensor) -> None:
        """q [B, HQ, D] bf16; k/v_cache [B, HKV, cap, D]; pos [B] int32; out [B, HQ, D] bf16."""
        _split_kernel[(self.B, self.HKV, self.NSPLIT)](
            q, k_cache, v_cache, pos, self.o_part, self.m_part, self.l_part, out,
            self.cap, self.scale,
            HQ=self.HQ, HKV=self.HKV, G=self.G, GP=self.GP, D=self.D,
            BLOCK_N=self.BLOCK_N, NSPLIT=self.NSPLIT, SPLIT_LEN=self.SPLIT_LEN,
            FINAL=self.NSPLIT == 1,
            num_warps=4, num_stages=2,
        )
        if self.NSPLIT == 1:
            return
        _reduce_kernel[(self.B, self.HQ)](
            self.o_part, self.m_part, self.l_part, out,
            HQ=self.HQ, D=self.D, NSPLIT=self.NSPLIT, NSP=self.NSP, num_warps=1,
        )
