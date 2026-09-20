"""Exact argmax plus approximate top-k over the vocabulary in one bandwidth-bound pass.

``torch.topk`` over 151,936 columns is a radix select that costs many times the
read bandwidth. Rank 0 here is the exact argmax (lowest index on ties, like
``torch.argmax``), which is all correctness needs; ranks 1..K-1 only seed the
draft table, so they come from the per-lane maxima of a strided sweep: lane j
keeps the best value it saw at columns j, j+BLOCK, j+2*BLOCK, ... and the
top-K of those lane maxima is taken at the end. A true top-8 token is only
missed when two of them share a lane, which merely lowers draft quality.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _topk_kernel(x_ptr, val_ptr, idx_ptr, V, stride_row, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    best = tl.full([BLOCK], float("-inf"), tl.float32)
    best_idx = tl.zeros([BLOCK], tl.int32)
    base = x_ptr + row * stride_row
    for start in range(0, V, BLOCK):
        cols = start + lane
        v = tl.load(base + cols, mask=cols < V, other=float("-inf")).to(tl.float32)
        take = v > best  # strict: the first occurrence of a value wins its lane
        best = tl.where(take, v, best)
        best_idx = tl.where(take, cols, best_idx)
    for k in tl.static_range(K):
        m = tl.max(best, axis=0)
        # lowest column index among lanes holding the maximum (exact torch.argmax tie rule for k == 0)
        cand = tl.where(best == m, best_idx, 2147483647)
        i = tl.min(cand, axis=0)
        tl.store(val_ptr + row * K + k, m)
        tl.store(idx_ptr + row * K + k, i)
        best = tl.where(best_idx == i, float("-inf"), best)


def fast_topk(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, V] -> (values [N, k] float32, indices [N, k] int32); indices[:, 0] is the exact argmax."""
    N, V = logits.shape
    vals = torch.empty((N, k), dtype=torch.float32, device=logits.device)
    idxs = torch.empty((N, k), dtype=torch.int32, device=logits.device)
    _topk_kernel[(N,)](logits, vals, idxs, V, logits.stride(0), K=k, BLOCK=2048, num_warps=8)
    return vals, idxs
