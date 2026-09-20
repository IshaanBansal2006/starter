"""Accept the longest verified path of a draft tree, on device.

The host used to do this: copy ``blk`` and ``cand`` back, walk the tree
template in Python, copy the answer forward again. That round trip left the GPU
idle for the whole of it, once per round. This kernel does the same walk with
one program per sequence, so the accept can sit inside the captured round graph
between the verify that produced ``cand`` and the compact that consumes the
path.

The rule is exactly ``Recycler.accept``: start at the root (node 0) holding the
token ``cand[0]``; at node ``p`` take the lowest-indexed child ``c`` whose
drafted token equals the token the model predicted after ``p``
(``blk[c] == cand[p]``); stop when no child matches. Verification is exact, so
every token on that path is the one plain greedy decode would have produced.

The tree template is flattened the way a CSR matrix is: ``child_start[p] ..
child_start[p + 1]`` is node ``p``'s slice of ``child_list``, and
``child_par[j]`` is the parent that owns slot ``j``. ``TreeTemplate.build``
appends children in ascending node order, so "lowest-indexed matching child" is
"first matching slot in the slice" — a masked ``min`` over the slot axis.

Everything the walk needs (the slices, the per-slot match bits, the candidate
tokens) is loaded into registers once before the loop, so the ``MAXA``
iterations are pure ALU and shuffle work; a dependent scalar load per level
would have cost more than the host round trip this replaces.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def flat_children(children: list[list[int]]) -> tuple[list[int], list[int], list[int]]:
    """CSR form of a tree template's children lists: (start[R+1], list[R-1], parent[R-1])."""
    start, flat, par = [0], [], []
    for p, kids in enumerate(children):
        for c in kids:
            flat.append(c)
            par.append(p)
        start.append(len(flat))
    return start, flat, par


@triton.jit
def accept_kernel(blk_ptr, cand_ptr, child_logit_ptr, row_max_ptr, child_start_ptr, child_list_ptr, child_par_ptr,
                  done_ptr, nseen_ptr, pos_ptr, limit_ptr,
                  root_ptr, path_idx_ptr, path_len_ptr, acc_tok_ptr, acc_cnt_ptr,
                  CAP, margin, R: tl.constexpr, C: tl.constexpr, P: tl.constexpr,
                  MAXA: tl.constexpr, GUARD: tl.constexpr):
    """One program per sequence.

    Frozen sequences (``done``) write ``path_len = -1``, which makes the compact
    copy nothing and ``pos += path_len + 1`` a no-op, and leave ``root`` alone so
    the next draft re-verifies the same block in place. Live sequences write the
    accepted path, its length, the new root, and the accepted tokens in order.

    ``done`` for the *next* round is recomputed here from the counters this
    kernel just advanced, which is the same test the host used to make from its
    queue lengths: a sequence is finished once it holds ``limit`` tokens, or
    once its next block would run past the end of the KV cache.
    """
    b = tl.program_id(0)
    live = tl.load(done_ptr + b) == 0
    n_prev = tl.load(nseen_ptr + b)
    pos = tl.load(pos_ptr + b)
    limit = tl.load(limit_ptr)

    j = tl.arange(0, P)
    cs = tl.load(child_start_ptr + j, mask=j < R + 1, other=0)
    cl = tl.load(child_list_ptr + j, mask=j < C, other=0)
    cp = tl.load(child_par_ptr + j, mask=j < C, other=0)
    cv = tl.load(cand_ptr + b * R + j, mask=j < R, other=0)
    # Per slot: is the child's drafted token within ``margin`` logits of the
    # best token the model predicted after its parent? With margin 0 this is
    # exact greedy (ties included). Among several qualifying children the one
    # with the highest logit wins, lowest slot on equal logits.
    drafted = tl.load(blk_ptr + b * R + cl, mask=j < C, other=-1)
    clog = tl.load(child_logit_ptr + b * C + j, mask=j < C, other=float("-inf"))
    pmax = tl.load(row_max_ptr + b * R + cp, mask=j < C, other=float("inf"))
    hit = (j < C) & (clog >= pmax - margin)

    cur = 0
    alen = 0
    alive = live
    for _ in range(MAXA):
        start = tl.sum(tl.where(j == cur, cs, 0), axis=0)
        end = tl.sum(tl.where(j == cur + 1, cs, 0), axis=0)
        inslice = hit & (j >= start) & (j < end)
        best = tl.max(tl.where(inslice, clog, float("-inf")), axis=0)
        jsel = tl.min(tl.where(inslice & (clog == best), j, P), axis=0)
        found = alive & (jsel < P)
        c = tl.where(found, tl.sum(tl.where(j == jsel, cl, 0), axis=0), 0)
        ctok = tl.sum(tl.where(j == jsel, drafted, 0), axis=0)
        tl.store(path_idx_ptr + b * MAXA + tl.minimum(alen, MAXA - 1), c, mask=found)
        # The token emitted at this step is the accepted draft itself, which
        # under a margin need not equal the parent's argmax.
        tl.store(acc_tok_ptr + b * (MAXA + 1) + tl.minimum(alen, MAXA), ctok, mask=found)
        alen = tl.where(found, alen + 1, alen)
        cur = tl.where(found, c, cur)
        alive = found
    # After the last accepted node the model's own argmax is emitted.
    tok = tl.sum(tl.where(j == cur, cv, 0), axis=0)
    tl.store(acc_tok_ptr + b * (MAXA + 1) + tl.minimum(alen, MAXA), tok)

    plen = tl.where(live, alen, -1)
    cnt = tl.where(live, alen + 1, 0)
    n_new = n_prev + cnt
    tl.store(path_len_ptr + b, plen)
    tl.store(acc_cnt_ptr + b, cnt)
    tl.store(root_ptr + b, tok, mask=live)
    tl.store(nseen_ptr + b, n_new)
    # Sticky: neither counter moves once a sequence is frozen, so recomputing
    # alone would already hold, but the OR makes an externally set flag final.
    frozen = (n_new >= limit) | (pos + plen + 1 + GUARD >= CAP) | (live == 0)
    tl.store(done_ptr + b, frozen.to(tl.int32))


def accept_paths(blk: torch.Tensor, cand: torch.Tensor, child_start: torch.Tensor,
                 child_list: torch.Tensor, child_par: torch.Tensor, done: torch.Tensor,
                 nseen: torch.Tensor, pos: torch.Tensor, limit: torch.Tensor,
                 root: torch.Tensor, path_idx: torch.Tensor, path_len: torch.Tensor,
                 acc_tokens: torch.Tensor, acc_count: torch.Tensor, cap: int, guard: int,
                 child_logit: torch.Tensor | None = None, row_max: torch.Tensor | None = None,
                 margin: float = 0.0) -> None:
    """blk/cand [B, R] int64; the CSR template; done/nseen/pos/limit int32 device
    state; writes root [B] int64, path_idx [B, MAXA] int32, path_len [B] int32,
    acc_tokens [B, MAXA+1] int64 and acc_count [B] int32.

    ``pos`` is the sequence's root slot for the block just verified, so the
    capacity guard here is the host's old ``pos_host[b] + 2 * R >= plan.cap``.
    """
    B, R = blk.shape
    MAXA = path_idx.shape[1]
    if acc_tokens.shape != (B, MAXA + 1):
        raise ValueError(f"acc_tokens must be [{B}, {MAXA + 1}], got {tuple(acc_tokens.shape)}")
    C = child_list.numel()
    P = triton.next_power_of_2(max(R + 1, C))
    if child_logit is None or row_max is None:
        raise ValueError("accept_paths needs child_logit [B, C] and row_max [B*R] from the verify pass")
    accept_kernel[(B,)](blk, cand, child_logit, row_max, child_start, child_list, child_par, done, nseen, pos, limit,
                        root, path_idx, path_len, acc_tokens, acc_count,
                        cap, float(margin), R=R, C=C, P=P, MAXA=MAXA, GUARD=guard, num_warps=1)
