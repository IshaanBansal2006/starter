# 003 — Accept the draft tree on device, one graph per recycling round
Date: 2026-09-20
Status: accepted

## Context

The token-recycling loop (`GraphPlan.run_recycle`) ran two graph replays per
round and then stopped dead: it copied `cand` and `blk` back to the host,
synchronised the stream, walked the tree template in Python
(`Recycler.accept`), and copied `path_idx`, `path_len`, `root` and the n-gram
spine forward again. The GPU was idle for the whole host part — about 0.2 to
0.4 ms, or 5 to 8% of a round on the H100 — and that fraction grows as the rest
of the round gets faster.

Nothing in that host work needs to be on the host. The acceptance rule is a
walk down a fixed tree comparing `blk[c]` against `cand[parent(c)]`; the
template is known at plan time; the only per-round inputs are two `[B, R]`
tensors that already live on device. The freeze test the host applied
(`len(queues[b]) >= max_new` or `pos_host[b] + 2 * R >= cap`) is two counters
the device can keep just as well.

The one genuinely host-side input is the n-gram spine: `NGramDrafter` indexes
the whole sequence in Python and its continuation is only known once the host
has read the round's tokens. Folding accept into the round graph means the next
round is queued before the host has read the previous one, so the spine is
necessarily at least a round stale.

## Options considered

**Leave the loop alone.** Keep the per-round round trip.
- Pros: nothing to get wrong; the spine stays exact.
- Cons: leaves 5-8% of every round on the floor, and the share only grows as
  the kernels improve.

**Accept on device with the walk as a scalar loop over `child_list`.** One
program per sequence, a dependent scalar load per tree level.
- Pros: the most literal transcription of `Recycler.accept`.
- Cons: about 500 cycles of memory latency per level, times up to 8 levels,
  times a gather per level — 20 to 30 µs a round, worse than the round trip it
  replaces.

**Accept on device, register-resident.** Load the flattened template, the
per-slot match bits and the candidate tokens into registers once, then run the
walk as masked `min`/`sum` reductions over the slot axis.
- Pros: the loop is pure ALU and shuffles, a couple of µs at most; fits in one
  program per sequence with `num_warps=1`.
- Cons: needs the template in CSR form (`child_start`, `child_list`,
  `child_par`) and a vector width of `next_pow2(R + 1)`; the walk is no longer
  a line-by-line copy of the Python.

**Spine: drop it.** Remove `NGramDrafter` from the recycling path.
- Pros: no staleness question at all; less host work per round.
- Cons: gives back whatever the spine was buying (added in #23).

**Spine: write it a round late, unshifted.** Send the continuation the host
last computed and use it as-is.
- Pros: trivial.
- Cons: actively harmful. The pool's first `a` tokens are the ones the round in
  flight just accepted, so depths 1..a of the tree get already-consumed tokens
  and the rank-0 chain — the most likely path — is destroyed.

**Spine: write it a round late, shifted by the device's own token counter.**
Tag the pool with the token count it was taken at; the draft kernel reads it at
`off = nseen - anchor`.
- Pros: the continuation lands where it belongs, so the spine keeps its value;
  `off == 0` reduces exactly to the old behaviour.
- Cons: the shift is only right if the model followed the pool over those
  tokens, so it needs a validity check, and the pool needs `S + maxa + 1` slots.

## Decision

Accept on device, register-resident, as the last kernel of a single per-round
graph: compact → pos → draft → verify → accept. The accept consumes the verify
it follows; the compact at the head of the *next* replay acts on the path it
wrote. The host's only per-round job is to read `acc_tokens`/`acc_count` out of
a pinned double buffer, one round behind, so the next round is already queued
while it reads.

Keep the spine, a round late, shifted by `nseen - anchor`, with
`pool[off - 1] == root` as the validity check — a miss drops the whole spine
back to the adjacency table for that round. `ENGINE_SPINE=0` disables it.

Rounds are queued only when one more is provably needed (the padding floor
still owes rounds, or the round in flight cannot fill the shortest queue even
at `maxa + 1` tokens), so the pipeline never burns a round it does not use.

## Consequences

- `Recycler.accept` stays as the host reference the kernel is tested against
  (`test_accept_kernel_matches_host_walk`), not as production code.
- The freeze test moves into the kernel and stays exact: it reads `pos` after
  the round's own `pos` update, which is the same value `pos_host[b]` held at
  the matching host accept, and it is sticky.
- `g_advance` and `g_verify` are gone from the recycling path, so the plan
  captures one graph there instead of two — less graph-pool memory, which the
  90% memory gate cares about.
- Output tokens are unchanged by construction: verification still decides every
  token, and the accept kernel reproduces `Recycler.accept` node for node.
- The spine is now approximate in a second way (the shift may be stale), which
  can only cost draft quality, never correctness.
