# 001 — Custom forward with static KV cache and CUDA graphs
Date: 2026-09-19
Status: accepted

## Context

The competition ("Kernel Rush", Dryft's Qwen3-4B decode benchmark) is judged, not
just benchmarked: after a run, the judge replays every token our engine emitted,
teacher-forced, through native `Qwen/Qwen3-4B-Instruct-2507` (rev `cdbee75f`),
and each token must be native's own argmax or within a 2.0-logit tie margin. The
margin exists only to absorb BF16 reordering noise — it was calibrated by
observing native's own greedy tokens sit up to 0.75 logits below the
teacher-forced replay's argmax at a handful of positions per twelve thousand —
so it is headroom for arithmetic reordering, not license to approximate.

On top of correctness, an official run gates on: TTFT and TPOT each ≤ 1.10x
native's median, ≤ 25% spread across the five official samples, ≤ 90% peak GPU
memory, and load + one warmup ≤ 300 s (300 s per sample too). Score is the
geometric mean of tok/s over three *hidden* workloads; the three public
workloads (batch/prompt/output of (1,512,32), (4,2048,32), (16,512,128)) are
feedback only and never rank.

The runtime is pinned and closed: Python 3.11, CUDA 12.4, torch 2.5.1, triton
3.1.0, transformers 4.51.3, no network in the run container, and the submitted
archive is Python/Triton source only (no compiled binaries, no extra weights
beyond the mounted checkpoint). Numerically, `AGENTS.md` and
`OPTIMIZATION_GUIDE.md` are explicit that we may "reorder arithmetic, never
reformulate" — changing *where* a value gets rounded to BF16 is a
reformulation even when the alternative is more accurate, because it computes
a different function than the one the judge is replaying against (see the
cast-placement comment in `engine/kernels/rmsnorm.py`). Cast placement is
sacred.

## Options considered

**A — Keep Transformers, swap leaf modules.** Wrap individual submodules
(e.g. `Qwen3RMSNorm` → a Triton `FusedRMSNorm`, per `OPTIMIZATION_GUIDE.md`'s
worked example) while leaving `Qwen3ForCausalLM`, its dynamic
`past_key_values`, and its per-step Python dispatch in place.
- Pros: smallest diff from a known-correct baseline; each swap validates
  against the untouched module in isolation; lowest risk of an
  `incorrect_output` failure.
- Cons: `OPTIMIZATION_GUIDE.md` calls out full Python + Transformers dispatch
  as the single largest cost at batch 1 and 4 — leaf swaps don't touch it.
  Dynamic cache growth and per-step host round-trips remain, so there is no
  clean path to CUDA-graph capture (the object graph and control flow change
  shape every call). Unlikely to clear a real speedup, let alone the 1.10x
  TTFT/TPOT gates.

**B — `torch.compile(mode="reduce-overhead")`** over the existing (or lightly
modified) HF forward, letting the compiler fuse ops and capture CUDA graphs
itself.
- Pros: far less hand-written kernel code; a mature path in the general case;
  the compiler, not us, owns fusion correctness.
- Cons: guard-triggered recompilation is exactly the "sometimes takes a slow
  path" hazard `AGENTS.md` warns about for the 25% spread gate, and a
  recompile inside the 300 s load+warmup budget (or inside a timed sample)
  risks a `latency_limit`/`timeout` failure outright. Compiler-chosen fusions
  are opaque to the "cast placement is sacred" rule — we cannot easily audit
  or pin exactly where a fusion rounds to BF16, and a silently different cast
  boundary is precisely the kind of reformulation that can push a logit past
  the 2.0 tie margin on some prompt. Guaranteeing "every Triton specialization
  launched during warmup" is also harder when the compiler decides
  specializations dynamically. The pinned stack (torch 2.5.1 / triton 3.1.0)
  is old enough that `reduce-overhead` mode is less battle-tested than on
  current releases.

**C — Custom forward over static buffers**, with a preallocated KV cache and
hand-captured CUDA graphs for prefill/decode/verify, calling hand-written
Triton kernels where warmup-time measurement says they win. (What
`engine/model.py` and `engine/engine.py` implement.)
- Pros: every buffer, kernel launch, and cast is explicit and inspectable
  against the reference line by line (see the cast-placement comments in
  `engine/kernels/rmsnorm.py`, `add_rmsnorm.py`, `rope.py`). Static shapes
  make CUDA-graph capture direct instead of incidental. Per-shape warmup can
  time cuBLAS against Triton and simply keep the faster one, per operation,
  per shape.
- Cons: by far the most code and validation burden of the three — there is no
  compiler or library backstop, so every fused kernel needs its own numerical
  check against the untouched reference (this is what `tests/test_engine.py`
  and `tests/test_kernels.py` do, on a tiny random-weight config, since the
  local RTX 4070 laptop GPU cannot hold the real 4B model).

## Decision

C. Decode and verification matmuls are chosen per shape at warmup by timing
cuBLAS against a Triton skinny-GEMM/gate-up, including a norm-fused variant
that folds the preceding residual-add + RMSNorm into the GEMM's own prologue
(`pick_matmul`, `pick_gateup`, `pick_normed` in `engine/kernels/gemm.py`), and
decode/verify attention is chosen the same way (`pick_attention` in
`engine/kernels/attention.py`). Prefill keeps cuBLAS for its projections and
calls the reference's own SDPA op directly, since prefill's row count sits far
outside the skinny-GEMM regime the decode kernels are tuned for. Speculative
decoding (n-gram drafting with exact verification, `engine/spec.py`, see
[002](002-speculative-decoding.md)) is implemented but sits behind
`ENGINE_SPEC_K`, off by default, because its win depends on acceptance rate
and its risk is the 25% spread gate — both are properties of real H100 timing
on real prompts, neither of which the local dev GPU can produce.

## Consequences

- Because option C has no compiler or library backstop for correctness, the
  engine carries its own safety net: `Engine.__init__` falls back to a
  verbatim Transformers implementation (`BaselineEngine` in
  `engine/baseline.py`) if loading the custom `Model` throws, and the first
  `generate` call runs an untimed, in-budget self-check
  (`Engine._run_self_check`) that teacher-forces the custom prefill/decode
  against `BaselineEngine` for several steps on the real warmup prompt. If the
  custom engine's greedy token or top-10 logits drift past the judge's own
  2.0-logit tie margin, every subsequent `generate` call for that process is
  served by the baseline instead. This makes the custom path fail toward
  *slower*, never toward *wrong*.
- Every Triton specialization actually used must be launched during warmup,
  before graph capture: `GraphPlan._warm_eager()` runs a prefill, several
  decode steps, and (if speculative decoding is on) one verify pass eagerly,
  and the `pick_*` searches in `Plan.__init__`/`VerifyPlan.__init__` only ever
  run while building a new shape's plan — which only happens during that
  workload's untimed warmup.
- Plans are cached by `(B, T)` in `Engine.plans`, with KV-cache capacity
  padded to `((T + max(max_new, 256) + 63) // 64) * 64` specifically so a
  sample asking for a few more tokens than warmup did still fits without a
  rebuild — a mid-workload rebuild would both cost time and reintroduce the
  "sometimes slow" pattern the spread gate punishes.
- The residual stream stays BF16 end to end (`add_rms_norm` and the
  norm-fused GEMM prologue in `gemm.py` both round their `x + y` sum to BF16
  before writing it back), matching the reference's own residual dtype
  rather than accumulating in a wider type and rounding later.
- Prefill attention calls `F.scaled_dot_product_attention(..., is_causal=True,
  enable_gqa=True)` — the same op the reference uses — optionally pinned to a
  specific backend via `ENGINE_PREFILL_SDPA`; only decode and verification use
  the hand-written split-K Triton attention kernel.
- Both graph capture and kernel selection fail closed, not loud:
  `Engine._plan` catches a failed `plan.capture()` and clears the graph
  handles so the plan runs eagerly instead (same math, no replay), and
  `pick_attention` falls back to `TorchDecodeAttention` — a graph-capturable
  SDPA-with-explicit-mask implementation — if no Triton attention
  configuration compiles or matches the reference within tolerance on the
  run's hardware.
