# 002 — N-gram speculative decoding with exact verification
Date: 2026-09-19
Status: accepted

## Context

Decode emits one token per sequence per forward pass, and every decode-step
matmul is a GEMM with `M` (row count) tied to batch size against the full
weight matrix. At BF16, Qwen3-4B's roughly 4B parameters are about 8 GB of
weights that must stream from HBM on every step, so a lone decode step is
weight-bandwidth bound rather than compute bound — this is exactly why
`engine/kernels/gemm.py` describes decode's projections as "pure weight
streaming" and `pick_matmul`/`pick_gateup` report their winner in GB/s, not
GFLOP/s. Because the cost is dominated by reading the weights once rather than
by how many query rows ride along, a verify pass over `K + 1` rows
(`engine/model.py`'s `VerifyPlan`, `R = K + 1`) costs about the same as one
ordinary decode step, as long as `R` stays small enough to stay inside the
skinny-GEMM/attention regime those kernels are tuned for. That gap — many
candidate tokens checked for roughly the price of one — is what makes
speculative decoding worth considering here at all.

Rule 3 of the contract (`AGENTS.md`, `QWEN_ENGINE_CONTRACT.md`) is unforgiving
of anything that changes the answer: quantization, cache eviction,
approximate/sparse attention, and an unverified draft model are all explicitly
called out as disallowed because they shift logits by whole units. Any
speculative scheme here has to keep the judge's teacher-forced replay passing
by construction, not by getting lucky on the tie margin.

## Options considered

**No speculation.** Plain greedy decode, one token per step.
- Pros: nothing to get wrong; no extra verify pass, no extra state to reset
  between `generate` calls.
- Cons: leaves the "many tokens per weight-read" opportunity on the table
  entirely.

**Prompt-lookup / n-gram drafts.** Draft the next `K` tokens by looking up the
longest suffix of the sequence generated so far inside that same sequence,
and reusing whatever followed it last time — no second model, no extra
weights.
- Pros: zero extra parameters (trivially satisfies "no weights in archive"),
  draft cost is a Python dict lookup rather than a GPU forward, and it works
  well precisely on the kind of repetitive or structured continuations decode
  benchmarks tend to produce.
- Cons: acceptance rate depends entirely on how self-repetitive the generated
  text is — on genuinely novel continuations the draft degenerates to
  repeating the last token, so the win is prompt-dependent by construction.

**Layer-skip self-draft.** Draft by running a subset of the model's own
layers (early-exit) for a cheap approximate next token, then verify with the
full model.
- Pros: still uses only the model's own weights, and unlike n-gram lookup can
  propose plausible tokens on novel text since it is still consulting the
  model.
- Cons: needs its own calibration of which layers to skip and how that
  interacts with this specific checkpoint's weights, plus a second code path
  through the transformer stack (partial stack vs. full) to build, validate,
  and keep in sync with every future change to the full forward — meaningfully
  more surface area than a host-side lookup table.

**Lookahead decoding.** Generate multiple candidate n-grams per step via
Jacobi-style parallel iteration and verify them together.
- Pros: does not depend on the generated text already containing repeats the
  way prompt-lookup does; a published technique with demonstrated speedups.
- Cons: materially more implementation and tuning surface (n-gram pool
  management, Jacobi trajectory bookkeeping) than a suffix-lookup table, for a
  benchmark whose workloads are short enough (32-128 output tokens) that a
  lookahead trajectory may have limited room to pay off its own setup cost.

**Draft model.** A smaller model proposes tokens; the main model verifies
them.
- Not allowed: the archive ships Python/Triton source only, with no weights
  beyond the mounted checkpoint and no network in the run container — there
  is nowhere to put a second model's parameters.

## Decision

N-gram (prompt-lookup) drafts with exact verification. `engine/spec.py`'s
`NGramDrafter` proposes `K` tokens from the longest matching earlier suffix
(falling back to repeating the last token when nothing matches), and
`engine/model.py`'s `VerifyPlan` runs one real forward over all `K + 1`
candidate rows through a block-causal multi-row attention kernel
(`engine/kernels/attention.py`'s `DecodeAttention` with `R = K + 1`).
`GraphPlan.run_spec` in `engine/engine.py` then does per-round, host-side
acceptance: it walks the verified rows and keeps the longest prefix that
matches what the model itself predicted, so the emitted sequence is
byte-for-byte what plain greedy decode would have emitted — verification is
exact, not approximate, so this passes rule 3 by construction rather than by
tolerance. It ships behind `ENGINE_SPEC_K` (default `0`, i.e. off).

## Consequences

- Acceptance rate — and therefore the actual speedup — varies per prompt by
  construction (repetitive continuations accept long drafts, novel ones fall
  back to near one token per round), which is exactly the "cost varies with
  the prompt" pattern `AGENTS.md` flags as a spread-gate risk rather than a
  latency-gate risk.
- Because verification is exact, correctness never depends on acceptance
  rate: a run with poor acceptance is merely slow (down toward plain-decode
  speed, one round per token), never wrong.
- Only enable this (`ENGINE_SPEC_K` > 0) once public runs on the real H100
  show a throughput gain on the workloads we can see, and — since public runs
  report but do not enforce the latency gates — a manual check that the
  official 25% spread gate still holds; both acceptance rate and spread are
  properties of real timing that the local tiny-model tests cannot measure.
