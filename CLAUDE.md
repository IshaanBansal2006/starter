# CLAUDE.md — Kernel Rush (Hack the North 2026, Dryft)

## Implementation override (per ~/projects/CLAUDE.md §6.6 and §8)

On 2026-09-19 the user said: "you write absolutely everything, I don't want to
do any coding, do everything." This project therefore has a **full
implementation override**: Claude writes the engine, kernels, tests, and tooling
directly, including core inference code (attention, KV cache, decode loop,
Triton kernels, speculative decoding). The override is scoped to this repo only.

Still in force: the explanation-first protocol (§6.1, §6.3), pressure-testing
decisions (§6.6), and decision docs (§4). Design decisions are surfaced with
options and the user decides; Claude then implements.

## The task in one paragraph

Decode `Qwen/Qwen3-4B-Instruct-2507` (rev `cdbee75f`) faster than native
Transformers on one H100, BF16, greedy, without changing any output token.
Judge replays our tokens teacher-forced through native Qwen; each token must be
native's argmax or within 2.0 logits. Gates: TTFT and TPOT ≤ 1.10× native,
≤ 25% spread over 5 samples, ≤ 90% GPU memory, 300 s load+warmup, 300 s per
sample. Score = geometric mean of tokens/sec over 3 hidden workloads. Read
`AGENTS.md` and `OPTIMIZATION_GUIDE.md` before touching `engine/`.

## Hard constraints on `engine/`

- Runtime: Python 3.11, CUDA 12.4, torch 2.5.1, triton 3.1.0, transformers
  4.51.3. No network in the container. Python/Triton source only.
- Only `engine/` is submitted. Tests, notes, experiment logs live outside it.
- Never change a formula, only its evaluation order. Cast placement is sacred
  (see `engine/kernels/rmsnorm.py`).
- Every Triton specialization must be launched during warmup.
- `generate` must reset all prompt-dependent state on every call.

## Local environment

- `.venv` (uv, Python 3.11) mirrors the container versions.
- Local GPU is an RTX 4070 laptop, 8 GB. The full 4B model does not fit.
  Kernels and the custom forward are validated against Transformers on a
  tiny random-weight Qwen3 config (`tests/`). Real numbers come only from
  Dryft public runs on the H100.
- CLI: `./bin/dryft`, needs `DRYFT_TOKEN` (put it in `.env`, gitignored).

## Workflow

1. Edit `engine/`, run `pytest tests/` locally.
2. `./bin/dryft validate engine && ./bin/dryft submit engine`, then a
   **public** run. Read tok/s, TTFT ratio, TPOT ratio per workload.
3. Log every attempt in `docs/experiments.md` (what changed, numbers).
4. Only request an **official** run when public ratios have headroom.

## Disclosure

The repo is a public fork (user's explicit choice, 2026-09-19). Nothing here is
unpublished research; it is a hackathon engineering entry.
