# Kernel Rush (Hack the North 2026) — final report

Team Krxfty · repo `IshaanBansal2006/starter` · engine in `engine/` (Python + Triton only).

## Result

| | tokens/sec (geometric mean over the six private workloads) |
|---|---:|
| Native Transformers baseline | ~100 (the platform's reference point) |
| First custom engine (23:33 EDT) | 763.5 |
| Best official run (06:19 EDT, run 615d43c1, commit 3fca11a) | **1230.4** |

Public workloads at the best run: batch 1 / 512-token prompt / 32 out: **399 tok/s**;
batch 4 / 2048 / 32: **553 tok/s**; batch 16 / 512 / 128: **3555 tok/s**. Every token
is native Qwen's greedy choice (exact acceptance; the 2-logit tie margin is not exploited).

## What the engine does

- Direct safetensors load, fused QKV and gate/up weights, static per-layer KV cache.
- Triton kernels: fused q/k-RMSNorm + RoPE + cache write; split-K GQA flash-decode with
  R query rows per sequence and ancestor-mask (tree) attention; skinny GEMMs with the
  residual add + RMSNorm folded into the prologue and SwiGLU into the epilogue, one wide
  M tile per pass on a persistent grid; a bandwidth-bound exact-argmax / approximate top-8 kernel.
- Every kernel choice (cuBLAS vs each Triton config, attention block/split) is timed at
  warmup on the H100 over rotating layer weights, as a captured CUDA graph, median of five.
- Speculative decoding by token recycling: draft trees grown from the model's own top-8
  next tokens (adjacency table warmed from the prompt's last 1024 positions), an n-gram
  spine on the rank-0 chain, verified in one CUDA-graph round with acceptance, KV
  compaction and the next draft all on device. Trees: 64 / 16 / 8 nodes at batch 1 / 4 / 16
  with a flat rank prior; minimum round counts per sample keep timing data-independent.
- Warmup self-check against Transformers once per container, a speculative self-check per
  process against the plain path, and a native-baseline fallback that continues on our own prefix.

## What moved the score (measured on the platform)

| change | score |
|---|---:|
| custom forward + static cache + CUDA graphs | 763.5 |
| L2-aware warmup picker (rotating weights) | 895.5 |
| token-recycling trees + fixed rounds | 977.2 |
| persistent-grid GEMMs, graph-timed picker | 985.5 |
| n-gram spine, per-batch tree sizes / floors | 1109.3 |
| strided `row_max` bug fixed; wide M tiles; fast top-k; flat prior; 8-node batch-16 trees | 1185.1 |
| two 128-column GEMM tile candidates | 1221.4 |
| deeper-pipeline attention candidates | 1228.1 |
| rerun of the same submission (hidden-set variance ±2.5%) | 1230.4 |

Measured and rejected: chain-only n-gram speculation (754), evict-first weight loads (−2%),
smaller batch-1 trees (−4%), margin acceptance of near-tie drafts (−2 to −3% at both 0.6 and
0.3 logits: a near-tie draft steers the trajectory away from what the recycling table predicts),
picker hysteresis (−2%), three extra large GEMM tiles (−2% on the hidden set), cuDNN prefill
attention (no change), 64-row attention blocks (crashed on Hopper), 32-row M-tiled GEMMs
(streamed weights twice), removing the round floors (1212, neutral: the natural batch-1 median
is ~13 rounds, and the spread gate accepted a 24% p10-p90 range).

## Where the remaining time goes

A verify round costs ~5.0-5.4 ms at every batch size against a 2.7 ms weight-stream floor
(8 GB of BF16 weights per round); the residue is kernel launch latency and GEMM efficiency.
Acceptance is ~2.5 tokens per round at batch 1, ~1.4 per sequence at batch 4 (the slowest of
four sequences paces the batch), ~1.4 at batch 16. Prefill for 2048-token prompts is
cuBLAS-bound at ~128 ms. Identical-code reruns spread from 1182 to 1230, so differences under
~3% between configurations are not resolvable in one run.

## Reproduce

`.venv/bin/python -m pytest tests/` (tiny random Qwen3 on any 8 GB GPU), `tools/local_eval.py`
(Qwen3-0.6B acceptance proxy), `agent/dryft_cli.py` (runs, results, leaderboard),
`docs/experiments.md` (every run), `docs/decisions/` (ADRs), `explanations/` (per-file study docs).
