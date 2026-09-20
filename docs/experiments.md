# Experiments

Every Dryft run is appended below automatically by `agent/dryft_cli.py`
(`log_experiment`, called from its `go`/`run`/`result` subcommands): each
entry is a timestamped heading naming the run id, git revision, and mode,
followed by a per-workload table of status, tok/s, speedup over native, and
TTFT/TPOT ratios. The numbers are the platform's, read back from its API, not
locally estimated.

## Planned A/B toggles

The platform has no way to set an environment variable per run — there is no
manifest and no per-run config, just `engine.py` and whatever it imports (see
`QWEN_ENGINE_CONTRACT.md`). So each toggle below is exercised by committing a
one-line default change in `engine/engine.py` or `engine/model.py` (flip the
default, submit a public run, record the result here, then flip it back or
keep it), not by setting the variable at submission time.

| Env var | Default | What it tests |
| --- | --- | --- |
| `ENGINE_SPEC_K` | `0` (off) | Speculative-decode draft length `K`. `0` disables speculation; `>0` verifies `K` n-gram-drafted tokens per round against one real forward. |
| `ENGINE_SPEC_MAX_ROWS` | `64` | Caps `B * (K + 1)` before speculative decoding is allowed for a shape; above this it falls back to plain greedy so the verify pass doesn't fall off the skinny-GEMM/attention regime those kernels are tuned for. |
| `ENGINE_FORCE_CUBLAS` | unset (off) | Skips the warmup-timed Triton skinny-GEMM/gate-up search for decode and verify projections and always uses cuBLAS, to isolate how much the Triton path is actually winning. |
| `ENGINE_ROPE_FUSED` | `1` (on) | Whether per-head Q/K RMSNorm + RoPE + KV-cache write run as one fused Triton launch or two separate ones. |
| `ENGINE_ATTN_DEFAULT` | unset (off) | Skips the warmup search over attention kernel configs/split counts (`pick_attention`) and uses the untuned default `DecodeAttention` directly. |
| `ENGINE_PREFILL_SDPA` | unset (framework default) | Pins prefill's `scaled_dot_product_attention` backend to `flash`, `cudnn`, or `efficient` instead of letting PyTorch choose. |
| `ENGINE_NO_GRAPHS` | unset (graphs on) | Disables CUDA graph capture/replay entirely; every step runs eagerly through the same kernels. Useful for isolating whether a regression is in the kernels themselves or in graph capture/replay. |
| `ENGINE_NORM_FUSED` | `1` (on) | Whether the residual-add + RMSNorm ahead of a decode/verify projection is fused into that GEMM kernel's own prologue (`pick_normed`) or run as a separate kernel launch before an unfused matmul. Found while reading the current `engine/model.py` / `engine/kernels/gemm.py` (not in the original toggle list) — same kind of fusion A/B as the others, included here for completeness. |

Each row is a hypothesis, not a result: "what it tests" is what the toggle
isolates, not a claim that it wins. Flip one at a time, keep everything else
fixed, and log the public-run numbers here before ever requesting an official
run on the change.

## Leaderboard snapshot 2026-09-19 23:25 EDT

Top: Segfault 1280.4 tok/s; places 2-5 at 1137-1144; 56 ranked teams.
