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

## 2026-09-20 03:33 UTC — official run `f0f62422-9c3b-4476-98e1-7d7a73826f1d` @ `ae33a6e` — succeeded

v1 default config, first run on the H100 (push-created submission)

score: 763.5456265036153  failure:  

| workload | status | tok/s | speedup | ttft ratio | tpot ratio | message |
|---|---|---:|---:|---:|---:|---|
| public-0 | passed | 201.0 | 4.42x | 0.44 | 0.22 |  |
| public-1 | passed | 415.5 | 3.02x | 0.62 | 0.25 |  |
| public-2 | passed | 2468.4 | 3.80x | 0.59 | 0.24 |  |

## 2026-09-20 03:44 UTC — official run `b6937394-23b3-4aab-8939-63bf154c1f5b` @ `1f5175f` — succeeded

spec K=3 default (n-gram drafts), otherwise v1

score: 754.1165863097582  failure:  

| workload | status | tok/s | speedup | ttft ratio | tpot ratio | message |
|---|---|---:|---:|---:|---:|---|
| public-0 | passed | 193.6 | 6.89x | 0.25 | 0.14 |  |
| public-1 | passed | 396.5 | 4.34x | 0.61 | 0.17 |  |
| public-2 | passed | 2704.2 | 6.67x | 0.59 | 0.13 |  |

## 2026-09-20 03:44 UTC — official run `b6937394-23b3-4aab-8939-63bf154c1f5b` @ `1f5175f` — succeeded

commit 396f4a9

score: 754.1165863097582  failure:  

| workload | status | tok/s | speedup | ttft ratio | tpot ratio | message |
|---|---|---:|---:|---:|---:|---|
| public-0 | passed | 193.6 | 6.89x | 0.25 | 0.14 |  |
| public-1 | passed | 396.5 | 4.34x | 0.61 | 0.17 |  |
| public-2 | passed | 2704.2 | 6.67x | 0.59 | 0.13 |  |

## 2026-09-20 03:52 UTC — official run `d604c2ee-05e4-4646-9695-4e2ef2b72301` @ `1f5175f` — succeeded

commit adc053d

score: 764.2689895695596  failure:  

| workload | status | tok/s | speedup | ttft ratio | tpot ratio | message |
|---|---|---:|---:|---:|---:|---|
| public-0 | passed | 180.5 | 5.02x | 0.34 | 0.19 |  |
| public-1 | passed | 386.0 | 3.41x | 0.61 | 0.22 |  |
| public-2 | passed | 2648.0 | 5.08x | 0.59 | 0.18 |  |

## 2026-09-20 03:59 UTC — official run `3e4946f1-f3bf-446a-9661-8c3f8e2b88be` @ `1f5175f` — succeeded

commit 1f5175f

score: 895.4586535213505  failure:  

| workload | status | tok/s | speedup | ttft ratio | tpot ratio | message |
|---|---|---:|---:|---:|---:|---|
| public-0 | passed | 215.0 | 3.88x | 0.48 | 0.25 |  |
| public-1 | passed | 422.7 | 2.78x | 0.61 | 0.28 |  |
| public-2 | passed | 2806.9 | 4.19x | 0.58 | 0.22 |  |

## 2026-09-20 04:11 UTC — official run `8521efe1-3cc1-4ecd-8de5-1c66d276118c` @ `1282d00` — succeeded

commit 2c3892d

score: 977.1500695109738  failure:  

| workload | status | tok/s | speedup | ttft ratio | tpot ratio | message |
|---|---|---:|---:|---:|---:|---|
| public-0 | passed | 325.0 | 7.42x | 0.46 | 0.12 |  |
| public-1 | passed | 429.2 | 3.27x | 0.70 | 0.20 |  |
| public-2 | passed | 2658.2 | 4.51x | 0.68 | 0.19 |  |

## 2026-09-20 04:19 UTC — official run `8fbe09a1-11f0-4900-9785-df01aa71dd8e` @ `93201ec` — failed

commit 1282d00

score: None  failure: incorrect_output The engine's tokens did not match native Qwen's greedy choice.

| workload | status | tok/s | speedup | ttft ratio | tpot ratio | message |
|---|---|---:|---:|---:|---:|---|
| public-0 | passed | 336.8 | 5.90x | 0.55 | 0.16 |  |
| public-1 | passed | 459.4 | 3.04x | 0.71 | 0.21 |  |
| public-2 | passed | 2752.6 | 4.12x | 0.69 | 0.21 |  |

## 2026-09-20 04:31 UTC — official run `9453d270-b697-4e50-afa5-2bb7d1f969dd` @ `51756fc` — succeeded

commit 93201ec

score: 985.5187265426272  failure:  

| workload | status | tok/s | speedup | ttft ratio | tpot ratio | message |
|---|---|---:|---:|---:|---:|---|
| public-0 | passed | 332.0 | 12.03x | 0.27 | 0.08 |  |
| public-1 | passed | 436.5 | 4.80x | 0.70 | 0.12 |  |
| public-2 | passed | 2758.9 | 6.86x | 0.69 | 0.12 |  |

## 2026-09-20 04:34 UTC — official run `cee14abd-9f47-4ca7-8fc2-a85a69db381b` @ `51756fc` — failed

commit a0c2df9

score: None  failure: candidate_error The model benchmark could not complete; detailed diagnostics are available to operators.

| workload | status | tok/s | speedup | ttft ratio | tpot ratio | message |
|---|---|---:|---:|---:|---:|---|

## 2026-09-20 05:11 UTC — official run `19001d22-f5e5-4d33-b743-c74935d86340` @ `9ca99c0` — succeeded

commit a853c8b

score: 1085.2874764441376  failure:  

| workload | status | tok/s | speedup | ttft ratio | tpot ratio | message |
|---|---|---:|---:|---:|---:|---|
| public-0 | passed | 369.9 | 6.63x | 0.54 | 0.14 |  |
| public-1 | passed | 460.9 | 3.04x | 0.70 | 0.21 |  |
| public-2 | passed | 3003.7 | 4.50x | 0.68 | 0.19 |  |
