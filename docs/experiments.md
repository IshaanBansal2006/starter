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

## 2026-09-20 05:21 UTC — official run `353c24ec-8ba2-4ac4-836b-45e83730b278` @ `7f0f98c` — failed

commit 9ca99c0

score: None  failure: incorrect_output The engine's tokens did not match native Qwen's greedy choice.

| workload | status | tok/s | speedup | ttft ratio | tpot ratio | message |
|---|---|---:|---:|---:|---:|---|
| public-0 | passed | 350.0 | 6.98x | 0.53 | 0.13 |  |
| public-1 | passed | 451.0 | 3.10x | 0.66 | 0.22 |  |
| public-2 | passed | 2937.8 | 4.39x | 0.69 | 0.20 |  |

## 2026-09-20 05:32 UTC — official run `823d4533-3e4a-45a0-8e6a-e20add798c26` @ `8db3965` — succeeded

commit 7f0f98c

score: 1109.2767773288615  failure:  

| workload | status | tok/s | speedup | ttft ratio | tpot ratio | message |
|---|---|---:|---:|---:|---:|---|
| public-0 | passed | 333.8 | 11.74x | 0.29 | 0.08 |  |
| public-1 | passed | 518.1 | 5.58x | 0.65 | 0.10 |  |
| public-2 | passed | 3019.1 | 7.32x | 0.69 | 0.11 |  |

## 2026-09-20 09:26 UTC — official run `bc7ba174-8b29-44e0-b221-7782a66fbd9c` @ `7c8a578` — succeeded

rerun of best (92d6da7)

score: 1192.6502770148777  failure:  

| workload | status | tok/s | speedup | ttft ratio | tpot ratio | message |
|---|---|---:|---:|---:|---:|---|
| public-0 | passed | 392.1 | 10.25x | 0.40 | 0.09 |  |
| public-1 | passed | 536.8 | 4.48x | 0.64 | 0.13 |  |
| public-2 | passed | 3380.2 | 5.81x | 0.65 | 0.14 |  |

## 2026-09-20 09:36 UTC — official run `22e03353-fa06-4ccc-a962-4a544e7797f5` @ `7c8a578` — succeeded

rerun of best (3fca11a)

score: 1217.9429253934245  failure:  

| workload | status | tok/s | speedup | ttft ratio | tpot ratio | message |
|---|---|---:|---:|---:|---:|---|
| public-0 | passed | 424.5 | 10.44x | 0.41 | 0.08 |  |
| public-1 | passed | 535.8 | 4.15x | 0.64 | 0.14 |  |
| public-2 | passed | 3451.1 | 5.60x | 0.64 | 0.15 |  |

## 2026-09-20 09:44 UTC — official run `f84f2838-1ae7-4e05-b3b4-7b1fca56373c` @ `7c8a578` — succeeded

median timing, no hysteresis (1228 base)

score: 1225.7615001341674  failure:  

| workload | status | tok/s | speedup | ttft ratio | tpot ratio | message |
|---|---|---:|---:|---:|---:|---|
| public-0 | passed | 403.3 | 7.59x | 0.52 | 0.12 |  |
| public-1 | passed | 543.6 | 3.66x | 0.63 | 0.17 |  |
| public-2 | passed | 3372.3 | 5.05x | 0.64 | 0.17 |  |

## 2026-09-20 09:52 UTC — official run `2390cf29-e544-478d-8dcf-da0b00309f15` @ `7c8a578` — succeeded

rerun 2 of best (3fca11a)

score: 1186.5338591981802  failure:  

| workload | status | tok/s | speedup | ttft ratio | tpot ratio | message |
|---|---|---:|---:|---:|---:|---|
| public-0 | passed | 394.0 | 8.70x | 0.45 | 0.10 |  |
| public-1 | passed | 530.7 | 3.86x | 0.64 | 0.15 |  |
| public-2 | passed | 3360.7 | 5.23x | 0.64 | 0.16 |  |

## 2026-09-20 10:01 UTC — official run `ec3dec86-a494-4d8a-8e56-4ba4c9bcee50` @ `7c8a578` — succeeded

rerun 3 of best (3fca11a)

score: 1181.6512931189113  failure:  

| workload | status | tok/s | speedup | ttft ratio | tpot ratio | message |
|---|---|---:|---:|---:|---:|---|
| public-0 | passed | 374.3 | 9.26x | 0.41 | 0.10 |  |
| public-1 | passed | 517.5 | 4.09x | 0.64 | 0.14 |  |
| public-2 | passed | 3340.0 | 6.01x | 0.64 | 0.14 |  |

## 2026-09-20 10:10 UTC — official run `cfa557f8-52ba-4827-b305-74752c7f2b59` @ `7c8a578` — succeeded

rerun of 7c8a578 (median timing base)

score: 1224.8272892223451  failure:  

| workload | status | tok/s | speedup | ttft ratio | tpot ratio | message |
|---|---|---:|---:|---:|---:|---|
| public-0 | passed | 425.2 | 11.16x | 0.38 | 0.08 |  |
| public-1 | passed | 539.3 | 4.79x | 0.63 | 0.12 |  |
| public-2 | passed | 3516.7 | 6.50x | 0.63 | 0.13 |  |

## 2026-09-20 10:19 UTC — official run `615d43c1-c813-473d-a2fe-d50cc2153578` @ `7c8a578` — succeeded

rerun 4 of best (3fca11a)

score: 1230.3939429523302  failure:  

| workload | status | tok/s | speedup | ttft ratio | tpot ratio | message |
|---|---|---:|---:|---:|---:|---|
| public-0 | passed | 399.1 | 9.22x | 0.47 | 0.10 |  |
| public-1 | passed | 553.3 | 4.20x | 0.63 | 0.13 |  |
| public-2 | passed | 3555.1 | 5.84x | 0.64 | 0.14 |  |

## 2026-09-20 10:28 UTC — official run `9a571964-2ee4-4885-8c00-9d97d527830a` @ `7c8a578` — succeeded

rerun 5 of best (3fca11a)

score: 1223.9902890477847  failure:  

| workload | status | tok/s | speedup | ttft ratio | tpot ratio | message |
|---|---|---:|---:|---:|---:|---|
| public-0 | passed | 354.1 | 10.03x | 0.37 | 0.09 |  |
| public-1 | passed | 528.4 | 4.66x | 0.64 | 0.12 |  |
| public-2 | passed | 3381.2 | 6.06x | 0.65 | 0.14 |  |
