"""Kernel Rush engine for Qwen3 4B: custom forward, static KV cache, CUDA graphs.

Per shape (batch, prompt length, output length) the engine builds a ``Plan``
holding every buffer, runs it once eagerly so all Triton specialisations are
compiled, then captures the prefill and the decode step into CUDA graphs. A
sample is then one prefill replay plus ``max_new_tokens - 1`` decode replays,
each followed by a single device-to-host copy of the ``B`` chosen tokens.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _ensure_triton_cache() -> None:
    """Triton compiles at runtime and needs a writable cache; the run container's
    home directory may not be writable for the unprivileged engine user."""
    target = os.environ.get("TRITON_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".triton", "cache")
    try:
        os.makedirs(target, exist_ok=True)
        probe = os.path.join(target, ".write-probe")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
    except OSError:
        os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="triton-cache-")


_ensure_triton_cache()

import torch

try:
    import numpy as np
except ImportError:  # the container ships numpy with transformers; this only keeps the engine importable without it
    np = None

from model import Model, Plan, VerifyPlan
from spec import NGramDrafter


def _log(msg: str) -> None:
    print(f"[engine] {msg}", file=sys.stderr, flush=True)


def _ids_tensor(input_ids: list[list[int]]) -> torch.Tensor:
    """Nested-list to int64 tensor; numpy walks the lists several times faster than torch.tensor."""
    if np is not None:
        return torch.from_numpy(np.asarray(input_ids, dtype=np.int64))
    return torch.tensor(input_ids, dtype=torch.int64)


class GraphPlan:
    def __init__(self, model: Model, B: int, T: int, max_new: int, spec_k: int | None = None):
        self.plan = Plan(model, B, T, max_new)
        self.B, self.T, self.max_new = B, T, max_new
        self.g_prefill: torch.cuda.CUDAGraph | None = None
        self.g_decode: torch.cuda.CUDAGraph | None = None
        self.g_verify: torch.cuda.CUDAGraph | None = None
        self.host_tok = torch.empty((2, B), dtype=torch.int64, pin_memory=True)
        self.events = [torch.cuda.Event() for _ in range(2)]
        self.spec_k = spec_k
        self.verify: VerifyPlan | None = None
        self.stats: dict[str, float] = {}
        if spec_k:
            self.verify = VerifyPlan(self.plan, spec_k)
            self.cand = torch.empty((B, spec_k + 1), dtype=torch.int64, device=model.device)
            self.host_blk = torch.empty((B, spec_k + 1), dtype=torch.int64, pin_memory=True)
            self.host_pos = torch.empty((B,), dtype=torch.int32, pin_memory=True)
            self.host_cand = torch.empty((B, spec_k + 1), dtype=torch.int64, pin_memory=True)

    def _warm_eager(self, steps: int = 3) -> None:
        plan = self.plan
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            plan.tok.copy_(plan.prefill().argmax(dim=-1))
            for _ in range(steps):
                plan.tok.copy_(plan.decode().argmax(dim=-1))
            if self.verify is not None:
                self.verify.pos.copy_(plan.pos)
                self.verify.blk.copy_(plan.tok[:, None].expand(-1, self.verify.R))
                self.cand.copy_(self.verify.verify())
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

    def capture(self) -> None:
        plan = self.plan
        t0 = time.perf_counter()
        self._warm_eager()
        self.g_prefill = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g_prefill):
            plan.tok.copy_(plan.prefill().argmax(dim=-1))
        self.g_decode = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g_decode):
            plan.tok.copy_(plan.decode().argmax(dim=-1))
        if self.verify is not None:
            self.g_verify = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g_verify):
                self.cand.copy_(self.verify.verify())
        torch.cuda.synchronize()
        _log(f"captured graphs for B={self.B} T={self.T} new={self.max_new} in {time.perf_counter() - t0:.1f}s")

    def _step_prefill(self) -> None:
        if self.g_prefill is None:
            self.plan.tok.copy_(self.plan.prefill().argmax(dim=-1))
        else:
            self.g_prefill.replay()

    def _step_decode(self) -> None:
        if self.g_decode is None:
            self.plan.tok.copy_(self.plan.decode().argmax(dim=-1))
        else:
            self.g_decode.replay()

    def _step_verify(self) -> None:
        if self.g_verify is None:
            self.cand.copy_(self.verify.verify())
        else:
            self.g_verify.replay()

    def run_spec(self, input_ids: list[list[int]], max_new_tokens: int):
        """Speculative loop: verify K drafts per sequence per round, yield steps
        as soon as every sequence has a token for them. Output is identical to
        plain greedy decode because only model-predicted tokens are kept."""
        plan, ver = self.plan, self.verify
        B, K, R = self.B, self.spec_k, self.verify.R
        plan.ids.copy_(_ids_tensor(input_ids))
        self._step_prefill()
        first = plan.tok.tolist()
        yield first
        yielded = 1
        queues = [[first[b]] for b in range(B)]
        drafters = [NGramDrafter(input_ids[b] + [first[b]], K) for b in range(B)]
        pos = [self.T] * B
        blk = [[first[b]] + drafters[b].draft() for b in range(B)]
        rounds = accepted = 0
        while yielded < max_new_tokens:
            self.host_blk.copy_(torch.tensor(blk, dtype=torch.int64))
            self.host_pos.copy_(torch.tensor(pos, dtype=torch.int32))
            ver.blk.copy_(self.host_blk, non_blocking=True)
            ver.pos.copy_(self.host_pos, non_blocking=True)
            self._step_verify()
            self.host_cand.copy_(self.cand, non_blocking=True)
            torch.cuda.current_stream().synchronize()
            cand = self.host_cand.tolist()
            rounds += 1
            for b in range(B):
                if len(queues[b]) >= max_new_tokens:
                    continue
                a = 0
                while a < K and blk[b][a + 1] == cand[b][a]:
                    a += 1
                new = cand[b][:a + 1]
                queues[b].extend(new)
                drafters[b].extend(new)
                accepted += a
                pos[b] += a + 1
                blk[b] = [cand[b][a]] + drafters[b].draft()
            while yielded < max_new_tokens and all(len(q) > yielded for q in queues):
                yield [q[yielded] for q in queues]
                yielded += 1
        self.stats = {"rounds": rounds, "accepted": accepted, "steps": max_new_tokens}
        _log(f"spec: {rounds} rounds for {max_new_tokens} steps x {B} seqs, {accepted} drafts accepted "
             f"({accepted / max(1, rounds * B * K):.2f} per draft slot)")

    def run(self, input_ids: list[list[int]], max_new_tokens: int):
        """Yield one token list per step, keeping the GPU one step ahead of the host.

        Step t's tokens are copied to pinned host memory and fenced with an
        event; step t+1 is launched *before* waiting on that event, so the
        harness's read of step t overlaps the compute of step t+1.
        """
        if self.verify is not None:
            yield from self.run_spec(input_ids, max_new_tokens)
            return
        plan = self.plan
        plan.ids.copy_(_ids_tensor(input_ids))
        host = self.host_tok
        events = self.events
        self._step_prefill()
        host[0].copy_(plan.tok, non_blocking=True)
        events[0].record()
        for step in range(1, max_new_tokens):
            cur, prev = step % 2, (step - 1) % 2
            self._step_decode()
            host[cur].copy_(plan.tok, non_blocking=True)
            events[cur].record()
            events[prev].synchronize()
            yield host[prev].tolist()
        last = (max_new_tokens - 1) % 2
        events[last].synchronize()
        yield host[last].tolist()


class Engine:
    def __init__(self, model_path: str) -> None:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        t0 = time.perf_counter()
        self.model = Model(model_path)
        torch.cuda.synchronize()
        _log(f"loaded {model_path} in {time.perf_counter() - t0:.1f}s")
        self.plans: dict[tuple[int, int], GraphPlan] = {}
        self.use_graphs = os.environ.get("ENGINE_NO_GRAPHS") is None
        self.spec_k = int(os.environ.get("ENGINE_SPEC_K", "0")) or None
        self.spec_max_rows = int(os.environ.get("ENGINE_SPEC_MAX_ROWS", "64"))

    def _plan(self, B: int, T: int, max_new: int) -> GraphPlan:
        key = (B, T)
        plan = self.plans.get(key)
        if plan is not None and T + max_new > plan.plan.cap:
            _log(f"max_new_tokens={max_new} exceeds planned capacity {plan.plan.cap}; rebuilding")
            plan = None
        if plan is None:
            self.plans.clear()
            torch.cuda.empty_cache()
            spec_k = self.spec_k if self.spec_k and B * (self.spec_k + 1) <= self.spec_max_rows else None
            plan = GraphPlan(self.model, B, T, max_new, spec_k=spec_k)
            if self.use_graphs:
                try:
                    plan.capture()
                except Exception as exc:  # eager execution is slower but produces the same tokens
                    _log(f"CUDA graph capture failed ({exc!r}); running eagerly")
                    plan.g_prefill = plan.g_decode = plan.g_verify = None
                    torch.cuda.synchronize()
            self.plans[key] = plan
        return plan

    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        B, T = len(input_ids), len(input_ids[0])
        if any(len(row) != T for row in input_ids):
            raise ValueError("all prompts in a batch must have the same length")
        if max_new_tokens < 1:
            return
        plan = self._plan(B, T, max_new_tokens)
        yield from plan.run(input_ids, max_new_tokens)
