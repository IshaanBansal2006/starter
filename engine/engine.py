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

from model import Model, Plan


def _log(msg: str) -> None:
    print(f"[engine] {msg}", file=sys.stderr, flush=True)


class GraphPlan:
    def __init__(self, model: Model, B: int, T: int, max_new: int):
        self.plan = Plan(model, B, T, max_new)
        self.B, self.T, self.max_new = B, T, max_new
        self.g_prefill: torch.cuda.CUDAGraph | None = None
        self.g_decode: torch.cuda.CUDAGraph | None = None
        self.host_tok = torch.empty((2, B), dtype=torch.int64, pin_memory=True)
        self.events = [torch.cuda.Event() for _ in range(2)]

    def _warm_eager(self, steps: int = 3) -> None:
        plan = self.plan
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            plan.tok.copy_(plan.prefill().argmax(dim=-1))
            for _ in range(steps):
                plan.tok.copy_(plan.decode().argmax(dim=-1))
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

    def run(self, input_ids: list[list[int]], max_new_tokens: int):
        """Yield one token list per step, keeping the GPU one step ahead of the host.

        Step t's tokens are copied to pinned host memory and fenced with an
        event; step t+1 is launched *before* waiting on that event, so the
        harness's read of step t overlaps the compute of step t+1.
        """
        plan = self.plan
        plan.ids.copy_(torch.tensor(input_ids, dtype=torch.int64), non_blocking=False)
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
        self.plans: dict[tuple[int, int, int], GraphPlan] = {}
        self.use_graphs = os.environ.get("ENGINE_NO_GRAPHS") is None

    def _plan(self, B: int, T: int, max_new: int) -> GraphPlan:
        key = (B, T)
        plan = self.plans.get(key)
        if plan is not None and T + max_new > plan.plan.cap:
            _log(f"max_new_tokens={max_new} exceeds planned capacity {plan.plan.cap}; rebuilding")
            plan = None
        if plan is None:
            self.plans.clear()
            torch.cuda.empty_cache()
            plan = GraphPlan(self.model, B, T, max_new)
            if self.use_graphs:
                plan.capture()
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
