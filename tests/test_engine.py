"""Engine vs Transformers 4.51.3 on a tiny checkpoint, teacher-forced.

The judge replays our tokens through native Qwen and compares logits; these
tests do the same locally: feed both models the same prefix and require the
per-position logits to agree to bf16 noise, then require identical argmax
streams from ``Engine.generate`` and from the pure-eager path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tiny import load_reference, make_tiny

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

TINY = Path(__file__).resolve().parent / "_tiny_model"


@pytest.fixture(scope="module")
def tiny_path() -> Path:
    return make_tiny(TINY)


@pytest.fixture(scope="module")
def reference(tiny_path):
    return load_reference(tiny_path)


@pytest.fixture(scope="module")
def engine(tiny_path):
    from engine import Engine
    return Engine(str(tiny_path))


def reference_logits(reference, ids: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        return reference(input_ids=ids, use_cache=False, return_dict=True).logits.float()


def prompts(B: int, T: int, vocab: int, seed: int) -> list[list[int]]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (B, T), generator=g).tolist()


def assert_close(mine: torch.Tensor, ref: torch.Tensor, where: str, tol: float = 0.03) -> None:
    scale = ref.abs().max().item()
    err = (mine.float() - ref).abs().max().item()
    assert err <= tol * scale, f"{where}: max|diff| {err:.4g} vs scale {scale:.4g}"


@pytest.mark.parametrize("B,T,new", [(1, 37, 5), (3, 130, 6), (2, 300, 4)])
def test_teacher_forced_logits(engine, reference, B, T, new):
    from engine import GraphPlan
    vocab = engine.model.cfg.vocab
    ids = prompts(B, T, vocab, seed=B * 1000 + T)
    plan = GraphPlan(engine.model, B, T, new).plan
    plan.ids.copy_(torch.tensor(ids))
    seq = torch.tensor(ids, device="cuda:0")
    ref = reference_logits(reference, seq)
    mine = plan.prefill()
    assert_close(mine, ref[:, -1], "prefill last position")
    forced = ref[:, -1].argmax(-1)
    for step in range(new - 1):
        seq = torch.cat([seq, forced[:, None]], dim=1)
        ref = reference_logits(reference, seq)
        plan.tok.copy_(forced)
        mine = plan.decode()
        assert_close(mine, ref[:, -1], f"decode step {step}")
        forced = ref[:, -1].argmax(-1)


def test_generate_matches_eager_and_reference(engine, reference):
    B, T, new = 2, 64, 8
    vocab = engine.model.cfg.vocab
    ids = prompts(B, T, vocab, seed=7)
    out = list(engine.generate(ids, new))
    assert len(out) == new and all(len(step) == B for step in out)
    from engine import GraphPlan
    eager = GraphPlan(engine.model, B, T, new)
    eager_out = list(eager.run(ids, new))
    assert out == eager_out, "CUDA graph replay diverged from the eager path"
    seq = torch.tensor(ids, device="cuda:0")
    for step, toks in enumerate(out):
        ref = reference_logits(reference, seq)[:, -1]
        top = ref.max(-1).values
        mine = ref[torch.arange(B), torch.tensor(toks, device="cuda:0")]
        assert torch.all(top - mine <= 2.0), f"step {step}: token outside the tie margin"
        seq = torch.cat([seq, torch.tensor(toks, device="cuda:0")[:, None]], dim=1)


def test_second_call_resets_cache(engine):
    B, T, new = 2, 64, 8
    vocab = engine.model.cfg.vocab
    a = prompts(B, T, vocab, seed=1)
    b = prompts(B, T, vocab, seed=2)
    first_b = list(engine.generate(b, new))
    list(engine.generate(a, new))
    again_b = list(engine.generate(b, new))
    assert first_b == again_b


def test_longer_output_reuses_plan(engine):
    B, T = 2, 40
    vocab = engine.model.cfg.vocab
    ids = prompts(B, T, vocab, seed=11)
    list(engine.generate(ids, 4))
    plan = engine.plans[(B, T)]
    out = list(engine.generate(ids, 12))
    assert engine.plans[(B, T)] is plan
    assert len(out) == 12
