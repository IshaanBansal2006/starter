"""Drive the engine the way the platform does: a fresh interpreter, ``engine/``
as the import root, one warmup generation, then samples of the same shape."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tiny import make_tiny

ROOT = Path(__file__).resolve().parents[1]
TINY = Path(__file__).resolve().parent / "_tiny_model"

DRIVER = r"""
import json, sys, time
sys.path.insert(0, sys.argv[1])
from engine import Engine
model_path, B, T, new = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
import torch
eng = Engine(model_path)
g = torch.Generator().manual_seed(0)
def sample(seed):
    return torch.randint(0, 1024, (B, T), generator=torch.Generator().manual_seed(seed)).tolist()
t0 = time.perf_counter(); warm = list(eng.generate(sample(1), new)); t_warm = time.perf_counter() - t0
outs = []
for seed in (2, 3):
    t0 = time.perf_counter(); out = list(eng.generate(sample(seed), new)); outs.append((out, time.perf_counter() - t0))
again = list(eng.generate(sample(2), new))
print(json.dumps({"warm_s": t_warm, "steps": [len(o) for o, _ in outs], "widths": [len(o[0]) for o, _ in outs],
                  "sample_s": [t for _, t in outs], "deterministic": again == outs[0][0]}))
"""


def run_driver(env_extra: dict, B: int, T: int, new: int) -> dict:
    make_tiny(TINY)
    import os
    env = dict(os.environ, **env_extra)
    proc = subprocess.run(
        [sys.executable, "-c", DRIVER, str(ROOT / "engine"), str(TINY), str(B), str(T), str(new)],
        capture_output=True, text=True, env=env, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_fresh_process_protocol():
    r = run_driver({}, 3, 50, 7)
    assert r["steps"] == [7, 7] and r["widths"] == [3, 3] and r["deterministic"]


def test_fresh_process_spec_protocol():
    r = run_driver({"ENGINE_SPEC_K": "3"}, 2, 40, 9)
    assert r["steps"] == [9, 9] and r["widths"] == [2, 2] and r["deterministic"]
