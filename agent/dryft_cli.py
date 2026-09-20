"""Minimal Dryft CLI over ``client.py`` for machines the official binary cannot run on.

    python agent/dryft_cli.py submit [engine_dir]           -> prints submission id
    python agent/dryft_cli.py run <submission_id> [--official] [--no-wait]
    python agent/dryft_cli.py result <run_id> [--wait]
    python agent/dryft_cli.py logs <run_id>
    python agent/dryft_cli.py runs | submissions | benchmark
    python agent/dryft_cli.py go [engine_dir] [--official]   -> submit + run + wait + report

Reads DRYFT_TOKEN (and optional DRYFT_API, default https://htn.dryft.ai) from
the environment or a ``.env`` file in the repo root. Every finished run is
appended to ``docs/experiments.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))

from client import Dryft, TERMINAL  # noqa: E402
from package import package  # noqa: E402

LATENCY_GATE = 1.10


def load_env() -> None:
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    os.environ.setdefault("DRYFT_API", "https://htn.dryft.ai")


def client() -> Dryft:
    load_env()
    return Dryft()


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def report(detail: dict) -> bool:
    state = detail.get("state")
    result = detail.get("result") or {}
    shapes = result.get("shapes") or []
    lines = [f"run {detail.get('id')}: {state}  mode={detail.get('mode')}"]
    if result.get("score") is not None:
        lines.append(f"score {result['score']:.1f}  (100 is native)")
    ok = state == "succeeded"
    for shape in shapes:
        metrics = shape.get("modelMetrics") or {}
        cols = [f"{shape.get('id', '?'):<10}", f"{shape.get('caseStatus', '?'):<10}"]
        if shape.get("metricMs") and metrics.get("referenceMs"):
            cols.append(f"{shape['metricMs']:9.1f} ms  {metrics['referenceMs'] / shape['metricMs']:5.2f}x native")
        for name, mine, native in (
            ("ttft", metrics.get("ttftMs"), metrics.get("referenceTtftMs")),
            ("tpot", metrics.get("tpotMs"), metrics.get("referenceTpotMs")),
        ):
            if mine and native:
                ratio = mine / native
                cols.append(f"{name} {ratio:4.2f}x" + ("  OVER GATE" if ratio > LATENCY_GATE else ""))
        if shape.get("tokensPerSecond"):
            cols.append(f"{shape['tokensPerSecond']:8.1f} tok/s")
        if metrics.get("peakMemoryBytes"):
            cols.append(f"{metrics['peakMemoryBytes'] / 2**30:5.1f} GiB")
        lines.append("  " + "  ".join(cols))
        if shape.get("caseMessage"):
            lines.append(f"    {shape['caseMessage']}")
        if shape.get("caseStatus") == "failed":
            ok = False
    for label, value in (
        ("failure", result.get("failureMessage") or result.get("failureCode")),
        ("error", detail.get("errorMessage") or detail.get("errorCode")),
        ("not ranked", result.get("rankingReason")),
    ):
        if value:
            lines.append(f"  {label}: {value}")
    text = "\n".join(lines)
    print(text)
    return ok


def log_experiment(detail: dict, note: str) -> None:
    path = ROOT / "docs" / "experiments.md"
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        path.write_text("# Experiments\n\nEvery Dryft run, newest last. Numbers are the platform's.\n")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    result = detail.get("result") or {}
    rows = []
    for shape in result.get("shapes") or []:
        metrics = shape.get("modelMetrics") or {}
        rows.append(
            f"| {shape.get('id')} | {shape.get('caseStatus')} | {shape.get('tokensPerSecond') or 0:.1f} | "
            f"{(metrics.get('referenceMs') or 0) / (shape.get('metricMs') or 1):.2f}x | "
            f"{(metrics.get('ttftMs') or 0) / (metrics.get('referenceTtftMs') or 1):.2f} | "
            f"{(metrics.get('tpotMs') or 0) / (metrics.get('referenceTpotMs') or 1):.2f} | {shape.get('caseMessage') or ''} |"
        )
    body = [
        f"\n## {stamp} — {detail.get('mode')} run `{detail.get('id')}` @ `{git_rev()}` — {detail.get('state')}",
        f"\n{note}\n" if note else "",
        f"score: {result.get('score')}  failure: {result.get('failureCode') or ''} {result.get('failureMessage') or ''}\n",
        "| workload | status | tok/s | speedup | ttft ratio | tpot ratio | message |",
        "|---|---|---:|---:|---:|---:|---|",
        *rows,
        "",
    ]
    with path.open("a") as f:
        f.write("\n".join(body))


def print_logs(api: Dryft, run_id: str) -> None:
    after = -1
    while True:
        page = api.logs(run_id, after=after, limit=500)
        items = page.get("items") or []
        for item in items:
            seq = item.get("sequence", item.get("seq"))
            print(f"[{item.get('stream', '?')}] {item.get('message', item.get('line', ''))}")
            if seq is not None:
                after = seq
        if not items or not page.get("hasMore", False):
            break


def wait_verbose(api: Dryft, run_id: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    last_state = None
    while True:
        cur = api.run(run_id)
        state = cur.get("state")
        if state != last_state:
            print(f"  {datetime.now().strftime('%H:%M:%S')} run {run_id}: {state}", flush=True)
            last_state = state
        if state in TERMINAL:
            return cur
        if time.monotonic() > deadline:
            raise TimeoutError(f"run {run_id} still {state} after {timeout}s")
        time.sleep(10)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit"); s.add_argument("engine_dir", nargs="?", default="engine")
    r = sub.add_parser("run"); r.add_argument("submission_id"); r.add_argument("--official", action="store_true"); r.add_argument("--no-wait", action="store_true"); r.add_argument("--note", default="")
    g = sub.add_parser("go"); g.add_argument("engine_dir", nargs="?", default="engine"); g.add_argument("--official", action="store_true"); g.add_argument("--note", default="")
    res = sub.add_parser("result"); res.add_argument("run_id"); res.add_argument("--wait", action="store_true"); res.add_argument("--note", default="")
    lg = sub.add_parser("logs"); lg.add_argument("run_id")
    sub.add_parser("runs"); sub.add_parser("submissions"); sub.add_parser("benchmark")
    args = ap.parse_args(argv)
    api = client()

    if args.cmd == "benchmark":
        print(json.dumps(api.benchmark(), indent=2)); return 0
    if args.cmd == "runs":
        print(json.dumps(api._send("GET", "/api/v1/runs"), indent=2)); return 0
    if args.cmd == "submissions":
        print(json.dumps(api._send("GET", "/api/v1/submissions"), indent=2)); return 0
    if args.cmd == "logs":
        print_logs(api, args.run_id); return 0
    if args.cmd in ("submit", "go"):
        archive = package(ROOT / args.engine_dir)
        sid = api.submit(archive)
        print(f"submission {sid} ({len(archive)} bytes)")
        if args.cmd == "submit":
            return 0
        args.submission_id = sid
        args.no_wait = False
    if args.cmd in ("run", "go"):
        mode = "official" if args.official else "public"
        run = api.start_run(args.submission_id, mode=mode)
        print(f"{mode} run {run['id']} started")
        if args.no_wait:
            return 0
        args.run_id = run["id"]
        args.wait = True
    detail = wait_verbose(api, args.run_id, 3600) if args.wait else api.run(args.run_id)
    ok = report(detail)
    if detail.get("state") in TERMINAL:
        log_experiment(detail, args.note)
        if not ok:
            print("--- logs ---")
            print_logs(api, args.run_id)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
