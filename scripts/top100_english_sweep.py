"""Train / plan the top-100 English Dale sweep (100 independent N~1..20 draws)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment import checkpoint_path
from vocab_top100_english import (
    COMPARISON_NAME,
    DEFAULT_SEEDS,
    N_RUNS,
    iter_runs,
    write_run_manifest,
)
from viz.compare.sweep_output import sweep_data_dir


def cmd_plan(_args: argparse.Namespace) -> None:
    manifest = write_run_manifest(sweep_data_dir(COMPARISON_NAME) / "run_manifest.json")
    import json

    data = json.loads(manifest.read_text(encoding="utf-8"))
    dfas = [r["n_dfa_states"] for r in data["runs"]]
    ns = [r["n_words"] for r in data["runs"]]
    print(f"top100 English sweep: {N_RUNS} runs")
    print(f"comparison: {COMPARISON_NAME}")
    print(f"manifest: {manifest}")
    print(f"N words: min={min(ns)} median={sorted(ns)[len(ns)//2]} max={max(ns)}")
    print(f"DFA states: min={min(dfas)} median={sorted(dfas)[len(dfas)//2]} max={max(dfas)}")
    print(f"N histogram: {data['n_words_histogram']}")


def _train_one(task: str, seeds: tuple[int, ...], *, smoke: bool, device: str,
               force: bool) -> None:
    need: list[int] = []
    for s in seeds:
        ckpt = checkpoint_path(task, "rnn_dale", seed=s)
        if force or not ckpt.is_file():
            need.append(s)
    if not need:
        return
    cmd = [
        sys.executable, "scripts/run_task.py", task,
        "--models", "rnn_dale",
        "--seeds", *[str(s) for s in need],
        "--skip-viz",
        "--device", device,
    ]
    if smoke:
        cmd.append("--smoke")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def cmd_train(args: argparse.Namespace) -> None:
    seeds = tuple(args.seeds) if args.seeds else DEFAULT_SEEDS
    tasks = [e["task"] for e in iter_runs()]
    if args.runs is not None:
        want = set(args.runs)
        tasks = [e["task"] for e in iter_runs() if int(e["run_id"]) in want]
    jobs = max(1, int(args.jobs))
    if jobs == 1:
        for task in tasks:
            _train_one(task, seeds, smoke=args.smoke, device=args.device, force=args.force)
        return
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futs = [
            pool.submit(_train_one, task, seeds, smoke=args.smoke, device=args.device, force=args.force)
            for task in tasks
        ]
        for fut in as_completed(futs):
            fut.result()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("plan", "train"))
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--runs", nargs="+", type=int, default=None)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.command == "plan":
        cmd_plan(args)
    else:
        cmd_train(args)


if __name__ == "__main__":
    main()
