#!/usr/bin/env python3
"""Train paper-critical Dale's-law RNNs (demo + mixed DFA sweep)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import vocab_mixed_dfa
from experiment import model_path

iter_runs = vocab_mixed_dfa.iter_runs

DEMO = "eight_word_ate_at_demo_ns"
DEMO_SEEDS = (1, 2, 3, 5, 7, 8)


def _run(cmd: list[str]) -> None:
    print(f"\n>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def train_demo(
    *,
    seeds: tuple[int, ...] = DEMO_SEEDS,
    save_learning_snaps: bool = True,
    skip_existing: bool = True,
) -> None:
    for seed in seeds:
        out = model_path(DEMO, "rnn_dale", seed=seed)
        if skip_existing and out.is_file() and out.stat().st_size > 1000:
            print(f"skip demo seed {seed} (exists {out})", flush=True)
            continue
        cmd = [
            sys.executable, "scripts/run_task.py", DEMO,
            "--models", "rnn_dale",
            "--seeds", str(seed),
            "--skip-viz",
        ]
        if save_learning_snaps:
            cmd.append("--save-learning-snaps")
        _run(cmd)


def train_mixed(
    *,
    seeds: tuple[int, ...] = (1,),
    save_learning_snaps: bool = True,
    skip_existing: bool = True,
    run_ids: list[int] | None = None,
) -> None:
    for entry in iter_runs():
        rid = int(entry["run_id"])
        if run_ids is not None and rid not in run_ids:
            continue
        task = str(entry["task"])
        for seed in seeds:
            out = model_path(task, "rnn_dale", seed=seed)
            if skip_existing and out.is_file() and out.stat().st_size > 1000:
                print(f"skip {task} seed {seed} (exists)", flush=True)
                continue
            cmd = [
                sys.executable, "scripts/run_task.py", task,
                "--models", "rnn_dale",
                "--seeds", str(seed),
                "--skip-viz",
            ]
            if save_learning_snaps:
                cmd.append("--save-learning-snaps")
            _run(cmd)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--demo", action="store_true", help="train eight_word Dale demo seeds")
    p.add_argument("--mixed", action="store_true", help="train mixeddfa_rXX_ns Dale runs")
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--run-ids", nargs="+", type=int, default=None)
    p.add_argument("--no-learning-snaps", action="store_true")
    p.add_argument("--force", action="store_true", help="retrain even if checkpoint exists")
    p.add_argument(
        "--sweep", default="mixed", choices=("mixed", "fixed_grid", "top100"),
        help="which sweep --mixed trains (fixed_grid / top100)",
    )
    args = p.parse_args()
    if args.sweep == "fixed_grid":
        import vocab_fixed_letters_grid

        global iter_runs
        iter_runs = vocab_fixed_letters_grid.iter_runs
    elif args.sweep == "top100":
        import vocab_top100_english

        global iter_runs
        iter_runs = vocab_top100_english.iter_runs
    if not args.demo and not args.mixed:
        args.demo = True
        args.mixed = True
    snaps = not args.no_learning_snaps
    skip = not args.force
    if args.demo:
        seeds = tuple(args.seeds) if args.seeds else DEMO_SEEDS
        train_demo(seeds=seeds, save_learning_snaps=snaps, skip_existing=skip)
    if args.mixed:
        seeds = tuple(args.seeds) if args.seeds else (1,)
        train_mixed(
            seeds=seeds,
            save_learning_snaps=snaps,
            skip_existing=skip,
            run_ids=args.run_ids,
        )


if __name__ == "__main__":
    main()
