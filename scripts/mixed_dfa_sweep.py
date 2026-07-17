"""Train and analyze mixed-length English vocab runs (organized by DFA size)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment import EXPERIMENTS_ROOT, checkpoint_path, comparison_dir
from vocab_mixed_dfa import (
    COMPARISON_NAME,
    DEFAULT_SEEDS,
    HIDDEN_SIZE_ABLATION,
    N_RUNS,
    comparison_name_for_h,
    iter_runs,
    iter_tasks_for_h,
    task_name,
    write_run_manifest,
)
from rnn.learning_snaps import list_learning_snaps
from viz.compare.mixed_dfa_viz import (
    collect_learning_decode,
    collect_learning_decode_by_dfa,
    plot_learning_decode,
    plot_learning_decode_by_dfa_bins,
    plot_mixed_dfa_trajectory_vocab_grid,
    plot_mixed_dfa_within_corr_vs_dfa,
    run_all_mixed_dfa_plots,
)
from viz.compare.sweep_output import sweep_data_dir

LEARNING_DECODE_SEEDS: tuple[int, ...] = tuple(range(1, 16))
HARD_RUN_ID = 41  # DFA=49, 21 words — hardest in the 50-run manifest


def _train_one(task: str, seeds: tuple[int, ...], *, smoke: bool, device: str,
               save_learning_snaps: bool = False, force_retrain: bool = False) -> None:
    need: list[int] = []
    for s in seeds:
        ckpt = checkpoint_path(task, "rnn", seed=s)
        if force_retrain:
            need.append(s)
        elif save_learning_snaps:
            if not list_learning_snaps(ckpt):
                need.append(s)
        elif not ckpt.is_file():
            need.append(s)
    if not need:
        return
    cmd = [
        sys.executable, "scripts/run_task.py", task,
        "--models", "rnn",
        "--seeds", *[str(s) for s in need],
        "--skip-viz",
        "--device", device,
    ]
    if smoke:
        cmd.append("--smoke")
    if save_learning_snaps:
        cmd.append("--save-learning-snaps")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def cmd_plan(args: argparse.Namespace) -> None:
    seeds = tuple(args.seeds) if args.seeds else DEFAULT_SEEDS
    manifest = write_run_manifest(sweep_data_dir(COMPARISON_NAME) / "run_manifest.json")
    tasks = [e["task"] for e in iter_runs()]
    missing = sum(
        1 for t in tasks for s in seeds
        if not checkpoint_path(t, "rnn", seed=s).is_file()
    )
    print(f"mixed vocab DFA sweep: {N_RUNS} runs  seeds={list(seeds)}")
    print(f"comparison: {COMPARISON_NAME}")
    print(f"manifest: {manifest}")
    print(f"missing checkpoints: {missing} / {len(tasks) * len(seeds)}")
    # Brief DFA spread.
    import json
    data = json.loads(manifest.read_text(encoding="utf-8"))
    dfas = [r["n_dfa_states"] for r in data["runs"]]
    print(f"DFA states: min={min(dfas)} median={sorted(dfas)[len(dfas)//2]} max={max(dfas)}")


def cmd_train(args: argparse.Namespace) -> None:
    seeds = tuple(args.seeds) if args.seeds else DEFAULT_SEEDS
    hidden_size = int(args.hidden_size) if args.hidden_size is not None else None
    if hidden_size is None:
        tasks = [e["task"] for e in iter_runs()]
        if args.runs is not None:
            want = set(args.runs)
            tasks = [task_name(i) for i in sorted(want)]
    else:
        tasks = [e["task"] for e in iter_tasks_for_h(hidden_size)]
        if args.runs is not None:
            want = set(args.runs)
            tasks = [
                task_name(i, hidden_size=hidden_size) for i in sorted(want)
            ]
        print(
            f"training H={hidden_size} -> {comparison_name_for_h(hidden_size)} "
            f"({len(tasks)} tasks)",
            flush=True,
        )
    jobs = max(1, int(args.jobs))
    if jobs == 1:
        for task in tasks:
            _train_one(task, seeds, smoke=args.smoke, device=args.device)
        return
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futs = [
            pool.submit(_train_one, task, seeds, smoke=args.smoke, device=args.device)
            for task in tasks
        ]
        for fut in as_completed(futs):
            fut.result()


def cmd_train_h_ablation(args: argparse.Namespace) -> None:
    """Train H=50 and H=150 (H=100 already exists under mixed_vocab_dfa_ns)."""
    hs = tuple(args.hidden_sizes) if args.hidden_sizes else HIDDEN_SIZE_ABLATION
    for h in hs:
        print(f"\n=== train hidden_size={h} ===", flush=True)
        args.hidden_size = int(h)
        cmd_train(args)


def cmd_plot_h_ablation(args: argparse.Namespace) -> None:
    from viz.compare.mixed_dfa_h_ablation import plot_mixed_dfa_h_ablation

    hs = tuple(args.hidden_sizes) if args.hidden_sizes else (50, 100, 150)
    out = plot_mixed_dfa_h_ablation(
        seed=(args.seeds[0] if args.seeds else 1),
        recompute=not args.replot_only,
        hs=hs,
    )
    print(f"wrote {out}", flush=True)


def cmd_plot(args: argparse.Namespace) -> None:
    seeds = tuple(args.seeds) if args.seeds else DEFAULT_SEEDS
    run_all_mixed_dfa_plots(seeds=seeds, recompute=not args.replot_only)


def cmd_learning_decode(args: argparse.Namespace) -> None:
    run_id = int(args.run_id)
    task = task_name(run_id)
    seeds = tuple(args.seeds) if args.seeds else LEARNING_DECODE_SEEDS
    need: list[int] = []
    for s in seeds:
        ckpt = checkpoint_path(task, "rnn", seed=s)
        if args.retrain or not list_learning_snaps(ckpt):
            need.append(int(s))
    if need:
        jobs = max(1, int(args.jobs))
        print(f"training {len(need)} seeds with learning snaps (jobs={jobs})", flush=True)
        if jobs == 1:
            _train_one(
                task, tuple(need), smoke=False, device=args.device,
                save_learning_snaps=True, force_retrain=args.retrain,
            )
        else:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futs = [
                    pool.submit(
                        _train_one,
                        task,
                        (seed,),
                        smoke=False,
                        device=args.device,
                        save_learning_snaps=True,
                        force_retrain=args.retrain,
                    )
                    for seed in need
                ]
                for fut in as_completed(futs):
                    fut.result()
    json_path = collect_learning_decode(task, seeds=seeds)
    out = plot_learning_decode(task, json_path=json_path)
    print(f"wrote {out}", flush=True)


def cmd_learning_decode_bins(args: argparse.Namespace) -> None:
    """Train seed-1 learning snaps for all mixed runs, then binned learning curves."""
    seeds = tuple(args.seeds) if args.seeds else (1,)
    seed = int(seeds[0])
    tasks = [e["task"] for e in iter_runs()]
    if args.runs is not None:
        want = set(args.runs)
        tasks = [task_name(i) for i in sorted(want)]
    need = []
    for task in tasks:
        ckpt = checkpoint_path(task, "rnn", seed=seed)
        if args.retrain or not list_learning_snaps(ckpt):
            need.append(task)
    if need:
        jobs = max(1, int(args.jobs))
        print(f"training {len(need)} runs seed={seed} with learning snaps (jobs={jobs})", flush=True)
        if jobs == 1:
            for task in need:
                _train_one(
                    task, (seed,), smoke=False, device=args.device,
                    save_learning_snaps=True, force_retrain=args.retrain,
                )
        else:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futs = [
                    pool.submit(
                        _train_one,
                        task,
                        (seed,),
                        smoke=False,
                        device=args.device,
                        save_learning_snaps=True,
                        force_retrain=args.retrain,
                    )
                    for task in need
                ]
                for fut in as_completed(futs):
                    fut.result()
    # seeds=None → all snaps on disk (seed 1 × 50 vocabs + any multi-seed runs).
    json_path = collect_learning_decode_by_dfa(seeds=None, recompute=True)
    out = plot_learning_decode_by_dfa_bins(json_path=json_path)
    print(f"wrote {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "plan", "train", "plot", "all",
            "learning-decode", "learning-decode-bins", "trajectory-grid", "within-corr",
            "train-h-ablation", "plot-h-ablation", "hard-dfa-geometry",
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto", "gpu"))
    parser.add_argument("--runs", type=int, nargs="+", default=None, help="subset of run ids")
    parser.add_argument("--replot-only", action="store_true")
    parser.add_argument("--run-id", type=int, default=HARD_RUN_ID, help="run id for learning-decode")
    parser.add_argument(
        "--retrain", action="store_true",
        help="with learning-decode: force retrain with --save-learning-snaps",
    )
    parser.add_argument(
        "--hidden-size", type=int, default=None,
        help="with train: use mixeddfa_h{H}_* tasks (default: legacy H=100 names)",
    )
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="+", default=None,
        help="with train-h-ablation / plot-h-ablation (default: 50 150 or 50 100 150)",
    )
    args = parser.parse_args()

    (EXPERIMENTS_ROOT / "comparisons" / COMPARISON_NAME).mkdir(parents=True, exist_ok=True)

    if args.command == "plan":
        cmd_plan(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "train-h-ablation":
        cmd_train_h_ablation(args)
    elif args.command == "plot":
        cmd_plot(args)
    elif args.command == "plot-h-ablation":
        cmd_plot_h_ablation(args)
    elif args.command == "hard-dfa-geometry":
        from viz.compare.mixed_dfa_viz import plot_hard_dfa_state_geometry

        outs = plot_hard_dfa_state_geometry(run_id=int(args.run_id), seed=(args.seeds[0] if args.seeds else 1))
        for p in outs:
            print(f"wrote {p}", flush=True)
    elif args.command == "learning-decode":
        cmd_learning_decode(args)
    elif args.command == "learning-decode-bins":
        cmd_learning_decode_bins(args)
    elif args.command == "trajectory-grid":
        out = plot_mixed_dfa_trajectory_vocab_grid()
        print(f"wrote {out}", flush=True)
    elif args.command == "within-corr":
        out = plot_mixed_dfa_within_corr_vs_dfa(recompute=True)
        print(f"wrote {out}", flush=True)
    elif args.command == "all":
        cmd_plan(args)
        cmd_train(args)
        cmd_plot(args)


if __name__ == "__main__":
    main()
