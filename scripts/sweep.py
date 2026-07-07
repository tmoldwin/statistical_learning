"""Word-count × letter-length sweep: plan, train, and heatmap plots."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment import TASKS, checkpoint_path
from vocab_sweep import (
    SWEEP_DEFAULT_SEEDS,
    SWEEP_LENGTHS,
    SWEEP_WORD_COUNTS,
    iter_sweep_cells,
    task_name,
)
from viz.compare.sweep_heatmap import replot_sweep_heatmaps, run_sweep_plots, run_sweep_training_plots
from viz.compare.sweep_spectrum import replot_sweep_spectra, run_sweep_spectrum_plots
from viz.compare.sweep_decoding import replot_sweep_decoding, run_sweep_decoding_plots


def _tasks() -> tuple[str, ...]:
    return tuple(task_name(n, L) for n, L in iter_sweep_cells())


def cmd_plan(args: argparse.Namespace) -> None:
    seeds = tuple(args.seeds) if args.seeds else SWEEP_DEFAULT_SEEDS
    tasks = _tasks()
    missing = 0
    for task in tasks:
        for seed in seeds:
            if not checkpoint_path(task, "rnn", seed=seed).is_file():
                missing += 1
    print(f"sweep grid: {len(SWEEP_WORD_COUNTS)} word counts × {len(SWEEP_LENGTHS)} lengths = {len(tasks)} tasks")
    print(f"seeds: {len(seeds)}  total jobs: {len(tasks) * len(seeds)}  missing checkpoints: {missing}")


def cmd_train(args: argparse.Namespace) -> None:
    seeds = tuple(args.seeds) if args.seeds else SWEEP_DEFAULT_SEEDS
    for task in _tasks():
        need = [
            s for s in seeds
            if not checkpoint_path(task, "rnn", seed=s).is_file()
        ]
        if not need:
            continue
        cmd = [
            sys.executable, "scripts/run_task.py", task,
            "--models", "rnn",
            "--seeds", *[str(s) for s in need],
            "--skip-viz",
        ]
        if args.smoke:
            cmd.append("--smoke")
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def cmd_plot(args: argparse.Namespace) -> None:
    seeds = tuple(args.seeds) if args.seeds else SWEEP_DEFAULT_SEEDS
    if args.decoding_only:
        if args.replot_only:
            curves_pca, curves_neu = replot_sweep_decoding()
            print(f"wrote {curves_pca}")
            print(f"wrote {curves_neu}")
        else:
            run_sweep_decoding_plots(seeds=seeds, recompute=True)
        return
    if args.spectrum_only:
        if args.replot_only:
            path = replot_sweep_spectra()
            print(f"wrote {path}")
        else:
            run_sweep_spectrum_plots(seeds=seeds, recompute=True)
        return
    if args.replot_only:
        path = replot_sweep_heatmaps()
        print(f"wrote {path}")
        return
    if args.training_only:
        run_sweep_plots(seeds=seeds, geometry=False, training=True)
        geom_path = REPO_ROOT / "experiments/comparisons/word_length_sweep_ns/trajectories/sweep_geometry.json"
        if geom_path.is_file():
            path = replot_sweep_heatmaps()
            print(f"wrote {path}")
    elif args.geometry_only:
        run_sweep_plots(seeds=seeds, geometry=True, training=False)
    else:
        run_sweep_plots(seeds=seeds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan", help="show job counts")
    p_plan.add_argument("--seeds", nargs="+", type=int)
    p_plan.set_defaults(func=cmd_plan)

    p_train = sub.add_parser("train", help="train all sweep tasks")
    p_train.add_argument("--seeds", nargs="+", type=int)
    p_train.add_argument("--smoke", action="store_true")
    p_train.set_defaults(func=cmd_train)

    p_plot = sub.add_parser("plot", help="write sweep_geometry.json + heatmaps")
    p_plot.add_argument("--seeds", nargs="+", type=int)
    p_plot.add_argument(
        "--decoding-only",
        action="store_true",
        help="compute/plot linear decoding from hidden states (PCA + neurons)",
    )
    p_plot.add_argument(
        "--spectrum-only",
        action="store_true",
        help="compute/plot closed-loop PC variance spectra",
    )
    p_plot.add_argument(
        "--replot-only",
        action="store_true",
        help="replot heatmaps from existing sweep_geometry.json + sweep_training.json",
    )
    p_plot.add_argument(
        "--training-only",
        action="store_true",
        help="refresh sweep_training.json and replot combined heatmaps (fast)",
    )
    p_plot.add_argument(
        "--geometry-only",
        action="store_true",
        help="only regenerate geometry heatmaps",
    )
    p_plot.set_defaults(func=cmd_plot)

    args = parser.parse_args()
    # Ensure sweep tasks are registered
    assert task_name(4, 3) in TASKS
    args.func(args)


if __name__ == "__main__":
    main()
