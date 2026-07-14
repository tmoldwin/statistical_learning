"""Shared plan/train/plot driver for pow2 sweep variants."""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]

from experiment import TASKS, checkpoint_path
from viz.compare.pow2_sweep_decoding import replot_pow2_sweep_decoding, run_pow2_sweep_decoding_plots
from viz.compare.pow2_sweep_heatmap import replot_pow2_sweep_heatmaps, run_pow2_sweep_plots
from viz.compare.pow2_sweep_spec import POW2_SWEEP_SPEC_H100, POW2_SWEEP_SPEC_NS, Pow2SweepSpec
from viz.compare.pow2_sweep_spectrum import replot_pow2_sweep_spectra, run_pow2_sweep_spectrum_plots
from viz.compare.pow2_sweep_viz import (
    run_pow2_sweep_closed_loop_plots,
    run_pow2_sweep_demo_sequence_plots,
    run_pow2_sweep_learning_curve_plots,
    run_pow2_sweep_seed_comparison_plots,
)
def _tasks(spec: Pow2SweepSpec) -> tuple[str, ...]:
    return tuple(spec.task_name(n, length) for n, length in spec.iter_cells())


def _train_one_task(
    task: str,
    seeds: tuple[int, ...],
    *,
    smoke: bool,
    device: str,
) -> None:
    need = [s for s in seeds if not checkpoint_path(task, "rnn", seed=s).is_file()]
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
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def cmd_plan(spec: Pow2SweepSpec, args: argparse.Namespace) -> None:
    seeds = tuple(args.seeds) if args.seeds else spec.default_seeds
    tasks = _tasks(spec)
    missing = sum(
        1 for task in tasks for seed in seeds
        if not checkpoint_path(task, "rnn", seed=seed).is_file()
    )
    print(
        f"pow2 sweep ({spec.comparison_name}): "
        f"{len(spec.word_counts)} word counts x {len(spec.lengths)} lengths = {len(tasks)} tasks"
    )
    print(f"word counts: {list(spec.word_counts)}")
    print(f"lengths: {list(spec.lengths)}")
    print(f"seeds: {len(seeds)}  total jobs: {len(tasks) * len(seeds)}  missing checkpoints: {missing}")


def cmd_train(spec: Pow2SweepSpec, args: argparse.Namespace) -> None:
    seeds = tuple(args.seeds) if args.seeds else spec.default_seeds
    tasks = _tasks(spec)
    jobs = max(1, int(args.jobs))
    if jobs == 1:
        for task in tasks:
            _train_one_task(task, seeds, smoke=args.smoke, device=args.device)
        return

    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [
            pool.submit(_train_one_task, task, seeds, smoke=args.smoke, device=args.device)
            for task in tasks
        ]
        for fut in as_completed(futures):
            fut.result()


def cmd_plot(spec: Pow2SweepSpec, args: argparse.Namespace) -> None:
    seeds = tuple(args.seeds) if args.seeds else spec.default_seeds
    seed_cmp_seeds = tuple(args.seeds) if args.seeds else spec.seed_comparison_seeds

    if args.learning_curves_only:
        run_pow2_sweep_learning_curve_plots(seeds=seeds, spec=spec)
        return
    if args.trajectories_only:
        run_pow2_sweep_closed_loop_plots(seeds=seeds, spec=spec)
        return
    if args.seed_comparison_only:
        run_pow2_sweep_seed_comparison_plots(seeds=seed_cmp_seeds, spec=spec)
        return
    if args.sequences_only:
        run_pow2_sweep_demo_sequence_plots(seeds=seeds, spec=spec)
        return
    if args.decoding_only:
        if args.replot_only:
            curves_pca, curves_neu = replot_pow2_sweep_decoding(spec=spec)
            print(f"wrote {curves_pca}")
            print(f"wrote {curves_neu}")
        else:
            run_pow2_sweep_decoding_plots(seeds=seeds, recompute=True, spec=spec)
        return
    if args.spectrum_only:
        if args.replot_only:
            path = replot_pow2_sweep_spectra(spec=spec)
            print(f"wrote {path}")
        else:
            run_pow2_sweep_spectrum_plots(seeds=seeds, recompute=True, spec=spec)
        return
    if args.replot_only:
        path = replot_pow2_sweep_heatmaps(spec=spec)
        print(f"wrote {path}")
        return
    if args.training_only:
        run_pow2_sweep_plots(seeds=seeds, geometry=False, training=True, spec=spec)
        geom_path = REPO_ROOT / "experiments/comparisons" / spec.comparison_name / "data/sweep_geometry.json"
        if geom_path.is_file():
            path = replot_pow2_sweep_heatmaps(spec=spec)
            print(f"wrote {path}")
    elif args.geometry_only:
        run_pow2_sweep_plots(seeds=seeds, geometry=True, training=False, spec=spec)
    else:
        run_pow2_sweep_plots(seeds=seeds, spec=spec)
        run_pow2_sweep_spectrum_plots(seeds=seeds, spec=spec)
        run_pow2_sweep_learning_curve_plots(seeds=seeds, spec=spec)
        run_pow2_sweep_closed_loop_plots(seeds=seeds, spec=spec)
        run_pow2_sweep_demo_sequence_plots(seeds=seeds, spec=spec)
        run_pow2_sweep_seed_comparison_plots(seeds=seed_cmp_seeds, spec=spec)
        run_pow2_sweep_decoding_plots(seeds=seeds, recompute=True, spec=spec)


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name, help_text in (
        ("plan", "show job counts"),
        ("train", "train all pow2 sweep tasks"),
        ("plot", "write sweep JSON + figures"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--seeds", nargs="+", type=int)
        if name == "train":
            p.add_argument("--smoke", action="store_true")
            p.add_argument("--jobs", type=int, default=1, help="parallel training workers")
            p.add_argument(
                "--device",
                default="auto",
                choices=["cpu", "cuda", "auto"],
                help="RNN trainer device (auto uses cuda when available)",
            )
        if name == "plot":
            p.add_argument("--decoding-only", action="store_true")
            p.add_argument("--spectrum-only", action="store_true")
            p.add_argument("--replot-only", action="store_true")
            p.add_argument("--training-only", action="store_true")
            p.add_argument("--learning-curves-only", action="store_true")
            p.add_argument("--trajectories-only", action="store_true")
            p.add_argument("--seed-comparison-only", action="store_true")
            p.add_argument("--sequences-only", action="store_true")
            p.add_argument("--geometry-only", action="store_true")
    return parser


def main_for_spec(spec: Pow2SweepSpec, *, description: str, assert_task: tuple[int, int | str]) -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    parser = build_parser(description)
    args = parser.parse_args()
    n_words, length = assert_task
    assert spec.task_name(n_words, length) in TASKS
    handlers = {"plan": cmd_plan, "train": cmd_train, "plot": cmd_plot}
    handlers[args.cmd](spec, args)


def main_h100() -> None:
    main_for_spec(
        POW2_SWEEP_SPEC_H100,
        description="H100 sweep (L1-6, word counts 5-25 step 5): plan, train, plot",
        assert_task=(5, 3),
    )


def main_ns() -> None:
    main_for_spec(
        POW2_SWEEP_SPEC_NS,
        description="Powers-of-2 word-count x letter-length sweep: plan, train, plot",
        assert_task=(4, 3),
    )
