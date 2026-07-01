"""Generate cross-task comparison figures under experiments/comparisons/<name>/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment import TASKS, checkpoint_path
from viz.compare.run import COMPARISON_KINDS, run_comparison
from viz.compare.spec import COMPARISON_PRESETS, ComparisonSpec
from viz.compare.train import train_comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/compare.py --preset sixteen_word_lengths_ns\n"
            "  python scripts/compare.py --preset sixteen_word_lengths_ns --seeds 42 43 44 --train\n"
            "  python scripts/compare.py --preset sixteen_word_lengths_ns "
            "--seeds 1 2 3 5 7 8 11 13 17 19 23 29 31 37 53 --train --kinds trajectory_geometry\n"
        ),
    )
    parser.add_argument("--preset", choices=sorted(COMPARISON_PRESETS))
    parser.add_argument("--name")
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS.keys()))
    parser.add_argument("--labels", nargs="*")
    parser.add_argument("--title", default="")
    parser.add_argument("--model-type", default="rnn", choices=["rnn", "rnn_dale", "transformer"])
    parser.add_argument(
        "--kinds",
        nargs="+",
        default=["closed_loop_trajectories"],
        choices=sorted(COMPARISON_KINDS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, help="RNG seeds (default: 42)")
    parser.add_argument("--train", action="store_true", help="train all tasks at each seed first")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--truncate-to-plateau", action="store_true")
    args = parser.parse_args()

    if args.preset:
        spec = COMPARISON_PRESETS[args.preset]
        if args.name and args.name != spec.name:
            parser.error("--name conflicts with --preset")
        if args.tasks:
            parser.error("--tasks cannot be used with --preset")
    else:
        if not args.name or not args.tasks:
            parser.error("provide --preset or both --name and --tasks")
        labels: dict[str, str] = {}
        if args.labels:
            if len(args.labels) != len(args.tasks):
                parser.error("--labels must match --tasks length")
            labels = dict(zip(args.tasks, args.labels, strict=True))
        spec = ComparisonSpec(
            name=args.name,
            tasks=tuple(args.tasks),
            labels=labels,
            title=args.title,
            model_type=args.model_type,
        )

    seeds = tuple(args.seeds) if args.seeds else spec.seeds
    spec = ComparisonSpec(
        name=spec.name,
        tasks=spec.tasks,
        labels=spec.labels,
        title=spec.title,
        model_type=args.model_type or spec.model_type,
        row_groups=spec.row_groups,
        seeds=seeds,
    )

    if args.train:
        train_comparison(spec, seeds=seeds, smoke=args.smoke)

    missing = [
        (task, seed)
        for seed in seeds
        for task in spec.tasks
        if not checkpoint_path(task, spec.model_type, seed=seed).is_file()
    ]
    if missing:
        print(
            f"warning: {len(missing)} missing checkpoints "
            f"(e.g. {missing[0][0]} seed {missing[0][1]}) — use --train"
        )

    run_comparison(spec, args.kinds, seeds=seeds, truncate_to_plateau=args.truncate_to_plateau)


if __name__ == "__main__":
    main()
