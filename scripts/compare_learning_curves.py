"""Generate a side-by-side learning-curve comparison for multiple tasks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment import TASKS
from viz.compare.learning_curves import plot_learning_curves
from viz.compare.spec import ComparisonSpec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="comparison folder under experiments/comparisons/")
    parser.add_argument("--tasks", nargs="+", required=True, choices=list(TASKS.keys()))
    parser.add_argument("--labels", nargs="*", help="panel labels (one per task, optional)")
    parser.add_argument("--title", default="", help="figure suptitle")
    parser.add_argument("--model-type", default="rnn", choices=["rnn", "transformer"])
    parser.add_argument("--truncate-to-plateau", action="store_true")
    args = parser.parse_args()

    labels: dict[str, str] = {}
    if args.labels:
        if len(args.labels) != len(args.tasks):
            parser.error("--labels must have the same length as --tasks")
        labels = dict(zip(args.tasks, args.labels, strict=True))

    spec = ComparisonSpec(
        name=args.name,
        tasks=tuple(args.tasks),
        labels=labels,
        title=args.title,
        model_type=args.model_type,
    )
    out = plot_learning_curves(spec, truncate_to_plateau=args.truncate_to_plateau)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
