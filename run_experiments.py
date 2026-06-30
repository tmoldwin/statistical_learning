"""
Train and visualize statistical-learning tasks.

Prefer `python scripts/run_task.py <task>` for a single task.
This entry point runs all active tasks in experiment.TASKS.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from experiment import TASKS


def run(cmd: list[str]) -> None:
    print(f"\n>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(TASKS.keys()),
        help="subset of tasks to run",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["rnn", "transformer"],
        default=["rnn"],
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-viz", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--trajectories-only", action="store_true")
    args = parser.parse_args()

    names = args.only if args.only else list(TASKS.keys())
    for name in names:
        cmd = [sys.executable, "scripts/run_task.py", name, "--seed", str(args.seed)]
        if args.skip_train:
            cmd.append("--skip-train")
        if args.skip_viz:
            cmd.append("--skip-viz")
        if args.smoke:
            cmd.append("--smoke")
        if args.trajectories_only:
            cmd.append("--trajectories-only")
        for model_type in args.models:
            cmd.extend(["--models", model_type])
        run(cmd)


if __name__ == "__main__":
    main()
