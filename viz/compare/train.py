"""Train all tasks in a comparison spec at one or more seeds."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from viz.compare.spec import ComparisonSpec

REPO_ROOT = Path(__file__).resolve().parents[2]


def train_comparison(
    spec: ComparisonSpec,
    *,
    seeds: tuple[int, ...],
    smoke: bool = False,
    skip_viz: bool = True,
) -> None:
    for task in spec.tasks:
        cmd = [
            sys.executable, "scripts/run_task.py", task,
            "--models", spec.model_type,
        ]
        if len(seeds) > 1:
            cmd.extend(["--seeds", *[str(s) for s in seeds]])
        else:
            cmd.extend(["--seed", str(seeds[0])])
        if smoke:
            cmd.append("--smoke")
        if skip_viz:
            cmd.append("--skip-viz")
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
