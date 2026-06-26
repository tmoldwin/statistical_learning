"""Train (if needed) and plot word trajectories for the micro curriculum."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment import EXPERIMENT_CONFIG, MICRO_CURRICULUM, spaced_experiment_name


def run(cmd: list[str]) -> None:
    print(f"\n>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> None:
    smoke = "--smoke" in sys.argv
    skip_train = "--skip-train" in sys.argv
    exps = [spaced_experiment_name(r) for r in MICRO_CURRICULUM]

    for exp in exps:
        cfg = EXPERIMENT_CONFIG[exp]
        regime = cfg["regime"]
        print(f"\n=== {exp} ===")

        if not skip_train:
            run([
                sys.executable, "task.py", regime,
                "--exp", exp,
                "--chars", str(cfg["chars"]),
                "--seed", "42",
            ])
            tf_cmd = [sys.executable, "-m", "transformer.train", "--exp", exp, "--seed", "42"]
            if smoke:
                tf_cmd.extend(["--steps", "500"])
            run(tf_cmd)

        run([
            sys.executable, "visualize.py",
            "--exp", exp,
            "--model-type", "transformer",
            "--length", str(cfg["viz_length"]),
            "--trajectories-only",
        ])

    run([sys.executable, "scripts/plot_micro_curriculum_trajectory_panels.py"])
    run([
        sys.executable, "scripts/plot_micro_curriculum_closed_loop_panels.py",
        "--steps", "80",
    ])


if __name__ == "__main__":
    main()
