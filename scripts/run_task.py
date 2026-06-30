"""Train and visualize a single task folder."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment import TASKS, DALE_RNN_DEFAULTS, experiment_regime, input_path, model_path, model_uses_dale
from vocab_diagrams import write_vocabulary_diagrams_for_experiment


def run(cmd: list[str]) -> None:
    print(f"\n>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def train_task(
    name: str,
    *,
    smoke: bool = False,
    seed: int = 42,
    model_type: str = "rnn",
) -> None:
    cfg = TASKS[name]
    regime = experiment_regime(name)
    print(f"\n=== {name} ===")

    run([
        sys.executable, "task.py", regime,
        "--exp", name,
        "--chars", str(cfg["chars"]),
        "--seed", str(seed),
    ])

    if model_type in ("rnn", "rnn_dale"):
        train_cmd = [
            sys.executable, "rnn/min_char_rnn.py",
            "--input", str(input_path(name)),
            "--model", str(model_path(name, model_type)),
            "--exp", name,
            "--seed", str(seed),
        ]
        if "hidden_size" in cfg:
            train_cmd.extend(["--hidden-size", str(cfg["hidden_size"])])
        if model_uses_dale(model_type) or cfg.get("dale"):
            train_cmd.append("--dale")
            e_frac = cfg.get("e_fraction", DALE_RNN_DEFAULTS["e_fraction"])
            train_cmd.extend(["--e-fraction", str(e_frac)])
        if "sequence_length" in cfg:
            train_cmd.extend(["--sequence-length", str(cfg["sequence_length"])])
        if "learning_rate" in cfg:
            train_cmd.extend(["--learning-rate", str(cfg["learning_rate"])])
        if model_uses_dale(model_type) and "dale_steps" in cfg:
            steps = 500 if smoke else int(cfg["dale_steps"])
        else:
            steps = 500 if smoke else int(cfg["steps"])
        train_cmd.extend(["--steps", str(steps)])
        run(train_cmd)
    elif model_type == "transformer":
        tf_cmd = [sys.executable, "-m", "transformer.train", "--exp", name, "--seed", str(seed)]
        if smoke:
            tf_cmd.extend(["--steps", "500"])
        run(tf_cmd)
    else:
        raise ValueError(f"unknown model_type {model_type!r}")


def visualize_task(name: str, *, model_type: str = "rnn", trajectories_only: bool = False) -> None:
    cfg = TASKS[name]
    viz_cmd = [
        sys.executable, "visualize.py",
        "--exp", name,
        "--model-type", model_type,
        "--length", str(cfg["viz_length"]),
    ]
    if trajectories_only:
        viz_cmd.append("--trajectories-only")
    run(viz_cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=list(TASKS.keys()), help="task folder name")
    parser.add_argument("--models", nargs="+", default=["rnn"], choices=["rnn", "rnn_dale", "transformer"])
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-viz", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trajectories-only", action="store_true")
    args = parser.parse_args()

    write_vocabulary_diagrams_for_experiment(args.task)

    for model_type in args.models:
        if not args.skip_train:
            train_task(args.task, smoke=args.smoke, seed=args.seed, model_type=model_type)
        if not args.skip_viz:
            visualize_task(
                args.task,
                model_type=model_type,
                trajectories_only=args.trajectories_only,
            )


if __name__ == "__main__":
    main()
