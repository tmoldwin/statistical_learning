"""Train and visualize a single task folder."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment import (
    DALE_RNN_DEFAULTS,
    DEFAULT_SEED,
    TASKS,
    experiment_regime,
    input_path,
    model_path,
    model_uses_dale,
)
from vocab_diagrams import write_vocabulary_diagrams_for_experiment


def run(cmd: list[str]) -> None:
    print(f"\n>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def train_task(
    name: str,
    *,
    smoke: bool = False,
    seed: int = DEFAULT_SEED,
    model_type: str = "rnn",
    multi_seed: bool = False,
    save_snapshots: bool = False,
    device: str = "cpu",
) -> None:
    cfg = TASKS[name]
    regime = experiment_regime(name)
    print(f"\n=== {name} (seed {seed}) ===")

    corpus_out = input_path(name)
    corpus_out.parent.mkdir(parents=True, exist_ok=True)

    run([
        sys.executable, "task.py", regime,
        "--exp", name,
        "--chars", str(cfg["chars"]),
        "--seed", str(seed),
        "--out", str(corpus_out),
    ])

    if model_type in ("rnn", "rnn_dale"):
        model_out = model_path(name, model_type, seed=seed) if multi_seed else model_path(name, model_type)
        model_out.parent.mkdir(parents=True, exist_ok=True)
        use_dale = model_uses_dale(model_type) or bool(cfg.get("dale"))
        use_gpu = device in ("cuda", "gpu", "auto")
        if use_gpu and use_dale:
            print("warning: Dale training has no GPU trainer; using CPU (min_char_rnn.py)")
            use_gpu = False
        if use_gpu and save_snapshots:
            raise ValueError("GPU trainer does not support --save-snapshots")
        train_script = "rnn/torch_char_rnn.py" if use_gpu else "rnn/min_char_rnn.py"
        train_cmd = [
            sys.executable, train_script,
            "--input", str(corpus_out),
            "--model", str(model_out),
            "--exp", name,
            "--seed", str(seed),
        ]
        if use_gpu:
            train_cmd.extend(["--device", "cuda" if device == "gpu" else device])
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
        if "target_word_error_frac" in cfg:
            train_cmd.extend(["--target-word-error", str(cfg["target_word_error_frac"])])
        if save_snapshots:
            train_cmd.append("--save-snapshots")
        if model_uses_dale(model_type) and "dale_steps" in cfg:
            steps = 500 if smoke else int(cfg["dale_steps"])
        else:
            steps = 500 if smoke else int(cfg["steps"])
        train_cmd.extend(["--steps", str(steps)])
        run(train_cmd)
        if multi_seed and seed == DEFAULT_SEED:
            shutil.copy2(model_out, model_path(name, model_type))
            print(f"promoted seed {seed} -> model.npz")
    elif model_type == "transformer":
        model_out = model_path(name, model_type, seed=seed) if multi_seed else model_path(name, model_type)
        tf_cmd = [
            sys.executable, "-m", "transformer.train",
            "--exp", name,
            "--seed", str(seed),
            "--model", str(model_out),
        ]
        if smoke:
            tf_cmd.extend(["--steps", "500"])
        run(tf_cmd)
        if multi_seed and seed == DEFAULT_SEED:
            shutil.copy2(model_out, model_path(name, model_type))
    else:
        raise ValueError(f"unknown model_type {model_type!r}")


def visualize_task(
    name: str,
    *,
    model_type: str = "rnn",
    trajectories_only: bool = False,
    seed: int | None = None,
) -> None:
    cfg = TASKS[name]
    viz_cmd = [
        sys.executable, "visualize.py",
        "--exp", name,
        "--model-type", model_type,
        "--length", str(cfg["viz_length"]),
    ]
    if seed is not None:
        viz_cmd.extend(["--seed", str(seed)])
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
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"RNG seed (default: {DEFAULT_SEED})")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="train multiple seeds; saves model_seed<N>.npz (and model.npz for seed 42)",
    )
    parser.add_argument("--trajectories-only", action="store_true")
    parser.add_argument(
        "--save-snapshots", action="store_true",
        help="record weight snapshots during training (large model files; "
             "only needed for learning-dynamics videos/analysis)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="training device for RNN tasks (cuda uses rnn/torch_char_rnn.py)",
    )
    args = parser.parse_args()

    write_vocabulary_diagrams_for_experiment(args.task)

    seeds = list(args.seeds) if args.seeds else [args.seed]
    multi_seed = args.seeds is not None

    for seed in seeds:
        for model_type in args.models:
            if not args.skip_train:
                train_task(
                    args.task,
                    smoke=args.smoke,
                    seed=seed,
                    model_type=model_type,
                    multi_seed=multi_seed,
                    save_snapshots=args.save_snapshots,
                    device=args.device,
                )
            if not args.skip_viz and not multi_seed:
                visualize_task(
                    args.task,
                    model_type=model_type,
                    trajectories_only=args.trajectories_only,
                )


if __name__ == "__main__":
    main()
