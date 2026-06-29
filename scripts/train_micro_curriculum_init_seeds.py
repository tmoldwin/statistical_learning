"""Train micro-curriculum models with multiple weight-init seeds."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment import (
    EXPERIMENT_CONFIG,
    MICRO_CURRICULUM,
    MICRO_CURRICULUM_INIT_SEEDS,
    MODEL_TYPES,
    input_path,
    model_path,
    spaced_experiment_name,
)


def run(cmd: list[str]) -> None:
    print(f"\n>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _experiments_for_spacing(spaced: bool) -> list[str]:
    if spaced:
        return [spaced_experiment_name(r) for r in MICRO_CURRICULUM]
    return list(MICRO_CURRICULUM)


def _rnn_train_cmd(exp: str, cfg: dict, *, seed: int, smoke: bool) -> list[str]:
    cmd = [
        sys.executable, "rnn/min_char_rnn.py",
        "--input", str(input_path(exp)),
        "--model", str(model_path(exp, "rnn", seed=seed)),
        "--steps", str(500 if smoke else cfg["steps"]),
        "--exp", exp,
        "--seed", str(seed),
    ]
    if "hidden_size" in cfg:
        cmd.extend(["--hidden-size", str(cfg["hidden_size"])])
    if cfg.get("dale"):
        cmd.append("--dale")
        cmd.extend(["--e-fraction", str(cfg.get("e_fraction", 0.8))])
    if "sequence_length" in cfg:
        cmd.extend(["--sequence-length", str(cfg["sequence_length"])])
    if "learning_rate" in cfg:
        cmd.extend(["--learning-rate", str(cfg["learning_rate"])])
    if "timestep_noise_std" in cfg:
        cmd.extend(["--noise-std", str(cfg["timestep_noise_std"])])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--no-word-space",
        action="store_true",
        help="unspaced corpora only (_ns); default trains both _s and _ns",
    )
    parser.add_argument(
        "--both-spacing",
        action="store_true",
        help="train both spaced and unspaced (default when neither flag is set)",
    )
    parser.add_argument(
        "--model-type",
        choices=list(MODEL_TYPES),
        default="rnn",
        help="which model to train (default: rnn)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(MICRO_CURRICULUM_INIT_SEEDS),
        help="weight-init / training RNG seeds",
    )
    args = parser.parse_args()

    if args.no_word_space:
        spacing_variants = [False]
    elif args.both_spacing:
        spacing_variants = [True, False]
    else:
        spacing_variants = [True, False]

    for spaced in spacing_variants:
        for exp in _experiments_for_spacing(spaced):
            cfg = EXPERIMENT_CONFIG[exp]
            regime = cfg["regime"]
            if not input_path(exp).is_file():
                run([
                    sys.executable, "task.py", regime,
                    "--exp", exp,
                    "--chars", str(cfg["chars"]),
                    "--seed", "42",
                ])
            for seed in args.seeds:
                print(f"\n=== {exp} seed={seed} ===")
                if args.model_type == "rnn":
                    run(_rnn_train_cmd(exp, cfg, seed=seed, smoke=args.smoke))
                else:
                    tf_cmd = [
                        sys.executable, "-m", "transformer.train",
                        "--exp", exp,
                        "--seed", str(seed),
                        "--model", str(model_path(exp, "transformer", seed=seed)),
                    ]
                    if args.smoke:
                        tf_cmd.extend(["--steps", "500"])
                    run(tf_cmd)


if __name__ == "__main__":
    main()
