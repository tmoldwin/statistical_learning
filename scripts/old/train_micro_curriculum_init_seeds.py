"""Train micro-curriculum models with multiple weight-init seeds."""

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
    EXPERIMENT_CONFIG,
    MICRO_CURRICULUM_INIT_SEEDS,
    MODEL_TYPES,
    VALIDATION_SUITES,
    input_path,
    model_path,
    spaced_experiment_name,
    validation_suite_curriculum,
)


def run(cmd: list[str]) -> None:
    print(f"\n>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _experiments_for_spacing(suite: str, spaced: bool) -> list[str]:
    curriculum = validation_suite_curriculum(suite)
    if spaced:
        return [spaced_experiment_name(r) for r in curriculum]
    return list(curriculum)


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


def _promote_default_checkpoint(exp: str, model_type: str, *, seed: int) -> None:
    """Copy the canonical init-seed checkpoint to model.npz / model.pt for analysis scripts."""
    src = model_path(exp, model_type, seed=seed)
    dst = model_path(exp, model_type)
    if not src.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if model_type == "transformer":
        seed_suffix = f"seed{seed}"
        meta_src = src.parent / f"training_meta_{seed_suffix}.json"
        meta_dst = src.parent / "training_meta.json"
        if meta_src.is_file():
            shutil.copy2(meta_src, meta_dst)
    print(f"promoted {src.name} -> {dst.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=list(VALIDATION_SUITES),
        default="micro",
        help="validation suite to train (default: micro)",
    )
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="retrain even when a seeded checkpoint already exists",
    )
    args = parser.parse_args()

    if args.no_word_space:
        spacing_variants = [False]
    elif args.both_spacing:
        spacing_variants = [True, False]
    else:
        spacing_variants = [True, False]

    for spaced in spacing_variants:
        for exp in _experiments_for_spacing(args.suite, spaced):
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
                ckpt = model_path(exp, args.model_type, seed=seed)
                if ckpt.is_file() and not args.force:
                    print(f"skip {ckpt}: already exists (use --force to retrain)")
                    continue
                print(f"\n=== {exp} seed={seed} ===")
                if args.model_type == "rnn":
                    run(_rnn_train_cmd(exp, cfg, seed=seed, smoke=args.smoke))
                else:
                    tf_cmd = [
                        sys.executable, "-m", "transformer.train",
                        "--exp", exp,
                        "--seed", str(seed),
                        "--model", str(ckpt),
                    ]
                    if args.smoke:
                        tf_cmd.extend(["--steps", "500"])
                    elif "steps" in cfg:
                        tf_cmd.extend(["--steps", str(cfg["steps"])])
                    run(tf_cmd)
            _promote_default_checkpoint(exp, args.model_type, seed=int(args.seeds[0]))


if __name__ == "__main__":
    main()
