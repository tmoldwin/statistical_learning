"""
Train and visualize statistical-learning experiments for RNN and Transformer.

    python run_experiments.py
    python run_experiments.py --only ten_word_overlap_s
    python run_experiments.py --only ten_word_overlap_s --models rnn
    python run_experiments.py --only ten_word_overlap_s --models transformer
    python run_experiments.py --skip-train --models transformer
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from experiment import EXPERIMENT_CONFIG, experiment_regime, input_path, model_path


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
        choices=list(EXPERIMENT_CONFIG.keys()),
        help="subset of experiments to run",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["rnn", "transformer"],
        default=["rnn", "transformer"],
        help="which model(s) to train and visualize (default: both)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="only run visualize.py (requires existing checkpoints)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="short training run for quick verification",
    )
    args = parser.parse_args()

    names = args.only if args.only else list(EXPERIMENT_CONFIG.keys())

    for name in names:
        cfg = EXPERIMENT_CONFIG.get(name, next(iter(EXPERIMENT_CONFIG.values())))
        regime = experiment_regime(name)
        print(f"\n=== {name} ===")

        if not args.skip_train:
            run([
                sys.executable, "task.py", regime,
                "--exp", name,
                "--chars", str(cfg["chars"]),
                "--seed", str(args.seed),
            ])

            if "rnn" in args.models:
                train_cmd = [
                    sys.executable, "rnn/min_char_rnn.py",
                    "--input", str(input_path(name)),
                    "--model", str(model_path(name, "rnn")),
                    "--steps", str(500 if args.smoke else cfg["steps"]),
                ]
                if "hidden_size" in cfg:
                    train_cmd.extend(["--hidden-size", str(cfg["hidden_size"])])
                if cfg.get("dale"):
                    train_cmd.append("--dale")
                    train_cmd.extend(["--e-fraction", str(cfg.get("e_fraction", 0.8))])
                if "sequence_length" in cfg:
                    train_cmd.extend(["--sequence-length", str(cfg["sequence_length"])])
                run(train_cmd)

            if "transformer" in args.models:
                tf_cmd = [
                    sys.executable, "-m", "transformer.train",
                    "--exp", name,
                    "--seed", str(args.seed),
                ]
                if args.smoke:
                    tf_cmd.extend(["--steps", "500"])
                run(tf_cmd)

        for model_type in args.models:
            run([
                sys.executable, "visualize.py",
                "--exp", name,
                "--model-type", model_type,
                "--length", str(cfg["viz_length"]),
            ])

        if name == "ten_word_overlap_s" and "rnn" in args.models:
            run([sys.executable, "scripts/build_readme.py"])

    print("\nDone. Outputs under experiments/<name>/{rnn,transformer}/")


if __name__ == "__main__":
    main()
