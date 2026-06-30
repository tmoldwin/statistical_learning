"""Train (RNN only) and plot sixteen-word validation figures.

Two vocabularies × spaced/unspaced; single init (seed 42); 5k steps.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment import (
    EXPERIMENT_CONFIG,
    model_path,
    spaced_experiment_name,
    validation_suite_curriculum,
    validation_suite_root,
)

SUITE = "sixteen_word"
MODEL_TYPE = "rnn"


def run(cmd: list[str]) -> None:
    print(f"\n>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _experiments_for_spacing(spaced: bool) -> list[str]:
    curriculum = validation_suite_curriculum(SUITE)
    if spaced:
        return [spaced_experiment_name(r) for r in curriculum]
    return list(curriculum)


def _plot_validation_figures() -> None:
    model_flag = ["--model-type", MODEL_TYPE, "--suite", SUITE]
    run([sys.executable, "scripts/plot_micro_curriculum_trajectories.py", *model_flag])
    run([
        sys.executable, "scripts/plot_micro_curriculum_trajectories.py",
        "--no-word-space", *model_flag,
    ])
    run([sys.executable, "scripts/plot_micro_curriculum_learning_curves.py", *model_flag])
    run([
        sys.executable, "scripts/plot_micro_curriculum_learning_curves.py",
        "--no-word-space", *model_flag,
    ])
    run([sys.executable, "scripts/plot_micro_curriculum_states.py", *model_flag])
    run([
        sys.executable, "scripts/plot_micro_curriculum_states.py",
        "--no-word-space", *model_flag,
    ])
    run([
        sys.executable, "scripts/analyze_micro_curriculum_unit_selectivity.py",
        "--no-no-word-space", *model_flag,
    ])
    run([
        sys.executable, "scripts/analyze_micro_curriculum_unit_selectivity.py",
        *model_flag,
    ])
    run([
        sys.executable, "scripts/analyze_micro_curriculum_dfa_sensitivity.py",
        *model_flag,
    ])
    run([
        sys.executable, "scripts/analyze_micro_curriculum_dfa_sensitivity.py",
        "--no-word-space", *model_flag,
    ])


def _train_models(*, smoke: bool = False) -> None:
    for spaced in (True, False):
        for exp in _experiments_for_spacing(spaced):
            cfg = EXPERIMENT_CONFIG[exp]
            regime = cfg["regime"]
            print(f"\n=== {exp} ===")
            run([
                sys.executable, "task.py", regime,
                "--exp", exp,
                "--chars", str(cfg["chars"]),
                "--seed", "42",
            ])
            rnn_cmd = [
                sys.executable, "rnn/min_char_rnn.py",
                "--input", str(REPO_ROOT / "experiments" / exp / "input.txt"),
                "--model", str(model_path(exp, MODEL_TYPE)),
                "--exp", exp,
                "--seed", "42",
            ]
            if "hidden_size" in cfg:
                rnn_cmd.extend(["--hidden-size", str(cfg["hidden_size"])])
            if "sequence_length" in cfg:
                rnn_cmd.extend(["--sequence-length", str(cfg["sequence_length"])])
            if "learning_rate" in cfg:
                rnn_cmd.extend(["--learning-rate", str(cfg["learning_rate"])])
            if smoke:
                rnn_cmd.extend(["--steps", "500"])
            elif "steps" in cfg:
                rnn_cmd.extend(["--steps", str(cfg["steps"])])
            run(rnn_cmd)


def _cleanup_stale_figures() -> None:
    """Remove legacy flat PNG/JSON files superseded by typed subfolders."""
    root = validation_suite_root(SUITE)
    stale_names = {
        "dfa_sensitivity_curriculum.json",
        "dfa_sensitivity_curriculum.png",
        "micro_curriculum_closed_loop_panels.png",
        "micro_curriculum_closed_loop_panels_3d.png",
        "micro_curriculum_trajectories_panels.png",
        "unit_selectivity_curriculum.json",
        "unit_selectivity_curriculum_heatmap.png",
        "unit_exemplars_char.png",
        "unit_exemplars_dfa.png",
        "unit_exemplars_next_char.png",
        "unit_exemplars_position.png",
        "unit_exemplars_prefix.png",
        "unit_exemplars_word_end.png",
        "unit_exemplars_word_start.png",
        # superseded by by_vocab naming for sixteen-word suite
        "by_init.png",
        "by_init_3d.png",
    }
    stale_globs = ("*_by_init.png",)
    stale_dirs = {"closed_loop", "unit_exemplars", "state_endpoints"}

    for spacing_dir in root.rglob("*"):
        if not spacing_dir.is_dir():
            continue
        if spacing_dir.name not in ("_s", "_ns"):
            continue
        for name in stale_names:
            path = spacing_dir / name
            if path.is_file():
                path.unlink()
                print(f"removed stale {path.relative_to(REPO_ROOT)}")
        for sub in spacing_dir.iterdir():
            if sub.is_dir():
                for stale in sub.glob("*_by_init.png"):
                    stale.unlink()
                    print(f"removed stale {stale.relative_to(REPO_ROOT)}")
        for dirname in stale_dirs:
            path = spacing_dir / dirname
            if path.is_dir():
                import shutil
                shutil.rmtree(path)
                print(f"removed stale dir {path.relative_to(REPO_ROOT)}")


def main() -> None:
    smoke = "--smoke" in sys.argv
    skip_train = "--skip-train" in sys.argv

    _cleanup_stale_figures()

    if not skip_train:
        _train_models(smoke=smoke)

    _plot_validation_figures()


if __name__ == "__main__":
    main()
