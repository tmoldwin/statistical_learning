"""Experiment layout: one task folder per run under experiments/<name>/.

experiments/<name>/
    input.txt
    shared/                   # vocabulary trie + DFA SVGs
    rnn/
        model.npz
        plots/
            training/
            weights/
            activations/
            states/
            trajectories/
            unit_selectivity/
            learning_dynamics/
    transformer/              # optional
        model.pt
        plots/
            ...
    rnn_dale/                 # optional Dale's-law RNN (ReLU + signed outgoing weights)
        model.npz
        plots/
            ...

Side-by-side comparison figures live under experiments/comparisons/<name>/.
Archived runs are under experiments/old/.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
COMPARISONS_ROOT = EXPERIMENTS_ROOT / "comparisons"
OLD_EXPERIMENTS_ROOT = EXPERIMENTS_ROOT / "old"

MODEL_TYPES = ("rnn", "rnn_dale", "transformer")

DALE_RNN_DEFAULTS: dict[str, object] = {
    "e_fraction": 0.8,
}

COMPARISON_VIZ_KINDS: tuple[str, ...] = (
    "learning_curves",
    "trajectories",
    "states",
    "unit_selectivity",
    "dfa_sensitivity",
)

_SIXTEEN_WORD_DEFAULTS: dict[str, object] = {
    "regime": "sixteen_word",
    "word_space": True,
    "chars": 50_000,
    "steps": 10_000,
    "viz_length": 50,
    "hidden_size": 50,
    "sequence_length": 12,
    "eval_interval": 50,
    "eval_iterations": 20,
    "metric_rollout_len": 1000,
}

_MIXED_LENGTH_DEFAULTS: dict[str, object] = {
    "regime": "sixteen_word_mixed",
    "word_space": True,
    "chars": 50_000,
    "steps": 12_000,
    "dale_steps": 24_000,
    "viz_length": 60,
    "hidden_size": 50,
    "sequence_length": 16,
    "eval_interval": 50,
    "eval_iterations": 20,
    "metric_rollout_len": 1000,
}

# Active tasks — each name is both the folder under experiments/ and the CLI key.
TASKS: dict[str, dict] = {
    "sixteen_word": dict(_SIXTEEN_WORD_DEFAULTS),
    "sixteen_word_ns": {
        **dict(_SIXTEEN_WORD_DEFAULTS),
        "word_space": False,
    },
    "sixteen_word_mixed": dict(_MIXED_LENGTH_DEFAULTS),
    "sixteen_word_mixed_ns": {
        **dict(_MIXED_LENGTH_DEFAULTS),
        "word_space": False,
    },
}

# Backward-compatible alias used by training / visualization entry points.
EXPERIMENT_CONFIG: dict[str, dict] = TASKS


def comparison_dir(comparison_name: str, kind: str) -> Path:
    if kind not in COMPARISON_VIZ_KINDS:
        raise ValueError(f"kind must be one of {COMPARISON_VIZ_KINDS}, got {kind!r}")
    return COMPARISONS_ROOT / comparison_name / kind


def experiment_regime(name: str) -> str:
    cfg = TASKS.get(name) or EXPERIMENT_CONFIG.get(name)
    if cfg and "regime" in cfg:
        return str(cfg["regime"])
    base = name
    if base.endswith("_dale"):
        base = base[: -len("_dale")]
    if base.endswith("_s"):
        base = base[:-2]
    return base


def experiment_uses_word_space(name: str) -> bool:
    cfg = TASKS.get(name) or EXPERIMENT_CONFIG.get(name)
    if cfg is not None:
        return bool(cfg.get("word_space", False))
    return name.endswith("_s")


def spaced_experiment_name(regime: str) -> str:
    """Legacy helper: spaced folder name for a regime (regime + '_s')."""
    return f"{regime}_s"


def experiment_dir(name: str) -> Path:
    return EXPERIMENTS_ROOT / name


def input_path(name: str) -> Path:
    return experiment_dir(name) / "input.txt"


def model_uses_dale(model_type: str) -> bool:
    return model_type == "rnn_dale"


def model_dir(name: str, model_type: str = "rnn") -> Path:
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}, got {model_type!r}")
    return experiment_dir(name) / model_type


def model_path(name: str, model_type: str = "rnn", *, seed: int | None = None) -> Path:
    if model_type in ("rnn", "rnn_dale"):
        fname = f"model_seed{seed}.npz" if seed is not None else "model.npz"
        return model_dir(name, model_type) / fname
    fname = f"model_seed{seed}.pt" if seed is not None else "model.pt"
    return model_dir(name, model_type) / fname


def checkpoint_path(name: str, model_type: str = "rnn", *, seed: int | None = None) -> Path:
    if seed is not None:
        seeded = model_path(name, model_type, seed=seed)
        if seeded.is_file():
            return seeded
    return model_path(name, model_type)


def model_config_path(name: str) -> Path:
    return model_dir(name, "transformer") / "model_config.json"


def training_meta_path(name: str) -> Path:
    return model_dir(name, "transformer") / "training_meta.json"


def shared_dir(name: str) -> Path:
    return experiment_dir(name) / "shared"


def plots_dir(name: str, model_type: str = "rnn") -> Path:
    return model_dir(name, model_type) / "plots"


def learning_dynamics_dir(name: str, model_type: str = "rnn") -> Path:
    return plots_dir(name, model_type) / "learning_dynamics"


def ensure_experiment_dirs(name: str, model_type: str = "rnn") -> None:
    experiment_dir(name).mkdir(parents=True, exist_ok=True)
    shared_dir(name).mkdir(parents=True, exist_ok=True)
    plots_dir(name, model_type).mkdir(parents=True, exist_ok=True)
    learning_dynamics_dir(name, model_type).mkdir(parents=True, exist_ok=True)
