"""Experiment directory layout with separate RNN and Transformer outputs.

experiments/<name>/
    input.txt                 # shared training corpus
    shared/                   # vocabulary trie + DFA (model-agnostic)
    rnn/
        model.npz
        plots/                # RNN-specific analysis figures
        learning_dynamics/
    transformer/
        model.pt
        model_config.json
        training_meta.json
        plots/                # transformer-specific analysis figures
        learning_dynamics/
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"

MODEL_TYPES = ("rnn", "transformer")

_BASE_CONFIG: dict[str, dict] = {
    # Tiny 2D sandbox: 3 words, short corpus, fast training for transformer debugging.
    "three_word_overlap": {
        "chars": 3_000,
        "steps": 500,
        "viz_length": 40,
        "hidden_size": 12,
        "sequence_length": 8,
        "num_heads": 1,
        "n_layer": 1,
        "eval_interval": 100,
        "eval_iterations": 10,
        "metric_rollout_len": 400,
    },
    "ten_word_overlap": {
        "chars": 50_000,
        "steps": 15_000,
        "viz_length": 50,
        "hidden_size": 32,
        "sequence_length": 40,
    },
    "twelve_word_overlap": {"chars": 50_000, "steps": 1_500, "viz_length": 150},
    "sixteen_word_overlap": {
        "chars": 50_000,
        "steps": 15_000,
        "viz_length": 50,
        "hidden_size": 32,
        "sequence_length": 40,
    },
    "six_word_overlap": {"chars": 50_000, "steps": 1_500, "viz_length": 150},
    "six_word_overlap_sin": {"chars": 50_000, "steps": 1_500, "viz_length": 150},
    "ten_four_letter_overlap": {
        "chars": 50_000,
        "steps": 15_000,
        "viz_length": 50,
        "hidden_size": 32,
        "sequence_length": 40,
    },
    "ten_four_letter_overlap_dale": {
        "regime": "ten_four_letter_overlap",
        "chars": 50_000,
        "steps": 15_000,
        "viz_length": 50,
        "hidden_size": 50,
        "dale": True,
        "e_fraction": 0.8,
        "sequence_length": 40,
    },
}

# Transformer defaults aligned with RNN regimes (char-level LM).
TRANSFORMER_DEFAULTS: dict[str, object] = {
    "n_embd": 32,
    "block_size": 40,
    "num_heads": 1,
    "head_size": 32,
    "n_layer": 2,
    "use_layernorm": True,
    "use_residual": True,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "eval_interval": 500,
    "eval_iterations": 50,
}


def spaced_experiment_name(regime: str) -> str:
    return f"{regime}_s"


def experiment_folder_name(regime: str, *, word_space: bool, dale: bool) -> str:
    name = spaced_experiment_name(regime) if word_space else regime
    return f"{name}_dale" if dale else name


def _build_experiment_config() -> dict[str, dict]:
    configs: dict[str, dict] = {}
    for regime, cfg in _BASE_CONFIG.items():
        task_regime = cfg.get("regime", regime)
        dale = bool(cfg.get("dale", False))
        for word_space in (False, True):
            exp_name = experiment_folder_name(task_regime, word_space=word_space, dale=dale)
            configs[exp_name] = {
                **cfg,
                "regime": task_regime,
                "word_space": word_space,
                "dale": dale,
            }
    return configs


EXPERIMENT_CONFIG: dict[str, dict] = _build_experiment_config()


def experiment_uses_word_space(name: str) -> bool:
    return bool(EXPERIMENT_CONFIG.get(name, {}).get("word_space", False))


def experiment_regime(name: str) -> str:
    cfg = EXPERIMENT_CONFIG.get(name)
    if cfg and "regime" in cfg:
        return cfg["regime"]
    base = name
    if base.endswith("_dale"):
        base = base[: -len("_dale")]
    if base.endswith("_s"):
        base = base[:-2]
    return base


def experiment_dir(name: str) -> Path:
    return EXPERIMENTS_ROOT / name


def input_path(name: str) -> Path:
    return experiment_dir(name) / "input.txt"


def model_dir(name: str, model_type: str = "rnn") -> Path:
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}, got {model_type!r}")
    return experiment_dir(name) / model_type


def model_path(name: str, model_type: str = "rnn") -> Path:
    if model_type == "rnn":
        return model_dir(name, model_type) / "model.npz"
    return model_dir(name, model_type) / "model.pt"


def model_config_path(name: str) -> Path:
    return model_dir(name, "transformer") / "model_config.json"


def training_meta_path(name: str) -> Path:
    return model_dir(name, "transformer") / "training_meta.json"


def shared_dir(name: str) -> Path:
    return experiment_dir(name) / "shared"


def plots_dir(name: str, model_type: str = "rnn") -> Path:
    return model_dir(name, model_type) / "plots"


def learning_dynamics_dir(name: str, model_type: str = "rnn") -> Path:
    return model_dir(name, model_type) / "learning_dynamics"


def plot_path(name: str, plot_name: str, model_type: str = "rnn") -> Path:
    return plots_dir(name, model_type) / plot_name


def ensure_experiment_dirs(name: str, model_type: str = "rnn") -> None:
    experiment_dir(name).mkdir(parents=True, exist_ok=True)
    shared_dir(name).mkdir(parents=True, exist_ok=True)
    plots_dir(name, model_type).mkdir(parents=True, exist_ok=True)
    learning_dynamics_dir(name, model_type).mkdir(parents=True, exist_ok=True)
