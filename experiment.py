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

# Default seed for single-run workflows (input.txt + model.npz).
DEFAULT_SEED = 42

DALE_RNN_DEFAULTS: dict[str, object] = {
    "e_fraction": 0.8,
}

COMPARISON_VIZ_KINDS: tuple[str, ...] = (
    "data",
    "learning_curves",
    "trajectories",
    "states",
    "unit_selectivity",
    "dfa_sensitivity",
)

_SIX_WORD_NS_DEFAULTS: dict[str, object] = {
    "regime": "six_word_overlap",
    "word_space": False,
    "chars": 30_000,
    "steps": 25_000,
    "target_word_error_frac": 0.03,
    "early_stop_patience": 3,
    "min_checkpoint_iter": 2_000,
    "viz_length": 24,
    "hidden_size": 50,
    "sequence_length": 12,
    "eval_interval": 50,
    "eval_iterations": 20,
    "metric_rollout_len": 500,
}

_SIXTEEN_WORD_NS_DEFAULTS: dict[str, object] = {
    "regime": "sixteen_word",
    "word_space": False,
    "chars": 50_000,
    "steps": 50_000,
    "target_word_error_frac": 0.03,
    "early_stop_patience": 3,
    "min_checkpoint_iter": 4_000,
    "viz_length": 50,
    "hidden_size": 128,
    "sequence_length": 12,
    "eval_interval": 50,
    "eval_iterations": 20,
    "metric_rollout_len": 1000,
    "train_ratio": 0.9,
    "dropout": 0.25,
    "l2_lambda": 1e-4,
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

_SIXTEEN_WORD_SPACED_DEFAULTS: dict[str, object] = {
    **_SIXTEEN_WORD_NS_DEFAULTS,
    "word_space": True,
}

_MICRO_CURRICULUM_NS_DEFAULTS: dict[str, object] = {
    "word_space": False,
    "chars": 20_000,
    "steps": 15_000,
    "target_word_error_frac": 0.03,
    "early_stop_patience": 3,
    "min_checkpoint_iter": 1_000,
    "viz_length": 18,
    "demo_snippet_len": 80,
    "hidden_size": 50,
    "sequence_length": 8,
    "eval_interval": 50,
    "eval_iterations": 20,
    "metric_rollout_len": 200,
    "train_ratio": 0.9,
    "dropout": 0.15,
    "l2_lambda": 1e-4,
}

_FIFTY_WORD_NS_DEFAULTS: dict[str, object] = {
    "regime": "fifty_word",
    "word_space": False,
    "chars": 100_000,
    "steps": 100_000,
    "target_word_error_frac": 0.03,
    "early_stop_patience": 3,
    "viz_length": 151,  # 50 words × 3 chars + 1
    "hidden_size": 150,
    "sequence_length": 12,
    "eval_interval": 50,
    "eval_iterations": 20,
    "metric_rollout_len": 1000,
}

# Active tasks — each name is both the folder under experiments/ and the CLI key.
TASKS: dict[str, dict] = {
    "six_word_overlap_ns": dict(_SIX_WORD_NS_DEFAULTS),
    "six_word_four_letter_ns": {
        **dict(_SIX_WORD_NS_DEFAULTS),
        "regime": "six_word_four_letter",
        "viz_length": 30,
        "sequence_length": 16,
    },
    "six_word_five_letter_ns": {
        **dict(_SIX_WORD_NS_DEFAULTS),
        "regime": "six_word_five_letter",
        "hidden_size": 64,
        "learning_rate": 0.04,
        "viz_length": 36,
        "sequence_length": 20,
        "steps": 35_000,
    },
    "sixteen_word": dict(_SIXTEEN_WORD_SPACED_DEFAULTS),
    "sixteen_word_ns": dict(_SIXTEEN_WORD_NS_DEFAULTS),
    "sixteen_word_mixed": dict(_MIXED_LENGTH_DEFAULTS),
    "sixteen_word_mixed_ns": {
        **dict(_SIXTEEN_WORD_NS_DEFAULTS),
        "regime": "sixteen_word_mixed",
        "hidden_size": 128,
        "learning_rate": 0.04,
        "viz_length": 115,
        "sequence_length": 48,
        "steps": 100_000,
    },
    "sixteen_word_four_letter_ns": {
        **dict(_SIXTEEN_WORD_NS_DEFAULTS),
        "regime": "sixteen_word_four_letter",
        "viz_length": 64,
        "sequence_length": 16,
    },
    "sixteen_word_five_letter_ns": {
        **dict(_SIXTEEN_WORD_NS_DEFAULTS),
        "regime": "sixteen_word_five_letter",
        "hidden_size": 128,
        "learning_rate": 0.04,
        "viz_length": 81,
        "sequence_length": 32,
        "steps": 75_000,
    },
    "sixteen_word_six_letter_ns": {
        **dict(_SIXTEEN_WORD_NS_DEFAULTS),
        "regime": "sixteen_word_six_letter",
        "hidden_size": 128,
        "learning_rate": 0.04,
        "viz_length": 97,
        "sequence_length": 40,
        "steps": 90_000,
    },
    "sixteen_word_seven_letter_ns": {
        **dict(_SIXTEEN_WORD_NS_DEFAULTS),
        "regime": "sixteen_word_seven_letter",
        "hidden_size": 128,
        "learning_rate": 0.04,
        "viz_length": 113,
        "sequence_length": 48,
        "steps": 100_000,
    },
    "sixteen_word_ns_h500": {
        **dict(_SIXTEEN_WORD_NS_DEFAULTS),
        "hidden_size": 500,
        "learning_rate": 0.04,
        "eval_interval": 500,
        "metric_rollout_len": 500,
        "metric_num_rollouts": 1,
    },
    "sixteen_word_mixed_ns_h500": {
        **dict(_SIXTEEN_WORD_NS_DEFAULTS),
        "regime": "sixteen_word_mixed",
        "hidden_size": 500,
        "learning_rate": 0.04,
        "viz_length": 115,
        "sequence_length": 48,
        "steps": 100_000,
        "eval_interval": 500,
        "metric_rollout_len": 500,
        "metric_num_rollouts": 1,
    },
    "sixteen_word_four_letter_ns_h500": {
        **dict(_SIXTEEN_WORD_NS_DEFAULTS),
        "regime": "sixteen_word_four_letter",
        "hidden_size": 500,
        "learning_rate": 0.04,
        "viz_length": 64,
        "sequence_length": 16,
        "eval_interval": 500,
        "metric_rollout_len": 500,
        "metric_num_rollouts": 1,
    },
    "sixteen_word_five_letter_ns_h500": {
        **dict(_SIXTEEN_WORD_NS_DEFAULTS),
        "regime": "sixteen_word_five_letter",
        "hidden_size": 500,
        "learning_rate": 0.04,
        "viz_length": 81,
        "sequence_length": 32,
        "steps": 75_000,
        "eval_interval": 500,
        "metric_rollout_len": 500,
        "metric_num_rollouts": 1,
    },
    "sixteen_word_six_letter_ns_h500": {
        **dict(_SIXTEEN_WORD_NS_DEFAULTS),
        "regime": "sixteen_word_six_letter",
        "hidden_size": 500,
        "learning_rate": 0.04,
        "viz_length": 97,
        "sequence_length": 40,
        "steps": 90_000,
        "eval_interval": 500,
        "metric_rollout_len": 500,
        "metric_num_rollouts": 1,
    },
    "sixteen_word_seven_letter_ns_h500": {
        **dict(_SIXTEEN_WORD_NS_DEFAULTS),
        "regime": "sixteen_word_seven_letter",
        "hidden_size": 500,
        "learning_rate": 0.04,
        "viz_length": 113,
        "sequence_length": 48,
        "steps": 100_000,
        "eval_interval": 500,
        "metric_rollout_len": 500,
        "metric_num_rollouts": 1,
    },
    "fifty_word_ns": dict(_FIFTY_WORD_NS_DEFAULTS),
    "fifty_word_four_letter_ns": {
        **dict(_FIFTY_WORD_NS_DEFAULTS),
        "regime": "fifty_word_four_letter",
        "viz_length": 201,  # 50 × 4 + 1
        "sequence_length": 16,
    },
    "fifty_word_five_letter_ns": {
        **dict(_FIFTY_WORD_NS_DEFAULTS),
        "regime": "fifty_word_five_letter",
        "hidden_size": 128,
        "learning_rate": 0.04,
        "viz_length": 251,  # 50 × 5 + 1
        "sequence_length": 32,
        "steps": 150_000,
    },
    "two_word_disjoint_ns": {
        **dict(_MICRO_CURRICULUM_NS_DEFAULTS),
        "regime": "two_word_disjoint",
        "viz_length": 12,
    },
    "two_word_pos_overlap_ns": {
        **dict(_MICRO_CURRICULUM_NS_DEFAULTS),
        "regime": "two_word_pos_overlap",
        "viz_length": 12,
    },
    "two_word_prefix_branch_ns": {
        **dict(_MICRO_CURRICULUM_NS_DEFAULTS),
        "regime": "two_word_prefix_branch",
        "viz_length": 12,
    },
    "three_word_overlap_ns": {
        **dict(_MICRO_CURRICULUM_NS_DEFAULTS),
        "regime": "three_word_overlap",
    },
    "three_word_permutation_ns": {
        **dict(_MICRO_CURRICULUM_NS_DEFAULTS),
        "regime": "three_word_permutation",
        "timestep_noise_std": 0.05,
    },
    "three_word_ca_hub_ns": {
        **dict(_MICRO_CURRICULUM_NS_DEFAULTS),
        "regime": "three_word_ca_hub",
    },
}

from vocab_sweep import register_sweep_tasks
from vocab_sweep_pow2 import register_pow2_sweep_tasks

register_sweep_tasks(TASKS)
register_pow2_sweep_tasks(TASKS)

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


def seeds_for_task(name: str, model_type: str = "rnn") -> set[int]:
    """RNG seeds with a saved checkpoint for this task."""
    found: set[int] = set()
    if model_path(name, model_type).is_file():
        found.add(DEFAULT_SEED)
    if model_type in ("rnn", "rnn_dale"):
        pattern = "model_seed*.npz"
    else:
        pattern = "model_seed*.pt"
    for path in model_dir(name, model_type).glob(pattern):
        found.add(int(path.stem.removeprefix("model_seed")))
    return found


def common_seeds(tasks: tuple[str, ...], model_type: str = "rnn") -> tuple[int, ...]:
    """Seeds that have checkpoints for every task in the comparison."""
    per_task = [seeds_for_task(task, model_type) for task in tasks]
    if not per_task:
        return (DEFAULT_SEED,)
    shared = set.intersection(*per_task)
    return tuple(sorted(shared)) if shared else (DEFAULT_SEED,)


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
