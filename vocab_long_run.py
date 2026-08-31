"""Long unconstrained RNN run on a hard mixed-English vocab (~100 DFA states)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from vocab_mixed_dfa import POOL_LENGTHS, WORD_BANKS

COMPARISON_NAME = "long_run_ns"
REGIME = "long_run"
TASK = "long_run_ns"
HIDDEN_SIZE = 100
DEFAULT_SEEDS: tuple[int, ...] = (1,)
TARGET_DFA_STATES = 100
STEPS = 250_000


def words() -> list[str]:
    """Full mixed English bank (80 words, lengths 3-6)."""
    return [w for length in POOL_LENGTHS for w in WORD_BANKS[length]]


def n_dfa_states() -> int:
    from vocab_diagrams import build_minimized_vocabulary_automaton

    return int(build_minimized_vocabulary_automaton(words()).dfa._n)


def long_run_task_config() -> dict[str, object]:
    vocab = words()
    n_words = len(vocab)
    mean_length = float(sum(len(w) for w in vocab)) / n_words
    max_length = max(len(w) for w in vocab)
    chars = max(80_000, int(n_words * mean_length * 600))
    viz_length = min(int(sum(len(w) for w in vocab) + 20), 500)
    sequence_length = max(12, 2 * max_length + 8)
    metric_rollout_len = min(1200, max(600, int(n_words * mean_length * 8)))
    return {
        "regime": REGIME,
        "word_space": False,
        "chars": int(chars),
        "steps": int(STEPS),
        "target_word_error_frac": 0.03,
        "early_stop_patience": 3,
        "min_checkpoint_iter": 8_000,
        "viz_length": int(viz_length),
        "hidden_size": int(HIDDEN_SIZE),
        "sequence_length": int(sequence_length),
        "eval_interval": 50,
        "eval_iterations": 20,
        "metric_rollout_len": int(metric_rollout_len),
        "train_ratio": 0.9,
        "dropout": 0.25,
        "l2_lambda": 1e-4,
        "learning_rate": 0.04,
        "stall_patience_evals": 0,
        "stall_min_delta": 0.0015,
        "stall_min_iter": 150_000,
        "sweep_n_words": int(n_words),
        "sweep_length": "mixed",
        "comparison": COMPARISON_NAME,
    }


def register_long_run_regimes(regimes: dict[str, list[str]]) -> None:
    regimes[REGIME] = words()


def register_long_run_tasks(tasks: dict[str, dict]) -> None:
    tasks[TASK] = long_run_task_config()


def write_run_manifest(out_path: Path) -> Path:
    vocab = words()
    n_dfa = n_dfa_states()
    payload = {
        "comparison": COMPARISON_NAME,
        "task": TASK,
        "regime": REGIME,
        "target_dfa_states": TARGET_DFA_STATES,
        "n_dfa_states": n_dfa,
        "n_words": len(vocab),
        "words": vocab,
        "hidden_size": HIDDEN_SIZE,
        "steps": STEPS,
        "length_counts": {
            str(L): sum(1 for w in vocab if len(w) == L) for L in POOL_LENGTHS
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def iter_task_names() -> Iterable[str]:
    yield TASK