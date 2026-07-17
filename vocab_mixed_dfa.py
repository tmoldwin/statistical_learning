"""Mixed-length English vocab runs indexed by DFA size (not word count × length).

Word banks: 20 real English words each of lengths 3, 4, 5, 6 (80 total).
Each run draws ``n_words`` uniformly from 1..25 by sampling the pooled bank
without replacement; analyses treat DFA state count as the primary axis.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

COMPARISON_NAME = "mixed_vocab_dfa_ns"
TASK_PREFIX = "mixeddfa"
N_RUNS = 50
N_WORDS_MIN = 1
N_WORDS_MAX = 25
RUNS_PER_COUNT = 2  # 2 × 25 sizes = 50 runs
HIDDEN_SIZE = 100
# Extra capacity ablation (separate comparison dirs / task names).
HIDDEN_SIZE_ABLATION: tuple[int, ...] = (50, 150)
DEFAULT_SEEDS: tuple[int, ...] = (1,)
# Fixed RNG so regimes are reproducible across machines.
BANK_SAMPLE_SEED = 20260714

WORD_BANKS: dict[int, tuple[str, ...]] = {
    3: (
        "cat", "hat", "mat", "rat", "bat",
        "met", "pet", "net", "bet", "wet",
        "can", "ban", "pan", "man", "tan",
        "car", "bar", "tar", "far", "jar",
    ),
    4: (
        "bake", "cake", "lake", "rake", "sake",
        "bank", "tank", "rank", "sank", "yank",
        "late", "mate", "rate", "gate", "hate",
        "line", "mine", "pine", "wine", "fine",
    ),
    5: (
        "light", "night", "right", "sight", "fight",
        "bound", "found", "hound", "pound", "round",
        "batch", "catch", "hatch", "match", "patch",
        "cream", "dream", "gleam", "steam", "paint",
    ),
    6: (
        "nation", "ration", "action", "motion",
        "bought", "fought", "sought", "taught",
        "master", "faster", "sister", "mister",
        "moment", "cement", "talent", "patent",
        "garden", "bottle", "window", "forest",
    ),
}

POOL_LENGTHS: tuple[int, ...] = (3, 4, 5, 6)


def _validate_banks() -> None:
    for length, words in WORD_BANKS.items():
        if len(words) != 20:
            raise ValueError(f"length-{length} bank must have 20 words, got {len(words)}")
        for w in words:
            if len(w) != length:
                raise ValueError(f"expected length {length}, got {w!r} ({len(w)})")
            if not w.isalpha() or not w.islower():
                raise ValueError(f"word must be lowercase alphabetic: {w!r}")
    pooled = [w for words in WORD_BANKS.values() for w in words]
    if len(set(pooled)) != len(pooled):
        raise ValueError("word banks contain duplicates across lengths")


_validate_banks()


def pooled_bank() -> list[str]:
    return [w for length in POOL_LENGTHS for w in WORD_BANKS[length]]


def regime_name(run_id: int) -> str:
    return f"{TASK_PREFIX}_r{run_id:02d}"


def task_name(run_id: int, *, hidden_size: int = HIDDEN_SIZE) -> str:
    """Task folder name. H=100 keeps legacy ``mixeddfa_rXX_ns`` names."""
    if int(hidden_size) == HIDDEN_SIZE:
        return f"{regime_name(run_id)}_ns"
    return f"{TASK_PREFIX}_h{int(hidden_size)}_r{run_id:02d}_ns"


def comparison_name_for_h(hidden_size: int) -> str:
    if int(hidden_size) == HIDDEN_SIZE:
        return COMPARISON_NAME
    return f"mixed_vocab_dfa_h{int(hidden_size)}_ns"


def _sample_vocab(n_words: int, rng: random.Random) -> list[str]:
    pool = pooled_bank()
    if n_words < 1 or n_words > len(pool):
        raise ValueError(f"n_words must be in 1..{len(pool)}, got {n_words}")
    words = rng.sample(pool, n_words)
    rng.shuffle(words)
    return words


def build_run_plan(*, n_runs: int = N_RUNS, seed: int = BANK_SAMPLE_SEED) -> list[dict]:
    """Deterministic list of run metadata: run_id, n_words, words."""
    if n_runs != N_RUNS:
        raise ValueError(f"this sweep is fixed at {N_RUNS} runs (got {n_runs})")
    rng = random.Random(seed)
    plan: list[dict] = []
    run_id = 0
    for n_words in range(N_WORDS_MIN, N_WORDS_MAX + 1):
        for _rep in range(RUNS_PER_COUNT):
            words = _sample_vocab(n_words, rng)
            plan.append({
                "run_id": run_id,
                "n_words": n_words,
                "words": words,
                "regime": regime_name(run_id),
                "task": task_name(run_id),
            })
            run_id += 1
    assert len(plan) == N_RUNS
    return plan


_PLAN: list[dict] | None = None


def run_plan() -> list[dict]:
    global _PLAN
    if _PLAN is None:
        _PLAN = build_run_plan()
    return _PLAN


def words_for_run(run_id: int) -> list[str]:
    return list(run_plan()[run_id]["words"])


def iter_runs() -> Iterable[dict]:
    yield from run_plan()


def iter_task_names() -> Iterable[str]:
    for entry in run_plan():
        yield entry["task"]


def mixed_dfa_task_config(run_id: int, *, hidden_size: int = HIDDEN_SIZE) -> dict[str, object]:
    entry = run_plan()[run_id]
    words: list[str] = list(entry["words"])
    n_words = int(entry["n_words"])
    mean_length = float(sum(len(w) for w in words)) / len(words)
    max_length = max(len(w) for w in words)

    chars = max(30_000, int(n_words * mean_length * 600))
    steps = min(120_000, max(15_000, int(n_words * 600 + mean_length * 2500)))
    if n_words >= 20:
        steps = max(steps, 80_000)

    viz_length = min(int(sum(len(w) for w in words) + 20), 500)
    sequence_length = max(8, 2 * max_length + (8 if n_words >= 20 else 4))
    metric_rollout_len = min(1000, max(300, int(n_words * mean_length * 2)))
    if n_words >= 15:
        metric_rollout_len = min(5000, max(metric_rollout_len, int(n_words * mean_length * 20)))
    stall_min_iter = max(
        int(max(1_000, steps * 0.30)),
        max(1_000, int(steps * 0.08)),
    )
    stall_patience_evals = 60 if n_words >= 20 else 36
    stall_min_delta = 0.0015

    return {
        "regime": regime_name(run_id),
        "word_space": False,
        "chars": int(chars),
        "steps": int(steps),
        "target_word_error_frac": 0.03,
        "early_stop_patience": 3,
        "min_checkpoint_iter": max(1_000, int(steps * 0.08)),
        "viz_length": int(viz_length),
        "hidden_size": int(hidden_size),
        "sequence_length": int(sequence_length),
        "eval_interval": 50,
        "eval_iterations": 20,
        "metric_rollout_len": int(metric_rollout_len),
        "train_ratio": 0.9,
        "dropout": 0.25,
        "l2_lambda": 1e-4,
        "learning_rate": 0.04 if max_length >= 5 or n_words >= 20 else 0.1,
        "stall_patience_evals": int(stall_patience_evals),
        "stall_min_delta": float(stall_min_delta),
        "stall_min_iter": int(stall_min_iter),
        "sweep_n_words": int(n_words),
        "sweep_length": "mixed",
        "mixed_dfa_run_id": int(run_id),
        "mixed_dfa_hidden_size": int(hidden_size),
        "comparison": comparison_name_for_h(int(hidden_size)),
    }


def register_mixed_dfa_regimes(regimes: dict[str, list[str]]) -> None:
    for entry in run_plan():
        regimes[entry["regime"]] = list(entry["words"])


def register_mixed_dfa_tasks(tasks: dict[str, dict]) -> None:
    for entry in run_plan():
        rid = int(entry["run_id"])
        tasks[task_name(rid)] = mixed_dfa_task_config(rid)
        for h in HIDDEN_SIZE_ABLATION:
            tasks[task_name(rid, hidden_size=h)] = mixed_dfa_task_config(
                rid, hidden_size=h,
            )


def iter_tasks_for_h(hidden_size: int = HIDDEN_SIZE) -> Iterable[dict]:
    """Run plan entries with task names rewritten for ``hidden_size``."""
    for entry in run_plan():
        rid = int(entry["run_id"])
        yield {
            **entry,
            "task": task_name(rid, hidden_size=hidden_size),
            "hidden_size": int(hidden_size),
            "comparison": comparison_name_for_h(hidden_size),
        }


def write_run_manifest(out_path: Path) -> Path:
    """Write JSON describing banks + every run (for analysis / paper)."""
    from vocab_diagrams import build_minimized_vocabulary_automaton

    runs = []
    for entry in run_plan():
        words = list(entry["words"])
        automaton = build_minimized_vocabulary_automaton(words)
        runs.append({
            **entry,
            "n_dfa_states": int(automaton.dfa._n),
            "length_counts": {
                str(L): sum(1 for w in words if len(w) == L) for L in POOL_LENGTHS
            },
        })
    payload = {
        "comparison": COMPARISON_NAME,
        "bank_sample_seed": BANK_SAMPLE_SEED,
        "n_runs": N_RUNS,
        "n_words_range": [N_WORDS_MIN, N_WORDS_MAX],
        "word_banks": {str(k): list(v) for k, v in WORD_BANKS.items()},
        "runs": runs,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
