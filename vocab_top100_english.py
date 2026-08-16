"""Top-100 English word sweeps (OEC frequency list).

Bank: Oxford English Corpus top-100 lemmas as listed on Wikipedia
(https://en.wikipedia.org/wiki/Most_common_words_in_English), lowercased.

Each of 100 runs independently:
  1. sample N uniformly from {1, ..., 20}
  2. sample N distinct words from the bank without replacement

Analyses treat minimized vocabulary DFA size as the primary axis.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

COMPARISON_NAME = "top100_english_ns"
TASK_PREFIX = "top100"
N_RUNS = 100
N_WORDS_MIN = 1
N_WORDS_MAX = 20
HIDDEN_SIZE = 200
DEFAULT_SEEDS: tuple[int, ...] = (1,)
BANK_SAMPLE_SEED = 20260816
DEFAULT_DALE_INIT = 0.01
DEFAULT_LR = 0.025
DEFAULT_DROPOUT = 0.0
DEFAULT_E_FRACTION = 0.8

# OEC top 100 (Wikipedia), lowercased spellings as used in the corpus list.
TOP100_WORDS: tuple[str, ...] = (
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
    "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
    "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
)


def _validate_bank() -> None:
    if len(TOP100_WORDS) != 100:
        raise ValueError(f"expected 100 words, got {len(TOP100_WORDS)}")
    if len(set(TOP100_WORDS)) != 100:
        raise ValueError("TOP100_WORDS contains duplicates")
    for w in TOP100_WORDS:
        if not w.isalpha() or not w.islower():
            raise ValueError(f"word must be lowercase alphabetic: {w!r}")


_validate_bank()


def regime_name(run_id: int) -> str:
    return f"{TASK_PREFIX}_r{run_id:02d}"


def task_name(run_id: int) -> str:
    return f"{regime_name(run_id)}_ns"


def _sample_run(rng: random.Random) -> tuple[int, list[str]]:
    n_words = rng.randint(N_WORDS_MIN, N_WORDS_MAX)
    words = rng.sample(list(TOP100_WORDS), n_words)
    rng.shuffle(words)
    return n_words, words


def build_run_plan(*, n_runs: int = N_RUNS, seed: int = BANK_SAMPLE_SEED) -> list[dict]:
    """100 independent (N, vocab) draws; N ~ Uniform{1..20}."""
    if n_runs != N_RUNS:
        raise ValueError(f"this sweep is fixed at {N_RUNS} runs (got {n_runs})")
    rng = random.Random(seed)
    plan: list[dict] = []
    for run_id in range(n_runs):
        n_words, words = _sample_run(rng)
        plan.append({
            "run_id": run_id,
            "n_words": int(n_words),
            "words": words,
            "regime": regime_name(run_id),
            "task": task_name(run_id),
        })
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


def top100_task_config(run_id: int) -> dict[str, object]:
    """Dale v5-style recipe: input reaches E and I, H=200, LR=0.025."""
    entry = run_plan()[run_id]
    words: list[str] = list(entry["words"])
    n_words = int(entry["n_words"])
    mean_length = float(sum(len(w) for w in words)) / len(words)
    max_length = max(len(w) for w in words)

    chars = max(30_000, int(n_words * mean_length * 600))
    steps = min(120_000, max(20_000, int(n_words * 800 + mean_length * 3000)))
    if n_words >= 15:
        steps = max(steps, 80_000)
    if n_words >= 20:
        steps = max(steps, 100_000)

    viz_length = min(int(sum(len(w) for w in words) + 20), 500)
    sequence_length = max(8, 2 * max_length + (8 if n_words >= 15 else 4))
    metric_rollout_len = min(1000, max(300, int(n_words * mean_length * 2)))
    if n_words >= 12:
        metric_rollout_len = min(5000, max(metric_rollout_len, int(n_words * mean_length * 20)))

    # eval_interval=50, so patience in evals -> patience*50 iterations of no
    # best-word-error progress before the plateau rule fires.
    stall_patience_evals = 300

    return {
        "regime": regime_name(run_id),
        "word_space": False,
        "chars": int(chars),
        "steps": int(steps),
        "target_word_error_frac": 0.03,
        "early_stop_patience": 3,
        "min_checkpoint_iter": max(2_000, int(steps * 0.05)),
        "viz_length": int(viz_length),
        "hidden_size": int(HIDDEN_SIZE),
        "sequence_length": int(sequence_length),
        "eval_interval": 50,
        "eval_iterations": 20,
        "metric_rollout_len": int(metric_rollout_len),
        "train_ratio": 0.9,
        "dropout": float(DEFAULT_DROPOUT),
        "l2_lambda": 1e-4,
        "learning_rate": float(DEFAULT_LR),
        "dale_init_scale": float(DEFAULT_DALE_INIT),
        "e_fraction": float(DEFAULT_E_FRACTION),
        # Three-tier stopping. Success: 3% WE held for `early_stop_patience`
        # evals. Plateau: no >=0.5% best-WE gain for 15k iters, but only after
        # 30% of budget so slow starters are not culled (the v1 mistake).
        # Hopeless: best WE still above 50% at 40% of budget -- a net at chance
        # never recovers, and letting those run the full cap wasted hours.
        "stall_patience_evals": int(stall_patience_evals),
        "stall_min_delta": 0.005,
        "stall_min_iter": int(max(15_000, 0.30 * steps)),
        "hopeless_word_error_frac": 0.5,
        "hopeless_min_iter": int(max(20_000, 0.40 * steps)),
        "sweep_n_words": int(n_words),
        "sweep_length": "mixed",
        "top100_run_id": int(run_id),
        "comparison": COMPARISON_NAME,
    }


def register_top100_regimes(regimes: dict[str, list[str]]) -> None:
    for entry in run_plan():
        regimes[entry["regime"]] = list(entry["words"])


def register_top100_tasks(tasks: dict[str, dict]) -> None:
    for entry in run_plan():
        tasks[task_name(int(entry["run_id"]))] = top100_task_config(int(entry["run_id"]))


def write_run_manifest(out_path: Path) -> Path:
    from vocab_diagrams import build_minimized_vocabulary_automaton

    runs = []
    for entry in run_plan():
        words = list(entry["words"])
        automaton = build_minimized_vocabulary_automaton(words)
        runs.append({
            **entry,
            "n_dfa_states": int(automaton.dfa._n),
            "mean_word_length": float(sum(len(w) for w in words) / len(words)),
        })
    n_words_hist = {str(k): 0 for k in range(N_WORDS_MIN, N_WORDS_MAX + 1)}
    for r in runs:
        n_words_hist[str(int(r["n_words"]))] += 1
    payload = {
        "comparison": COMPARISON_NAME,
        "source": "https://en.wikipedia.org/wiki/Most_common_words_in_English",
        "source_note": "Oxford English Corpus top-100 lemmas (Wikipedia table)",
        "bank_sample_seed": BANK_SAMPLE_SEED,
        "n_runs": N_RUNS,
        "n_words_range": [N_WORDS_MIN, N_WORDS_MAX],
        "sampling": "independent: N~Unif{1..20}, then N words without replacement",
        "word_bank": list(TOP100_WORDS),
        "n_words_histogram": n_words_hist,
        "runs": runs,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
