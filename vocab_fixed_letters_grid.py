"""Fixed alphabet, systematic factorial vocab grid (length x word count).

Unlike ``vocab_fixed_letters_dfa`` (which rejection-samples candidates and picks
the closest to preset DFA targets), this sweep is a plain crossed design: for
every (word_length, n_words) cell, words are drawn uniformly without
replacement from all strings of that exact length over a fixed alphabet.
Minimized DFA size is measured, never targeted, so the generating procedure is
fully systematic and |Sigma| is identical across runs.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

COMPARISON_NAME = "fixed_letters_grid_ns"
TASK_PREFIX = "fixgrid"
ALPHABET = "abcd"  # |Sigma| = 4, fixed for every run
N_LETTERS = len(ALPHABET)
WORD_LENS: tuple[int, ...] = (3, 4, 5, 6)
N_WORDS_LEVELS: tuple[int, ...] = (3, 5, 8, 12, 18, 25)
RUNS_PER_CELL = 2
N_RUNS = len(WORD_LENS) * len(N_WORDS_LEVELS) * RUNS_PER_CELL  # 48
HIDDEN_SIZE = 200  # H=400+LR=0.04 plateau-stalled; H=200+LR=0.1+dropout=0 solved mid cells
DEFAULT_SEEDS: tuple[int, ...] = (1,)
BANK_SAMPLE_SEED = 20260810


def regime_name(run_id: int) -> str:
    return f"{TASK_PREFIX}_r{run_id:02d}"


def task_name(run_id: int) -> str:
    return f"{regime_name(run_id)}_ns"


def _all_words(length: int) -> list[str]:
    """Every string of exactly ``length`` over ALPHABET, in lexicographic order."""
    words = [""]
    for _ in range(length):
        words = [w + ch for w in words for ch in ALPHABET]
    return words


def _sample_vocab(n_words: int, length: int, rng: random.Random) -> list[str]:
    """Uniform sample of distinct words; resample until every letter appears."""
    pool = _all_words(length)
    if n_words > len(pool):
        raise ValueError(f"n_words={n_words} exceeds {len(pool)} length-{length} words")
    for _ in range(10_000):
        words = rng.sample(pool, n_words)
        if {ch for w in words for ch in w} == set(ALPHABET):
            rng.shuffle(words)
            return words
    raise RuntimeError(f"could not cover alphabet with {n_words} length-{length} words")


def build_run_plan(*, seed: int = BANK_SAMPLE_SEED) -> list[dict]:
    """Deterministic crossed design: length x n_words x replicate."""
    rng = random.Random(seed)
    plan: list[dict] = []
    run_id = 0
    for length in WORD_LENS:
        for n_words in N_WORDS_LEVELS:
            for rep in range(RUNS_PER_CELL):
                words = _sample_vocab(n_words, length, rng)
                plan.append({
                    "run_id": run_id,
                    "n_words": int(n_words),
                    "word_length": int(length),
                    "rep": int(rep),
                    "n_letters": int(N_LETTERS),
                    "alphabet": ALPHABET,
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


def fixed_grid_task_config(run_id: int) -> dict[str, object]:
    """Dale grid hyperparams: dropout off, LR=0.1, full step budget (no plateau kill)."""
    entry = run_plan()[run_id]
    words: list[str] = list(entry["words"])
    n_words = int(entry["n_words"])
    length = int(entry["word_length"])
    mean_length = float(length)
    max_length = length

    chars = max(40_000, int(n_words * mean_length * 800))
    # Plateau-stop was killing unsolved nets at 50% budget. Train to the cap;
    # only the 3% word-error success rule may stop early.
    steps = min(150_000, max(80_000, int(n_words * 1500 + mean_length * 5000)))
    if n_words >= 18:
        steps = max(steps, 120_000)
    if n_words >= 25 or length >= 6:
        steps = max(steps, 150_000)

    viz_length = min(int(sum(len(w) for w in words) + 20), 500)
    sequence_length = max(8, 2 * max_length + (8 if n_words >= 20 else 4))
    metric_rollout_len = min(1000, max(300, int(n_words * mean_length * 2)))
    if n_words >= 15:
        metric_rollout_len = min(5000, max(metric_rollout_len, int(n_words * mean_length * 20)))

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
        "dropout": 0.0,
        "l2_lambda": 1e-4,
        "learning_rate": 0.025,
        "e_fraction": 0.8,
        "stall_patience_evals": 0,
        "stall_min_delta": 0.001,
        "stall_min_iter": int(steps),
        "sweep_n_words": int(n_words),
        "sweep_length": str(length),
        "fixed_grid_run_id": int(run_id),
        "fixed_grid_alphabet": ALPHABET,
        "comparison": COMPARISON_NAME,
    }


def register_fixed_grid_regimes(regimes: dict[str, list[str]]) -> None:
    for entry in run_plan():
        regimes[entry["regime"]] = list(entry["words"])


def register_fixed_grid_tasks(tasks: dict[str, dict]) -> None:
    for entry in run_plan():
        tasks[task_name(int(entry["run_id"]))] = fixed_grid_task_config(int(entry["run_id"]))


def write_run_manifest(out_path: Path) -> Path:
    from vocab_diagrams import build_minimized_vocabulary_automaton

    runs = []
    for entry in run_plan():
        words = list(entry["words"])
        automaton = build_minimized_vocabulary_automaton(words)
        runs.append({
            **entry,
            "n_dfa_states": int(automaton.dfa._n),
        })
    payload = {
        "comparison": COMPARISON_NAME,
        "alphabet": ALPHABET,
        "n_letters_fixed": N_LETTERS,
        "bank_sample_seed": BANK_SAMPLE_SEED,
        "n_runs": N_RUNS,
        "word_lens": list(WORD_LENS),
        "n_words_levels": list(N_WORDS_LEVELS),
        "runs_per_cell": RUNS_PER_CELL,
        "note": "Crossed length x n_words grid; uniform words over fixed alphabet; DFA measured not targeted.",
        "runs": runs,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
