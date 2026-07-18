"""Fixed small alphabet, synthetic words, stratified by minimized DFA size.

Alphabet size is held fixed (default |Σ|=4); words are random strings over that
alphabet. Runs are chosen to span a wide range of minimized DFA state counts so
analyses can attribute geometry to DFA size rather than alphabet size.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

COMPARISON_NAME = "fixed_letters_dfa_ns"
TASK_PREFIX = "fixlettdfa"
ALPHABET = "abcd"  # |Σ| = 4
N_LETTERS = len(ALPHABET)
N_RUNS = 20
WORD_LENS = (2, 3, 4, 5, 6)
HIDDEN_SIZE = 100
DEFAULT_SEEDS: tuple[int, ...] = (1,)
BANK_SAMPLE_SEED = 20260718
# Evenly spaced DFA targets; sampler picks closest synthetic vocab to each.
DFA_TARGETS: tuple[int, ...] = (
    6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
    26, 28, 30, 32, 34, 36, 38, 40, 42, 44,
)


def regime_name(run_id: int) -> str:
    return f"{TASK_PREFIX}_r{run_id:02d}"


def task_name(run_id: int, *, hidden_size: int = HIDDEN_SIZE) -> str:
    if int(hidden_size) != HIDDEN_SIZE:
        raise ValueError("fixed-letters sweep is H=100 only for now")
    return f"{regime_name(run_id)}_ns"


def _n_dfa(words: list[str]) -> int:
    from vocab_diagrams import build_minimized_vocabulary_automaton

    return int(build_minimized_vocabulary_automaton(words).dfa._n)


def _rand_word(rng: random.Random) -> str:
    length = rng.choice(WORD_LENS)
    return "".join(rng.choice(ALPHABET) for _ in range(length))


def _sample_vocab(n_words: int, rng: random.Random, *, require_full_alphabet: bool = True) -> list[str]:
    """Sample unique synthetic words; optionally force every alphabet letter to appear."""
    if require_full_alphabet and n_words < N_LETTERS:
        raise ValueError(f"need n_words >= {N_LETTERS} to cover alphabet {ALPHABET!r}")
    words: set[str] = set()
    if require_full_alphabet:
        # Seed one dedicated letter word per alphabet char so |Σ| is exact.
        for ch in ALPHABET:
            words.add(ch * rng.choice((2, 3)))
    guard = 0
    while len(words) < n_words and guard < 50_000:
        words.add(_rand_word(rng))
        guard += 1
    if len(words) < n_words:
        raise RuntimeError(f"could not sample {n_words} unique words over {ALPHABET!r}")
    letters = {ch for w in words for ch in w}
    if require_full_alphabet and letters != set(ALPHABET):
        raise RuntimeError(f"alphabet leak/mismatch: {sorted(letters)} vs {ALPHABET}")
    out = list(words)
    rng.shuffle(out)
    return out


def _candidate_pool(rng: random.Random, *, n_candidates: int = 5000) -> list[tuple[int, int, list[str]]]:
    """Return (n_dfa, n_words, words) with exact alphabet usage."""
    pool: list[tuple[int, int, list[str]]] = []
    for n_words in range(N_LETTERS, 40):
        trials = max(50, n_candidates // 30)
        for _ in range(trials):
            try:
                words = _sample_vocab(n_words, rng, require_full_alphabet=True)
            except RuntimeError:
                continue
            dfa = _n_dfa(words)
            pool.append((dfa, n_words, words))
            if len(pool) >= n_candidates:
                return pool
    return pool


def build_run_plan(*, seed: int = BANK_SAMPLE_SEED) -> list[dict]:
    """One run per DFA target: closest synthetic vocab with exact |Σ|=4."""
    rng = random.Random(seed)
    pool = _candidate_pool(rng)
    if len(pool) < N_RUNS:
        raise RuntimeError(f"candidate pool too small: {len(pool)}")
    used_words: set[tuple[str, ...]] = set()
    used_dfa: set[int] = set()
    plan: list[dict] = []
    for run_id, target in enumerate(DFA_TARGETS):
        best = None
        best_key = None
        for dfa, n_words, words in pool:
            key = tuple(sorted(words))
            if key in used_words:
                continue
            err = abs(dfa - target)
            # Prefer unused DFA sizes so the axis isn't collapsed.
            dup = 1 if dfa in used_dfa else 0
            score = (dup, err, abs(n_words - max(N_LETTERS, target // 2)))
            if best_key is None or score < best_key:
                best_key = score
                best = (dfa, n_words, words)
        if best is None:
            raise RuntimeError(f"no candidate for DFA target {target}")
        dfa, n_words, words = best
        used_words.add(tuple(sorted(words)))
        used_dfa.add(int(dfa))
        letters = {ch for w in words for ch in w}
        if letters != set(ALPHABET):
            raise RuntimeError(f"run {run_id} alphabet {sorted(letters)} != {ALPHABET}")
        plan.append({
            "run_id": run_id,
            "n_words": int(n_words),
            "n_letters": int(N_LETTERS),
            "n_dfa_states": int(dfa),
            "dfa_target": int(target),
            "words": list(words),
            "alphabet": ALPHABET,
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


def fixed_letters_task_config(run_id: int) -> dict[str, object]:
    """Hyperparams mirrored from mixed_dfa_task_config (H=100)."""
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
    stall_min_iter = max(int(max(1_000, steps * 0.30)), max(1_000, int(steps * 0.08)))
    stall_patience_evals = 60 if n_words >= 20 else 36

    return {
        "regime": regime_name(run_id),
        "word_space": False,
        "chars": int(chars),
        "steps": int(steps),
        "target_word_error_frac": 0.03,
        "early_stop_patience": 3,
        "min_checkpoint_iter": max(1_000, int(steps * 0.08)),
        "viz_length": int(viz_length),
        "hidden_size": int(HIDDEN_SIZE),
        "sequence_length": int(sequence_length),
        "eval_interval": 50,
        "eval_iterations": 20,
        "metric_rollout_len": int(metric_rollout_len),
        "train_ratio": 0.9,
        "dropout": 0.25,
        "l2_lambda": 1e-4,
        "learning_rate": 0.04 if max_length >= 5 or n_words >= 20 else 0.1,
        "stall_patience_evals": int(stall_patience_evals),
        "stall_min_delta": 0.0015,
        "stall_min_iter": int(stall_min_iter),
        "sweep_n_words": int(n_words),
        "sweep_length": "synth",
        "fixed_letters_run_id": int(run_id),
        "fixed_letters_alphabet": ALPHABET,
        "comparison": COMPARISON_NAME,
    }


def register_fixed_letters_dfa_regimes(regimes: dict[str, list[str]]) -> None:
    """No-op stubs. Words are filled by ``finalize_registrations`` (via experiment)."""
    for run_id in range(N_RUNS):
        key = regime_name(run_id)
        # Don't clobber an already-finalized regime (task.py import order).
        if key not in regimes or not regimes[key]:
            regimes[key] = []


def register_fixed_letters_dfa_tasks(tasks: dict[str, dict]) -> None:
    """Stub tasks unless already finalized."""
    for run_id in range(N_RUNS):
        key = task_name(run_id)
        if key in tasks and tasks[key].get("fixed_letters_alphabet"):
            continue
        tasks[key] = {
            "regime": regime_name(run_id),
            "comparison": COMPARISON_NAME,
            "hidden_size": HIDDEN_SIZE,
            "fixed_letters_run_id": int(run_id),
            "word_space": False,
            "chars": 30_000,
            "steps": 15_000,
        }


def finalize_registrations(tasks: dict[str, dict], regimes: dict[str, list[str]]) -> None:
    """Fill regimes/tasks once ``experiment`` / ``vocab_diagrams`` can import cleanly."""
    for entry in run_plan():
        rid = int(entry["run_id"])
        regimes[entry["regime"]] = list(entry["words"])
        tasks[task_name(rid)] = fixed_letters_task_config(rid)


def write_run_manifest(out_path: Path) -> Path:
    runs = []
    for entry in run_plan():
        words = list(entry["words"])
        runs.append({
            **entry,
            "words": words,
            "length_counts": {
                str(L): sum(1 for w in words if len(w) == L) for L in WORD_LENS
            },
        })
    payload = {
        "comparison": COMPARISON_NAME,
        "alphabet": ALPHABET,
        "n_letters_fixed": N_LETTERS,
        "bank_sample_seed": BANK_SAMPLE_SEED,
        "n_runs": N_RUNS,
        "dfa_targets": list(DFA_TARGETS),
        "note": "Synthetic words over fixed alphabet; runs stratified by minimized DFA size.",
        "runs": runs,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
