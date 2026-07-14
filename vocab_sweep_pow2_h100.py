"""Word-count (5–25 step 5) x length 1-6 + mixed; fixed 100 hidden units."""

from __future__ import annotations

import string
from typing import Iterable

from vocab_sweep import build_vocab as _build_vocab_general

SweepLength = int | str

POW2_H100_WORD_COUNTS: tuple[int, ...] = (5, 10, 15, 20, 25)
POW2_H100_LENGTHS: tuple[SweepLength, ...] = (1, 2, 3, 4, 5, 6, "mixed")
POW2_H100_HIDDEN_SIZE = 100
POW2_H100_DEFAULT_SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5)
POW2_H100_SEED_COMPARISON_SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5)
POW2_H100_COMPARISON = "word_count_pow2_sweep_h100_ns"
POW2_H100_TASK_PREFIX = "pow2sweep_h100"

_SINGLE_CHARS: tuple[str, ...] = tuple(
    string.ascii_lowercase + string.ascii_uppercase + string.digits + "!@#$%^&*"
)
_MIXED_POOLS: dict[int, tuple[str, ...]] = {
    3: ("cat", "cet", "cit", "cot", "cut", "cab", "cad", "cam", "can", "cap"),
    4: ("dake", "dank", "date", "dant", "dine", "dock", "deed", "dore", "dorn", "dile"),
    5: ("fight", "found", "fatch", "faint", "fiver", "fress", "fling", "funch", "foard", "fable"),
    6: ("gation", "gought", "gaster", "gement", "gently", "gobble", "garden", "gilded", "grotto", "gasket"),
}


def _two_letter_words(n_words: int) -> list[str]:
    out: list[str] = []
    for c in "bcdfghjklmnpqrstvwxyz":
        for v in "aeiou":
            out.append(c + v)
            if len(out) >= n_words:
                return out
    raise ValueError(f"only {len(out)} distinct 2-letter tokens available, requested {n_words}")


def length_label(length: SweepLength) -> str:
    return "mixed" if length == "mixed" else f"{int(length)}-letter"


def regime_name(n_words: int, length: SweepLength) -> str:
    if length == "mixed":
        return f"{POW2_H100_TASK_PREFIX}_w{n_words}_lmix"
    return f"{POW2_H100_TASK_PREFIX}_w{n_words}_l{length}"


def task_name(n_words: int, length: SweepLength) -> str:
    return f"{regime_name(n_words, length)}_ns"


def _mixed_vocab(n_words: int) -> list[str]:
    n_buckets = 4
    counts = [n_words // n_buckets] * n_buckets
    for i in range(n_words % n_buckets):
        counts[i] += 1
    words: list[str] = []
    for idx, count in enumerate(counts, start=3):
        if count > 0:
            pool = _MIXED_POOLS[idx]
            if count > len(pool):
                raise ValueError(
                    f"mixed-length pool for length {idx} only has {len(pool)} words, requested {count}"
                )
            words.extend(pool[:count])
    return words


def build_vocab(n_words: int, length: SweepLength) -> list[str]:
    if n_words < 1:
        raise ValueError("need at least 1 word")
    if length == "mixed":
        return _mixed_vocab(n_words)
    if length < 1:
        raise ValueError("length must be >= 1")
    if length == 1:
        if n_words > len(_SINGLE_CHARS):
            raise ValueError(
                f"only {len(_SINGLE_CHARS)} distinct 1-letter tokens available, "
                f"requested {n_words}"
            )
        return list(_SINGLE_CHARS[:n_words])
    if length == 2:
        return _two_letter_words(n_words)
    if n_words == 1:
        words = _build_vocab_general(2, length)
        return [words[0]]
    return _build_vocab_general(n_words, length)


def pow2_h100_sweep_task_config(n_words: int, length: SweepLength) -> dict[str, object]:
    words = build_vocab(n_words, length)
    mean_length = float(sum(len(w) for w in words)) / len(words)
    max_length = max(len(w) for w in words)

    chars = max(30_000, int(n_words * mean_length * 600))
    steps = min(120_000, max(15_000, int(n_words * 600 + mean_length * 2500)))
    if n_words >= 20:
        steps = max(steps, 80_000)
    if length == 1:
        steps = min(steps, max(10_000, n_words * 500))

    viz_length = min(int(sum(len(w) for w in words) + 20), 500)
    if length == 1:
        viz_length = min(n_words + 10, 100)

    sequence_length = max(8, 2 * max_length + (8 if n_words >= 20 else 4))
    if length == 1:
        sequence_length = max(4, min(8, max(n_words, 4)))

    metric_rollout_len = min(1000, max(300, int(n_words * mean_length * 2)))
    if n_words >= 15:
        metric_rollout_len = min(5000, max(metric_rollout_len, int(n_words * mean_length * 20)))
    stall_min_iter = max(
        int(max(1_000, steps * 0.30)),
        max(1_000, int(steps * 0.08)),
    )
    stall_patience_evals = 24
    if n_words >= 20:
        stall_patience_evals = 40
    if length == "mixed":
        stall_patience_evals = 60 if n_words >= 20 else 36
    stall_min_delta = 0.0015 if length == "mixed" else 0.001

    return {
        "regime": regime_name(n_words, length),
        "word_space": False,
        "chars": int(chars),
        "steps": int(steps),
        "target_word_error_frac": 0.03,
        "early_stop_patience": 3,
        "min_checkpoint_iter": max(1_000, int(steps * 0.08)),
        "viz_length": int(viz_length),
        "hidden_size": POW2_H100_HIDDEN_SIZE,
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
        "sweep_length": length if length == "mixed" else int(length),
    }


def register_pow2_h100_sweep_regimes(regimes: dict[str, list[str]]) -> None:
    for n_words in POW2_H100_WORD_COUNTS:
        for length in POW2_H100_LENGTHS:
            regimes[regime_name(n_words, length)] = build_vocab(n_words, length)


def register_pow2_h100_sweep_tasks(tasks: dict[str, dict]) -> None:
    for n_words in POW2_H100_WORD_COUNTS:
        for length in POW2_H100_LENGTHS:
            tasks[task_name(n_words, length)] = pow2_h100_sweep_task_config(n_words, length)


def iter_pow2_h100_sweep_cells() -> Iterable[tuple[int, SweepLength]]:
    for n_words in POW2_H100_WORD_COUNTS:
        for length in POW2_H100_LENGTHS:
            yield n_words, length
