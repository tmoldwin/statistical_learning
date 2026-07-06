"""Word-count (powers of 2) × letter-length sweep: 1–64 words, lengths 1–7."""

from __future__ import annotations

import string
from typing import Iterable

from vocab_sweep import build_vocab as _build_vocab_general

POW2_WORD_COUNTS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
POW2_LENGTHS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
POW2_DEFAULT_SEEDS: tuple[int, ...] = tuple(range(1, 16))  # 15 seeds

# Enough distinct single-character tokens for the largest length-1 grid cell.
_SINGLE_CHARS: tuple[str, ...] = tuple(
    string.ascii_lowercase + string.ascii_uppercase + string.digits + "!@#$%^&*"
)


def _two_letter_words(n_words: int) -> list[str]:
    out: list[str] = []
    for c in "bcdfghjklmnpqrstvwxyz":
        for v in "aeiou":
            out.append(c + v)
            if len(out) >= n_words:
                return out
    raise ValueError(f"only {len(out)} distinct 2-letter tokens available, requested {n_words}")


def regime_name(n_words: int, length: int) -> str:
    return f"pow2sweep_w{n_words}_l{length}"


def task_name(n_words: int, length: int) -> str:
    return f"{regime_name(n_words, length)}_ns"


def build_vocab(n_words: int, length: int) -> list[str]:
    """Build ``n_words`` words of fixed ``length`` (handles length 1–2 and n_words 1)."""
    if n_words < 1:
        raise ValueError("need at least 1 word")
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


def pow2_sweep_task_config(n_words: int, length: int) -> dict[str, object]:
    """Training/viz defaults scaled to vocabulary size and word length."""
    hidden = 32 + n_words + 4 * length
    if n_words >= 8:
        hidden = max(64, hidden)
    if n_words >= 32:
        hidden = max(128, hidden)
    if n_words >= 64:
        hidden = max(192, hidden)
    hidden = min(int(hidden), 256)

    chars = max(30_000, n_words * max(length, 1) * 600)
    steps = min(120_000, max(15_000, n_words * 600 + length * 2500))
    if n_words >= 32:
        steps = max(steps, 80_000)
    if n_words >= 64:
        steps = max(steps, 100_000)
    if length == 1:
        steps = min(steps, max(10_000, n_words * 500))

    viz_length = min(n_words * max(length, 1) + 20, 500)
    if length == 1:
        viz_length = min(n_words + 10, 100)

    sequence_length = max(8, 2 * length + (8 if n_words >= 32 else 4))
    if length == 1:
        sequence_length = max(4, min(8, max(n_words, 4)))

    metric_rollout_len = min(1000, max(300, n_words * max(length, 1) * 2))
    if n_words >= 16:
        metric_rollout_len = min(5000, max(metric_rollout_len, n_words * max(length, 1) * 20))

    return {
        "regime": regime_name(n_words, length),
        "word_space": False,
        "chars": int(chars),
        "steps": int(steps),
        "target_word_error_frac": 0.03,
        "early_stop_patience": 3,
        "min_checkpoint_iter": max(1_000, int(steps * 0.08)),
        "viz_length": int(viz_length),
        "hidden_size": int(hidden),
        "sequence_length": int(sequence_length),
        "eval_interval": 50,
        "eval_iterations": 20,
        "metric_rollout_len": int(metric_rollout_len),
        "train_ratio": 0.9,
        "dropout": 0.25,
        "l2_lambda": 1e-4,
        "learning_rate": 0.04 if length >= 5 or n_words >= 32 else 0.1,
        "sweep_n_words": int(n_words),
        "sweep_length": int(length),
    }


def register_pow2_sweep_regimes(regimes: dict[str, list[str]]) -> None:
    for n_words in POW2_WORD_COUNTS:
        for length in POW2_LENGTHS:
            regimes[regime_name(n_words, length)] = build_vocab(n_words, length)


def register_pow2_sweep_tasks(tasks: dict[str, dict]) -> None:
    for n_words in POW2_WORD_COUNTS:
        for length in POW2_LENGTHS:
            tasks[task_name(n_words, length)] = pow2_sweep_task_config(n_words, length)


def iter_pow2_sweep_cells() -> Iterable[tuple[int, int]]:
    for n_words in POW2_WORD_COUNTS:
        for length in POW2_LENGTHS:
            yield n_words, length


def parse_pow2_sweep_task(task: str) -> tuple[int, int] | None:
    """Return (n_words, length) for ``pow2sweep_w{N}_l{L}_ns`` tasks."""
    if not task.startswith("pow2sweep_w") or not task.endswith("_ns"):
        return None
    core = task.removeprefix("pow2sweep_w").removesuffix("_ns")
    if "_l" not in core:
        return None
    n_s, l_s = core.split("_l", 1)
    try:
        return int(n_s), int(l_s)
    except ValueError:
        return None
