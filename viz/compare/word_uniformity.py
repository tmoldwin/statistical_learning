"""Compare model output word frequencies to uniform training distribution."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from vocab_diagrams import segmented_word_tokens
from visualize import (
    _closed_loop_summary_seed,
    _trajectory_seed_letters,
    corpus_uses_word_spacing,
    rnn_closed_loop_rollout,
)


def _rollout_chars_for_uniformity(
    *,
    n_words: int,
    word_len: int,
    min_words: int = 200,
) -> int:
    """Rollout length (chars) for stable word-count estimates."""
    cycle = n_words * max(word_len, 1)
    return max(cycle * 30, min_words * max(word_len, 1))


def empirical_word_probs(
    tokens: list[str],
    words: list[str],
) -> np.ndarray:
    """Normalized count vector aligned with ``words`` order."""
    if not tokens:
        return np.full(len(words), np.nan)
    counts = Counter(tokens)
    n = len(tokens)
    return np.array([counts.get(w, 0) / n for w in words], dtype=float)


def deviation_from_uniform(probs: np.ndarray) -> dict[str, float]:
    """Distance of empirical word probs from uniform 1/K."""
    k = len(probs)
    if k == 0 or not np.all(np.isfinite(probs)):
        return {
            "uniform_tv_distance": float("nan"),
            "uniform_kl_divergence": float("nan"),
            "uniform_l2_distance": float("nan"),
        }
    target = 1.0 / k
    tv = 0.5 * float(np.sum(np.abs(probs - target)))
    eps = 1e-12
    kl = float(np.sum(probs * np.log((probs + eps) / target)))
    l2 = float(np.sqrt(np.sum((probs - target) ** 2)))
    return {
        "uniform_tv_distance": tv,
        "uniform_kl_divergence": kl,
        "uniform_l2_distance": l2,
    }


def measure_output_word_uniformity(
    model: dict,
    words: list[str],
    *,
    task: str,
    seed: int = 42,
    rollout_chars: int | None = None,
    trim_edges: bool = True,
) -> dict[str, Any]:
    """Stochastic closed-loop rollout → word counts vs uniform training prior."""
    if not words:
        return {"error": "empty vocabulary"}

    spaced = corpus_uses_word_spacing("", task)
    vocab = set(words)
    word_len = max(len(w) for w in words)
    n_words = len(words)
    chars = rollout_chars or _rollout_chars_for_uniformity(
        n_words=n_words, word_len=word_len,
    )

    seed_letters = _trajectory_seed_letters(model, words)
    seed_text = _closed_loop_summary_seed(words, seed_letters, spaced=spaced)
    rng = np.random.default_rng(seed)

    _hidden, generated = rnn_closed_loop_rollout(
        model, seed_text=seed_text, steps=chars, rng=rng,
    )
    rollout_text = "".join(generated)
    tokens = segmented_word_tokens(
        rollout_text, vocab, spaced=spaced, trim_edges=trim_edges,
    )
    in_vocab = [t for t in tokens if t in vocab]
    probs = empirical_word_probs(in_vocab, words)
    metrics = deviation_from_uniform(probs)

    return {
        "n_rollout_chars": len(rollout_text),
        "n_rollout_words": len(in_vocab),
        "n_invalid_tokens": len(tokens) - len(in_vocab),
        "empirical_word_probs": {
            w: float(probs[i]) for i, w in enumerate(words)
        },
        **metrics,
    }
