"""Rollout sampling and word-validity metrics (numpy weights, shared by CPU/GPU trainers)."""

from __future__ import annotations

import numpy as np

from rnn.rnn_dyn import rnn_hidden_step, stable_softmax
from vocab_diagrams import invalid_word_fraction, segment_corpus_by_words

METRIC_ROLLOUT_LEN = 500
METRIC_NUM_ROLLOUTS = 2
METRIC_RNG_BASE = 42


def _sample_rollout(
    weights: dict[str, np.ndarray],
    *,
    hidden_size: int,
    vocab_size: int,
    index_to_char: dict[int, str],
    seed_index: int,
    num_chars: int,
    use_relu: bool,
    timestep_noise_std: float,
    rng: np.random.Generator,
) -> str:
    w_ih = weights["weights_input_to_hidden"]
    w_hh = weights["weights_hidden_to_hidden"]
    w_ho = weights["weights_hidden_to_output"]
    b_h = weights["bias_hidden"]
    b_o = weights["bias_output"]

    hidden_state = np.zeros((hidden_size, 1))
    input_one_hot = np.zeros((vocab_size, 1))
    input_one_hot[seed_index] = 1
    indices: list[int] = []
    for _ in range(num_chars):
        hidden_state, _ = rnn_hidden_step(
            hidden_state, input_one_hot, w_ih, w_hh, b_h,
            use_relu=use_relu, timestep_noise_std=timestep_noise_std, noise_rng=rng,
        )
        logits = w_ho @ hidden_state + b_o
        probs = stable_softmax(logits)
        next_index = int(rng.choice(range(vocab_size), p=probs.ravel()))
        input_one_hot = np.zeros((vocab_size, 1))
        input_one_hot[next_index] = 1
        indices.append(next_index)
    return "".join(index_to_char[i] for i in indices)


def valid_vocab_letter_fraction(sampled_text: str, vocab: set[str], *, use_word_segmentation: bool) -> float:
    if not vocab:
        return float("nan")
    if use_word_segmentation:
        tokens = [seg[2] for seg in segment_corpus_by_words(sampled_text, vocab)]
    else:
        tokens = [t for t in sampled_text.split(" ") if t]
    total_letters = sum(len(token) for token in tokens)
    valid_letters = sum(len(token) for token in tokens if token in vocab)
    return (valid_letters / total_letters) if total_letters > 0 else float("nan")


def stochastic_word_validity_metrics(
    weights: dict[str, np.ndarray],
    *,
    hidden_size: int,
    vocab_size: int,
    index_to_char: dict[int, str],
    seed_index: int,
    vocab: set[str],
    use_word_segmentation: bool,
    use_relu: bool,
    timestep_noise_std: float,
    rng: np.random.Generator,
    rollout_len: int | None = None,
    num_rollouts: int | None = None,
) -> tuple[float, float, str]:
    rollout_len = METRIC_ROLLOUT_LEN if rollout_len is None else rollout_len
    num_rollouts = METRIC_NUM_ROLLOUTS if num_rollouts is None else num_rollouts
    word_errs: list[float] = []
    letter_fracs: list[float] = []
    first_text = ""
    for r in range(num_rollouts):
        rollout_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        text = _sample_rollout(
            weights, hidden_size=hidden_size, vocab_size=vocab_size,
            index_to_char=index_to_char, seed_index=seed_index, num_chars=rollout_len,
            use_relu=use_relu, timestep_noise_std=timestep_noise_std, rng=rollout_rng,
        )
        if r == 0:
            first_text = text
        word_errs.append(invalid_word_fraction(text, vocab, spaced=not use_word_segmentation, trim_edges=True))
        letter_fracs.append(valid_vocab_letter_fraction(text, vocab, use_word_segmentation=use_word_segmentation))
    return float(np.nanmean(word_errs)), float(np.nanmean(letter_fracs)), first_text