"""Rollout sampling and word-validity metrics (numpy weights, shared by CPU/GPU trainers)."""

from __future__ import annotations

import numpy as np

from rnn.rnn_dyn import rnn_hidden_step, stable_softmax
from vocab_diagrams import oov_char_fraction, segment_corpus_by_words

METRIC_ROLLOUT_LEN = 500
METRIC_NUM_ROLLOUTS = 4
METRIC_RNG_BASE = 42


def _stochastic_rollout(
    weights: dict[str, np.ndarray],
    *,
    hidden_size: int,
    vocab_size: int,
    char_to_index: dict[str, int],
    index_to_char: dict[int, str],
    prompt: str,
    num_chars: int,
    use_relu: bool,
    timestep_noise_std: float,
    rng: np.random.Generator,
) -> str:
    """Teacher-force ``prompt`` from zero state, then sample ``num_chars`` at temperature 1.

    Warm-starting from corpus context puts the rollout on the same state
    distribution as the teacher-forced cross-entropy, so the two learning
    curves measure the same model behavior (prediction vs. generation).
    """
    w_ih = weights["weights_input_to_hidden"]
    w_hh = weights["weights_hidden_to_hidden"]
    w_ho = weights["weights_hidden_to_output"]
    b_h = weights["bias_hidden"]
    b_o = weights["bias_output"]

    if not prompt:
        return ""

    hidden_state = np.zeros((hidden_size, 1))
    input_one_hot = np.zeros((vocab_size, 1))
    input_one_hot[char_to_index[prompt[0]]] = 1
    for ch_next in prompt[1:]:
        hidden_state, _ = rnn_hidden_step(
            hidden_state, input_one_hot, w_ih, w_hh, b_h,
            use_relu=use_relu, timestep_noise_std=timestep_noise_std, noise_rng=rng,
        )
        input_one_hot = np.zeros((vocab_size, 1))
        input_one_hot[char_to_index[ch_next]] = 1

    indices: list[int] = []
    for _ in range(num_chars):
        hidden_state, _ = rnn_hidden_step(
            hidden_state, input_one_hot, w_ih, w_hh, b_h,
            use_relu=use_relu, timestep_noise_std=timestep_noise_std, noise_rng=rng,
        )
        logits = w_ho @ hidden_state + b_o
        probs = stable_softmax(logits)
        next_index = int(rng.choice(vocab_size, p=probs.ravel()))
        indices.append(next_index)
        input_one_hot = np.zeros((vocab_size, 1))
        input_one_hot[next_index] = 1
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


def rollout_word_validity_metrics(
    weights: dict[str, np.ndarray],
    *,
    hidden_size: int,
    vocab_size: int,
    index_to_char: dict[int, str],
    char_to_index: dict[str, int],
    seed_index: int,
    vocab: set[str],
    use_word_segmentation: bool,
    use_relu: bool,
    rng: np.random.Generator,
    corpus_text: str = "",
    prompt_len: int = 0,
    rollout_len: int | None = None,
    num_rollouts: int | None = None,
    timestep_noise_std: float = 0.0,
) -> tuple[float, float, str]:
    """
    Mean out-of-vocabulary character fraction over stochastic rollouts
    warm-started from random corpus prefixes (teacher-forced). Returned text is
    one example rollout (r=0). Only boundary-truncatable chars are trimmed.
    """
    rollout_len = METRIC_ROLLOUT_LEN if rollout_len is None else rollout_len
    num_rollouts = METRIC_NUM_ROLLOUTS if num_rollouts is None else num_rollouts
    word_errs: list[float] = []
    letter_fracs: list[float] = []
    example_text = ""
    spaced = not use_word_segmentation

    for r in range(num_rollouts):
        if corpus_text and prompt_len > 0 and len(corpus_text) > prompt_len:
            start = int(rng.integers(0, len(corpus_text) - prompt_len))
            prompt = corpus_text[start : start + prompt_len]
        else:
            prompt = index_to_char[seed_index]
        text = _stochastic_rollout(
            weights,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            char_to_index=char_to_index,
            index_to_char=index_to_char,
            prompt=prompt,
            num_chars=rollout_len,
            use_relu=use_relu,
            timestep_noise_std=timestep_noise_std,
            rng=rng,
        )
        if r == 0:
            example_text = text
        word_errs.append(oov_char_fraction(text, vocab, spaced=spaced))
        letter_fracs.append(valid_vocab_letter_fraction(text, vocab, use_word_segmentation=use_word_segmentation))
    return float(np.nanmean(word_errs)), float(np.nanmean(letter_fracs)), example_text


# Backward-compatible alias
stochastic_word_validity_metrics = rollout_word_validity_metrics
