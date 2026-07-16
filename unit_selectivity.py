"""Per-unit feature selectivity: lexical context, lookahead, and readout alignment."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

if TYPE_CHECKING:
    from vocab_diagrams import MinimizedVocabAutomaton
    from visualize import CondensedView

ANALYSIS_FEATURES = ("dfa", "char", "position", "position_from_end")
# Word identity is opt-in for dedicated readout figures (not default ANALYSIS).
WORD_ANALYSIS_FEATURES = (
    "dfa", "char", "position", "position_from_end", "word",
)
SELECTIVITY_FEATURES = ("dfa", "prefix", "char", "position", "position_from_end")
ALL_CATEGORICAL_FEATURES = SELECTIVITY_FEATURES
LEXICAL_FEATURES = SELECTIVITY_FEATURES

LEGACY_LEXICAL_FEATURES = (
    "prefix", "string", "dfa", "position", "char", "word_start", "word_end",
)
PREDICTION_FEATURES = ("next_char", "next_char_2", "next_bigram", "prediction_entropy")

FEATURE_DISPLAY = {
    "char": "current char",
    "position": "position from beginning",
    "position_from_end": "position from end",
    "dfa": "DFA state",
    "word": "word identity",
    "prefix": "prefix",
    "string": "string",
    "word_start": "word start",
    "word_end": "word end",
    "next_char": "next char (+1)",
    "next_char_2": "next char (+2)",
    "next_bigram": "next bigram",
    "prediction_entropy": "pred. entropy",
}

# Match DECODE_FEATURE_COLORS (Okabe–Ito) so every feature type is visually
# distinct in separation bars, selectivity summaries, and decoding curves.
FEATURE_COLORS = {
    "char": "#E69F00",
    "dfa": "#0072B2",
    "position": "#009E73",
    "position_from_end": "#CC79A7",
    "word": "#D55E00",
    "prefix": "#56B4E9",
    "string": "#8c564b",
    "word_start": "#17becf",
    "word_end": "#9edae5",
    "next_char": "#e377c2",
    "next_char_2": "#f7b6d2",
    "next_bigram": "#bcbd22",
    "prediction_entropy": "#ff7f0e",
}

# Distinct category palettes so feature panels don't share the same tab20 look.
FEATURE_CMAPS = {
    "dfa": "nipy_spectral",
    "char": "Dark2",
    "position": "YlGnBu",
    "position_from_end": "OrRd",
    "word": "tab10",
    "prefix": "Set3",
    "string": "Accent",
    "word_start": "Set2",
    "word_end": "Pastel2",
    "next_char": "tab10",
    "next_char_2": "Paired",
    "next_bigram": "tab20b",
    "prediction_entropy": "plasma",
}

LEXICAL_SET = frozenset(ANALYSIS_FEATURES)
PREDICTION_SET = frozenset(PREDICTION_FEATURES)


@dataclass
class TimestepLabels:
    chars: list[str]
    prefix: list[str]
    string: list[str]
    dfa: list[int]
    position: list[int | None]
    position_from_end: list[int | None]
    word: list[str | None]
    word_start: list[str]
    word_end: list[str]
    next_char: list[str]
    next_char_2: list[str | None]
    next_bigram: list[str | None]
    pred_entropy_bin: list[int | None]

    def feature_values(self, feature: str) -> tuple[list, list[bool] | None]:
        if feature == "prefix":
            return self.prefix, None
        if feature == "string":
            return self.string, None
        if feature == "dfa":
            return self.dfa, None
        if feature == "position":
            return self.position, [p is not None for p in self.position]
        if feature == "position_from_end":
            return self.position_from_end, [p is not None for p in self.position_from_end]
        if feature == "word":
            return self.word, [w is not None for w in self.word]
        if feature == "char":
            return self.chars, None
        if feature == "word_start":
            return self.word_start, [v != "space" for v in self.word_start]
        if feature == "word_end":
            return self.word_end, [v != "space" for v in self.word_end]
        if feature == "next_char":
            return self.next_char, None
        if feature == "next_char_2":
            return self.next_char_2, [v is not None for v in self.next_char_2]
        if feature == "next_bigram":
            return self.next_bigram, [v is not None for v in self.next_bigram]
        if feature == "prediction_entropy":
            return self.pred_entropy_bin, [v is not None for v in self.pred_entropy_bin]
        raise ValueError(f"unknown feature: {feature}")


@dataclass
class SelectivityResult:
    si: dict[str, np.ndarray]
    peak_gap: dict[str, np.ndarray]
    eta2: dict[str, np.ndarray]
    gap: dict[str, np.ndarray]
    peak_label: dict[str, list[str]]
    target_prob_r: np.ndarray
    max_logit_r: np.ndarray
    entropy_r: np.ndarray
    best_predicted_char: list[str]
    primary_feature: list[str]
    primary_category: list[str]
    primary_group: list[str]
    mixed: list[bool]
    n_points: int = 0


def _future_labels(text: str, n: int) -> tuple[list[str], list[str | None], list[str | None]]:
    next_char: list[str] = []
    next_char_2: list[str | None] = []
    next_bigram: list[str | None] = []
    for t in range(n):
        next_char.append(text[(t + 1) % n])
        next_char_2.append(text[t + 2] if t + 2 < n else None)
        next_bigram.append(text[t + 1 : t + 3] if t + 2 < n else None)
    return next_char, next_char_2, next_bigram


def _entropy_quartile_bins(entropy: np.ndarray) -> list[int | None]:
    valid = np.isfinite(entropy)
    if valid.sum() < 4:
        return [None] * len(entropy)
    qs = np.quantile(entropy[valid], [0.25, 0.5, 0.75])
    bins: list[int | None] = []
    for e in entropy:
        if not np.isfinite(e):
            bins.append(None)
        elif e <= qs[0]:
            bins.append(0)
        elif e <= qs[1]:
            bins.append(1)
        elif e <= qs[2]:
            bins.append(2)
        else:
            bins.append(3)
    return bins


def build_timestep_labels(
    text: str,
    automaton: MinimizedVocabAutomaton,
    *,
    spaced: bool,
    words: list[str] | None,
    label_words: list[str] | None = None,
    condensed: CondensedView | None = None,
    model: dict | None = None,
    activations: np.ndarray | None = None,
) -> TimestepLabels:
    from visualize import _corpus_vocab
    from vocab_diagrams import (
        dfa_state_at_position,
        dfa_state_for_prefix,
        in_word_prefix_at_position,
        in_word_prefix_before_current,
        position_from_end_at_index,
        position_in_word_at_index,
        position_in_word_for_prefix_label,
        prefix_before_from_string_label,
        word_boundary_flags_at_index,
        word_boundary_flags_for_prefix_label,
        word_identity_at_index,
    )

    nc2: list[str | None] = []
    bg: list[str | None] = []
    word_starts: list[str] = []
    word_ends: list[str] = []
    position_from_end_ids: list[int | None] = []
    word_ids: list[str | None] = []

    boundary_words = label_words if label_words is not None else words
    vocab = _corpus_vocab(text, boundary_words)

    if condensed is not None:
        compare_chars = condensed.input_chars
        state_ids = [
            dfa_state_for_prefix(l, automaton, spaced=spaced) for l in condensed.labels
        ]
        position_ids = [position_in_word_for_prefix_label(l) for l in condensed.labels]
        position_from_end_ids = [
            position_from_end_at_index(text, idx, spaced=spaced, vocab=vocab)
            for idx in condensed.timestep_indices
        ]
        word_ids = [
            word_identity_at_index(text, idx, spaced=spaced, vocab=vocab)
            for idx in condensed.timestep_indices
        ]
        string_labels = list(condensed.labels)
        prefix_labels = [prefix_before_from_string_label(l) for l in condensed.labels]
        next_char = list(condensed.next_chars)
        n = len(condensed.labels)
        _, next_char_2_full, next_bigram_full = _future_labels(text, len(text))
        for label, idx in zip(condensed.labels, condensed.timestep_indices):
            ws, we = word_boundary_flags_for_prefix_label(label, boundary_words)
            word_starts.append(ws)
            word_ends.append(we)
            nc2.append(next_char_2_full[idx] if idx < len(next_char_2_full) else None)
            bg.append(next_bigram_full[idx] if idx < len(next_bigram_full) else None)
    else:
        n = len(text)
        compare_chars = list(text)
        state_ids = [
            dfa_state_at_position(text, t, automaton, spaced=spaced, vocab=vocab)
            for t in range(n)
        ]
        position_ids = [
            position_in_word_at_index(text, t, spaced=spaced, vocab=vocab) for t in range(n)
        ]
        position_from_end_ids = [
            position_from_end_at_index(text, t, spaced=spaced, vocab=vocab) for t in range(n)
        ]
        word_ids = [
            word_identity_at_index(text, t, spaced=spaced, vocab=vocab) for t in range(n)
        ]
        string_labels = [
            in_word_prefix_at_position(text, t, spaced=spaced, vocab=vocab) for t in range(n)
        ]
        prefix_labels = [
            in_word_prefix_before_current(text, t, spaced=spaced, vocab=vocab) for t in range(n)
        ]
        next_char, next_char_2, next_bigram = _future_labels(text, n)
        nc2 = next_char_2
        bg = next_bigram
        for t in range(n):
            ws, we = word_boundary_flags_at_index(
                text, t, spaced=spaced, vocab=vocab,
            )
            word_starts.append(ws)
            word_ends.append(we)

    pred_entropy_bin: list[int | None] = [None] * (len(compare_chars))
    if model is not None and activations is not None and activations.shape[0] == len(compare_chars):
        from visualize import prediction_entropy

        probs = _next_char_probabilities(model, activations)
        pred_entropy_bin = _entropy_quartile_bins(prediction_entropy(probs))

    return TimestepLabels(
        chars=compare_chars,
        prefix=prefix_labels,
        string=string_labels,
        dfa=state_ids,
        position=position_ids,
        position_from_end=position_from_end_ids,
        word=word_ids,
        word_start=word_starts,
        word_end=word_ends,
        next_char=next_char if condensed is None else list(condensed.next_chars),
        next_char_2=nc2,
        next_bigram=bg,
        pred_entropy_bin=pred_entropy_bin,
    )


def _next_char_logits(model: dict, hidden_states: np.ndarray) -> np.ndarray:
    if model.get("model_type") == "transformer":
        import torch

        torch_model = model["_torch_model"]
        h = torch.tensor(hidden_states, dtype=torch.float32)
        return torch_model.lm_head(h).detach().cpu().numpy()
    weights = model["weights_hidden_to_output"]
    bias = model["bias_output"].ravel()
    return hidden_states @ weights.T + bias


def _next_char_probabilities(model: dict, hidden_states: np.ndarray) -> np.ndarray:
    logits = _next_char_logits(model, hidden_states)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _unit_gap_score(
    unit_acts: np.ndarray,
    labels: list,
    valid_mask: list[bool] | None,
) -> float:
    within: list[float] = []
    between: list[float] = []
    all_pairs: list[float] = []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            if valid_mask is not None and (not valid_mask[i] or not valid_mask[j]):
                continue
            dist = abs(float(unit_acts[i] - unit_acts[j]))
            all_pairs.append(dist)
            if labels[i] == labels[j]:
                within.append(dist)
            else:
                between.append(dist)
    if not within or not between or not all_pairs:
        return float("nan")
    gap = float(np.median(between) - np.median(within))
    denom = max(float(np.median(all_pairs)), 1e-6)
    return gap / denom


def _unit_eta2(
    unit_acts: np.ndarray,
    labels: list,
    valid_mask: list[bool] | None,
) -> float:
    groups: dict[Any, list[float]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        if valid_mask is not None and not valid_mask[i]:
            continue
        groups[lbl].append(float(unit_acts[i]))
    if len(groups) < 2:
        return float("nan")
    values = np.array([v for g in groups.values() for v in g])
    grand_mean = float(values.mean())
    ss_total = float(((values - grand_mean) ** 2).sum())
    if ss_total <= 0:
        return 0.0
    ss_between = sum(
        len(g) * (float(np.mean(g)) - grand_mean) ** 2 for g in groups.values()
    )
    return ss_between / ss_total


def _unit_category_means(
    unit_acts: np.ndarray,
    labels: list,
    valid_mask: list[bool] | None,
) -> dict[Any, float]:
    return {
        lbl: mean
        for lbl, (mean, _) in _unit_category_stats(
            unit_acts, labels, valid_mask,
        ).items()
    }


def _unit_category_stats(
    unit_acts: np.ndarray,
    labels: list,
    valid_mask: list[bool] | None,
) -> dict[Any, tuple[float, int]]:
    """Category → (mean, count)."""
    groups: dict[Any, list[float]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        if valid_mask is not None and not valid_mask[i]:
            continue
        groups[lbl].append(float(unit_acts[i]))
    return {lbl: (float(np.mean(v)), len(v)) for lbl, v in groups.items() if v}


def peak_selectivity_index(
    category_means: np.ndarray,
    *,
    min_range: float = 0.25,
) -> float:
    """Peaked tuning: (r_max − r̄_others) / (r_max + r̄_others) on nonnegative rates.

    Category means are shifted by their minimum so rates are ≥ 0. Flat profiles
    (range < ``min_range``) score 0. One elevated category among lows → ~1.
    """
    means = np.asarray(category_means, dtype=float).ravel()
    if means.size < 2:
        return 0.0
    if float(means.max() - means.min()) < min_range:
        return 0.0
    rates = means - float(means.min())
    r_max = float(rates.max())
    if r_max <= 1e-12:
        return 0.0
    others = rates[rates < r_max - 1e-15]
    if others.size == 0:
        # ties for max across all → not selective
        return 0.0
    r_rest = float(others.mean())
    return (r_max - r_rest) / (r_max + r_rest + 1e-12)


def _unit_peak_metrics(
    unit_acts: np.ndarray,
    labels: list,
    valid_mask: list[bool] | None,
    *,
    min_std: float = 0.1,
    min_range: float = 0.25,
    min_peak_count: int = 2,
) -> tuple[float, float, float, str | None]:
    """
    Selectivity for one unit vs one feature.

    Returns (SI, normalized_peak_gap, eta2, peak_category_label).

    SI is peak-vs-rest on category means (nonnegative after min-shift).
    Peak category is argmax |mean − grand_mean| among categories with at least
    ``min_peak_count`` samples (falls back to all categories if none qualify).
    Near-flat units score SI = 0 but still report population ``eta2``.
    """
    stats = _unit_category_stats(unit_acts, labels, valid_mask)
    if len(stats) < 2:
        return float("nan"), float("nan"), float("nan"), None

    values = np.array([
        float(unit_acts[i])
        for i in range(len(labels))
        if valid_mask is None or valid_mask[i]
    ])
    std = float(np.std(values))
    grand = float(values.mean())
    eta = _unit_eta2(unit_acts, labels, valid_mask)

    supported = {
        lbl: (mean, count)
        for lbl, (mean, count) in stats.items()
        if count >= min_peak_count
    }
    rank_pool = supported if len(supported) >= 2 else stats

    ranked = sorted(
        rank_pool.items(),
        key=lambda kv: abs(kv[1][0] - grand),
        reverse=True,
    )
    top_lbl, (top_mean, top_count) = ranked[0]
    _, (second_mean, _) = ranked[1]

    mean_vec = np.array([m for m, _ in stats.values()], dtype=float)
    if std < min_std or top_count < min_peak_count:
        return 0.0, 0.0, eta, str(top_lbl)

    si = peak_selectivity_index(mean_vec, min_range=min_range)
    if si <= 0.0:
        return 0.0, 0.0, eta, str(top_lbl)

    peak_gap_norm = (
        (abs(top_mean - grand) - abs(second_mean - grand)) / std
        if std > 1e-12
        else 0.0
    )
    return si, peak_gap_norm, eta, str(top_lbl)


def compute_unit_selectivity_matrix(
    activations: np.ndarray,
    labels: TimestepLabels,
    features: tuple[str, ...] = ALL_CATEGORICAL_FEATURES,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, list[str]],
]:
    n_units = activations.shape[1]
    si: dict[str, np.ndarray] = {}
    peak_gap: dict[str, np.ndarray] = {}
    eta2: dict[str, np.ndarray] = {}
    gap: dict[str, np.ndarray] = {}
    peak_label: dict[str, list[str]] = {}
    for feat in features:
        vals, mask = labels.feature_values(feat)
        si[feat] = np.full(n_units, np.nan)
        peak_gap[feat] = np.full(n_units, np.nan)
        eta2[feat] = np.full(n_units, np.nan)
        gap[feat] = np.full(n_units, np.nan)
        labels_out: list[str] = [""] * n_units
        for u in range(n_units):
            s, pg, e2, pl = _unit_peak_metrics(activations[:, u], vals, mask)
            si[feat][u] = s
            peak_gap[feat][u] = pg
            eta2[feat][u] = e2
            gap[feat][u] = _unit_gap_score(activations[:, u], vals, mask)
            labels_out[u] = pl or ""
        peak_label[feat] = labels_out
    return si, peak_gap, eta2, gap, peak_label


def compute_prediction_correlation_scores(
    activations: np.ndarray,
    model: dict,
    text: str,
    *,
    next_chars: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    n_units = activations.shape[1]
    n = activations.shape[0]
    probs = _next_char_probabilities(model, activations)
    logits = _next_char_logits(model, activations)
    chars = model["chars"]

    if next_chars is not None and len(next_chars) == n:
        targets = list(next_chars)
    else:
        targets = [text[(t + 1) % len(text)] for t in range(n)]
    target_idx = np.array([chars.index(c) for c in targets])
    target_prob = probs[np.arange(n), target_idx]

    from visualize import prediction_entropy

    entropy = prediction_entropy(probs)

    target_prob_r = np.full(n_units, np.nan)
    entropy_r = np.full(n_units, np.nan)
    max_logit_r = np.full(n_units, np.nan)
    best_char: list[str] = [""] * n_units

    for u in range(n_units):
        h = activations[:, u]
        if np.std(h) > 1e-12 and np.std(target_prob) > 1e-12:
            target_prob_r[u] = float(np.corrcoef(h, target_prob)[0, 1])
        if np.std(h) > 1e-12 and np.std(entropy) > 1e-12:
            entropy_r[u] = abs(float(np.corrcoef(h, entropy)[0, 1]))
        corrs = []
        for c in range(logits.shape[1]):
            if np.std(logits[:, c]) > 1e-12 and np.std(h) > 1e-12:
                corrs.append(float(np.corrcoef(h, logits[:, c])[0, 1]))
            else:
                corrs.append(0.0)
        corrs_arr = np.array(corrs)
        best_ix = int(np.argmax(np.abs(corrs_arr)))
        max_logit_r[u] = corrs_arr[best_ix]
        best_char[u] = chars[best_ix] if best_ix < len(chars) else ""

    return target_prob_r, max_logit_r, entropy_r, best_char


def _assign_primary_features(
    si: dict[str, np.ndarray],
    peak_label: dict[str, list[str]],
    features: tuple[str, ...] = ALL_CATEGORICAL_FEATURES,
) -> tuple[list[str], list[str], list[str], list[bool]]:
    feat_arr = np.stack([si[f] for f in features], axis=1)
    n_units = feat_arr.shape[0]
    primary: list[str] = []
    categories: list[str] = []
    groups: list[str] = []
    mixed: list[bool] = []

    for u in range(n_units):
        scores = feat_arr[u]
        valid = np.isfinite(scores)
        if not valid.any():
            primary.append("none")
            categories.append("")
            groups.append("none")
            mixed.append(False)
            continue
        order = np.argsort(scores)[::-1]
        top = int(order[0])
        second = int(order[1]) if len(order) > 1 else top
        feat_name = features[top]
        primary.append(feat_name)
        categories.append(peak_label.get(feat_name, [""] * n_units)[u])
        groups.append("lexical" if feat_name in LEXICAL_SET else "predictive")
        s0, s1 = scores[top], scores[second]
        mixed.append(
            s0 >= 0.35
            and s1 >= 0.35
            and s1 >= 0.7 * s0
            and features[top] != features[second]
        )

    return primary, categories, groups, mixed


def compute_selectivity(
    activations: np.ndarray,
    labels: TimestepLabels,
    model: dict,
    text: str,
) -> SelectivityResult:
    si, peak_gap, eta2, gap, peak_label = compute_unit_selectivity_matrix(
        activations, labels,
    )
    target_prob_r, max_logit_r, entropy_r, best_char = compute_prediction_correlation_scores(
        activations, model, text, next_chars=labels.next_char,
    )
    primary, categories, groups, mixed = _assign_primary_features(si, peak_label)
    return SelectivityResult(
        si=si,
        peak_gap=peak_gap,
        eta2=eta2,
        gap=gap,
        peak_label=peak_label,
        target_prob_r=target_prob_r,
        max_logit_r=max_logit_r,
        entropy_r=entropy_r,
        best_predicted_char=best_char,
        primary_feature=primary,
        primary_category=categories,
        primary_group=groups,
        mixed=mixed,
        n_points=activations.shape[0],
    )


def _top_units(scores: np.ndarray, k: int = 3) -> list[int]:
    valid = np.where(np.isfinite(scores))[0]
    if len(valid) == 0:
        return []
    order = valid[np.argsort(scores[valid])[::-1]]
    return list(order[:k])


def _top_selective_units(
    scores: np.ndarray,
    activations: np.ndarray,
    k: int = 2,
    *,
    min_std: float = 0.05,
    exclude: set[int] | frozenset[int] | None = None,
) -> list[int]:
    """Top-k by score among units with nontrivial activation variance."""
    skip = exclude or set()
    valid = [
        i for i in range(len(scores))
        if i not in skip
        and np.isfinite(scores[i])
        and float(np.std(activations[:, i])) >= min_std
    ]
    if not valid:
        # Fall back: allow excluded units if nothing else qualifies.
        valid = [
            i for i in range(len(scores))
            if np.isfinite(scores[i]) and float(np.std(activations[:, i])) >= min_std
        ]
    if not valid:
        return _top_units(scores, k=k)
    order = sorted(valid, key=lambda i: float(scores[i]), reverse=True)
    return order[:k]


def plot_selectivity_heatmap(
    scores: dict[str, np.ndarray],
    features: tuple[str, ...],
    save_path: str,
    *,
    title: str,
    unit_labels: list[str],
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    mat = np.stack([scores[f] for f in features], axis=1)
    if vmin is None:
        vmin = float(np.nanpercentile(mat, 2)) if np.isfinite(mat).any() else 0.0
    if vmax is None:
        vmax = float(np.nanpercentile(mat, 98)) if np.isfinite(mat).any() else 1.0
    vmax = max(vmax, vmin + 1e-6)

    fig_h = max(6, 0.12 * len(unit_labels))
    fig, ax = plt.subplots(figsize=(10, fig_h), constrained_layout=True)
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels([FEATURE_DISPLAY[f] for f in features], rotation=45, ha="right")
    step = max(1, len(unit_labels) // 40)
    yticks = list(range(0, len(unit_labels), step))
    ax.set_yticks(yticks)
    ax.set_yticklabels([unit_labels[i] for i in yticks], fontsize=7)
    ax.set_ylabel("unit")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_prediction_encoding_heatmap(
    result: SelectivityResult,
    save_path: str,
    *,
    title: str,
    unit_labels: list[str],
) -> None:
    mat = np.stack(
        [result.target_prob_r, result.max_logit_r, result.entropy_r],
        axis=1,
    )
    vmax = float(np.nanpercentile(np.abs(mat), 98)) if np.isfinite(mat).any() else 1.0
    vmax = max(vmax, 0.1)

    fig_h = max(6, 0.12 * len(unit_labels))
    fig, ax = plt.subplots(figsize=(6, fig_h), constrained_layout=True)
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["P(target) corr", "max logit corr", "|entropy| corr"], rotation=30, ha="right")
    step = max(1, len(unit_labels) // 40)
    yticks = list(range(0, len(unit_labels), step))
    ax.set_yticks(yticks)
    ax.set_yticklabels([unit_labels[i] for i in yticks], fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def _si_arrays(
    result: SelectivityResult | dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if isinstance(result, dict):
        return result
    return result.si


def plot_selectivity_si_on_ax(
    ax,
    result: SelectivityResult | dict[str, np.ndarray],
    *,
    features: tuple[str, ...] = ANALYSIS_FEATURES,
    show_legend: bool = True,
    min_si: float = 1e-6,
) -> None:
    """Overlapped smooth SI density curves (Gaussian KDE) for one unit population.

    Exact zeros (flat / gated units) are omitted so the density reflects the
    distribution among units with peaked tuning.
    """
    from scipy.stats import gaussian_kde

    from viz.compare.decoding import DECODE_FEATURE_COLORS

    si = _si_arrays(result)
    xs = np.linspace(0.0, 1.0, 256)
    y_hi = 0.0
    for feat in features:
        vals = np.asarray(si.get(feat, []), dtype=float)
        vals = vals[np.isfinite(vals) & (vals > min_si)]
        if vals.size < 2:
            continue
        color = DECODE_FEATURE_COLORS.get(feat, FEATURE_COLORS.get(feat, "#888"))
        # Slightly wider than Scott so curves stay smooth with n≈10–50.
        try:
            kde = gaussian_kde(vals, bw_method=lambda k: max(k.scotts_factor() * 1.6, 0.08))
        except Exception:
            continue
        dens = kde(xs)
        dens = np.clip(dens, 0.0, None)
        ax.fill_between(xs, dens, color=color, alpha=0.18, linewidth=0)
        ax.plot(xs, dens, color=color, lw=1.8, label=FEATURE_DISPLAY.get(feat, feat))
        y_hi = max(y_hi, float(np.nanmax(dens)) if dens.size else 0.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, max(y_hi * 1.12, 0.1))
    ax.set_xlabel("selectivity index (peak vs rest)", fontsize=9)
    ax.set_ylabel("density", fontsize=9)
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    if show_legend:
        ax.legend(fontsize=7, frameon=False, loc="upper right")


def plot_selectivity_distributions(
    result: SelectivityResult,
    save_path: str,
    *,
    title: str,
    features: tuple[str, ...] = ANALYSIS_FEATURES,
) -> None:
    """Single-panel overlapped SI density curves (per-unit peak-vs-rest index)."""
    from viz.plot_layout import finalize_grid_figure, save_figure

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    plot_selectivity_si_on_ax(ax, result, features=features)
    finalize_grid_figure(fig, top=0.86, bottom=0.16, left=0.12, right=0.97)
    fig.suptitle(title, fontsize=11, y=0.98)
    save_figure(fig, save_path, dpi=150)
    print(f"wrote {save_path}")


def plot_primary_feature_pie(
    result: SelectivityResult,
    save_path: str,
    *,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

    feat_counts: dict[str, int] = defaultdict(int)
    for f in result.primary_feature:
        if f != "none":
            feat_counts[f] += 1
    if feat_counts:
        feats = sorted(feat_counts, key=lambda k: -feat_counts[k])
        axes[0].pie(
            [feat_counts[f] for f in feats],
            labels=[FEATURE_DISPLAY.get(f, f) for f in feats],
            colors=[FEATURE_COLORS.get(f, "#ccc") for f in feats],
            autopct="%1.0f%%",
            startangle=90,
        )
    axes[0].set_title("Primary feature (max SI)")

    grp_counts: dict[str, int] = defaultdict(int)
    for g in result.primary_group:
        if g != "none":
            grp_counts[g] += 1
    if grp_counts:
        grps = list(grp_counts.keys())
        axes[1].pie(
            [grp_counts[g] for g in grps],
            labels=[g.capitalize() for g in grps],
            colors=["#4c72b0", "#e377c2"][: len(grps)],
            autopct="%1.0f%%",
            startangle=90,
        )
    axes[1].set_title("Lexical vs predictive primary")

    fig.suptitle(title)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_feature_mixture_scatter(
    result: SelectivityResult,
    save_path: str,
    *,
    title: str,
) -> None:
    pairs = [
        ("dfa", "position", "DFA vs position from beginning"),
        ("char", "position_from_end", "current char vs position from end"),
        ("position", "dfa", "position from beginning vs DFA"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for ax, (x_key, y_key, subtitle) in zip(axes, pairs):
        x = result.si[x_key]
        y = result.si[y_key]
        mask = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[mask], y[mask], s=12, alpha=0.5, c="#4c72b0")
        for u, is_mix in enumerate(result.mixed):
            if is_mix and mask[u]:
                ax.scatter(
                    x[u], y[u], s=40, facecolors="none", edgecolors="crimson", linewidths=1.5,
                )
        ax.set_xlabel(FEATURE_DISPLAY.get(x_key, x_key))
        ax.set_ylabel(FEATURE_DISPLAY.get(y_key, y_key))
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_title(subtitle)
        ax.grid(True, linestyle=":", alpha=0.35)
    fig.suptitle(title)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def _category_means(
    unit_acts: np.ndarray,
    labels: list,
    valid_mask: list[bool] | None,
) -> tuple[list, list[float], list[float]]:
    groups: dict[Any, list[float]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        if valid_mask is not None and not valid_mask[i]:
            continue
        groups[lbl].append(float(unit_acts[i]))
    cats = sorted(groups.keys(), key=lambda k: (str(type(k)), str(k)))
    means = [float(np.mean(groups[c])) for c in cats]
    sems = [
        float(np.std(groups[c]) / np.sqrt(len(groups[c]))) if len(groups[c]) > 1 else 0.0
        for c in cats
    ]
    return cats, means, sems


def _category_color_map(
    feature: str,
    vals: list,
    visible: list[int],
    automaton: MinimizedVocabAutomaton | None,
) -> tuple[dict, dict]:
    cmap = plt.get_cmap(FEATURE_CMAPS.get(feature, "tab20"))
    cat_to_color, legend_labels, _ = _panel_feature_colors(
        feature, vals, visible, automaton, cmap,
    )
    return cat_to_color, legend_labels


def _colors_for_categories(feature: str, cats: list, cmap) -> dict:
    """Assign one color per category using the feature's palette."""
    n = len(cats)
    if n == 0:
        return {}
    # Ordered numeric features: walk a sequential scale by value.
    if feature in ("position", "position_from_end", "prediction_entropy") and all(
        isinstance(c, (int, float, np.integer, np.floating)) for c in cats
    ):
        nums = [float(c) for c in cats]
        lo, hi = min(nums), max(nums)
        span = max(hi - lo, 1e-9)
        return {c: cmap(0.15 + 0.75 * ((float(c) - lo) / span)) for c in cats}
    # Qualitative: spaced samples (avoid wrapping the same first colors).
    if n == 1:
        return {cats[0]: cmap(0.35)}
    return {c: cmap(i / (n - 1)) for i, c in enumerate(cats)}


def _timestep_tick_labels(
    labels: TimestepLabels,
    indices: list[int],
    *,
    feature: str | None = None,
    automaton: MinimizedVocabAutomaton | None = None,
) -> list[str]:
    # Always show the input character: panels share one chronological sequence.
    # Feature identity is carried by point/bar color, not by rewriting the x-axis.
    del feature, automaton
    return [labels.chars[i] for i in indices]


def _set_all_timestep_xticks(
    ax,
    labels: TimestepLabels,
    indices: list[int] | None = None,
    *,
    feature: str | None = None,
    automaton: MinimizedVocabAutomaton | None = None,
    show_labels: bool = True,
    show_xlabel: bool = True,
    fontsize: float = 8,
) -> None:
    n = len(labels.chars)
    if indices is None:
        indices = list(range(n))
    ax.set_xticks(indices)
    if show_labels:
        ax.set_xticklabels(
            _timestep_tick_labels(
                labels, indices, feature=feature, automaton=automaton,
            ),
            rotation=0,
            va="top",
            ha="center",
            fontsize=fontsize,
            fontfamily="monospace",
        )
        if show_xlabel:
            ax.set_xlabel("input character (same sequence in every panel)", fontsize=7)
        else:
            ax.set_xlabel("")
    else:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    ax.tick_params(axis="x", pad=1, labelbottom=True)


def _example_panel_figsize(
    n_timesteps: int,
    n_rows: int,
    n_cols: int = 2,
    *,
    compact: bool = False,
) -> tuple[float, float]:
    width = max(7.2, 0.10 * n_timesteps + 2.0) * (n_cols / 2)
    if compact:
        row_h = max(1.85, 1.45 + 0.006 * n_timesteps)
    else:
        row_h = max(1.7, 1.2 + 0.012 * n_timesteps)
    height = row_h * n_rows
    return width, height


def _plot_unit_timestep_trace(
    ax,
    unit_acts: np.ndarray,
    labels: TimestepLabels,
    feature: str,
    unit_label: str,
    *,
    automaton: MinimizedVocabAutomaton | None = None,
    target_prob: np.ndarray | None = None,
    compact: bool = False,
    show_xticks: bool = True,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Lollipop (stem) plot of activation vs timestep; color = feature category."""
    vals, mask = labels.feature_values(feature)
    n = len(unit_acts)
    # Keep every timestep so all panels share the same x sequence.
    xs = list(range(n))
    if not xs:
        ax.set_axis_off()
        return

    cat_to_color, legend_labels = _category_color_map(
        feature, vals, xs, automaton,
    )
    ys = [float(unit_acts[i]) for i in xs]
    colors = [
        cat_to_color[vals[i]]
        if (mask is None or mask[i]) and vals[i] in cat_to_color
        else "#bbbbbb"
        for i in xs
    ]

    lw = 1.0 if compact else 1.4
    pt = 18 if compact else 32
    for i, y, color in zip(xs, ys, colors):
        ax.plot(
            [i, i], [0.0, y],
            color=color, linewidth=lw, solid_capstyle="round", zorder=2,
        )
        ax.scatter(
            i, y, c=[color], s=pt, zorder=3, edgecolors="0.25", linewidths=0.3,
        )
    ax.axhline(0.0, color="0.8", linewidth=0.6, zorder=1)
    ax.set_xlim(-0.6, n - 0.4)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel("activation", fontsize=7 if compact else 9)
    if compact:
        ax.set_title(unit_label, fontsize=8, loc="left")
        _set_all_timestep_xticks(
            ax, labels, feature=None, automaton=None,
            show_labels=show_xticks, show_xlabel=False, fontsize=5.5,
        )
        ax.tick_params(axis="both", labelsize=6, labelbottom=show_xticks)
    else:
        ax.set_title(
            f"{unit_label} — chronological input\n"
            f"lollipop color = {FEATURE_DISPLAY[feature]}",
            fontsize=9,
        )
        _set_all_timestep_xticks(ax, labels, feature=None, automaton=None)
        cats = list(cat_to_color.keys())
        if len(cats) <= 10:
            handles = [
                Patch(facecolor=cat_to_color[c], label=legend_labels[c])
                for c in cats
            ]
            ax.legend(
                handles=handles,
                title=FEATURE_DISPLAY[feature],
                fontsize=7,
                title_fontsize=7,
                loc="upper right",
                framealpha=0.9,
            )

    if target_prob is not None and feature in PREDICTION_SET:
        ax2 = ax.twinx()
        ax2.plot(
            xs, [target_prob[i] for i in xs],
            color="0.55", linewidth=0.8, linestyle="--", alpha=0.8,
        )
        ax2.set_ylabel("P(correct next char)", fontsize=7 if compact else 8, color="0.45")
        ax2.tick_params(axis="y", labelsize=6 if compact else 7, colors="0.45")
        ax2.set_ylim(0, 1)


def _plot_unit_category_bars(
    ax,
    unit_acts: np.ndarray,
    vals: list,
    mask: list[bool] | None,
    cat_to_color: dict,
    legend_labels: dict,
    *,
    compact: bool = False,
    show_xticks: bool = True,
) -> None:
    bar_cats, means, sems = _category_means(unit_acts, vals, mask)
    x = np.arange(len(bar_cats))
    colors = [cat_to_color.get(c, "#888") for c in bar_cats]
    cap = 2 if compact else 4
    ax.bar(x, means, yerr=sems, color=colors, capsize=cap, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    if show_xticks:
        rot = 45 if compact and len(bar_cats) > 4 else 30
        fs = 5.5 if compact else 8
        labels = [legend_labels.get(c, str(c)) for c in bar_cats]
        if len(labels) > 10:
            labels = [lab if i % 2 == 0 else "" for i, lab in enumerate(labels)]
        ax.set_xticklabels(labels, rotation=rot, ha="right", fontsize=fs)
    else:
        ax.set_xticklabels([])
    ax.set_ylabel("mean" if compact else "mean activation", fontsize=7 if compact else 9)
    if compact:
        ax.set_title("mean by category", fontsize=7)
    else:
        ax.set_title("average over timesteps in each category")
    ax.axhline(0, color="0.8", linewidth=0.5)
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    if compact:
        ax.tick_params(axis="y", labelsize=6)


def _plot_unit_example_panel(
    ax_trace,
    ax_tune,
    *,
    unit_ix: int,
    unit_label: str,
    activations: np.ndarray,
    labels: TimestepLabels,
    feature: str,
    target_prob: np.ndarray | None,
    automaton: MinimizedVocabAutomaton | None = None,
    compact: bool = False,
    show_xticks: bool = True,
    ylim: tuple[float, float] | None = None,
) -> None:
    unit_acts = activations[:, unit_ix]
    vals, mask = labels.feature_values(feature)
    visible = list(range(len(unit_acts)))
    cat_to_color, legend_labels = _category_color_map(
        feature, vals, visible, automaton,
    )

    _plot_unit_timestep_trace(
        ax_trace, unit_acts, labels, feature, unit_label,
        automaton=automaton,
        target_prob=target_prob,
        compact=compact,
        show_xticks=show_xticks,
        ylim=ylim,
    )
    _plot_unit_category_bars(
        ax_tune, unit_acts, vals, mask, cat_to_color, legend_labels,
        compact=compact,
        show_xticks=show_xticks,
    )


def _target_prob_array(
    model: dict,
    activations: np.ndarray,
    text: str,
    *,
    next_chars: list[str] | None = None,
) -> np.ndarray:
    probs = _next_char_probabilities(model, activations)
    n = activations.shape[0]
    if next_chars is not None and len(next_chars) == n:
        targets = list(next_chars)
    else:
        targets = [text[(t + 1) % len(text)] for t in range(n)]
    target_idx = [model["chars"].index(c) for c in targets]
    return probs[np.arange(n), target_idx]


def plot_example_units_for_feature(
    result: SelectivityResult,
    activations: np.ndarray,
    labels: TimestepLabels,
    feature: str,
    save_path: str,
    *,
    unit_labels: list[str],
    text: str,
    model: dict,
    automaton: MinimizedVocabAutomaton | None = None,
    k: int = 6,
) -> None:
    scores = result.si[feature]
    units = _top_selective_units(scores, activations, k=k)
    if not units:
        return
    target_prob = _target_prob_array(
        model, activations, text, next_chars=labels.next_char,
    )

    fig, axes = plt.subplots(
        len(units), 2,
        figsize=_example_panel_figsize(len(activations), len(units)),
        constrained_layout=True,
    )
    if len(units) == 1:
        axes = np.array([axes])
    for row, u in enumerate(units):
        _plot_unit_example_panel(
            axes[row, 0], axes[row, 1],
            unit_ix=u,
            unit_label=f"{unit_labels[u]} (SI={float(scores[u]):.2f})",
            activations=activations,
            labels=labels,
            feature=feature,
            target_prob=target_prob if feature in PREDICTION_SET else None,
            automaton=automaton,
            show_xticks=(row == len(units) - 1),
        )
    fig.suptitle(
        f"Top units for {FEATURE_DISPLAY[feature]} "
        f"(peak vs rest SI)",
        fontsize=11,
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def _si_scores_for_feature(
    activations: np.ndarray,
    labels: TimestepLabels,
    feature: str,
) -> np.ndarray:
    """Per-unit peaked selectivity (SI) on the given timestep cloud."""
    n_units = activations.shape[1]
    vals, mask = labels.feature_values(feature)
    out = np.full(n_units, np.nan)
    for u in range(n_units):
        si, _, _, _ = _unit_peak_metrics(activations[:, u], vals, mask)
        if not np.isfinite(si):
            continue
        out[u] = float(si)
    return out


def plot_example_units_combined(
    result: SelectivityResult,
    activations: np.ndarray,
    labels: TimestepLabels,
    save_path: str,
    *,
    unit_labels: list[str],
    text: str,
    model: dict,
    automaton: MinimizedVocabAutomaton | None = None,
    features: tuple[str, ...] = ("dfa", "char", "position", "position_from_end"),
    k: int = 2,
    rank_si: dict[str, np.ndarray] | None = None,
) -> None:
    """One figure: top-``k`` distinct units per feature on a shared input sequence.

    Ranking uses ``rank_si`` when provided (typically condensed-prefix SI);
    traces always use the chronological ``activations`` / ``labels`` here.
    """
    scores = rank_si if rank_si is not None else {
        feat: _si_scores_for_feature(activations, labels, feat) for feat in features
    }

    rows: list[tuple[str, int, float]] = []
    used: set[int] = set()
    for feat in features:
        picks = _top_selective_units(
            scores[feat], activations, k=k, exclude=used,
        )
        used.update(picks)
        for u in picks:
            rows.append((feat, u, float(scores[feat][u])))
    if not rows:
        return

    unit_ixs = [u for _, u, _ in rows]
    y_stack = np.concatenate([activations[:, u] for u in unit_ixs])
    y_lo = float(np.min(y_stack))
    y_hi = float(np.max(y_stack))
    pad = 0.06 * max(y_hi - y_lo, 0.2)
    ylim = (y_lo - pad, y_hi + pad)

    target_prob = _target_prob_array(
        model, activations, text, next_chars=labels.next_char,
    )
    fig, axes = plt.subplots(
        len(rows), 2,
        figsize=_example_panel_figsize(len(activations), len(rows), compact=True),
    )
    if len(rows) == 1:
        axes = np.array([axes])

    prev_feat: str | None = None
    for row, (feat, u, si) in enumerate(rows):
        rank = 1 + sum(1 for f, uu, _ in rows[:row] if f == feat)
        title = f"{FEATURE_DISPLAY[feat]} #{rank} · {unit_labels[u]} (SI={si:.2f})"
        _plot_unit_example_panel(
            axes[row, 0], axes[row, 1],
            unit_ix=u,
            unit_label=title,
            activations=activations,
            labels=labels,
            feature=feat,
            target_prob=target_prob if feat in PREDICTION_SET else None,
            automaton=automaton,
            compact=True,
            show_xticks=True,
            ylim=ylim,
        )
        for col in (0, 1):
            axes[row, col].tick_params(axis="x", labelbottom=True, labelsize=5.5)
            # Labels sit in the inter-row gap; don't let the next axes cover them.
            for tick in axes[row, col].get_xticklabels():
                tick.set_clip_on(False)
                tick.set_zorder(10)
        if row == len(rows) - 1:
            axes[row, 0].set_xlabel(
                "input character (same sequence in every panel)", fontsize=7,
            )
            axes[row, 0].xaxis.label.set_clip_on(False)
        if feat != prev_feat:
            short = {
                "dfa": "DFA",
                "char": "char",
                "position": "pos→",
                "position_from_end": "←pos",
            }.get(feat, FEATURE_DISPLAY[feat])
            axes[row, 0].set_ylabel(
                f"{short}\nact.",
                fontsize=7,
                color=FEATURE_COLORS.get(feat, "#333"),
            )
            prev_feat = feat
        else:
            axes[row, 0].set_ylabel("act.", fontsize=6)
        axes[row, 1].set_ylabel("")

    fig.suptitle(
        f"Top-{k} units per feature on one corpus window "
        f"(lollipop color = feature category; peak SI)",
        fontsize=11,
        y=0.995,
    )
    fig.subplots_adjust(
        left=0.07, right=0.99, top=0.96, bottom=0.05, hspace=1.05, wspace=0.22,
    )
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"wrote {save_path}")
    del result


def plot_example_predictor_units(
    result: SelectivityResult,
    activations: np.ndarray,
    labels: TimestepLabels,
    save_path: str,
    *,
    unit_labels: list[str],
    text: str,
    model: dict,
    automaton: MinimizedVocabAutomaton | None = None,
    k: int = 6,
) -> None:
    units = _top_units(np.abs(result.max_logit_r), k=k)
    if not units:
        return
    target_prob = _target_prob_array(
        model, activations, text, next_chars=labels.next_char,
    )

    fig, axes = plt.subplots(
        len(units), 2,
        figsize=_example_panel_figsize(len(activations), len(units)),
        constrained_layout=True,
    )
    if len(units) == 1:
        axes = np.array([axes])
    nc = labels.next_char
    cats = sorted(set(nc))
    cmap = plt.get_cmap(FEATURE_CMAPS.get("next_char", "tab10"))
    cat_to_color = _colors_for_categories("next_char", cats, cmap)
    legend_labels = {c: str(c) for c in cats}
    n = len(activations)

    for row, u in enumerate(units):
        unit_acts = activations[:, u]
        ax0, ax1 = axes[row]
        y0 = 0.0
        for i in range(n):
            color = cat_to_color[nc[i]]
            y = float(unit_acts[i])
            ax0.plot([i, i], [y0, y], color=color, linewidth=1.4, solid_capstyle="round", zorder=2)
            ax0.scatter(i, y, c=[color], s=36, zorder=3, edgecolors="0.25", linewidths=0.4)
        ax0.axhline(y0, color="0.8", linewidth=0.6, zorder=1)
        ax0.set_xlim(-0.8, n - 0.2)
        ax0.set_ylabel("activation")
        ax0.set_title(
            f"{unit_labels[u]} — stem color = next char\n"
            f"best logit: '{result.best_predicted_char[u]}' "
            f"(r={result.max_logit_r[u]:.2f})",
            fontsize=9,
        )
        _set_all_timestep_xticks(ax0, labels)
        handles = [Patch(facecolor=cat_to_color[c], label=legend_labels[c]) for c in cats[:12]]
        if len(cats) <= 12:
            ax0.legend(handles=handles, title="next char", fontsize=7, title_fontsize=7, loc="upper right")

        ax2 = ax0.twinx()
        ax2.plot(
            range(n), target_prob, color="0.55", linewidth=0.8, linestyle="--", alpha=0.8,
        )
        ax2.set_ylabel("P(correct next char)", fontsize=8, color="0.45")
        ax2.tick_params(axis="y", labelsize=7, colors="0.45")
        ax2.set_ylim(0, 1)

        _plot_unit_category_bars(ax1, unit_acts, nc, None, cat_to_color, legend_labels)
    fig.suptitle("Top predictor units (|logit correlation|)", fontsize=11)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_example_mixed_units(
    result: SelectivityResult,
    activations: np.ndarray,
    labels: TimestepLabels,
    save_path: str,
    *,
    unit_labels: list[str],
    automaton: MinimizedVocabAutomaton | None = None,
    k: int = 6,
) -> None:
    mixed_units = [u for u, m in enumerate(result.mixed) if m]
    if not mixed_units:
        score = result.si["dfa"] + result.si["position"]
        mixed_units = _top_units(score, k=k)
    else:
        scores = np.array([
            result.si["dfa"][u] + result.si["position"][u] for u in mixed_units
        ])
        order = np.argsort(scores)[::-1]
        mixed_units = [mixed_units[i] for i in order[:k]]
    if not mixed_units:
        print(f"skip mixed units: none found for {save_path}")
        return

    n = len(mixed_units)
    fig, axes = plt.subplots(
        n, 4,
        figsize=(_example_panel_figsize(len(activations), n, n_cols=4)[0], 3.2 * n),
        constrained_layout=True,
    )
    if n == 1:
        axes = np.array([axes])
    for row, u in enumerate(mixed_units):
        feat_scores = sorted(
            ((f, result.si[f][u]) for f in ALL_CATEGORICAL_FEATURES),
            key=lambda x: -x[1] if np.isfinite(x[1]) else float("-inf"),
        )
        top_feats = [f for f, _ in feat_scores[:2]]
        while len(top_feats) < 2:
            top_feats.append(ALL_CATEGORICAL_FEATURES[len(top_feats) % len(ALL_CATEGORICAL_FEATURES)])
        for col, feat in enumerate(top_feats):
            _plot_unit_example_panel(
                axes[row, col * 2],
                axes[row, col * 2 + 1],
                unit_ix=u,
                unit_label=unit_labels[u],
                activations=activations,
                labels=labels,
                feature=feat,
                target_prob=None,
                automaton=automaton,
            )
    fig.suptitle("Mixed units (top two analysis features)", fontsize=11)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def _colored_prefix_labels_at_points(
    ax,
    projected: np.ndarray,
    prefix_labels: list[str],
    point_colors: list,
) -> None:
    """Prefix text at each embedding point, colored (no scatter, no leaders)."""
    for xy, prefix, color in zip(projected, prefix_labels, point_colors):
        label = "␣" if prefix == " " else prefix
        ax.text(
            xy[0], xy[1], label,
            fontsize=6,
            color=color,
            ha="center",
            va="center",
            zorder=5,
        )


def _compact_dfa_state_label(
    state: int,
    automaton: MinimizedVocabAutomaton,
    *,
    max_chars: int = 18,
) -> str:
    """Short DFA-state tick/legend label from the state's prefix set."""
    from vocab_diagrams import display_prefix

    prefixes = automaton.state_prefixes.get(int(state), set())
    shown = sorted({display_prefix(p) for p in prefixes}, key=lambda s: (len(s), s))
    if not shown:
        return f"s{int(state)}"
    if len(shown) == 1:
        lab = shown[0]
    elif len(shown) == 2:
        lab = f"{shown[0]},{shown[1]}"
    else:
        lab = f"{shown[0]}+{len(shown) - 1}"
    if len(lab) > max_chars:
        return f"s{int(state)}"
    return lab


def _panel_feature_colors(
    feat: str,
    vals: list,
    visible: list[int],
    automaton: MinimizedVocabAutomaton | None,
    cmap,
) -> tuple[dict, dict, list]:
    """Map feature categories to colors and legend labels for one panel."""
    cats = sorted(set(vals[j] for j in visible), key=lambda x: (str(type(x)), str(x)))
    if feat == "dfa" and automaton is not None:
        from visualize import _state_id_colors

        # Prefer project DFA state colors when available; otherwise feature cmap.
        try:
            state_colors = _state_id_colors([int(c) for c in cats])
            cat_to_color = {c: state_colors[int(c)] for c in cats}
        except Exception:
            cat_to_color = _colors_for_categories(feat, cats, cmap)
        if len(cats) > 8:
            legend_labels = {c: f"s{int(c)}" for c in cats}
        else:
            legend_labels = {
                c: _compact_dfa_state_label(int(c), automaton) for c in cats
            }
    else:
        cat_to_color = _colors_for_categories(feat, cats, cmap)
        legend_labels = {c: str(c) for c in cats}
    vis_colors = [cat_to_color[vals[j]] for j in visible]
    return cat_to_color, legend_labels, vis_colors


def _plot_dimred_by_feature(
    activations: np.ndarray,
    labels: TimestepLabels,
    save_path: str,
    *,
    title: str,
    projected: np.ndarray,
    x_label: str,
    y_label: str,
    method_name: str,
    automaton: MinimizedVocabAutomaton | None,
    features: tuple[str, ...] = ALL_CATEGORICAL_FEATURES,
) -> None:
    n = activations.shape[0]
    prefix_labels = labels.string

    n_feat = len(features)
    ncols = 4
    nrows = (n_feat + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.4 * ncols, 3.0 * nrows), constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    for i, feat in enumerate(features):
        ax = axes_flat[i]
        vals, mask = labels.feature_values(feat)
        visible = [j for j in range(n) if mask is None or mask[j]]
        if not visible:
            ax.axis("off")
            continue
        cmap = plt.get_cmap(FEATURE_CMAPS.get(feat, "tab20"))
        cat_to_color, legend_labels, vis_colors = _panel_feature_colors(
            feat, vals, visible, automaton, cmap,
        )
        vis_xy = projected[visible]
        vis_prefix = [prefix_labels[j] for j in visible]
        _colored_prefix_labels_at_points(ax, vis_xy, vis_prefix, vis_colors)
        pad_x = max((vis_xy[:, 0].max() - vis_xy[:, 0].min()) * 0.08, 0.05)
        pad_y = max((vis_xy[:, 1].max() - vis_xy[:, 1].min()) * 0.08, 0.05)
        ax.set_xlim(vis_xy[:, 0].min() - pad_x, vis_xy[:, 0].max() + pad_x)
        ax.set_ylim(vis_xy[:, 1].min() - pad_y, vis_xy[:, 1].max() + pad_y)
        ax.set_title(FEATURE_DISPLAY.get(feat, feat), fontsize=9)
        ax.set_xlabel(x_label, fontsize=7)
        ax.set_ylabel(y_label, fontsize=7)
        cats = list(cat_to_color.keys())
        if len(cats) <= 12:
            handles = [
                Patch(facecolor=cat_to_color[c], label=legend_labels[c])
                for c in cats
            ]
            ax.legend(
                handles=handles,
                fontsize=5 if feat == "dfa" else 6,
                loc="upper right",
                framealpha=0.85,
            )

    for j in range(n_feat, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        f"{title} — {method_name}: prefix labels colored by panel feature "
        "(DFA panel uses min-DFA state colors)",
        fontsize=11,
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_pca_by_feature(
    activations: np.ndarray,
    labels: TimestepLabels,
    save_path: str,
    *,
    title: str,
    text: str,
    spaced: bool,
    automaton: MinimizedVocabAutomaton | None = None,
    features: tuple[str, ...] = ALL_CATEGORICAL_FEATURES,
) -> None:
    from visualize import fit_pca_2d_with_evr

    _ = text, spaced
    if activations.shape[0] < 3 or activations.shape[1] < 2:
        return
    pca_xy, _, _, evr = fit_pca_2d_with_evr(activations)
    pc1 = 100.0 * float(evr[0]) if len(evr) > 0 else 0.0
    pc2 = 100.0 * float(evr[1]) if len(evr) > 1 else 0.0
    _plot_dimred_by_feature(
        activations, labels, save_path,
        title=title,
        projected=pca_xy,
        x_label=f"PC1 ({pc1:.0f}%)",
        y_label=f"PC2 ({pc2:.0f}%)",
        method_name="PCA",
        automaton=automaton,
        features=features,
    )


def plot_ica_by_feature(
    activations: np.ndarray,
    labels: TimestepLabels,
    save_path: str,
    *,
    title: str,
    automaton: MinimizedVocabAutomaton | None = None,
    features: tuple[str, ...] = ALL_CATEGORICAL_FEATURES,
) -> None:
    from visualize import fit_ica_2d

    if activations.shape[0] < 3 or activations.shape[1] < 2:
        return
    ica_xy, _, _ = fit_ica_2d(activations)
    _plot_dimred_by_feature(
        activations, labels, save_path,
        title=title,
        projected=ica_xy,
        x_label="IC1",
        y_label="IC2",
        method_name="ICA",
        automaton=automaton,
        features=features,
    )


def plot_unit_selectivity_summary(
    result: SelectivityResult,
    save_path: str,
    *,
    repr_label: str,
) -> None:
    """Population summary: peak SI, η², peak gap, primary features."""
    feats = list(ALL_CATEGORICAL_FEATURES)
    x = np.arange(len(feats))
    colors = [FEATURE_COLORS.get(f, "#888") for f in feats]
    tick = [FEATURE_DISPLAY.get(f, f) for f in feats]

    fig, axes = plt.subplots(2, 3, figsize=(10.0, 6.2), constrained_layout=True)
    fig.suptitle(
        f"Unit selectivity summary ({repr_label}, n={result.n_points} points, "
        f"{len(result.primary_feature)} units)",
        fontsize=9,
        y=1.02,
    )

    def _bar(ax, values, ylabel, ylim=None) -> None:
        ax.bar(x, values, color=colors, edgecolor="0.2", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(tick, rotation=28, ha="right", fontsize=6.5)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, axis="y", linestyle=":", alpha=0.35)
        if ylim is not None:
            ax.set_ylim(*ylim)

    si_med = [float(np.nanmedian(result.si[f])) for f in feats]
    si_p90 = [float(np.nanpercentile(result.si[f], 90)) for f in feats]
    _bar(axes[0, 0], si_med, "Median SI", (0, 1.05))
    axes[0, 0].set_title("Peak vs rest SI (p90 above)", fontsize=9)
    for i, p in enumerate(si_p90):
        y = min(si_med[i] + 0.06, 0.98)
        if si_med[i] < 0.08:
            y = 0.12
        axes[0, 0].text(i, y, f"{p:.2f}", ha="center", va="bottom", fontsize=5.5, rotation=0)

    eta_med = [float(np.nanmedian(result.eta2[f])) for f in feats]
    _bar(axes[0, 1], eta_med, "Median η²", (0, 1.05))
    axes[0, 1].set_title("Overall labeled variance", fontsize=9)

    pg_med = [float(np.nanmedian(result.peak_gap[f])) for f in feats]
    _bar(axes[0, 2], pg_med, "Median peak gap / σ(unit)")
    axes[0, 2].set_title("Top − runner-up category", fontsize=9)

    frac_strong = [
        float(np.mean(result.si[f] >= 0.5)) if np.isfinite(result.si[f]).any() else 0.0
        for f in feats
    ]
    _bar(axes[1, 0], frac_strong, "Fraction of units", (0, 1.05))
    axes[1, 0].set_title("Units with SI ≥ 0.5", fontsize=9)

    feat_counts: dict[str, int] = defaultdict(int)
    for f in result.primary_feature:
        if f != "none":
            feat_counts[f] += 1
    if feat_counts:
        pf_feats = sorted(feat_counts, key=lambda k: -feat_counts[k])
        px = np.arange(len(pf_feats))
        axes[1, 1].bar(
            px,
            [feat_counts[f] for f in pf_feats],
            color=[FEATURE_COLORS.get(f, "#888") for f in pf_feats],
            edgecolor="0.2",
            linewidth=0.6,
        )
        axes[1, 1].set_xticks(px)
        axes[1, 1].set_xticklabels(
            [FEATURE_DISPLAY.get(f, f) for f in pf_feats],
            rotation=35, ha="right", fontsize=8,
        )
        axes[1, 1].set_ylabel("# units")
        axes[1, 1].set_title("Primary feature (argmax SI)")
        axes[1, 1].grid(True, axis="y", linestyle=":", alpha=0.35)

    ax = axes[1, 2]
    for f in feats:
        si_v = result.si[f]
        e2 = result.eta2[f]
        mask = np.isfinite(si_v) & np.isfinite(e2)
        ax.scatter(
            e2[mask], si_v[mask],
            s=18, alpha=0.45, c=FEATURE_COLORS.get(f, "#888"), label=FEATURE_DISPLAY[f],
        )
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.35)
    ax.set_xlabel("η²")
    ax.set_ylabel("SI")
    ax.set_title("Per-unit SI vs η²", fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(
        fontsize=5.5, loc="center left", bbox_to_anchor=(1.04, 0.5),
        borderaxespad=0.0, frameon=False, handletextpad=0.3,
    )
    ax.grid(True, linestyle=":", alpha=0.35)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def _save_summary_json(result: SelectivityResult, save_path: str, unit_labels: list[str]) -> None:
    payload = {
        "units": unit_labels,
        "n_points": result.n_points,
        "si": {k: v.tolist() for k, v in result.si.items()},
        "peak_gap": {k: v.tolist() for k, v in result.peak_gap.items()},
        "eta2": {k: v.tolist() for k, v in result.eta2.items()},
        "gap": {k: v.tolist() for k, v in result.gap.items()},
        "peak_label": result.peak_label,
        "target_prob_r": result.target_prob_r.tolist(),
        "max_logit_r": result.max_logit_r.tolist(),
        "entropy_r": result.entropy_r.tolist(),
        "best_predicted_char": result.best_predicted_char,
        "primary_feature": result.primary_feature,
        "primary_category": result.primary_category,
        "primary_group": result.primary_group,
        "mixed": [bool(m) for m in result.mixed],
        "population_median_si": {
            f: float(np.nanmedian(result.si[f])) for f in ALL_CATEGORICAL_FEATURES
        },
        "population_median_eta2": {
            f: float(np.nanmedian(result.eta2[f])) for f in ALL_CATEGORICAL_FEATURES
        },
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {save_path}")


def plot_unit_selectivity_suite(
    activations: np.ndarray,
    text: str,
    automaton: MinimizedVocabAutomaton,
    save_dir: str | Path,
    *,
    model: dict,
    spaced: bool,
    words: list[str] | None,
    condensed: CondensedView | None,
    repr_label: str,
    unit_labels: list[str] | None = None,
    label_words: list[str] | None = None,
    example_k: int = 6,
    output_probs: np.ndarray | None = None,
) -> SelectivityResult | None:
    os.makedirs(save_dir, exist_ok=True)

    # Exemplars use the chronological corpus window so every panel shares one
    # readable input sequence. Selectivity scores stay on condensed prefixes.
    chrono_activations = activations
    chrono_labels = build_timestep_labels(
        text, automaton,
        spaced=spaced, words=words, label_words=label_words, condensed=None,
        model=model, activations=chrono_activations,
    )

    if condensed is None:
        from visualize import condense_hidden_states_by_prefix

        condensed = condense_hidden_states_by_prefix(
            text, activations, output_probs, spaced=spaced, words=words,
        )
        print(
            f"unit selectivity: auto-condensed to {len(condensed.labels)} prefixes "
            f"(from {sum(condensed.counts)} timesteps)"
        )

    activations = condensed.hidden_states

    if activations.shape[0] < 2:
        print("unit selectivity: need at least 2 condensed points")
        return None

    n_units = activations.shape[1]
    if unit_labels is None:
        unit_labels = [f"u{i}" for i in range(n_units)]

    labels = build_timestep_labels(
        text, automaton,
        spaced=spaced, words=words, label_words=label_words, condensed=condensed,
        model=model, activations=activations,
    )
    result = compute_selectivity(activations, labels, model, text)

    plot_unit_selectivity_summary(
        result,
        os.path.join(save_dir, "unit_selectivity_summary.png"),
        repr_label=repr_label,
    )
    plot_selectivity_heatmap(
        result.si, ALL_CATEGORICAL_FEATURES,
        os.path.join(save_dir, "selectivity_heatmap_si.png"),
        title=f"{repr_label} — Peak vs rest SI",
        unit_labels=unit_labels,
        vmin=0.0,
        vmax=1.0,
    )
    plot_selectivity_heatmap(
        result.eta2, ALL_CATEGORICAL_FEATURES,
        os.path.join(save_dir, "selectivity_heatmap_eta2.png"),
        title=f"{repr_label} — η² per unit",
        unit_labels=unit_labels,
        vmin=0.0,
        vmax=1.0,
    )
    plot_selectivity_heatmap(
        result.peak_gap, ALL_CATEGORICAL_FEATURES,
        os.path.join(save_dir, "selectivity_heatmap_peak_gap.png"),
        title=f"{repr_label} — normalized top − runner-up",
        unit_labels=unit_labels,
    )
    plot_selectivity_heatmap(
        result.gap, ALL_CATEGORICAL_FEATURES,
        os.path.join(save_dir, "selectivity_heatmap_gap.png"),
        title=f"{repr_label} — legacy pairwise gap",
        unit_labels=unit_labels,
    )
    plot_prediction_encoding_heatmap(
        result,
        os.path.join(save_dir, "prediction_encoding_heatmap.png"),
        title=f"{repr_label} — readout alignment",
        unit_labels=unit_labels,
    )
    plot_selectivity_distributions(
        result,
        os.path.join(save_dir, "selectivity_distributions.png"),
        title="Per-unit selectivity index",
    )
    plot_primary_feature_pie(
        result,
        os.path.join(save_dir, "primary_feature_pie.png"),
        title=f"{repr_label} — population primary features",
    )
    plot_feature_mixture_scatter(
        result,
        os.path.join(save_dir, "feature_mixture_scatter.png"),
        title=f"{repr_label} — feature mixtures",
    )
    plot_pca_by_feature(
        activations, labels,
        os.path.join(save_dir, "pca_by_feature.png"),
        title=repr_label,
        text=text,
        spaced=spaced,
        automaton=automaton,
    )
    plot_ica_by_feature(
        activations, labels,
        os.path.join(save_dir, "ica_by_feature.png"),
        title=repr_label,
        automaton=automaton,
    )

    for feat in ALL_CATEGORICAL_FEATURES:
        plot_example_units_for_feature(
            result, chrono_activations, chrono_labels, feat,
            os.path.join(save_dir, f"example_units_{feat}.png"),
            unit_labels=unit_labels, text=text, model=model,
            automaton=automaton,
            k=example_k,
        )

    plot_example_units_combined(
        result, chrono_activations, chrono_labels,
        os.path.join(save_dir, "example_units_combined.png"),
        unit_labels=unit_labels, text=text, model=model,
        automaton=automaton,
        features=ANALYSIS_FEATURES,
        k=2,
        rank_si=result.si,
    )

    plot_example_predictor_units(
        result, chrono_activations, chrono_labels,
        os.path.join(save_dir, "example_units_predictors.png"),
        unit_labels=unit_labels, text=text, model=model,
        automaton=automaton,
        k=example_k,
    )
    plot_example_mixed_units(
        result, chrono_activations, chrono_labels,
        os.path.join(save_dir, "example_units_mixed.png"),
        unit_labels=unit_labels,
        automaton=automaton,
        k=example_k,
    )
    _save_summary_json(result, os.path.join(save_dir, "selectivity_summary.json"), unit_labels)
    return result
