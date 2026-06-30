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

ANALYSIS_FEATURES = ("char", "position", "position_from_end", "dfa")
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
    "prefix": "prefix",
    "string": "string",
    "word_start": "word start",
    "word_end": "word end",
    "next_char": "next char (+1)",
    "next_char_2": "next char (+2)",
    "next_bigram": "next bigram",
    "prediction_entropy": "pred. entropy",
}

FEATURE_COLORS = {
    "char": "#55a868",
    "position": "#8172b3",
    "position_from_end": "#9372b3",
    "dfa": "#4c72b0",
    "prefix": "#9467bd",
    "string": "#8c564b",
    "word_start": "#17becf",
    "word_end": "#9edae5",
    "next_char": "#e377c2",
    "next_char_2": "#f7b6d2",
    "next_bigram": "#bcbd22",
    "prediction_entropy": "#ff7f0e",
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
    )

    nc2: list[str | None] = []
    bg: list[str | None] = []
    word_starts: list[str] = []
    word_ends: list[str] = []
    position_from_end_ids: list[int | None] = []

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
    groups: dict[Any, list[float]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        if valid_mask is not None and not valid_mask[i]:
            continue
        groups[lbl].append(float(unit_acts[i]))
    return {lbl: float(np.mean(v)) for lbl, v in groups.items() if v}


def _unit_peak_metrics(
    unit_acts: np.ndarray,
    labels: list,
    valid_mask: list[bool] | None,
) -> tuple[float, float, float, str | None]:
    """
    Peaked selectivity for one unit vs one feature.

    Returns (selectivity_index, normalized_peak_gap, eta2, peak_category_label).
    Categories are ranked by |mean − grand_mean| so inverted tuning counts.
    """
    means = _unit_category_means(unit_acts, labels, valid_mask)
    if len(means) < 2:
        return float("nan"), float("nan"), float("nan"), None

    values = np.array([
        float(unit_acts[i])
        for i in range(len(labels))
        if valid_mask is None or valid_mask[i]
    ])
    grand = float(values.mean())
    scale = max(float(np.std(unit_acts)), 1e-9)

    ranked = sorted(means.items(), key=lambda kv: abs(kv[1] - grand), reverse=True)
    top_lbl, top_mean = ranked[0]
    _, second_mean = ranked[1]
    peak_gap_norm = (abs(top_mean - grand) - abs(second_mean - grand)) / scale

    abs_devs = sorted((abs(m - grand) for m in means.values()), reverse=True)
    r_max = abs_devs[0]
    r_others = float(np.mean(abs_devs[1:])) if len(abs_devs) > 1 else 0.0
    si = (r_max - r_others) / (r_max + r_others + 1e-9)

    return si, peak_gap_norm, _unit_eta2(unit_acts, labels, valid_mask), str(top_lbl)


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


def plot_selectivity_distributions(
    result: SelectivityResult,
    save_path: str,
    *,
    title: str,
) -> None:
    n_feat = len(ALL_CATEGORICAL_FEATURES)
    ncols = 3
    nrows = (n_feat + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for i, feat in enumerate(ALL_CATEGORICAL_FEATURES):
        ax = axes_flat[i]
        vals = result.si[feat]
        vals = vals[np.isfinite(vals)]
        ax.hist(vals, bins=20, color=FEATURE_COLORS.get(feat, "#888"), alpha=0.85, edgecolor="white")
        ax.set_title(FEATURE_DISPLAY[feat])
        ax.set_xlabel("selectivity index (peak vs rest)")
        ax.set_ylabel("units")
        ax.set_xlim(0, 1.05)
    for j in range(n_feat, len(axes_flat)):
        axes_flat[j].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
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
    axes[0].set_title("Primary feature (max selectivity index)")

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
    cmap = plt.cm.tab20
    cat_to_color, legend_labels, _ = _panel_feature_colors(
        feature, vals, visible, automaton, cmap,
    )
    return cat_to_color, legend_labels


def _timestep_tick_labels(labels: TimestepLabels, indices: list[int]) -> list[str]:
    return [labels.chars[i] for i in indices]


def _set_all_timestep_xticks(ax, labels: TimestepLabels, indices: list[int] | None = None) -> None:
    n = len(labels.chars)
    if indices is None:
        indices = list(range(n))
    ax.set_xticks(indices)
    ax.set_xticklabels(
        _timestep_tick_labels(labels, indices),
        rotation=90,
        va="top",
        ha="center",
        fontsize=7,
    )
    ax.tick_params(axis="x", pad=1)
    ax.set_xlabel("character at each timestep")


def _example_panel_figsize(
    n_timesteps: int,
    n_rows: int,
    n_cols: int = 2,
    *,
    compact: bool = False,
) -> tuple[float, float]:
    width = max(12.0, 0.14 * n_timesteps + 4.0) * (n_cols / 2)
    if compact:
        row_h = max(1.6, 0.9 + 0.012 * n_timesteps)
    else:
        row_h = max(3.4, 2.6 + 0.04 * n_timesteps)
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
) -> None:
    """Stem plot of unit activation per corpus timestep; color = feature category."""
    vals, mask = labels.feature_values(feature)
    n = len(unit_acts)
    visible = [i for i in range(n) if mask is None or mask[i]]
    if not visible:
        ax.set_axis_off()
        return

    cat_to_color, legend_labels = _category_color_map(
        feature, vals, visible, automaton,
    )

    y0 = 0.0
    lw = 1.0 if compact else 1.4
    pt = 18 if compact else 36
    for i in visible:
        color = cat_to_color[vals[i]]
        y = float(unit_acts[i])
        ax.plot([i, i], [y0, y], color=color, linewidth=lw, solid_capstyle="round", zorder=2)
        ax.scatter(i, y, c=[color], s=pt, zorder=3, edgecolors="0.25", linewidths=0.3)

    ax.axhline(y0, color="0.8", linewidth=0.6, zorder=1)
    ax.set_xlim(-0.8, n - 0.2)
    ax.set_ylabel("activation", fontsize=7 if compact else 9)
    if compact:
        ax.set_title(unit_label, fontsize=8)
        _set_all_timestep_xticks(ax, labels)
        ax.tick_params(axis="both", labelsize=7)
        ax.set_xlabel("character at each timestep", fontsize=7)
    else:
        ax.set_title(
            f"{unit_label} — each timestep is one character read\n"
            f"stem color = {FEATURE_DISPLAY[feature]}",
            fontsize=9,
        )
        _set_all_timestep_xticks(ax, labels)
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
            visible, [target_prob[i] for i in visible],
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
) -> None:
    bar_cats, means, sems = _category_means(unit_acts, vals, mask)
    x = np.arange(len(bar_cats))
    colors = [cat_to_color.get(c, "#888") for c in bar_cats]
    cap = 2 if compact else 4
    ax.bar(x, means, yerr=sems, color=colors, capsize=cap, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    rot = 45 if compact and len(bar_cats) > 4 else 30
    fs = 6 if compact else 8
    ax.set_xticklabels(
        [legend_labels.get(c, str(c)) for c in bar_cats],
        rotation=rot,
        ha="right",
        fontsize=fs,
    )
    ax.set_ylabel("mean activation", fontsize=7 if compact else 9)
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
) -> None:
    unit_acts = activations[:, unit_ix]
    vals, mask = labels.feature_values(feature)
    visible = [i for i in range(len(unit_acts)) if mask is None or mask[i]]
    cat_to_color, legend_labels = _category_color_map(
        feature, vals, visible, automaton,
    )

    _plot_unit_timestep_trace(
        ax_trace, unit_acts, labels, feature, unit_label,
        automaton=automaton,
        target_prob=target_prob,
        compact=compact,
    )
    _plot_unit_category_bars(
        ax_tune, unit_acts, vals, mask, cat_to_color, legend_labels,
        compact=compact,
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
    units = _top_units(result.si[feature], k=k)
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
            unit_label=unit_labels[u],
            activations=activations,
            labels=labels,
            feature=feature,
            target_prob=target_prob if feature in PREDICTION_SET else None,
            automaton=automaton,
        )
    fig.suptitle(
        f"Top units for {FEATURE_DISPLAY[feature]} (selectivity index)",
        fontsize=11,
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


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
    cmap = plt.cm.tab20
    cat_to_color = {c: cmap(i % 20) for i, c in enumerate(cats)}
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


def _panel_feature_colors(
    feat: str,
    vals: list,
    visible: list[int],
    automaton: MinimizedVocabAutomaton | None,
    cmap,
) -> tuple[dict, dict, list]:
    """Map feature categories to colors and legend labels for one panel."""
    from vocab_diagrams import dfa_state_label

    cats = sorted(set(vals[j] for j in visible), key=lambda x: (str(type(x)), str(x)))
    if feat == "dfa" and automaton is not None:
        from visualize import _state_id_colors

        state_colors = _state_id_colors([int(c) for c in cats])
        cat_to_color = {c: state_colors[int(c)] for c in cats}
        legend_labels = {c: dfa_state_label(int(c), automaton) for c in cats}
    else:
        cat_to_color = {c: cmap(ci % 20) for ci, c in enumerate(cats)}
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
        nrows, ncols, figsize=(5.5 * ncols, 5.0 * nrows), constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()
    cmap = plt.cm.tab20

    for i, feat in enumerate(features):
        ax = axes_flat[i]
        vals, mask = labels.feature_values(feat)
        visible = [j for j in range(n) if mask is None or mask[j]]
        if not visible:
            ax.axis("off")
            continue
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
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
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
    """Population summary: peaked selectivity (SI), η², peak gap, primary features."""
    feats = list(ALL_CATEGORICAL_FEATURES)
    x = np.arange(len(feats))
    colors = [FEATURE_COLORS.get(f, "#888") for f in feats]
    tick = [FEATURE_DISPLAY.get(f, f) for f in feats]

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0), constrained_layout=True)
    fig.suptitle(
        f"Unit selectivity summary ({repr_label}, n={result.n_points} points, "
        f"{len(result.primary_feature)} units)",
        fontsize=12,
        y=1.02,
    )

    def _bar(ax, values, ylabel, ylim=None) -> None:
        ax.bar(x, values, color=colors, edgecolor="0.2", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(tick, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, axis="y", linestyle=":", alpha=0.35)
        if ylim is not None:
            ax.set_ylim(*ylim)

    si_med = [float(np.nanmedian(result.si[f])) for f in feats]
    si_p90 = [float(np.nanpercentile(result.si[f], 90)) for f in feats]
    _bar(axes[0, 0], si_med, "Median selectivity index", (0, 1.05))
    axes[0, 0].set_title("Peaked tuning (top vs rest)")
    for i, p in enumerate(si_p90):
        axes[0, 0].text(i, si_med[i], f"p90={p:.2f}", ha="center", va="bottom", fontsize=6)

    eta_med = [float(np.nanmedian(result.eta2[f])) for f in feats]
    _bar(axes[0, 1], eta_med, "Median η²", (0, 1.05))
    axes[0, 1].set_title("Overall labeled variance")

    pg_med = [float(np.nanmedian(result.peak_gap[f])) for f in feats]
    _bar(axes[0, 2], pg_med, "Median peak gap / σ(unit)")
    axes[0, 2].set_title("Top − runner-up category")

    frac_strong = [
        float(np.mean(result.si[f] >= 0.5)) if np.isfinite(result.si[f]).any() else 0.0
        for f in feats
    ]
    _bar(axes[1, 0], frac_strong, "Fraction of units", (0, 1.05))
    axes[1, 0].set_title("Units with SI ≥ 0.5")

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
    ax.set_ylabel("selectivity index")
    ax.set_title("Per-unit SI vs η²")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6, loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.35)

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
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
        title=f"{repr_label} — selectivity index (peak vs rest)",
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
        title=f"{repr_label} — selectivity index distributions",
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
            result, activations, labels, feat,
            os.path.join(save_dir, f"example_units_{feat}.png"),
            unit_labels=unit_labels, text=text, model=model,
            automaton=automaton,
            k=example_k,
        )

    plot_example_predictor_units(
        result, activations, labels,
        os.path.join(save_dir, "example_units_predictors.png"),
        unit_labels=unit_labels, text=text, model=model,
        automaton=automaton,
        k=example_k,
    )
    plot_example_mixed_units(
        result, activations, labels,
        os.path.join(save_dir, "example_units_mixed.png"),
        unit_labels=unit_labels,
        automaton=automaton,
        k=example_k,
    )
    _save_summary_json(result, os.path.join(save_dir, "selectivity_summary.json"), unit_labels)
    return result
