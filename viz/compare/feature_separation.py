"""Cross-task feature separation comparison (grouped bars, multi-seed)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiment import TASKS, checkpoint_path, comparison_dir
from task import corpus_for_experiment, label_extensions_for_experiment
from unit_selectivity import FEATURE_DISPLAY
from vocab_diagrams import (
    build_minimized_vocabulary_automaton,
    select_analysis_window,
    vocabulary_for_experiment,
)
from viz.compare.spec import ComparisonSpec
from viz.plot_layout import (
    apply_category_tick_labels,
    condition_bar_colors,
    finalize_grid_figure,
    save_figure,
    set_ylabel_multiline,
)
from visualize import (
    SEPARATION_FEATURES,
    FeatureSeparationStats,
    compute_feature_separation_stats,
    condense_hidden_states_by_prefix,
    corpus_uses_word_spacing,
    load_model_for_viz,
    run_forward_pass,
)


def _pairwise_ratio(stats: FeatureSeparationStats, feature: str) -> float:
    w_med = stats.pairwise_within_median[feature]
    b_med = stats.pairwise_between_median[feature]
    if not np.isfinite(w_med) or not np.isfinite(b_med) or b_med <= 0:
        return float("nan")
    return float(w_med / b_med)


def _metric_value(stats: FeatureSeparationStats, metric_key: str, feature: str) -> float:
    if metric_key == "pairwise_ratio":
        return _pairwise_ratio(stats, feature)
    bucket = getattr(stats, metric_key)
    return float(bucket[feature])


_PANEL_SPECS: tuple[tuple[str, str], ...] = (
    ("centroid_gap", "between − within spread"),
    ("silhouette", "mean silhouette"),
    ("eta2", "η²"),
    ("pairwise_ratio", "within / between"),
    ("shuffle_z", "centroid gap\nshuffle z"),
    ("shuffle_p", "pairwise ratio\nshuffle p"),
)


def compute_task_feature_separation(
    task: str,
    *,
    model_type: str = "rnn",
    seed: int,
) -> FeatureSeparationStats | None:
    from unit_selectivity import build_timestep_labels

    ckpt = checkpoint_path(task, model_type, seed=seed)
    if not ckpt.is_file():
        return None

    cfg = TASKS[task]
    model = load_model_for_viz(str(ckpt), model_type)
    full_text = corpus_for_experiment(task, seed=seed)
    spaced = corpus_uses_word_spacing(full_text, task)
    words = vocabulary_for_experiment(task)
    length = min(int(cfg.get("viz_length", 50)), len(full_text))

    if words and not spaced:
        extensions = label_extensions_for_experiment(task)
        _win_start, text, label_words = select_analysis_window(
            full_text, words, length, spaced=spaced, extensions=extensions,
        )
    else:
        text = full_text[:length]
        label_words = None

    automaton = build_minimized_vocabulary_automaton(words) if words else None
    if automaton is None:
        return None

    hidden_states, output_probs = run_forward_pass(model, text, model_type)
    condensed = condense_hidden_states_by_prefix(
        text, hidden_states, output_probs, spaced=spaced, words=words,
    )
    ts_labels = build_timestep_labels(
        text, automaton,
        spaced=spaced, words=words, label_words=label_words, condensed=condensed,
    )
    return compute_feature_separation_stats(condensed.hidden_states, ts_labels)


def _plot_grouped_feature_bars(
    ax,
    *,
    features: tuple[str, ...],
    condition_labels: list[str],
    values_by_condition: list[list[float]],
    errs_by_condition: list[list[float]],
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    n_features = len(features)
    n_conditions = len(condition_labels)
    x = np.arange(n_features, dtype=float)
    width = min(0.18, 0.72 / max(n_conditions, 1))
    colors = condition_bar_colors(n_conditions)

    for ci in range(n_conditions):
        offset = (ci - (n_conditions - 1) / 2.0) * width
        means = np.asarray(values_by_condition[ci], dtype=float)
        errs = np.asarray(errs_by_condition[ci], dtype=float)
        ax.bar(
            x + offset,
            means,
            width,
            yerr=errs,
            label=condition_labels[ci],
            color=colors[ci],
            edgecolor="#333333",
            linewidth=0.6,
            capsize=2.5,
            error_kw={"elinewidth": 0.9, "ecolor": "#333333", "capthick": 0.9},
            zorder=2,
        )

    tick_labels = [FEATURE_DISPLAY.get(f, f) for f in features]
    apply_category_tick_labels(ax, tick_labels, fontsize=7)
    set_ylabel_multiline(ax, ylabel, fontsize=8)
    ax.grid(True, axis="y", linestyle=":", alpha=0.35, zorder=0)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if ylabel.startswith("mean silhouette"):
        ax.axhline(0.0, color="0.3", linewidth=0.8, linestyle=":")
    if ylabel.startswith("between"):
        ax.axhline(0.0, color="0.3", linewidth=0.8, linestyle=":")
    if ylabel.startswith("within /"):
        ax.axhline(1.0, color="0.3", linewidth=0.8, linestyle=":")


def plot_feature_separation_comparison(
    spec: ComparisonSpec,
    *,
    seeds: tuple[int, ...] | None = None,
) -> Path:
    """Grouped barplots: x = feature, bars = word-length condition, error = SEM over seeds."""
    run_seeds = seeds if seeds is not None else spec.seeds
    tasks = tuple(spec.tasks)
    condition_labels = [spec.label_for(t) for t in tasks]
    features = tuple(SEPARATION_FEATURES)

    # stats[task][seed] -> FeatureSeparationStats
    per_task_seed: dict[str, dict[int, FeatureSeparationStats]] = {t: {} for t in tasks}
    for task in tasks:
        for run_seed in run_seeds:
            stats = compute_task_feature_separation(task, model_type=spec.model_type, seed=run_seed)
            if stats is not None:
                per_task_seed[task][run_seed] = stats

    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.2))
    for ax, (metric_key, ylabel) in zip(axes.ravel(), _PANEL_SPECS):
        values_by_condition: list[list[float]] = []
        errs_by_condition: list[list[float]] = []
        for task in tasks:
            feat_vals: list[list[float]] = [[] for _ in features]
            for run_seed in run_seeds:
                stats = per_task_seed[task].get(run_seed)
                if stats is None:
                    continue
                for fi, feat in enumerate(features):
                    val = _metric_value(stats, metric_key, feat)
                    if np.isfinite(val):
                        feat_vals[fi].append(val)
            means = [
                float(np.mean(v)) if v else float("nan")
                for v in feat_vals
            ]
            sems = [
                float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
                for v in feat_vals
            ]
            values_by_condition.append(means)
            errs_by_condition.append(sems)

        ylim = (0.0, 1.05) if metric_key == "eta2" else None
        if metric_key == "silhouette":
            ylim = (-0.05, 1.05)
        _plot_grouped_feature_bars(
            ax,
            features=features,
            condition_labels=condition_labels,
            values_by_condition=values_by_condition,
            errs_by_condition=errs_by_condition,
            ylabel=ylabel,
            ylim=ylim,
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=len(condition_labels),
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    finalize_grid_figure(
        fig,
        suptitle=(
            f"{spec.display_title}: feature separation "
            f"({spec.model_type}, n={len(run_seeds)} seeds, prefix-condensed)"
        ),
        top=0.90,
        bottom=0.12,
        hspace=0.42,
    )

    out_dir = comparison_dir(spec.name, "feature_separation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.png"
    save_figure(fig, out_path, dpi=150)
    print(f"wrote {out_path}")

    json_path = out_dir / "summary.json"

    def _json_float(v: float) -> float | None:
        return None if isinstance(v, float) and not np.isfinite(v) else v

    payload: dict[str, Any] = {
        "tasks": list(tasks),
        "condition_labels": condition_labels,
        "features": list(features),
        "metrics": [m for m, _ in _PANEL_SPECS],
        "seeds": list(run_seeds),
        "per_task_seed": {},
    }
    for task in tasks:
        payload["per_task_seed"][task] = {}
        for run_seed, stats in per_task_seed[task].items():
            payload["per_task_seed"][task][str(run_seed)] = {
                metric_key: {
                    feat: _json_float(_metric_value(stats, metric_key, feat))
                    for feat in features
                }
                for metric_key, _ in _PANEL_SPECS
            }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {json_path}")
    return out_path
