"""Per-letter hidden states in PCA, one feature per figure (regimes × inits)."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment import (
    EXPERIMENT_CONFIG,
    MICRO_CURRICULUM,
    MICRO_CURRICULUM_INIT_SEEDS,
    MICRO_CURRICULUM_REGIME_LABELS,
    MODEL_TYPES,
    micro_curriculum_repr_label,
    micro_curriculum_viz_dir,
    model_path,
    spaced_experiment_name,
)
from task import REGIMES
from transformer.adapter import extract_transformer_activations
from unit_selectivity import (
    FEATURE_DISPLAY,
    TimestepLabels,
    _panel_feature_colors,
    build_timestep_labels,
)
from vocab_diagrams import MinimizedVocabAutomaton, build_minimized_vocabulary_automaton
from visualize import (
    PAIR_DISTANCE_PALETTE,
    _draw_annotation_groups,
    _expand_limits_for_annotations,
    _layout_group_label_positions,
    _longest_vocabulary_word_length,
    _square_data_limits,
    _trajectory_vocabulary_words,
    argmax_next_char,
    fit_pca_2d_with_evr,
    load_model_for_viz,
    median_mad,
    pairwise_hidden_state_distance_groups,
    run_forward_pass,
)

REGIME_LABELS = MICRO_CURRICULUM_REGIME_LABELS

STATE_FEATURES: tuple[str, ...] = (
    "char",
    "position",
    "prefix",
    "dfa",
    "next_char",
    "predicted_char",
    "word_end",
    "prediction_entropy",
)

# Color by this TimestepLabels field (prefix column uses string = annotated partial word).
COLOR_FEATURE: dict[str, str] = {
    "prefix": "string",
}

GRID_COLOR_FEATURES: tuple[str, ...] = (
    "char",
    "position",
    "prefix",
    "dfa",
    "next_char",
    "predicted_char",
)

GRID_DISTANCE_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("prefix", "Within prefix", "Between prefixes"),
    ("dfa", "Within DFA state", "Between DFA states"),
)

FEATURE_TITLES: dict[str, str] = {
    **FEATURE_DISPLAY,
    "predicted_char": "predicted next char",
}


@dataclass
class WordLetterPath:
    word: str
    projected: np.ndarray
    labels: TimestepLabels
    predicted_char: list[str]


@dataclass
class StatesPanelData:
    paths: list[WordLetterPath]
    automaton: MinimizedVocabAutomaton
    hidden: np.ndarray
    labels: TimestepLabels
    predicted_char: list[str]


def _fix_labels_for_isolated_word(snippet: str, labels: TimestepLabels) -> TimestepLabels:
    """Teacher-forced single words: no wrap-around next-char at final letter."""
    n = len(snippet)
    next_char = [snippet[i + 1] if i + 1 < n else "·" for i in range(n)]
    return TimestepLabels(
        chars=list(labels.chars),
        prefix=list(labels.prefix),
        string=list(labels.string),
        dfa=list(labels.dfa),
        position=list(labels.position),
        word_start=list(labels.word_start),
        word_end=list(labels.word_end),
        next_char=next_char,
        next_char_2=list(labels.next_char_2),
        next_bigram=list(labels.next_bigram),
        pred_entropy_bin=list(labels.pred_entropy_bin),
    )


def _append_timestep_labels(acc: TimestepLabels | None, labels: TimestepLabels) -> TimestepLabels:
    if acc is None:
        return TimestepLabels(
            chars=list(labels.chars),
            prefix=list(labels.prefix),
            string=list(labels.string),
            dfa=list(labels.dfa),
            position=list(labels.position),
            word_start=list(labels.word_start),
            word_end=list(labels.word_end),
            next_char=list(labels.next_char),
            next_char_2=list(labels.next_char_2),
            next_bigram=list(labels.next_bigram),
            pred_entropy_bin=list(labels.pred_entropy_bin),
        )
    return TimestepLabels(
        chars=acc.chars + list(labels.chars),
        prefix=acc.prefix + list(labels.prefix),
        string=acc.string + list(labels.string),
        dfa=acc.dfa + list(labels.dfa),
        position=acc.position + list(labels.position),
        word_start=acc.word_start + list(labels.word_start),
        word_end=acc.word_end + list(labels.word_end),
        next_char=acc.next_char + list(labels.next_char),
        next_char_2=acc.next_char_2 + list(labels.next_char_2),
        next_bigram=acc.next_bigram + list(labels.next_bigram),
        pred_entropy_bin=acc.pred_entropy_bin + list(labels.pred_entropy_bin),
    )


def _word_hidden(model: dict, snippet: str, *, model_type: str) -> np.ndarray:
    if model_type == "transformer":
        return extract_transformer_activations(model, snippet).block_output
    hidden, _ = run_forward_pass(model, snippet, model_type)
    return hidden


def _predicted_chars(model: dict, hidden: np.ndarray) -> list[str]:
    chars = list(model["chars"])
    idxs = argmax_next_char(model, hidden)
    return [chars[int(i)] for i in idxs]


def _collect_states_panel(
    exp: str,
    *,
    model_type: str,
    seed: int,
) -> StatesPanelData | None:
    ckpt = model_path(exp, model_type, seed=seed)
    if not ckpt.is_file():
        return None

    cfg = EXPERIMENT_CONFIG[exp]
    regime = cfg["regime"]
    words = REGIMES[regime]
    spaced = bool(cfg.get("word_space", False))
    model = load_model_for_viz(str(ckpt), model_type)
    automaton = build_minimized_vocabulary_automaton(words)
    vocab_words = _trajectory_vocabulary_words("", words)
    max_word_len = _longest_vocabulary_word_length(vocab_words)

    word_hidden: list[np.ndarray] = []
    word_names: list[str] = []
    word_labels: list[TimestepLabels] = []
    word_predicted: list[list[str]] = []
    flat_labels: TimestepLabels | None = None
    flat_predicted: list[str] = []

    for word in sorted(set(vocab_words)):
        snippet = word[:max_word_len]
        if not snippet:
            continue
        hidden = _word_hidden(model, snippet, model_type=model_type)
        if hidden.size == 0:
            continue
        labels = build_timestep_labels(
            snippet,
            automaton,
            spaced=spaced,
            words=words,
            model=model,
            activations=hidden,
        )
        labels = _fix_labels_for_isolated_word(snippet, labels)
        predicted = _predicted_chars(model, hidden)
        word_names.append(word)
        word_hidden.append(hidden)
        word_labels.append(labels)
        word_predicted.append(predicted)
        flat_labels = _append_timestep_labels(flat_labels, labels)
        flat_predicted.extend(predicted)

    if not word_names or flat_labels is None:
        return None

    stacked = np.vstack(word_hidden)
    _projected, mean, components, _evr = fit_pca_2d_with_evr(stacked)

    paths: list[WordLetterPath] = []
    for word, hidden, labels, predicted in zip(
        word_names, word_hidden, word_labels, word_predicted, strict=True,
    ):
        z = (hidden - mean) @ components.T
        paths.append(WordLetterPath(word=word, projected=z, labels=labels, predicted_char=predicted))

    return StatesPanelData(
        paths=paths,
        automaton=automaton,
        hidden=stacked,
        labels=flat_labels,
        predicted_char=flat_predicted,
    )


def _feature_values_for_color(
    data: StatesPanelData,
    feature: str,
    path: WordLetterPath,
) -> tuple[list, list[bool] | None]:
    color_feature = COLOR_FEATURE.get(feature, feature)
    if color_feature == "predicted_char":
        return path.predicted_char, None
    return path.labels.feature_values(color_feature)


def _gather_labeled_points(
    data: StatesPanelData,
    *,
    feature: str,
) -> tuple[np.ndarray, list[str], list[str], list, list[int], dict, dict] | None:
    """Flatten panel points with one global feature→color map."""
    projected_rows: list[np.ndarray] = []
    group_keys: list[str] = []
    display_prefixes: list[str] = []
    feature_vals: list = []
    visible_global: list[int] = []

    for path in data.paths:
        vals, mask = _feature_values_for_color(data, feature, path)
        for j in range(len(path.projected)):
            if mask is not None and not mask[j]:
                continue
            gi = len(projected_rows)
            visible_global.append(gi)
            projected_rows.append(path.projected[j])
            prefix = path.labels.string[j]
            display_prefixes.append(prefix)
            group_keys.append(f"{path.word}\0{prefix}")
            feature_vals.append(vals[j])

    if not projected_rows:
        return None

    projected = np.asarray(projected_rows)
    cmap = plt.cm.tab20
    color_feature = COLOR_FEATURE.get(feature, feature)
    cat_to_color, legend_labels, point_colors = _panel_feature_colors(
        color_feature,
        feature_vals,
        visible_global,
        data.automaton,
        cmap,
    )
    return (
        projected,
        group_keys,
        display_prefixes,
        feature_vals,
        point_colors,
        cat_to_color,
        legend_labels,
    )


def _plot_states_panel(
    ax: plt.Axes,
    data: StatesPanelData | None,
    *,
    feature: str,
    show_axis_labels: bool,
    show_legend: bool,
    point_size: float = 24,
    label_fontsize: float = 5.5,
    leader_linewidth: float = 0.45,
) -> bool:
    if data is None or not data.paths:
        ax.set_xticks([])
        ax.set_yticks([])
        return False

    gathered = _gather_labeled_points(data, feature=feature)
    if gathered is None:
        ax.set_xticks([])
        ax.set_yticks([])
        return False

    projected, group_keys, display_prefixes, _feature_vals, point_colors, cat_to_color, legend_labels_map = gathered

    by_group: dict[str, list[int]] = defaultdict(list)
    for i, key in enumerate(group_keys):
        by_group[key].append(i)
    label_positions = _layout_group_label_positions(projected, by_group)
    label_text = {
        key: ("␣" if display_prefixes[idxs[0]] == " " else display_prefixes[idxs[0]])
        for key, idxs in by_group.items()
    }

    text_positions = _draw_annotation_groups(
        ax,
        projected,
        by_group,
        label_positions,
        point_colors,
        label_text,
        point_size=point_size,
        label_fontsize=label_fontsize,
        leader_linewidth=leader_linewidth,
    )

    base_xlim, base_ylim = _square_data_limits(projected)
    _expand_limits_for_annotations(ax, projected, text_positions, base_xlim, base_ylim)
    ax.set_aspect("equal", adjustable="box")
    if show_axis_labels:
        ax.set_xlabel("PC1", fontsize=7)
        ax.set_ylabel("PC2", fontsize=7)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    ax.tick_params(labelsize=6)
    ax.grid(True, linestyle=":", alpha=0.3)

    if show_legend:
        cats = sorted(cat_to_color.keys(), key=lambda x: (str(type(x)), str(x)))
        if len(cats) <= 12:
            handles = [
                Patch(facecolor=cat_to_color[c], label=legend_labels_map[c])
                for c in cats
            ]
            ax.legend(handles=handles, fontsize=5, loc="upper right", framealpha=0.85)
    return True


def _plot_within_between_panel(
    ax: plt.Axes,
    data: StatesPanelData | None,
    *,
    within_key: str,
    between_key: str,
    show_axis_labels: bool,
    show_legend: bool = False,
) -> bool:
    if data is None or data.hidden.shape[0] < 2:
        ax.set_xticks([])
        ax.set_yticks([])
        return False

    labels = data.labels
    by_label = pairwise_hidden_state_distance_groups(
        labels.chars,
        data.hidden,
        labels.dfa,
        labels.position,
        labels.prefix,
        labels.string,
    )
    within_vals = by_label.get(within_key, np.asarray([], dtype=float))
    between_vals = by_label.get(between_key, np.asarray([], dtype=float))
    if len(within_vals) == 0 and len(between_vals) == 0:
        ax.set_xticks([])
        ax.set_yticks([])
        return False

    order = [within_key, between_key]
    specs = [(k, np.asarray(by_label.get(k, []), dtype=float)) for k in order]
    rng = np.random.default_rng(0)
    max_points = 120
    palette = PAIR_DISTANCE_PALETTE
    x = np.arange(len(order))
    ymax = 0.0
    for i, (label, vals) in enumerate(specs):
        if len(vals) == 0:
            continue
        if len(vals) > max_points:
            idx = rng.choice(len(vals), size=max_points, replace=False)
            vals = vals[idx]
        jitter = rng.uniform(-0.16, 0.16, size=len(vals))
        color = palette.get(label, "0.45")
        ax.scatter(
            x[i] + jitter,
            vals,
            c=color,
            alpha=0.4,
            s=10,
            linewidths=0,
            zorder=1,
        )
        if len(vals):
            med, mad = median_mad(vals)
            err_lo = min(med, mad)
            ax.errorbar(
                x[i],
                med,
                yerr=np.array([[err_lo], [mad]]),
                fmt="D",
                color=color,
                ecolor=color,
                elinewidth=1.2,
                capsize=4,
                capthick=1.2,
                markersize=5,
                markerfacecolor="white",
                markeredgecolor="0.15",
                markeredgewidth=1.0,
                zorder=4,
            )
            ymax = max(ymax, float(np.max(vals)))

    ax.set_xticks(x)
    short = {"Within prefix": "within", "Between prefixes": "between",
             "Within DFA state": "within", "Between DFA states": "between"}
    ax.set_xticklabels([short.get(k, k) for k in order], fontsize=6)
    ax.set_ylim(bottom=0, top=max(ymax * 1.08, 0.01))
    if show_axis_labels:
        ax.set_ylabel("L2 dist.", fontsize=6)
    else:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=5)
    ax.grid(True, axis="y", linestyle=":", alpha=0.3)

    if show_legend:
        handles = [
            Patch(facecolor=palette.get(k, "0.45"), label=k)
            for k in order if len(by_label.get(k, []))
        ]
        if handles:
            ax.legend(handles=handles, fontsize=4, loc="upper right", framealpha=0.85)
    return True


def _write_feature_figure(
    *,
    feature: str,
    exps: list[str],
    regimes: list[str],
    seeds: list[int],
    model_type: str,
    spaced: bool,
    out_path: Path,
) -> None:
    has_any = any(
        model_path(exp, model_type, seed=seed).is_file()
        for exp in exps
        for seed in seeds
    )
    if not has_any:
        print(f"skip {out_path}: no seeded {model_type} checkpoints for micro curriculum")
        return

    nrows = len(seeds)
    ncols = len(exps)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.6 * ncols, 2.4 * nrows),
        constrained_layout=True,
        squeeze=False,
    )

    for row, seed in enumerate(seeds):
        for col, (regime, exp) in enumerate(zip(regimes, exps, strict=True)):
            ax = axes[row, col]
            if not model_path(exp, model_type, seed=seed).is_file():
                ax.text(
                    0.5, 0.5, f"missing\nseed {seed}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=7,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            panel = _collect_states_panel(exp, model_type=model_type, seed=seed)
            _plot_states_panel(
                ax,
                panel,
                feature=feature,
                show_axis_labels=col == 0,
                show_legend=row == 0 and col == 0,
            )
            if row == 0:
                words = REGIMES[regime]
                tag = REGIME_LABELS.get(regime, regime)
                ax.set_title(f"{', '.join(words)}\n({tag})", fontsize=8)
            if col == 0:
                ax.text(
                    -0.28, 0.5, f"seed {seed}",
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                )

    spacing = "spaced" if spaced else "unspaced"
    repr_label = micro_curriculum_repr_label(model_type)
    feat_title = FEATURE_TITLES.get(feature, feature)
    fig.suptitle(
        f"Micro curriculum: per-letter states by {feat_title} "
        f"({spacing}, {model_type} {repr_label}, PCA)",
        fontsize=11,
        y=1.02,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def _seed_for_experiment_row(row: int, seeds: list[int]) -> int:
    """One init per experiment row, cycling through available seeds."""
    return seeds[row % len(seeds)]


def _write_experiments_by_feature_figure(
    *,
    exps: list[str],
    regimes: list[str],
    seeds: list[int],
    model_type: str,
    spaced: bool,
    out_path: Path,
) -> None:
    has_any = any(
        model_path(exp, model_type, seed=_seed_for_experiment_row(i, seeds)).is_file()
        for i, exp in enumerate(exps)
    )
    if not has_any:
        print(f"skip {out_path}: no seeded {model_type} checkpoints for micro curriculum")
        return

    color_features = list(GRID_COLOR_FEATURES)
    distance_cols = list(GRID_DISTANCE_COLUMNS)
    ncols = len(color_features) + len(distance_cols)
    nrows = len(exps)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.15 * ncols, 2.15 * nrows),
        constrained_layout=True,
        squeeze=False,
    )

    for row, (regime, exp) in enumerate(zip(regimes, exps, strict=True)):
        seed = _seed_for_experiment_row(row, seeds)
        panel = _collect_states_panel(exp, model_type=model_type, seed=seed)
        words = REGIMES[regime]
        tag = REGIME_LABELS.get(regime, regime)
        missing = panel is None or not model_path(exp, model_type, seed=seed).is_file()

        for col, feature in enumerate(color_features):
            ax = axes[row, col]
            if missing:
                ax.text(
                    0.5, 0.5, f"missing\nseed {seed}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=7,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            _plot_states_panel(
                ax,
                panel,
                feature=feature,
                show_axis_labels=col == 0,
                show_legend=False,
                point_size=18,
                label_fontsize=4.8,
                leader_linewidth=0.35,
            )
            if row == 0:
                ax.set_title(FEATURE_TITLES.get(feature, feature), fontsize=8)
            if col == 0:
                ax.text(
                    -0.32, 0.5,
                    f"{', '.join(words)}\n({tag})\ninit {seed}",
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=7,
                    fontweight="bold",
                )

        for j, (dist_key, within_key, between_key) in enumerate(distance_cols):
            col = len(color_features) + j
            ax = axes[row, col]
            if missing:
                ax.text(
                    0.5, 0.5, f"missing\nseed {seed}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=7,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            _plot_within_between_panel(
                ax,
                panel,
                within_key=within_key,
                between_key=between_key,
                show_axis_labels=col == 0,
                show_legend=row == 0 and j == 0,
            )
            if row == 0:
                title = "prefix dist." if dist_key == "prefix" else "DFA dist."
                ax.set_title(title, fontsize=8)

    spacing = "spaced" if spaced else "unspaced"
    repr_label = micro_curriculum_repr_label(model_type)
    fig.suptitle(
        f"Micro curriculum: per-letter states — experiments × features "
        f"({spacing}, {model_type} {repr_label}, PCA + within/between distances)",
        fontsize=11,
        y=1.01,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-word-space",
        action="store_true",
        help="use unspaced micro-curriculum experiments (output under _ns)",
    )
    parser.add_argument(
        "--model-type",
        default="rnn",
        choices=list(MODEL_TYPES),
        help="which model checkpoints to plot (default: rnn)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(MICRO_CURRICULUM_INIT_SEEDS),
        help="initialization seeds (rows)",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=list(STATE_FEATURES),
        choices=list(STATE_FEATURES),
        help="which feature coloring figures to write (default: all)",
    )
    args = parser.parse_args()

    spaced = not args.no_word_space
    regimes = list(MICRO_CURRICULUM)
    exps = (
        [spaced_experiment_name(r) for r in regimes]
        if spaced
        else regimes
    )
    out_dir = micro_curriculum_viz_dir(
        spaced=spaced, model_type=args.model_type, kind="states",
    )
    features = list(args.features)

    for feature in features:
        _write_feature_figure(
            feature=feature,
            exps=exps,
            regimes=regimes,
            seeds=list(args.seeds),
            model_type=args.model_type,
            spaced=spaced,
            out_path=out_dir / f"{feature}_by_init.png",
        )

    _write_experiments_by_feature_figure(
        exps=exps,
        regimes=regimes,
        seeds=list(args.seeds),
        model_type=args.model_type,
        spaced=spaced,
        out_path=out_dir / "experiments_by_feature.png",
    )


if __name__ == "__main__":
    main()
