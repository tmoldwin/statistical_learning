"""Compare transformer block_output unit selectivity across the micro curriculum."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment import EXPERIMENT_CONFIG, MICRO_CURRICULUM, input_path, model_path, spaced_experiment_name
from task import REGIMES
from transformer.adapter import extract_transformer_activations
from unit_selectivity import (
    FEATURE_DISPLAY,
    LEXICAL_FEATURES,
    PREDICTION_SET,
    SelectivityResult,
    TimestepLabels,
    _example_panel_figsize,
    _plot_unit_example_panel,
    _target_prob_array,
    _top_units,
    build_timestep_labels,
    compute_selectivity,
    plot_example_units_for_feature,
    plot_selectivity_heatmap,
)
from vocab_diagrams import MinimizedVocabAutomaton, build_minimized_vocabulary_automaton
from visualize import load_model_for_viz

PANEL_TITLES: dict[str, str] = {
    "two_word_disjoint": "disjoint",
    "two_word_pos_overlap": "same 2nd letter",
    "two_word_prefix_branch": "shared prefix",
    "three_word_overlap": "suffix family",
    "three_word_permutation": "permutation",
    "three_word_ca_hub": "3-way ca hub",
}

HEATMAP_FEATURES: tuple[str, ...] = LEXICAL_FEATURES + ("next_char",)
EXEMPLAR_FEATURES: tuple[str, ...] = (
    "dfa", "prefix", "char", "position", "word_start", "word_end", "next_char",
)


@dataclass
class ExperimentSelectivity:
    regime: str
    exp: str
    words: list[str]
    text: str
    model: dict
    automaton: MinimizedVocabAutomaton
    activations: np.ndarray
    labels: TimestepLabels
    result: SelectivityResult
    unit_labels: list[str]


def _curriculum_experiments(*, spaced: bool) -> list[str]:
    if spaced:
        return [spaced_experiment_name(regime) for regime in MICRO_CURRICULUM]
    return list(MICRO_CURRICULUM)


def _row_label(regime: str, words: list[str]) -> str:
    tag = PANEL_TITLES.get(regime, regime)
    return f"{tag}\n{', '.join(words)}"


def _feature_summary(gap: np.ndarray, *, stat: str = "p90") -> float:
    vals = gap[np.isfinite(gap)]
    if len(vals) == 0:
        return float("nan")
    if stat == "median":
        return float(np.median(vals))
    if stat == "max":
        return float(np.max(vals))
    return float(np.percentile(vals, 90))


def load_experiment_selectivity(exp: str) -> ExperimentSelectivity | None:
    ckpt = model_path(exp, "transformer")
    if not ckpt.is_file():
        print(f"skip {exp}: no transformer checkpoint at {ckpt}")
        return None

    cfg = EXPERIMENT_CONFIG[exp]
    regime = cfg["regime"]
    words = REGIMES[regime]
    spaced = bool(cfg.get("word_space", False))
    text = input_path(exp).read_text(encoding="utf-8")[: cfg["viz_length"]]
    model = load_model_for_viz(str(ckpt), "transformer")
    automaton = build_minimized_vocabulary_automaton(words)
    activations = extract_transformer_activations(model, text).block_output
    labels = build_timestep_labels(
        text, automaton,
        spaced=spaced, words=words,
        model=model, activations=activations,
    )
    result = compute_selectivity(activations, labels, model, text)
    unit_labels = [f"u{i}" for i in range(activations.shape[1])]
    return ExperimentSelectivity(
        regime=regime,
        exp=exp,
        words=words,
        text=text,
        model=model,
        automaton=automaton,
        activations=activations,
        labels=labels,
        result=result,
        unit_labels=unit_labels,
    )


def collect_rows(items: list[ExperimentSelectivity]) -> list[dict]:
    rows: list[dict] = []
    for item in items:
        row: dict = {
            "regime": item.regime,
            "exp": item.exp,
            "words": item.words,
        }
        for feat in HEATMAP_FEATURES:
            row[f"{feat}_gap_p90"] = _feature_summary(item.result.gap[feat], stat="p90")
            row[f"{feat}_gap_max"] = _feature_summary(item.result.gap[feat], stat="max")
            row[f"{feat}_eta2_p90"] = _feature_summary(item.result.eta2[feat], stat="p90")
        row["readout_max_logit_p90"] = _feature_summary(
            np.abs(item.result.max_logit_r), stat="p90",
        )
        rows.append(row)
    return rows


def plot_curriculum_heatmaps(
    items: list[ExperimentSelectivity],
    out_path: Path,
    *,
    spaced: bool,
) -> None:
    if not items:
        print("no experiments for heatmap")
        return

    labels = [_row_label(item.regime, item.words) for item in items]
    gap_mat = np.array([
        [_feature_summary(item.result.gap[feat], stat="p90") for feat in HEATMAP_FEATURES]
        for item in items
    ])
    eta2_mat = np.array([
        [_feature_summary(item.result.eta2[feat], stat="p90") for feat in HEATMAP_FEATURES]
        for item in items
    ])
    readout_mat = np.array([
        [
            _feature_summary(np.abs(item.result.max_logit_r), stat="p90"),
            _feature_summary(np.abs(item.result.target_prob_r), stat="p90"),
            _feature_summary(np.abs(item.result.entropy_r), stat="p90"),
        ]
        for item in items
    ])

    fig, axes = plt.subplots(
        1, 3,
        figsize=(20, max(6, 0.9 * len(items) + 2)),
        constrained_layout=True,
    )
    specs = [
        (axes[0], gap_mat, "YlOrRd", None, "normalized gap (p90 across units)"),
        (axes[1], eta2_mat, "YlOrRd", (0.0, 1.0), "η² (p90 across units)"),
        (axes[2], readout_mat, "RdBu_r", None, "readout |r| (p90 across units)"),
    ]
    feat_labels = [FEATURE_DISPLAY[f] for f in HEATMAP_FEATURES]
    readout_labels = ["max logit", "P(target)", "entropy"]

    for ax, mat, cmap, clim, cbar_label in specs:
        if clim is None:
            vmin = 0.0
            vmax = float(np.nanpercentile(mat, 98)) if np.isfinite(mat).any() else 1.0
            vmax = max(vmax, vmin + 1e-6)
            im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=clim[0], vmax=clim[1])
        ax.set_xticks(range(mat.shape[1]))
        ax.set_xticklabels(
            feat_labels if mat.shape[1] == len(HEATMAP_FEATURES) else readout_labels,
            rotation=45,
            ha="right",
            fontsize=8,
        )
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if not np.isfinite(val):
                    continue
                txt_color = "white" if clim is None and val > 0.65 * vmax else "0.15"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=txt_color)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=cbar_label)

    spacing = "spaced" if spaced else "unspaced"
    fig.suptitle(
        f"Unit selectivity across micro curriculum ({spacing})",
        fontsize=11,
        y=1.02,
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_per_regime_exemplars(
    item: ExperimentSelectivity,
    out_dir: Path,
    *,
    example_k: int,
) -> None:
    """Full exemplar pages per regime: top-k units, trace + category means."""
    regime_dir = out_dir / item.regime
    regime_dir.mkdir(parents=True, exist_ok=True)

    tag = PANEL_TITLES.get(item.regime, item.regime)
    repr_title = f"{tag} ({', '.join(item.words)})"

    plot_selectivity_heatmap(
        item.result.gap, HEATMAP_FEATURES,
        str(regime_dir / "selectivity_heatmap_gap.png"),
        title=f"{repr_title} — normalized gap per unit",
        unit_labels=item.unit_labels,
    )

    for feature in EXEMPLAR_FEATURES:
        plot_example_units_for_feature(
            item.result,
            item.activations,
            item.labels,
            feature,
            str(regime_dir / f"example_units_{feature}.png"),
            unit_labels=item.unit_labels,
            text=item.text,
            model=item.model,
            automaton=item.automaton,
            k=example_k,
        )


def plot_curriculum_exemplar_overview(
    items: list[ExperimentSelectivity],
    out_dir: Path,
    *,
    overview_k: int,
    spaced: bool,
    suffix: str,
) -> None:
    """Cross-regime comparison: 2×3 regimes, top-k units each."""
    if not items or overview_k < 1:
        return

    ncols = 3
    n_regime_rows = int(np.ceil(len(items) / ncols))
    n_timesteps = len(items[0].labels.chars)
    unit_w, unit_h = _example_panel_figsize(n_timesteps, 1, compact=True)
    title_row_h = 0.28
    regime_block_h = title_row_h + unit_h * overview_k
    fig_w = unit_w * ncols * 1.02
    fig_h = regime_block_h * n_regime_rows * 1.02 + 0.35

    block_h = [0.22] + [1.0] * overview_k
    height_ratios = block_h * n_regime_rows
    n_grid_rows = len(height_ratios)

    for feature in EXEMPLAR_FEATURES:
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = fig.add_gridspec(
            n_grid_rows,
            ncols * 2,
            height_ratios=height_ratios,
            hspace=0.14,
            wspace=0.18,
            top=0.94,
            bottom=0.03,
            left=0.04,
            right=0.98,
        )

        grid_row = 0
        for regime_row in range(n_regime_rows):
            for col in range(ncols):
                idx = regime_row * ncols + col
                if idx >= len(items):
                    continue
                item = items[idx]
                tag = PANEL_TITLES.get(item.regime, item.regime)
                ax_title = fig.add_subplot(gs[grid_row, col * 2 : (col + 1) * 2])
                ax_title.axis("off")
                ax_title.set_title(
                    f"{tag}  ({', '.join(item.words)})",
                    fontsize=10,
                    fontweight="bold",
                    loc="left",
                    pad=2,
                )

            grid_row += 1
            for _u_row in range(overview_k):
                for col in range(ncols):
                    idx = regime_row * ncols + col
                    if idx >= len(items):
                        continue
                    item = items[idx]
                    units = _top_units(item.result.gap[feature], k=overview_k)
                    if not units:
                        continue
                    unit_ix = units[_u_row] if _u_row < len(units) else None
                    if unit_ix is None:
                        continue

                    target_prob = (
                        _target_prob_array(item.model, item.activations, item.text)
                        if feature in PREDICTION_SET
                        else None
                    )
                    ax_trace = fig.add_subplot(gs[grid_row, col * 2])
                    ax_tune = fig.add_subplot(gs[grid_row, col * 2 + 1])
                    gap = float(item.result.gap[feature][unit_ix])
                    _plot_unit_example_panel(
                        ax_trace,
                        ax_tune,
                        unit_ix=unit_ix,
                        unit_label=f"{item.unit_labels[unit_ix]} · gap {gap:.2f}",
                        activations=item.activations,
                        labels=item.labels,
                        feature=feature,
                        target_prob=target_prob,
                        automaton=item.automaton,
                        compact=True,
                    )
                grid_row += 1

        spacing = "spaced" if spaced else "unspaced"
        fig.suptitle(
            f"Top {overview_k} units per regime ({spacing}) — "
            f"{FEATURE_DISPLAY[feature]}",
            fontsize=12,
            y=0.995,
        )
        out_path = out_dir / f"unit_exemplars_{feature}{suffix}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--example-k",
        type=int,
        default=4,
        help="top-k exemplar units in per-regime pages (default: 4)",
    )
    parser.add_argument(
        "--overview-k",
        type=int,
        default=4,
        help="top-k units per regime in cross-regime overview PNGs (default: 4)",
    )
    parser.add_argument(
        "--no-word-space",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use unspaced corpora (default: true); pass --no-word-space false for _s",
    )
    args = parser.parse_args()
    spaced = not args.no_word_space
    suffix = "" if spaced else "_no_space"

    items: list[ExperimentSelectivity] = []
    for exp in _curriculum_experiments(spaced=spaced):
        row = load_experiment_selectivity(exp)
        if row is not None:
            items.append(row)

    if not items:
        print("no transformer checkpoints found for micro curriculum")
        return

    out_dir = REPO_ROOT / "experiments" / "micro_curriculum_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(items)
    json_path = out_dir / f"unit_selectivity_curriculum{suffix}.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {json_path}")

    plot_curriculum_heatmaps(
        items, out_dir / f"unit_selectivity_curriculum_heatmap{suffix}.png", spaced=spaced,
    )

    exemplar_root = out_dir / f"unit_exemplars{suffix}"
    for item in items:
        plot_per_regime_exemplars(item, exemplar_root, example_k=args.example_k)

    plot_curriculum_exemplar_overview(
        items, out_dir, overview_k=args.overview_k, spaced=spaced, suffix=suffix,
    )


if __name__ == "__main__":
    main()
