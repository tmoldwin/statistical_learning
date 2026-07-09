"""Powers-of-2 sweep-scale linear decoding from hidden states and top-k PCA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from viz.compare.sweep_output import sweep_data_dir, sweep_decoding_dir
from viz.compare._data import load_task_decoding_context
from viz.compare.decoding import (
    DECODING_FEATURES,
    DECODE_FEATURE_COLORS,
    _DEFAULT_MAX_PCS,
    chance_corrected,
    compute_panel_decoding,
    feature_display_name,
    null_chances_for_vocab,
)
from viz.plot_layout import finalize_grid_figure, hide_x_tick_labels, save_figure
from viz.compare.pow2_sweep_spec import POW2_SWEEP_SPEC_NS, Pow2SweepSpec
from vocab_sweep_pow2 import (
    POW2_DEFAULT_SEEDS,
    POW2_LENGTHS,
    POW2_WORD_COUNTS,
    build_vocab,
    iter_pow2_sweep_cells,
    length_label,
    task_name,
)

POW2_SWEEP_COMPARISON_NAME = POW2_SWEEP_SPEC_NS.comparison_name


def _chance_length(words: list[str], length: object) -> int:
    if isinstance(length, int):
        return length
    return max((len(w) for w in words), default=1)


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _aggregate_feature_curves(
    seed_panels: list[dict[str, Any]],
    *,
    feature: str,
    max_k: int,
    by_k_field: str = "by_k",
) -> dict[str, Any]:
    """Mean/std accuracy curves for one sweep cell and feature."""
    chances: list[float] = []
    full_hiddens: list[float] = []
    by_k_rows: list[list[float]] = []

    for panel in seed_panels:
        feat = panel.get("features", {}).get(feature, {})
        if feat.get("error"):
            continue
        if "chance" in feat:
            chances.append(float(feat["chance"]))
        full_key = "full_hidden" if by_k_field == "by_k" else "full_hidden_neurons"
        if np.isfinite(feat.get(full_key, float("nan"))):
            full_hiddens.append(float(feat[full_key]))
        row = feat.get(by_k_field) or []
        if row:
            padded = list(row[:max_k])
            if len(padded) < max_k:
                padded.extend([float("nan")] * (max_k - len(padded)))
            by_k_rows.append(padded)

    out: dict[str, Any] = {"n_seeds": len(by_k_rows)}
    if chances:
        out["chance"] = float(np.mean(chances))
    if full_hiddens:
        out["full_hidden_mean"] = float(np.mean(full_hiddens))
        out["full_hidden_std"] = (
            float(np.std(full_hiddens, ddof=1)) if len(full_hiddens) > 1 else 0.0
        )
    if by_k_rows:
        arr = np.asarray(by_k_rows, dtype=float)
        valid = np.sum(np.isfinite(arr), axis=0)
        sums = np.nansum(arr, axis=0)
        means = np.full(max_k, np.nan, dtype=float)
        means[valid > 0] = sums[valid > 0] / valid[valid > 0]
        out["by_k_mean"] = means.tolist()
        if len(by_k_rows) > 1:
            centered = np.where(np.isfinite(arr), arr - means[None, :], np.nan)
            sq = np.nansum(centered ** 2, axis=0)
            std = np.full(max_k, np.nan, dtype=float)
            ok = valid > 1
            std[ok] = np.sqrt(sq[ok] / (valid[ok] - 1))
            out["by_k_std"] = std.tolist()
        else:
            out["by_k_std"] = [0.0 if v > 0 else float("nan") for v in valid]
    if chances:
        out["probe_n_classes"] = float(1.0 / np.mean(chances))
    return out


def _merge_aggregates(
    pca: dict[str, Any],
    neuron: dict[str, Any],
) -> dict[str, Any]:
    out = dict(pca)
    if neuron.get("by_k_mean"):
        out["by_k_neurons_mean"] = neuron["by_k_mean"]
    if neuron.get("by_k_std"):
        out["by_k_neurons_std"] = neuron["by_k_std"]
    if neuron.get("full_hidden_mean") is not None:
        out["full_hidden_neurons_mean"] = neuron["full_hidden_mean"]
        out["full_hidden_neurons_std"] = neuron.get("full_hidden_std", 0.0)
    return out


def write_pow2_sweep_decoding(
    *,
    seeds: tuple[int, ...] | None = None,
    model_type: str = "rnn",
    max_k: int = _DEFAULT_MAX_PCS,
    outfile: str = "sweep_decoding.json",
    spec: Pow2SweepSpec = POW2_SWEEP_SPEC_NS,
) -> Path:
    run_seeds = seeds if seeds is not None else spec.default_seeds
    agg_panels: list[dict[str, Any]] = []

    for n_words, length in spec.iter_cells():
        task = spec.task_name(n_words, length)
        cell_seed_panels: list[dict[str, Any]] = []
        hidden_size = 0
        n_samples = 0

        for run_seed in run_seeds:
            try:
                ctx = load_task_decoding_context(task, model_type=model_type, seed=run_seed)
            except (FileNotFoundError, KeyError):
                continue
            panel = compute_panel_decoding(ctx, max_k=max_k)
            panel["n_words"] = n_words
            panel["length"] = length
            cell_seed_panels.append(panel)
            hidden_size = int(panel.get("hidden_size", hidden_size))
            n_samples = max(n_samples, int(panel.get("n_samples", 0)))
            print(f"  {task} seed {run_seed}", flush=True)

        features_agg: dict[str, Any] = {}
        for feat in DECODING_FEATURES:
            pca_agg = _aggregate_feature_curves(
                cell_seed_panels, feature=feat, max_k=max_k, by_k_field="by_k",
            )
            neu_agg = _aggregate_feature_curves(
                cell_seed_panels, feature=feat, max_k=max_k, by_k_field="by_k_neurons",
            )
            features_agg[feat] = _merge_aggregates(pca_agg, neu_agg)

        agg_panels.append({
            "task": task,
            "n_words": n_words,
            "length": length,
            "n_seeds": len(cell_seed_panels),
            "hidden_size": hidden_size,
            "n_samples": n_samples,
            "null_chance": null_chances_for_vocab(
                spec.build_vocab(n_words, length),
                _chance_length(spec.build_vocab(n_words, length), length),
            ),
            "features": features_agg,
        })

    out_path = sweep_data_dir(spec.comparison_name) / outfile
    payload = {
        "comparison": spec.comparison_name,
        "model_type": model_type,
        "features": list(DECODING_FEATURES),
        "word_counts": list(spec.word_counts),
        "lengths": list(spec.lengths),
        "seeds": list(run_seeds),
        "max_k": max_k,
        "max_pcs": max_k,
        "panels": agg_panels,
    }
    out_path.write_text(
        json.dumps(_sanitize_for_json(payload), indent=2),
        encoding="utf-8",
    )
    return out_path


def _panel_feature_lookup(panels: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    return {(p["n_words"], p["length"]): p for p in panels}


def _trim_curve(y: np.ndarray) -> tuple[np.ndarray, int]:
    """Return finite prefix of ``y`` and its length."""
    arr = np.asarray(y, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return arr[:0], 0
    last = int(np.max(np.where(finite)[0])) + 1
    return arr[:last], last


def _null_chance_for_cell(
    panel: dict[str, Any],
    feat: str,
    *,
    n_words: int,
    length: int,
    build_vocab_fn=build_vocab,
) -> float | None:
    feat_data = panel.get("features", {}).get(feat, {})
    probe_chance = feat_data.get("chance")
    if probe_chance is not None and np.isfinite(probe_chance) and probe_chance < 1.0:
        return float(probe_chance)
    stored = feat_data.get("null_chance")
    if stored is not None and np.isfinite(stored):
        return float(stored)
    nc = panel.get("null_chance")
    if isinstance(nc, dict) and feat in nc:
        return float(nc[feat])
    words = build_vocab_fn(n_words, length)
    return null_chances_for_vocab(words, _chance_length(words, length)).get(feat)


def _by_k_mean_field(basis: str) -> str:
    return "by_k_mean" if basis == "pca" else "by_k_neurons_mean"


def _full_hidden_mean_field(basis: str) -> str:
    return "full_hidden_mean" if basis == "pca" else "full_hidden_neurons_mean"


def _correct_curve(y: np.ndarray, chance: float) -> np.ndarray:
    return np.asarray([chance_corrected(v, chance) for v in y], dtype=float)


def plot_pow2_sweep_decode_curves(
    panels: list[dict[str, Any]],
    *,
    word_counts: tuple[int, ...] | None = None,
    lengths: tuple[object, ...] | None = None,
    max_k: int = _DEFAULT_MAX_PCS,
    features: tuple[str, ...] = DECODING_FEATURES,
    basis: str = "pca",
    outfile: str | None = None,
    spec: Pow2SweepSpec = POW2_SWEEP_SPEC_NS,
) -> Path:
    """Grid: rows = letter length, cols = word count; lines = decoded features."""
    word_counts = spec.word_counts if word_counts is None else word_counts
    lengths = spec.lengths if lengths is None else lengths
    if outfile is None:
        outfile = (
            "sweep_decode_curves.png" if basis == "pca"
            else "sweep_decode_neuron_curves.png"
        )
    by_k_field = _by_k_mean_field(basis)
    full_field = _full_hidden_mean_field(basis)
    x_unit = "PCs" if basis == "pca" else "neurons"
    lookup = _panel_feature_lookup(panels)
    n_rows = len(lengths)
    n_cols = len(word_counts)
    full_x = float(max_k) + 1.5
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.8 * n_cols, 2.8 * n_rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    k_x = np.arange(1, max_k + 1, dtype=float)

    for li, length in enumerate(lengths):
        for wi, n_words in enumerate(word_counts):
            ax = axes[li, wi]
            panel = lookup.get((n_words, length), {})
            for feat in features:
                feat_data = panel.get("features", {}).get(feat, {})
                by_k = feat_data.get(by_k_field) or []
                if not by_k:
                    continue
                null_ch = _null_chance_for_cell(
                    panel, feat, n_words=n_words, length=length,
                    build_vocab_fn=spec.build_vocab,
                )
                if null_ch is None:
                    continue
                y, n_plot = _trim_curve(np.asarray(by_k, dtype=float))
                if n_plot == 0:
                    continue
                y = _correct_curve(y, null_ch)
                if not np.any(np.isfinite(y)):
                    continue
                color = DECODE_FEATURE_COLORS.get(feat, "#888888")
                ax.plot(
                    k_x[:n_plot],
                    y,
                    color=color,
                    linewidth=1.8,
                    alpha=0.95,
                    marker="o",
                    markersize=3.5,
                )
                full_acc = feat_data.get(full_field)
                if full_acc is not None and np.isfinite(full_acc):
                    corr_full = chance_corrected(float(full_acc), null_ch)
                    if np.isfinite(corr_full):
                        ax.scatter(
                            [full_x],
                            [corr_full],
                            color=color,
                            s=45,
                            marker="*",
                            zorder=5,
                            edgecolors="white",
                            linewidths=0.4,
                        )

            ax.axhline(0.0, color="0.35", linestyle="--", linewidth=0.8, alpha=0.8)
            if li == 0:
                ax.set_title(f"{n_words} words", fontsize=10, fontweight="medium")
            if wi == 0:
                ax.set_ylabel(spec.length_label(length), fontsize=9)
            ax.grid(axis="y", alpha=0.3, linewidth=0.5)
            if li == n_rows - 1:
                ax.set_xlabel(f"# {x_unit}  |  star = full hidden", fontsize=8)

    xticks = [1, 5, 10, 15, 20]
    xticks = [t for t in xticks if t <= max_k]
    if max_k not in xticks:
        xticks.append(max_k)
    xtick_labels = [*(str(t) for t in xticks), "full"]
    for li in range(n_rows):
        for wi in range(n_cols):
            ax = axes[li, wi]
            ax.set_xlim(0.6, full_x + 0.6)
            ax.set_xticks([*xticks, full_x])
            ax.set_ylim(-0.05, 1.05)
            if li == n_rows - 1:
                ax.set_xticklabels(xtick_labels, fontsize=7)
            else:
                hide_x_tick_labels(ax)

    fig.text(
        0.02, 0.5,
        "(accuracy - chance) / (1 - chance)",
        va="center",
        rotation="vertical",
        fontsize=9,
        color="0.25",
    )

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0], [0],
            color=DECODE_FEATURE_COLORS.get(feat, "#888888"),
            linewidth=1.8,
            marker="o",
            markersize=4,
            label=feature_display_name(feat),
        )
        for feat in features
    ]
    fig.legend(
        legend_handles,
        [feature_display_name(f) for f in features],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=4,
        fontsize=7,
        framealpha=0.9,
    )

    basis_label = "PCA" if basis == "pca" else "top-variance neurons"
    finalize_grid_figure(
        fig,
        suptitle=f"Chance-corrected decoding vs # {x_unit} ({basis_label})",
        top=0.96,
        hspace=0.42,
        wspace=0.30,
        bottom=0.14,
    )
    out_dir = sweep_decoding_dir(spec.comparison_name)
    out_path = out_dir / outfile
    save_figure(fig, out_path, dpi=160)
    return out_path


def replot_pow2_sweep_decoding(
    *,
    decoding_file: str = "sweep_decoding.json",
    spec: Pow2SweepSpec = POW2_SWEEP_SPEC_NS,
) -> tuple[Path, Path]:
    payload = json.loads(
        (sweep_data_dir(spec.comparison_name) / decoding_file).read_text(encoding="utf-8"),
    )
    max_k = int(payload.get("max_k", payload.get("max_pcs", _DEFAULT_MAX_PCS)))
    panels = payload["panels"]
    curves_pca = plot_pow2_sweep_decode_curves(panels, max_k=max_k, basis="pca", spec=spec)
    curves_neu = plot_pow2_sweep_decode_curves(panels, max_k=max_k, basis="neuron", spec=spec)
    return curves_pca, curves_neu


def run_pow2_sweep_decoding_plots(
    *,
    seeds: tuple[int, ...] | None = None,
    max_k: int = _DEFAULT_MAX_PCS,
    recompute: bool = True,
    decoding_file: str = "sweep_decoding.json",
    spec: Pow2SweepSpec = POW2_SWEEP_SPEC_NS,
) -> tuple[Path, Path, Path]:
    json_path = sweep_data_dir(spec.comparison_name) / decoding_file
    if recompute or not json_path.is_file():
        json_path = write_pow2_sweep_decoding(
            seeds=seeds, max_k=max_k, outfile=decoding_file, spec=spec,
        )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    max_k = int(payload.get("max_k", payload.get("max_pcs", max_k)))
    panels = payload["panels"]
    curves_pca = plot_pow2_sweep_decode_curves(panels, max_k=max_k, basis="pca", spec=spec)
    curves_neu = plot_pow2_sweep_decode_curves(panels, max_k=max_k, basis="neuron", spec=spec)
    print(f"wrote {json_path}")
    print(f"wrote {curves_pca}")
    print(f"wrote {curves_neu}")
    return json_path, curves_pca, curves_neu
