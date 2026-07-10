"""Linear readout decoding curves (PCA vs random neuron subsets) for one task."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from experiment import common_seeds
from viz.compare._data import load_task_decoding_context, load_task_viz_context
from viz.compare.decoding import (
    DECODING_FEATURES,
    DECODE_FEATURE_COLORS,
    _DEFAULT_MAX_PCS,
    _DEFAULT_NEURON_RANDOM_TRIALS,
    chance_corrected,
    compute_panel_decoding,
    feature_display_name,
)
from viz.compare.trajectories import _plot_task_closed_loop_panel
from readme_figures import numbered_plot_path
from viz.plot_layout import finalize_grid_figure, hide_x_tick_labels, save_figure


def _trim_curve(y: np.ndarray) -> tuple[np.ndarray, int]:
    arr = np.asarray(y, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return arr[:0], 0
    last = int(np.max(np.where(finite)[0])) + 1
    return arr[:last], last


def _null_chance(panel: dict[str, Any], feat: str) -> float | None:
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
    return None


def _chance_correct_curve(
    y: np.ndarray,
    y_std: np.ndarray | None,
    null_ch: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Chance-correct mean curve; scale raw std by 1/(1-chance)."""
    y_corr = np.asarray([chance_corrected(v, null_ch) for v in y], dtype=float)
    if y_std is None:
        return y_corr, None
    scale = 1.0 / max(1.0 - null_ch, 1e-12)
    return y_corr, np.asarray(y_std, dtype=float) * scale


def _apply_k_ticks(ax, max_k: int, *, fontsize: float = 8, show_labels: bool = True) -> None:
    full_x = float(max_k) + 1.5
    xticks = [t for t in (1, 5, 10, 15, 20) if t <= max_k]
    if max_k not in xticks:
        xticks.append(max_k)
    ax.set_xlim(0.6, full_x + 0.6)
    ax.set_xticks([*xticks, full_x])
    if show_labels:
        ax.set_xticklabels([*(str(t) for t in xticks), "full"], fontsize=fontsize)
    else:
        hide_x_tick_labels(ax)


def _panel_y_range(
    panel: dict[str, Any],
    *,
    max_k: int,
    features: tuple[str, ...],
    basis: str = "both",
) -> tuple[float, float]:
    """Finite y range (chance-corrected, incl. neuron std bands) for tight axes."""
    vals: list[float] = []
    field_specs: list[tuple[str, str | None]] = []
    if basis in ("both", "pca"):
        field_specs.append(("by_k", None))
    if basis in ("both", "neuron"):
        field_specs.append(("by_k_neurons", "by_k_neurons_std"))
    for feat in features:
        feat_data = panel.get("features", {}).get(feat, {})
        null_ch = _null_chance(panel, feat)
        if null_ch is None:
            continue
        for field, std_field in field_specs:
            by_k = feat_data.get(field) or []
            if not by_k:
                continue
            y, n_plot = _trim_curve(np.asarray(by_k, dtype=float))
            if n_plot == 0:
                continue
            y_std = None
            if std_field and feat_data.get(std_field):
                y_std, _ = _trim_curve(np.asarray(feat_data[std_field], dtype=float))
                y_std = y_std[:n_plot]
            y, y_std = _chance_correct_curve(y, y_std, null_ch)
            vals.extend(y[np.isfinite(y)].tolist())
            if y_std is not None:
                vals.extend((y - y_std)[np.isfinite(y)].tolist())
                vals.extend((y + y_std)[np.isfinite(y)].tolist())
    if not vals:
        return 0.0, 1.02
    ymin = float(np.min(vals))
    ymax = float(np.max(vals))
    pad = max(0.02, 0.03 * (ymax - ymin))
    return max(0.0, ymin - pad), min(1.02, ymax + pad)


def _plot_decode_pca_on_ax(
    ax,
    panel: dict[str, Any],
    *,
    max_k: int,
    features: tuple[str, ...] = DECODING_FEATURES,
    show_legend: bool = False,
    compact: bool = False,
    ylim: tuple[float, float] | None = None,
) -> None:
    full_x = float(max_k) + 1.5
    k_x = np.arange(1, max_k + 1, dtype=float)
    lw = 1.2 if compact else 1.8
    ms = 2.5 if compact else 4

    for feat in features:
        feat_data = panel.get("features", {}).get(feat, {})
        by_k = feat_data.get("by_k") or []
        if not by_k:
            continue
        null_ch = _null_chance(panel, feat)
        if null_ch is None:
            continue
        y, n_plot = _trim_curve(np.asarray(by_k, dtype=float))
        if n_plot == 0:
            continue
        y, _ = _chance_correct_curve(y, None, null_ch)
        if not np.any(np.isfinite(y)):
            continue
        color = DECODE_FEATURE_COLORS.get(feat, "#888888")
        ax.plot(
            k_x[:n_plot], y, color=color, linewidth=lw,
            marker="o", markersize=ms,
            label=feature_display_name(feat) if show_legend else None,
        )
        full_acc = feat_data.get("full_hidden")
        if full_acc is not None and np.isfinite(full_acc):
            corr_full = chance_corrected(float(full_acc), null_ch)
            if np.isfinite(corr_full):
                ax.scatter(
                    [full_x], [corr_full], color=color, s=28 if compact else 55, marker="*",
                    zorder=5, edgecolors="white", linewidths=0.35,
                )

    ax.axhline(0.0, color="0.35", linestyle=":", linewidth=0.7, alpha=0.8)
    ax.set_ylim(ylim if ylim is not None else (-0.05, 1.05))
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    if show_legend:
        ax.legend(fontsize=6 if compact else 7, loc="lower left", framealpha=0.9)


def _plot_decode_neurons_on_ax(
    ax,
    panel: dict[str, Any],
    *,
    max_k: int,
    features: tuple[str, ...] = DECODING_FEATURES,
    show_legend: bool = False,
    compact: bool = False,
    ylim: tuple[float, float] | None = None,
) -> None:
    full_x = float(max_k) + 1.5
    k_x = np.arange(1, max_k + 1, dtype=float)
    lw = 1.2 if compact else 1.8
    ms = 2.5 if compact else 4

    for feat in features:
        feat_data = panel.get("features", {}).get(feat, {})
        by_k = feat_data.get("by_k_neurons") or []
        if not by_k:
            continue
        null_ch = _null_chance(panel, feat)
        if null_ch is None:
            continue
        y, n_plot = _trim_curve(np.asarray(by_k, dtype=float))
        if n_plot == 0:
            continue
        y_std_raw = feat_data.get("by_k_neurons_std")
        y_std = None
        if y_std_raw:
            y_std, _ = _trim_curve(np.asarray(y_std_raw, dtype=float))
            y_std = y_std[:n_plot]
        y, y_std = _chance_correct_curve(y, y_std, null_ch)
        if not np.any(np.isfinite(y)):
            continue
        color = DECODE_FEATURE_COLORS.get(feat, "#888888")
        x_plot = k_x[:n_plot]
        ax.plot(
            x_plot, y, color=color, linewidth=lw,
            marker="o", markersize=ms,
            label=feature_display_name(feat) if show_legend else None,
        )
        if y_std is not None and np.any(np.isfinite(y_std)):
            ax.fill_between(
                x_plot, y - y_std, y + y_std,
                color=color, alpha=0.16, linewidth=0,
            )
        full_acc = feat_data.get("full_hidden_neurons")
        if full_acc is not None and np.isfinite(full_acc):
            corr_full = chance_corrected(float(full_acc), null_ch)
            if np.isfinite(corr_full):
                ax.scatter(
                    [full_x], [corr_full], color=color, s=28 if compact else 55, marker="*",
                    zorder=5, edgecolors="white", linewidths=0.35,
                )

    ax.axhline(0.0, color="0.35", linestyle=":", linewidth=0.7, alpha=0.8)
    ax.set_ylim(ylim if ylim is not None else (-0.05, 1.05))
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    if show_legend:
        ax.legend(fontsize=6 if compact else 7, loc="lower left", framealpha=0.9)


def _add_trajectory_inset(
    ax,
    task: str,
    seed: int,
    *,
    model_type: str = "rnn",
    embed_method: str = "pca",
    average_trials: int = 8,
) -> None:
    ctx = load_task_viz_context(task, model_type=model_type, seed=seed)
    inset = inset_axes(
        ax, width="42%", height="42%", loc="lower right",
        borderpad=0.45,
    )
    inset.set_facecolor("white")
    inset.patch.set_alpha(0.92)
    for spine in inset.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("0.45")
    _plot_task_closed_loop_panel(
        inset, ctx,
        is_3d=False,
        rollout_seed=seed,
        embed_method=embed_method,
        average_trials=average_trials,
        minimal_axes=True,
    )
    inset.set_title("closed loop", fontsize=5.5, pad=1.5)


def plot_task_decode_curves(
    panel: dict[str, Any],
    save_path: str | Path,
    *,
    max_k: int = _DEFAULT_MAX_PCS,
    features: tuple[str, ...] = DECODING_FEATURES,
    n_neuron_trials: int = _DEFAULT_NEURON_RANDOM_TRIALS,
) -> Path:
    """Chance-corrected PCA vs random neuron subsets (mean ± std) per feature."""
    save_path = Path(save_path)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    finalize_grid_figure(
        fig,
        suptitle="Linear readout: chance-corrected decoding accuracy",
        top=0.90,
        wspace=0.22,
    )

    _plot_decode_pca_on_ax(axes[0], panel, max_k=max_k, features=features, show_legend=True)
    axes[0].set_xlabel("# PCs  |  star = full hidden", fontsize=9)
    axes[0].set_title("top-k PCA", fontsize=10)

    _plot_decode_neurons_on_ax(axes[1], panel, max_k=max_k, features=features, show_legend=True)
    axes[1].set_xlabel(
        f"# neurons  |  star = full hidden  |  {n_neuron_trials} random draws",
        fontsize=9,
    )
    axes[1].set_title(f"random neuron subsets (mean ± std, n={n_neuron_trials})", fontsize=10)

    axes[0].set_ylabel("(accuracy − chance) / (1 − chance)", fontsize=9)
    for ax in axes:
        _apply_k_ticks(ax, max_k, fontsize=8, show_labels=True)

    save_figure(fig, save_path)
    print(f"wrote {save_path}")
    return save_path


def plot_multi_seed_decode_curves(
    panels: dict[int, dict[str, Any]],
    save_path: str | Path,
    *,
    task: str,
    max_k: int = _DEFAULT_MAX_PCS,
    features: tuple[str, ...] = DECODING_FEATURES,
    n_neuron_trials: int = _DEFAULT_NEURON_RANDOM_TRIALS,
    model_type: str = "rnn",
) -> Path:
    """Grid: columns = seeds; rows = PCA / random neurons; closed-loop inset per column."""
    save_path = Path(save_path)
    seeds = tuple(sorted(panels))
    n_cols = len(seeds)
    if n_cols == 0:
        raise ValueError("need at least one seed panel")

    fig_w = max(2.05 * n_cols, 8.0)
    fig, axes = plt.subplots(2, n_cols, figsize=(fig_w, 4.9), squeeze=False)

    pca_mins, pca_maxs, neu_mins, neu_maxs = [], [], [], []
    for panel in panels.values():
        lo, hi = _panel_y_range(panel, max_k=max_k, features=features, basis="pca")
        pca_mins.append(lo)
        pca_maxs.append(hi)
        lo, hi = _panel_y_range(panel, max_k=max_k, features=features, basis="neuron")
        neu_mins.append(lo)
        neu_maxs.append(hi)
    pca_ylim = (min(pca_mins), max(pca_maxs))
    neu_ylim = (min(neu_mins), max(neu_maxs))

    for row in range(2):
        for col in range(1, n_cols):
            axes[row, col].sharey(axes[row, 0])

    legend_handles = None
    legend_labels = None

    for col, seed in enumerate(seeds):
        panel = panels[seed]
        ax_pca = axes[0, col]
        ax_neu = axes[1, col]

        _plot_decode_pca_on_ax(
            ax_pca, panel, max_k=max_k, features=features,
            show_legend=(col == 0), compact=True, ylim=pca_ylim,
        )
        if col == 0:
            legend_handles, legend_labels = ax_pca.get_legend_handles_labels()
            if ax_pca.get_legend() is not None:
                ax_pca.get_legend().remove()

        _plot_decode_neurons_on_ax(
            ax_neu, panel, max_k=max_k, features=features,
            show_legend=False, compact=True, ylim=neu_ylim,
        )
        _add_trajectory_inset(ax_neu, task, seed, model_type=model_type)

        ax_pca.set_title(f"seed {seed}", fontsize=8, fontweight="medium", pad=3)
        _apply_k_ticks(ax_pca, max_k, fontsize=6, show_labels=False)
        _apply_k_ticks(ax_neu, max_k, fontsize=6, show_labels=(col == n_cols // 2))

        if col == 0:
            ax_pca.set_ylabel("top-k PCA", fontsize=7)
            ax_neu.set_ylabel(f"random neurons ±std (n={n_neuron_trials})", fontsize=7)

    axes[1, n_cols // 2].set_xlabel("# dims  |  ★ = full hidden", fontsize=7)
    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc="upper center", bbox_to_anchor=(0.5, 1.01),
            ncol=4, fontsize=6.5, frameon=False, columnspacing=0.9,
        )
    fig.text(
        0.02, 0.5, "(accuracy − chance) / (1 − chance)",
        va="center", ha="left", rotation=90, fontsize=8,
    )
    finalize_grid_figure(
        fig,
        suptitle="Linear readout decoding across seeds (closed-loop trajectory inset)",
        suptitle_fontsize=11,
        top=0.90,
        bottom=0.10,
        hspace=0.32,
        wspace=0.18,
    )
    save_figure(fig, save_path, dpi=150)
    print(f"wrote {save_path}")
    return save_path


def run_task_decoding_analysis(
    task: str,
    out_dir: str | Path,
    *,
    model_type: str = "rnn",
    seed: int | None = None,
    max_k: int = _DEFAULT_MAX_PCS,
    n_neuron_trials: int = _DEFAULT_NEURON_RANDOM_TRIALS,
    neuron_rng_seed: int = 0,
) -> dict[str, Any]:
    """Compute and plot decoding curves for one checkpoint."""
    ctx = load_task_decoding_context(task, model_type=model_type, seed=seed)
    panel = compute_panel_decoding(
        ctx,
        max_k=max_k,
        neuron_sampling="random",
        n_random_trials=n_neuron_trials,
        neuron_rng_seed=neuron_rng_seed,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = numbered_plot_path(out_dir, "decoding_curves.png")
    json_path = png_path.with_name("decoding_curves.json")
    json_path.write_text(json.dumps(panel, indent=2), encoding="utf-8")
    print(f"wrote {json_path}")

    plot_task_decode_curves(
        panel,
        png_path,
        max_k=max_k,
        n_neuron_trials=n_neuron_trials,
    )
    return panel


def run_multi_seed_decoding_analysis(
    task: str,
    out_dir: str | Path,
    *,
    seeds: tuple[int, ...] | None = None,
    model_type: str = "rnn",
    max_k: int = _DEFAULT_MAX_PCS,
    n_neuron_trials: int = _DEFAULT_NEURON_RANDOM_TRIALS,
    neuron_rng_seed: int = 0,
) -> dict[str, Any]:
    """Decode and plot all seeds for one task."""
    run_seeds = seeds if seeds is not None else common_seeds((task,), model_type)
    if not run_seeds:
        raise RuntimeError(f"no checkpoints found for {task!r}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panels: dict[int, dict[str, Any]] = {}
    for seed in run_seeds:
        print(f"decoding seed {seed}...", flush=True)
        ctx = load_task_decoding_context(task, model_type=model_type, seed=seed)
        panels[seed] = compute_panel_decoding(
            ctx,
            max_k=max_k,
            neuron_sampling="random",
            n_random_trials=n_neuron_trials,
            neuron_rng_seed=neuron_rng_seed,
        )

    bundle = {"task": task, "seeds": list(run_seeds), "panels": panels}
    png_path = numbered_plot_path(out_dir, "decoding_curves_by_seed.png")
    json_path = png_path.with_name("decoding_curves_by_seed.json")
    json_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"wrote {json_path}")

    plot_multi_seed_decode_curves(
        panels,
        png_path,
        task=task,
        max_k=max_k,
        n_neuron_trials=n_neuron_trials,
        model_type=model_type,
    )
    return bundle
