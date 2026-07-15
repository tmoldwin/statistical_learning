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


def _full_marker_x(max_k: int) -> float:
    """X position for the ★ = full-hidden marker (just past the k curve)."""
    return float(max_k) + 3.0


def _staggered_full_xs(max_k: int, n_features: int) -> np.ndarray:
    """Spread ★ markers so near-ceiling full-hidden scores stay readable."""
    base = _full_marker_x(max_k)
    if n_features <= 1:
        return np.asarray([base], dtype=float)
    # One feature-width (~1.0) between adjacent ★ so char/DFA at ~1.0 don't cover.
    span = 1.05 * (n_features - 1)
    return base + np.linspace(-span / 2, span / 2, n_features)


def _apply_k_ticks(ax, max_k: int, *, fontsize: float = 10, show_labels: bool = True) -> None:
    # Keep ★/"full" clear of the last numeric k without a huge empty gap.
    full_x = _full_marker_x(max_k)
    n_feat_pad = 4  # room for staggered ★ fan
    xticks = [t for t in (1, 5, 10, 15, 20) if t <= max_k]
    if max_k not in xticks:
        xticks.append(max_k)
    ax.set_xlim(0.6, full_x + 0.55 * n_feat_pad + 0.8)
    ax.set_xticks([*xticks, full_x])
    if show_labels:
        ax.set_xticklabels(
            [*(str(t) for t in xticks), "full"],
            fontsize=max(8, fontsize - 1),
        )
        ax.tick_params(axis="x", length=4, width=0.9, pad=3)
    else:
        hide_x_tick_labels(ax)


def _plot_curve_to_full(
    ax,
    *,
    xs: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray | None,
    full_x: float,
    full_y: float,
    color: str,
    label: str | None,
    lw: float = 1.8,
) -> None:
    """Plot finite mean±std, then a dashed bridge to the full-hidden ★."""
    finite = np.isfinite(y_mean)
    if not np.any(finite):
        if np.isfinite(full_y):
            ax.scatter(
                [full_x], [full_y], color=color, marker="*", s=70,
                zorder=5, edgecolors="white", linewidths=0.4,
            )
        return
    x_plot = xs[finite]
    y_plot = y_mean[finite]
    ax.plot(x_plot, y_plot, color=color, lw=lw, label=label)
    if y_std is not None:
        s_plot = np.asarray(y_std, dtype=float)[finite]
        ax.fill_between(
            x_plot, y_plot - s_plot, y_plot + s_plot, color=color, alpha=0.18,
        )
    if np.isfinite(full_y):
        ax.plot(
            [float(x_plot[-1]), full_x], [float(y_plot[-1]), full_y],
            color=color, lw=lw, ls="-", alpha=0.9,
        )
        ax.scatter(
            [full_x], [full_y], color=color, marker="*", s=90,
            zorder=5, edgecolors="0.15", linewidths=0.55,
        )


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
    full_x = _full_marker_x(max_k)
    k_x = np.arange(1, max_k + 1, dtype=float)
    lw = 1.2 if compact else 1.8
    ms = 2.5 if compact else 4
    full_xs = _staggered_full_xs(max_k, len(features))

    for fi, feat in enumerate(features):
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
                fx = float(full_xs[fi])
                ax.plot(
                    [float(k_x[n_plot - 1]), fx], [float(y[-1]), corr_full],
                    color=color, lw=0.9, ls="--", alpha=0.7,
                )
                ax.scatter(
                    [fx], [corr_full], color=color, s=28 if compact else 55, marker="*",
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
    k_x = np.arange(1, max_k + 1, dtype=float)
    lw = 1.2 if compact else 1.8
    ms = 2.5 if compact else 4
    full_xs = _staggered_full_xs(max_k, len(features))

    for fi, feat in enumerate(features):
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
                fx = float(full_xs[fi])
                ax.plot(
                    [float(x_plot[-1]), fx], [float(y[-1]), corr_full],
                    color=color, lw=0.9, ls="--", alpha=0.7,
                )
                ax.scatter(
                    [fx], [corr_full], color=color, s=28 if compact else 55, marker="*",
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
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), sharey=True)
    finalize_grid_figure(
        fig,
        suptitle="Linear readout: chance-corrected decoding accuracy",
        top=0.86,
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


def _load_task_si_arrays(
    task: str,
    *,
    model_type: str = "rnn",
    features: tuple[str, ...] = DECODING_FEATURES,
) -> dict[str, np.ndarray] | None:
    """Per-unit SI from unit-selectivity summary JSON (default checkpoint)."""
    from experiment import plots_dir

    path = plots_dir(task, model_type) / "unit_selectivity" / "selectivity_summary.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    si = payload.get("si") or {}
    out: dict[str, np.ndarray] = {}
    for feat in features:
        vals = si.get(feat)
        if vals is None:
            continue
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            out[feat] = arr
    return out or None


def _pooled_si_arrays_across_seeds(
    task: str,
    seeds: tuple[int, ...],
    *,
    model_type: str = "rnn",
    features: tuple[str, ...] = DECODING_FEATURES,
) -> dict[str, np.ndarray] | None:
    """Concatenate per-unit SI across training seeds (one population per checkpoint)."""
    from unit_selectivity import build_timestep_labels, compute_selectivity
    from visualize import condense_hidden_states_by_prefix
    from vocab_diagrams import build_minimized_vocabulary_automaton

    buckets: dict[str, list[np.ndarray]] = {feat: [] for feat in features}
    for seed in seeds:
        try:
            ctx = load_task_viz_context(task, model_type=model_type, seed=seed)
        except Exception as exc:  # noqa: BLE001
            print(f"skip SI seed {seed}: {exc.__class__.__name__}")
            continue
        automaton = build_minimized_vocabulary_automaton(ctx.words)
        condensed = condense_hidden_states_by_prefix(
            ctx.text, ctx.hidden_states, spaced=ctx.spaced, words=ctx.words,
        )
        if condensed.hidden_states.shape[0] < 2:
            continue
        labels = build_timestep_labels(
            ctx.text, automaton,
            spaced=ctx.spaced, words=ctx.words, condensed=condensed,
            model=ctx.model, activations=condensed.hidden_states,
        )
        result = compute_selectivity(
            condensed.hidden_states, labels, ctx.model, ctx.text,
        )
        for feat in features:
            vals = result.si.get(feat)
            if vals is None:
                continue
            arr = np.asarray(vals, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                buckets[feat].append(arr)

    out: dict[str, np.ndarray] = {}
    for feat, parts in buckets.items():
        if parts:
            out[feat] = np.concatenate(parts)
    return out or None


def plot_aggregated_seed_decode_curves(
    panels: dict[int, dict[str, Any]],
    save_path: str | Path,
    *,
    task: str,
    max_k: int = _DEFAULT_MAX_PCS,
    features: tuple[str, ...] = DECODING_FEATURES,
    n_neuron_trials: int = _DEFAULT_NEURON_RANDOM_TRIALS,
    model_type: str = "rnn",
) -> Path:
    """Mean ± std decoding curves across seeds, plus pooled per-unit SI density."""
    save_path = Path(save_path)
    seeds = tuple(sorted(panels))
    if not seeds:
        raise ValueError("need at least one seed panel")

    si_by_feat = _pooled_si_arrays_across_seeds(
        task, seeds, model_type=model_type, features=features,
    )
    if si_by_feat is None:
        si_by_feat = _load_task_si_arrays(task, model_type=model_type, features=features)
    n_cols = 3 if si_by_feat else 2
    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(12.6 if si_by_feat else 9.6, 3.6),
        sharey=False,
    )
    axes = np.atleast_1d(axes)
    full_xs = _staggered_full_xs(max_k, len(features))
    xs = np.arange(1, max_k + 1, dtype=float)

    for fi, feat in enumerate(features):
        color = DECODE_FEATURE_COLORS.get(feat, "#333333")
        label = feature_display_name(feat)

        # PCA curves
        pca_rows = []
        full_pca = []
        for seed in seeds:
            feat_data = panels[seed]["features"][feat]
            null_ch = _null_chance(panels[seed], feat) or 0.0
            by_k, n = _trim_curve(np.asarray(feat_data["by_k"], dtype=float))
            y, _ = _chance_correct_curve(by_k, None, null_ch)
            row = np.full(max_k, np.nan)
            row[: min(n, max_k)] = y[: min(n, max_k)]
            pca_rows.append(row)
            full_pca.append(chance_corrected(float(feat_data["full_hidden"]), null_ch))
        pca_mat = np.vstack(pca_rows)
        pca_mean = np.nanmean(pca_mat, axis=0)
        pca_std = np.nanstd(pca_mat, axis=0)
        _plot_curve_to_full(
            axes[0],
            xs=xs, y_mean=pca_mean, y_std=pca_std,
            full_x=float(full_xs[fi]),
            full_y=float(np.nanmean(full_pca)),
            color=color, label=label,
        )

        # Neuron curves
        neu_rows = []
        full_neu = []
        for seed in seeds:
            feat_data = panels[seed]["features"][feat]
            null_ch = _null_chance(panels[seed], feat) or 0.0
            by_k, n = _trim_curve(np.asarray(feat_data["by_k_neurons"], dtype=float))
            y, _ = _chance_correct_curve(by_k, None, null_ch)
            row = np.full(max_k, np.nan)
            row[: min(n, max_k)] = y[: min(n, max_k)]
            neu_rows.append(row)
            full_neu.append(chance_corrected(float(feat_data["full_hidden_neurons"]), null_ch))
        neu_mat = np.vstack(neu_rows)
        neu_mean = np.nanmean(neu_mat, axis=0)
        neu_std = np.nanstd(neu_mat, axis=0)
        _plot_curve_to_full(
            axes[1],
            xs=xs, y_mean=neu_mean, y_std=neu_std,
            full_x=float(full_xs[fi]),
            full_y=float(np.nanmean(full_neu)),
            color=color, label=label,
        )

    axes[0].set_title("top-k PCA", fontsize=10)
    axes[0].set_xlabel("# PCs  |  ★ = full hidden", fontsize=9)
    axes[0].set_ylabel("(accuracy − chance) / (1 − chance)", fontsize=9)
    axes[1].set_title("random neurons", fontsize=10)
    axes[1].set_xlabel(f"# neurons  |  ★ = full  |  {n_neuron_trials} draws/seed", fontsize=9)
    for ax in axes[:2]:
        ax.set_ylim(-0.05, 1.12)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.tick_params(axis="y", which="both", labelleft=True, labelsize=8)
        _apply_k_ticks(ax, max_k, fontsize=11, show_labels=True)
    axes[0].set_ylabel("(accuracy − chance) / (1 − chance)", fontsize=9)
    axes[1].set_ylabel("(accuracy − chance) / (1 − chance)", fontsize=9)

    if si_by_feat is not None:
        from unit_selectivity import plot_selectivity_si_on_ax

        ax_si = axes[2]
        plot_selectivity_si_on_ax(
            ax_si, si_by_feat, features=features, show_legend=False,
        )
        ax_si.set_title(f"unit SI density · {len(seeds)} seeds (SI > 0)", fontsize=10)
        ax_si.set_xlabel("SI (peak vs rest)", fontsize=9)
        # Keep density curves fully inside the axes (no clipped peaks).
        y_top = 0.0
        for line in ax_si.lines:
            yd = np.asarray(line.get_ydata(), dtype=float)
            if yd.size:
                y_top = max(y_top, float(np.nanmax(yd)))
        if y_top > 0:
            ax_si.set_ylim(0.0, y_top * 1.12)

    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc="upper center", bbox_to_anchor=(0.5, 0.98),
            ncol=4, fontsize=8, frameon=False, columnspacing=1.2,
        )
    finalize_grid_figure(
        fig,
        suptitle=f"Linear readout decoding · mean ± std across {len(seeds)} seeds",
        top=0.80,
        bottom=0.18,
        left=0.08,
        right=0.99,
        wspace=0.32 if si_by_feat else 0.18,
    )
    # finalize_grid_figure / tight packing can hide y labels on non-left panels.
    for ax in axes[:2]:
        ax.tick_params(axis="y", which="both", labelleft=True, labelsize=8)
    save_figure(fig, save_path, dpi=150)
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
    agg_path = png_path.with_name("decoding_curves_seed_mean.png")
    plot_aggregated_seed_decode_curves(
        panels,
        agg_path,
        task=task,
        max_k=max_k,
        n_neuron_trials=n_neuron_trials,
        model_type=model_type,
    )
    return bundle