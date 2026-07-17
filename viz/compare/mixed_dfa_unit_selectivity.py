"""Pooled single-unit selectivity across mixed-vocab DFA sweep.

Goal:
1) Compute per-unit SI (peak-vs-rest selectivity index) for each run/seed.
2) Pool SI into DFA-size difficulty bins (10, 20, 30, ...).
3) Plot:
   - SI histograms conditioned on DFA difficulty.
   - Mean SI vs DFA size with regression per feature.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from experiment import comparison_dir
from unit_selectivity import (
    FEATURE_COLORS,
    FEATURE_DISPLAY,
    compute_unit_selectivity_matrix,
    build_timestep_labels,
)
from vocab_diagrams import build_minimized_vocabulary_automaton
from vocab_mixed_dfa import COMPARISON_NAME, iter_runs
from visualize import condense_hidden_states_by_prefix
from viz.compare._data import load_task_viz_context
from viz.compare.decoding import DECODE_FEATURE_COLORS
from viz.plot_layout import finalize_grid_figure, save_figure


DEFAULT_MODEL_TYPE = "rnn"
DEFAULT_FEATURES: tuple[str, ...] = ("dfa", "char", "position", "position_from_end", "word")
# Tiny automata (DFA < 10) are a separate regime — near-zero SI, not comparable.
MIN_DFA_FOR_ANALYSIS = 10


@dataclass(frozen=True)
class DifficultyBin:
    lo: int | None
    hi: int | None
    label: str

    def contains(self, x: int) -> bool:
        if self.lo is not None and x < self.lo:
            return False
        if self.hi is not None and x >= self.hi:
            return False
        return True


def _default_difficulty_bins() -> list[DifficultyBin]:
    """DFA bins from 10 upward (exclude <10 outlier regime)."""
    edges = [10, 20, 30, 40, 10**9]
    out: list[DifficultyBin] = []
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if hi >= 10**9:
            label = f"{lo}+ states"
            out.append(DifficultyBin(lo=lo, hi=None, label=label))
        else:
            label = f"{lo}-{hi-1} states"
            out.append(DifficultyBin(lo=lo, hi=hi, label=label))
    return out


def _compute_si_for_ctx(
    ctx,
    *,
    features: tuple[str, ...] = DEFAULT_FEATURES,
) -> tuple[int, dict[str, np.ndarray]]:
    """Return (n_dfa_states, si_per_feature) for this task/seed."""
    automaton = build_minimized_vocabulary_automaton(ctx.words)
    n_dfa = int(automaton.dfa._n)

    condensed = condense_hidden_states_by_prefix(
        ctx.text,
        ctx.hidden_states,
        output_probs=None,
        spaced=ctx.spaced,
        words=ctx.words,
    )
    activations = condensed.hidden_states

    # Build labels without model/activations so pred-entropy bookkeeping is skipped.
    labels = build_timestep_labels(
        ctx.text,
        automaton,
        spaced=ctx.spaced,
        words=ctx.words,
        condensed=condensed,
        model=None,
        activations=None,
    )

    si, _, _, _gap, _peak_label = compute_unit_selectivity_matrix(
        activations,
        labels,
        features=features,
    )
    return n_dfa, si


def collect_mixed_dfa_unit_selectivity(
    *,
    model_type: str = DEFAULT_MODEL_TYPE,
    features: tuple[str, ...] = DEFAULT_FEATURES,
    bins: list[DifficultyBin] | None = None,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    """Compute and pool SI across all mixed DFA sweep runs and available seeds."""
    if bins is None:
        bins = _default_difficulty_bins()

    pooled_si: dict[str, dict[str, list[float]]] = {
        b.label: {f: [] for f in features} for b in bins
    }
    # Regression input: one scalar per (task, seed).
    per_run: list[dict[str, Any]] = []

    tasks_seen = 0
    for entry in iter_runs():
        task = entry["task"]
        words = list(entry["words"])
        if max_tasks is not None and tasks_seen >= max_tasks:
            break

        from experiment import seeds_for_task

        seeds = sorted(seeds_for_task(task, model_type))
        if not seeds:
            continue
        tasks_seen += 1

        for seed in seeds:
            print(f"selectivity {task} seed {seed} ...", flush=True)
            ctx = load_task_viz_context(task, model_type=model_type, seed=seed)
            n_dfa, si = _compute_si_for_ctx(ctx, features=features)

            mean_si = {
                f: float(np.nanmean(si[f])) if si.get(f) is not None else float("nan")
                for f in features
            }
            per_run.append({
                "task": task,
                "seed": seed,
                "n_dfa_states": int(n_dfa),
                "n_words": int(entry["n_words"]),
                "words": words,
                "mean_si": mean_si,
            })

            for b in bins:
                if not b.contains(int(n_dfa)):
                    continue
                for f in features:
                    vals = np.asarray(si[f], dtype=float)
                    vals = vals[np.isfinite(vals)]
                    pooled_si[b.label][f].extend(vals.tolist())

    return {
        "comparison": COMPARISON_NAME,
        "model_type": model_type,
        "features": list(features),
        "bins": [b.__dict__ for b in bins],
        "pooled_si": pooled_si,
        "per_run": per_run,
    }


def plot_mixed_dfa_si_histograms(
    pooled_si: dict[str, dict[str, list[float]]],
    *,
    features: tuple[str, ...] = DEFAULT_FEATURES,
    bins: list[DifficultyBin] | None = None,
    min_si: float = 1e-6,
    outfile: str | None = None,
) -> plt.Figure:
    """One row of histogram panels, one panel per DFA difficulty bin (DFA ≥ 10)."""
    if bins is None:
        bins = _default_difficulty_bins()
    if outfile is None:
        outfile = "mixed_dfa_si_histograms.png"

    n_bins = len(bins)
    fig, axes = plt.subplots(1, n_bins, figsize=(3.55 * n_bins, 3.6), sharey=True)
    axes = np.atleast_1d(axes)

    xs = np.linspace(0.0, 1.0, 31)
    for ax, b in zip(axes, bins):
        for f in features:
            vals = np.asarray(pooled_si.get(b.label, {}).get(f, []), dtype=float)
            vals = vals[np.isfinite(vals) & (vals > min_si)]
            if vals.size == 0:
                continue
            ax.hist(
                vals,
                bins=xs,
                density=True,
                histtype="step",
                lw=1.7,
                color=FEATURE_COLORS.get(f, "#888"),
                alpha=0.95,
                label=FEATURE_DISPLAY.get(f, f) if ax is axes[0] else None,
            )
        ax.set_title(b.label, fontsize=8)
        ax.set_xlim(0.0, 1.0)
        ax.axvline(0.0, color="0.35", lw=0.7, ls=":")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("SI (peak vs rest)", fontsize=8)

    axes[0].set_ylabel("density (SI > 0)", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
            frameon=False,
            fontsize=8,
        )
    finalize_grid_figure(
        fig,
        suptitle=(
            f"Single-unit selectivity by DFA size "
            f"(DFA ≥ {MIN_DFA_FOR_ANALYSIS}; SI > 0 only)"
        ),
        top=0.84,
        bottom=0.16,
        left=0.08,
        right=0.98,
        hspace=0.35,
        wspace=0.22,
    )
    return fig


def _linear_fit_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return slope, intercept, R²."""
    if x.size < 3:
        return float("nan"), float("nan"), float("nan")
    b, a = np.polyfit(x, y, 1)
    yhat = a + b * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return float(b), float(a), float(r2)


def plot_mixed_dfa_si_regression(
    per_run: list[dict[str, Any]],
    *,
    features: tuple[str, ...] = DEFAULT_FEATURES,
    min_dfa: int = MIN_DFA_FOR_ANALYSIS,
    outfile: str | None = None,
) -> plt.Figure:
    """Mean SI per run vs DFA size (DFA ≥ min_dfa); flat trend expected."""
    if outfile is None:
        outfile = "mixed_dfa_si_regression.png"

    xs_all = np.asarray([r["n_dfa_states"] for r in per_run], dtype=float)
    n_feat = len(features)
    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(10.8, 6.2),
        squeeze=False,
    )
    axes_flat = axes.ravel().tolist()

    for i, feat in enumerate(features):
        ax = axes_flat[i]
        ys_all = np.asarray(
            [r["mean_si"].get(feat, float("nan")) for r in per_run], dtype=float,
        )
        mask_main = np.isfinite(xs_all) & np.isfinite(ys_all) & (xs_all >= min_dfa)
        mask_out = np.isfinite(xs_all) & np.isfinite(ys_all) & (xs_all < min_dfa)

        xs2 = xs_all[mask_main]
        ys2 = ys_all[mask_main]
        if xs2.size < 3:
            ax.set_axis_off()
            continue

        b, a, r2 = _linear_fit_r2(xs2, ys2)
        mean_y = float(np.mean(ys2))

        if mask_out.any():
            ax.scatter(
                xs_all[mask_out], ys_all[mask_out],
                s=22, alpha=0.45, facecolors="none",
                edgecolors="0.55", linewidths=0.8, zorder=2,
                label="DFA < 10 (excluded)" if i == 0 else None,
            )
        ax.scatter(
            xs2, ys2, s=18, alpha=0.55,
            color=FEATURE_COLORS.get(feat, "#888"),
            edgecolors="white", linewidths=0.25, zorder=3,
        )
        xgrid = np.linspace(float(np.min(xs2)), float(np.max(xs2)), 64)
        ax.plot(
            xgrid, a + b * xgrid,
            color=FEATURE_COLORS.get(feat, "#888"), lw=1.5, alpha=0.85, zorder=4,
        )
        ax.axhline(
            mean_y, color=FEATURE_COLORS.get(feat, "#888"),
            lw=1.0, ls="--", alpha=0.55, zorder=1,
        )

        ax.set_title(
            f"{FEATURE_DISPLAY.get(feat, feat)}\n"
            f"slope={b:+.4f}  R²={r2:.2f}  mean={mean_y:.2f}",
            fontsize=8.5,
            pad=4,
        )
        ax.set_xlabel("minimized DFA states", fontsize=8)
        ax.set_ylabel("mean SI across units", fontsize=8)
        ax.set_xlim(min_dfa - 1, float(np.max(xs_all)) + 1)
        y_lo = max(0.0, float(np.min(ys2)) - 0.04)
        y_hi = min(1.02, float(np.max(ys2)) + 0.06)
        ax.set_ylim(y_lo, y_hi)
        ax.axhline(0.0, color="0.35", lw=0.7, ls=":")
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=8)

    for j in range(n_feat, len(axes_flat)):
        axes_flat[j].set_axis_off()

    if axes_flat[0].get_legend_handles_labels()[0]:
        axes_flat[0].legend(fontsize=7, loc="lower right", framealpha=0.9)

    finalize_grid_figure(
        fig,
        suptitle=(
            f"Mean unit selectivity vs DFA size "
            f"(regression on DFA ≥ {min_dfa}; open circles = excluded)"
        ),
        top=0.88,
        bottom=0.10,
        left=0.10,
        right=0.98,
        hspace=0.42,
        wspace=0.24,
    )
    return fig


def run_mixed_dfa_unit_selectivity_analysis(
    *,
    model_type: str = DEFAULT_MODEL_TYPE,
    features: tuple[str, ...] = DEFAULT_FEATURES,
    recompute: bool = False,
    max_tasks: int | None = None,
    out_json: str = "mixed_dfa_unit_selectivity_pooled.json",
    out_hist: str = "mixed_dfa_unit_selectivity_histograms.png",
    out_reg: str = "mixed_dfa_unit_selectivity_regression.png",
) -> tuple[Path, Path, Path, Path]:
    """Compute pooled SI and write histogram/regression figures."""
    out_dir = comparison_dir(COMPARISON_NAME, "unit_selectivity")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json_path = out_dir / out_json
    out_hist_path = out_dir / out_hist
    out_reg_path = out_dir / out_reg

    if out_json_path.is_file() and not recompute:
        payload = json.loads(out_json_path.read_text(encoding="utf-8"))
    else:
        payload = collect_mixed_dfa_unit_selectivity(
            model_type=model_type,
            features=features,
            max_tasks=max_tasks,
        )
        out_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    bins = _default_difficulty_bins()
    old_pooled = payload.get("pooled_si", {})
    pooled_si = {
        b.label: {
            f: list(old_pooled.get(b.label, {}).get(f, []))
            for f in features
        }
        for b in bins
    }

    fig = plot_mixed_dfa_si_histograms(
        pooled_si,
        features=features,
        bins=bins,
    )
    save_figure(fig, out_hist_path, dpi=150)
    plt.close(fig)

    fig = plot_mixed_dfa_si_regression(
        payload["per_run"],
        features=features,
    )
    save_figure(fig, out_reg_path, dpi=150)
    plt.close(fig)

    fig15_path = plot_mixed_fig15_geometry_and_selectivity(si_json=out_json_path)

    return out_json_path, out_hist_path, out_reg_path, fig15_path


def _pool_si_all_runs(
    pooled_si: dict[str, dict[str, list[float]]],
    *,
    features: tuple[str, ...] = DEFAULT_FEATURES,
    min_si: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Pool all unit SI values across DFA ≥ 10 bins."""
    out: dict[str, list[float]] = {f: [] for f in features}
    for _label, feat_dict in pooled_si.items():
        for f in features:
            vals = np.asarray(feat_dict.get(f, []), dtype=float)
            vals = vals[np.isfinite(vals) & (vals > min_si)]
            out[f].extend(vals.tolist())
    return {f: np.asarray(v, dtype=float) for f, v in out.items()}


def _plot_pooled_si_density_on_ax(
    ax,
    pooled: dict[str, np.ndarray],
    *,
    features: tuple[str, ...] = DEFAULT_FEATURES,
) -> None:
    """Overlaid SI densities (SI > 0), one curve per feature."""
    from scipy.stats import gaussian_kde

    xs = np.linspace(0.0, 1.0, 256)
    y_hi = 0.0
    for feat in features:
        vals = pooled.get(feat, np.asarray([], dtype=float))
        if vals.size < 2:
            continue
        color = DECODE_FEATURE_COLORS.get(feat, FEATURE_COLORS.get(feat, "#888"))
        try:
            kde = gaussian_kde(vals, bw_method=lambda k: max(k.scotts_factor() * 1.4, 0.07))
        except Exception:
            continue
        dens = np.clip(kde(xs), 0.0, None)
        ax.plot(xs, dens, color=color, lw=1.8, label=FEATURE_DISPLAY.get(feat, feat))
        y_hi = max(y_hi, float(np.max(dens)))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, max(y_hi * 1.12, 0.5))
    ax.set_xlabel("SI (peak vs rest)", fontsize=8)
    ax.set_ylabel("density (SI > 0)", fontsize=8)
    ax.set_title(f"Pooled units (DFA ≥ {MIN_DFA_FOR_ANALYSIS})", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(labelsize=7)


def _plot_mean_si_scatter_on_ax(
    ax,
    per_run: list[dict[str, Any]],
    *,
    features: tuple[str, ...] = DEFAULT_FEATURES,
    min_dfa: int = MIN_DFA_FOR_ANALYSIS,
) -> None:
    """All features: mean SI per run vs DFA size (DFA ≥ min_dfa)."""
    xs_all = np.asarray([r["n_dfa_states"] for r in per_run], dtype=float)
    for feat in features:
        ys = np.asarray(
            [r["mean_si"].get(feat, float("nan")) for r in per_run], dtype=float,
        )
        mask_main = np.isfinite(xs_all) & np.isfinite(ys) & (xs_all >= min_dfa)
        color = DECODE_FEATURE_COLORS.get(feat, FEATURE_COLORS.get(feat, "#888"))
        ax.scatter(
            xs_all[mask_main], ys[mask_main],
            s=16, alpha=0.55, color=color,
            edgecolors="white", linewidths=0.25,
            label=FEATURE_DISPLAY.get(feat, feat), zorder=3,
        )
        if mask_main.sum() >= 3:
            b, a, _r2 = _linear_fit_r2(xs_all[mask_main], ys[mask_main])
            xgrid = np.linspace(float(xs_all[mask_main].min()), float(xs_all[mask_main].max()), 32)
            ax.plot(xgrid, a + b * xgrid, color=color, lw=1.0, alpha=0.65, zorder=2)
    ax.axhline(0.0, color="0.35", lw=0.6, ls=":")
    ax.set_xlabel("minimized DFA states", fontsize=8)
    ax.set_ylabel("mean SI across units", fontsize=8)
    ax.set_title(f"Per-run mean (DFA ≥ {min_dfa})", fontsize=9)
    ax.set_xlim(min_dfa - 1, float(np.max(xs_all)) + 1)
    ax.set_ylim(0.28, 0.50)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=7)


def _plot_cosine_within_on_ax(
    ax,
    panels: list[dict[str, Any]],
    *,
    features: tuple[str, ...],
) -> None:
    """Within-feature cosine vs DFA (observed + shuffle)."""
    from matplotlib.lines import Line2D
    from viz.compare.pow2_sweep_metric_board import _fit_trend

    xs = np.asarray([float(p["n_dfa_states"]) for p in panels], dtype=float)
    for feat in features:
        color = DECODE_FEATURE_COLORS.get(feat, FEATURE_COLORS.get(feat, "#666"))
        y_obs = np.asarray(
            [float(p.get("within_cosine", {}).get(feat, float("nan"))) for p in panels],
            dtype=float,
        )
        y_sh = np.asarray(
            [float(p.get("shuffle_within_cosine", {}).get(feat, float("nan"))) for p in panels],
            dtype=float,
        )
        mask_o = np.isfinite(xs) & np.isfinite(y_obs)
        mask_s = np.isfinite(xs) & np.isfinite(y_sh)
        ax.scatter(
            xs[mask_o], y_obs[mask_o], s=20, color=color, marker="o", alpha=0.78,
            edgecolors="white", linewidths=0.3, zorder=3,
        )
        ax.scatter(
            xs[mask_s], y_sh[mask_s], s=18, facecolors="none", edgecolors=color,
            marker="o", alpha=0.5, linewidths=0.9, zorder=2,
        )
        x_fit_o, y_fit_o, _r2, _ = _fit_trend(xs[mask_o], y_obs[mask_o])
        x_fit_s, y_fit_s, _, _ = _fit_trend(xs[mask_s], y_sh[mask_s])
        if x_fit_o is not None:
            ax.plot(x_fit_o, y_fit_o, color=color, lw=1.3, zorder=4)
        if x_fit_s is not None:
            ax.plot(x_fit_s, y_fit_s, color=color, lw=1.0, ls="--", alpha=0.7, zorder=3)
    ax.set_xlabel("DFA states", fontsize=8)
    ax.set_ylabel("within-feature cosine", fontsize=8)
    ax.set_title("Condensed hidden states", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=7)
    ax.text(
        0.02, 0.02, "open/dashed = shuffle",
        transform=ax.transAxes, fontsize=6, color="0.45", va="bottom",
    )


def plot_mixed_fig15_geometry_and_selectivity(
    *,
    si_json: Path | str | None = None,
    within_json: Path | str | None = None,
    outfile: str | None = None,
    features: tuple[str, ...] = DEFAULT_FEATURES,
) -> Path:
    """Figure 15: cosine within + pooled unit SI + mean SI vs DFA (3 panels)."""
    from viz.compare.sweep_output import sweep_data_dir, sweep_decoding_dir

    si_path = Path(si_json or comparison_dir(COMPARISON_NAME, "unit_selectivity") / "mixed_dfa_unit_selectivity_pooled.json")
    within_path = Path(within_json or sweep_data_dir(COMPARISON_NAME) / "within_corr_vs_dfa.json")
    out = Path(outfile or sweep_decoding_dir(COMPARISON_NAME) / "fig15_geometry_and_selectivity.png")

    si_payload = json.loads(si_path.read_text(encoding="utf-8"))
    within_payload = json.loads(within_path.read_text(encoding="utf-8"))

    bins = _default_difficulty_bins()
    old_pooled = si_payload.get("pooled_si", {})
    pooled_si = {
        b.label: {f: list(old_pooled.get(b.label, {}).get(f, [])) for f in features}
        for b in bins
    }
    pooled_all = _pool_si_all_runs(pooled_si, features=features)
    within_panels = within_payload.get("panels", [])
    within_feats = tuple(within_payload.get("features", features[:4]))

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), squeeze=False)
    ax_cos, ax_den, ax_sc = axes[0]

    _plot_cosine_within_on_ax(ax_cos, within_panels, features=within_feats)
    _plot_pooled_si_density_on_ax(ax_den, pooled_all, features=features)
    _plot_mean_si_scatter_on_ax(ax_sc, si_payload.get("per_run", []), features=features)

    handles, labels = ax_sc.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="lower center", bbox_to_anchor=(0.5, 0.01),
            ncol=len(handles), fontsize=7, frameon=False, columnspacing=1.0,
        )

    finalize_grid_figure(
        fig,
        suptitle="Population geometry and unit selectivity vs DFA size (mixed vocabs)",
        top=0.86,
        bottom=0.18,
        left=0.07,
        right=0.98,
        wspace=0.32,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out

