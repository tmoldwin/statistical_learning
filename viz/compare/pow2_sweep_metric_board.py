"""Unified word-count x length metric heatmaps for a pow2 sweep."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from viz.compare.pow2_sweep_spec import POW2_SWEEP_SPEC_H100, Pow2SweepSpec
from viz.compare.sweep_output import sweep_data_dir
from viz.plot_layout import finalize_grid_figure, save_figure


def _norm_length(length: object) -> object:
    if isinstance(length, str) and length.isdigit():
        return int(length)
    return length


def _cell_key(n_words: object, length: object) -> tuple[int, object]:
    return int(n_words), _norm_length(length)


def _mean_grid(
    values_by_cell: dict[tuple[int, object], list[float]],
    *,
    word_counts: tuple[int, ...],
    lengths: tuple[object, ...],
) -> np.ndarray:
    mat = np.full((len(lengths), len(word_counts)), np.nan, dtype=float)
    for li, length in enumerate(lengths):
        for wi, n_words in enumerate(word_counts):
            vals = values_by_cell.get(_cell_key(n_words, length), [])
            if vals:
                mat[li, wi] = float(np.mean(vals))
    return mat


def _collect_from_seed_panels(
    panels: list[dict[str, Any]],
    *,
    seeds: set[int] | None,
    getter: Callable[[dict[str, Any]], float | None],
) -> dict[tuple[int, object], list[float]]:
    out: dict[tuple[int, object], list[float]] = {}
    for panel in panels:
        if "error" in panel:
            continue
        if seeds is not None and int(panel.get("seed", -1)) not in seeds:
            continue
        if "n_words" not in panel:
            continue
        val = getter(panel)
        if val is None or not np.isfinite(val):
            continue
        key = _cell_key(panel["n_words"], panel["length"])
        out.setdefault(key, []).append(float(val))
    return out


def _nested_get(panel: dict[str, Any], section: str, key: str) -> float | None:
    blob = panel.get(section)
    if not isinstance(blob, dict) or key not in blob:
        return None
    try:
        return float(blob[key])
    except (TypeError, ValueError):
        return None


def _weight_stage_maps(
    panels: list[dict[str, Any]],
    *,
    seeds: set[int] | None,
    family: str,
    key: str,
) -> tuple[dict[tuple[int, object], list[float]], dict[tuple[int, object], list[float]]]:
    init_map: dict[tuple[int, object], list[float]] = {}
    final_map: dict[tuple[int, object], list[float]] = {}
    for panel in panels:
        n_words = panel.get("n_words")
        length = panel.get("length")
        if n_words is None:
            continue
        cell = _cell_key(n_words, length)
        for row in panel.get("seeds", []):
            if "error" in row:
                continue
            seed = int(row.get("seed", -1))
            if seeds is not None and seed not in seeds:
                continue
            if family == "structure":
                init_blob, final_blob = row.get("init", {}), row.get("final", {})
            else:
                init_blob, final_blob = row.get("motif_init", {}), row.get("motif_final", {})
            if key in init_blob:
                try:
                    init_map.setdefault(cell, []).append(float(init_blob[key]))
                except (TypeError, ValueError):
                    pass
            if key in final_blob:
                try:
                    final_map.setdefault(cell, []).append(float(final_blob[key]))
                except (TypeError, ValueError):
                    pass
    return init_map, final_map


def _global_mean(values_by_cell: dict[tuple[int, object], list[float]]) -> float | None:
    vals = [v for lst in values_by_cell.values() for v in lst]
    if not vals:
        return None
    return float(np.mean(vals))


def _spectrum_metric_maps(
    panels: list[dict[str, Any]],
    *,
    metric: str,
) -> dict[tuple[int, object], list[float]]:
    out: dict[tuple[int, object], list[float]] = {}
    for panel in panels:
        spectrum = panel.get("spectrum_pct") or []
        if not spectrum:
            continue
        y = np.asarray(spectrum, dtype=float)
        if metric == "top1_pct":
            val = float(y[0]) if y.size else float("nan")
        elif metric == "top2_cum_pct":
            val = float(np.sum(y[:2])) if y.size else float("nan")
        elif metric == "dims_90pct":
            if y.size == 0:
                continue
            c = np.cumsum(y)
            idx = int(np.searchsorted(c, 90.0, side="left"))
            val = float(min(idx + 1, y.size))
        else:
            continue
        if not np.isfinite(val):
            continue
        key = _cell_key(panel["n_words"], panel["length"])
        # spectra panels are already mean-over-seeds; store as singleton list
        out.setdefault(key, []).append(val)
    return out


# (title, builder -> (final_map, init_map_or_None, cmap, log_scale, vmin, vmax))
MetricSpec = tuple[str, Callable[..., tuple[
    dict[tuple[int, object], list[float]],
    dict[tuple[int, object], list[float]] | None,
    str,
    bool,
    tuple[float, float] | None,
]]]


def _metric_catalog(
    *,
    geometry_panels: list[dict[str, Any]],
    training_panels: list[dict[str, Any]],
    spectra_panels: list[dict[str, Any]],
    weight_panels: list[dict[str, Any]],
    seeds: set[int] | None,
) -> list[tuple[
    str,
    dict[tuple[int, object], list[float]],
    dict[tuple[int, object], list[float]] | None,
    str,
    bool,
    tuple[float, float] | None,
]]:
    specs: list[tuple[
        str,
        dict[tuple[int, object], list[float]],
        dict[tuple[int, object], list[float]] | None,
        str,
        bool,
        tuple[float, float] | None,
    ]] = []

    geom_items = [
        ("shape", "polygon_score", "polygon score", "YlOrRd", False, (0.0, 1.0)),
        ("shape", "polygon_order", "polygon order m*", "YlOrRd", False, None),
        ("shape", "circularity", "circularity", "YlOrRd", False, (0.0, 1.0)),
        ("shape", "word_spread_over_diameter", "word spread / diameter", "YlOrRd", False, None),
        ("state_space", "loop_top2_variance_frac", "loop top-2 var frac", "YlOrRd", False, (0.0, 1.0)),
        ("state_space", "corpus_top2_variance_frac", "corpus top-2 var frac", "YlOrRd", False, (0.0, 1.0)),
        ("state_space", "loop_effective_dim", "loop effective dim", "YlOrRd", False, None),
        ("state_space", "corpus_effective_dim", "corpus effective dim", "YlOrRd", False, None),
        ("state_space", "loop_dims_90pct", "loop dims to 90%", "YlOrRd", False, None),
        ("state_space", "corpus_dims_90pct", "corpus dims to 90%", "YlOrRd", False, None),
        ("state_space", "corpus_mean_abs_corr", "corpus mean |r|", "YlOrRd", False, (0.0, 1.0)),
        ("full_space", "planarity_top2", "planarity top-2", "YlOrRd", False, (0.0, 1.0)),
        ("full_space", "turn_regularity", "turn regularity", "YlOrRd", False, (0.0, 1.0)),
        ("pca_2d", "pc1_variance_frac", "PC1 variance frac", "YlOrRd", False, (0.0, 1.0)),
        ("jpca", "omega", "jPCA rate", "YlOrRd", False, None),
    ]
    for section, key, title, cmap, log_scale, lim in geom_items:
        final_map = _collect_from_seed_panels(
            geometry_panels,
            seeds=seeds,
            getter=lambda p, s=section, k=key: _nested_get(p, s, k),
        )
        specs.append((title, final_map, None, cmap, log_scale, lim))

    train_items = [
        ("demo_word_error_pct", "word error (demo %)", "YlOrRd_r", False, None),
        ("best_metric_word_error_pct", "best word error %", "YlOrRd_r", False, None),
        ("iter_to_threshold", "iters to 3% word err", "YlOrRd_r", True, None),
        ("uniform_tv_distance", "TV dist from uniform", "YlGn_r", False, None),
    ]
    for key, title, cmap, log_scale, lim in train_items:
        final_map = _collect_from_seed_panels(
            training_panels,
            seeds=seeds,
            getter=lambda p, k=key: (
                float(p[k]) if k in p and p[k] is not None and np.isfinite(float(p[k])) else None
            ),
        )
        specs.append((title, final_map, None, cmap, log_scale, lim))

    for metric, title, lim in [
        ("top1_pct", "PC1 % variance (loop)", (0.0, 100.0)),
        ("top2_cum_pct", "PC1+2 % variance (loop)", (0.0, 100.0)),
        ("dims_90pct", "PCs to 90% variance (loop)", None),
    ]:
        final_map = _spectrum_metric_maps(spectra_panels, metric=metric)
        specs.append((title, final_map, None, "YlOrRd", False, lim))

    weight_items = [
        ("motif", "cluster_cohesion_xh", r"$W_{xh}$ cluster cohesion", (0.0, 1.0)),
        ("motif", "xh_top1_mass", r"$W_{xh}$ top-1 mass", (0.0, 1.0)),
        ("motif", "input_tuning_entropy", r"$W_{xh}$ input entropy", (0.0, 1.0)),
        ("motif", "hh_adjacent_corr", r"$W_{hh}$ adjacent |corr|", (0.0, 1.0)),
        ("motif", "hh_within_between_ratio", r"$W_{hh}$ within/between |w|", None),
        ("structure", "input_over_recurrent_norm", "input / recurrent Frobenius", None),
        ("structure", "mean_input_drive_fraction", "mean input-drive fraction", (0.0, 1.0)),
        ("structure", "spectral_radius_hh", r"$W_{hh}$ spectral radius", None),
    ]
    for family, key, title, lim in weight_items:
        init_map, final_map = _weight_stage_maps(
            weight_panels, seeds=seeds, family=family, key=key,
        )
        specs.append((title, final_map, init_map, "YlOrRd", False, lim))

    # Drop empty metrics
    return [s for s in specs if any(s[1].values())]


def plot_pow2_sweep_metric_board(
    *,
    outfile: str = "sweep_all_metrics.png",
    seeds: tuple[int, ...] | None = None,
    spec: Pow2SweepSpec = POW2_SWEEP_SPEC_H100,
    n_cols: int = 4,
) -> Path:
    data_dir = sweep_data_dir(spec.comparison_name)
    geometry = json.loads((data_dir / "sweep_geometry.json").read_text(encoding="utf-8"))
    training = json.loads((data_dir / "sweep_training.json").read_text(encoding="utf-8"))
    spectra = json.loads((data_dir / "sweep_spectra.json").read_text(encoding="utf-8"))
    weights = json.loads((data_dir / "sweep_weight_metrics.json").read_text(encoding="utf-8"))

    run_seeds = seeds if seeds is not None else spec.default_seeds
    seed_set = set(run_seeds)
    word_counts = tuple(int(w) for w in geometry["word_counts"])
    lengths = tuple(_norm_length(L) for L in geometry["lengths"])

    catalog = _metric_catalog(
        geometry_panels=geometry["panels"],
        training_panels=training["panels"],
        spectra_panels=spectra["panels"],
        weight_panels=weights["panels"],
        seeds=seed_set,
    )
    n_metrics = len(catalog)
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.15 * n_cols + 0.4, 2.55 * n_rows + 0.8),
        squeeze=False,
    )
    length_labels = [spec.length_label(L) for L in lengths]
    wc_labels = [str(w) for w in word_counts]

    for idx, (title, final_map, init_map, cmap, log_scale, lim) in enumerate(catalog):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        mat = _mean_grid(final_map, word_counts=word_counts, lengths=lengths)
        finite = mat[np.isfinite(mat)]
        if lim is not None:
            vmin, vmax = lim
        elif finite.size:
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
            if vmax <= vmin:
                vmax = vmin + 1e-6
        else:
            vmin, vmax = 0.0, 1.0

        norm = None
        if log_scale and finite.size and np.all(finite > 0):
            norm = LogNorm(vmin=max(vmin, float(np.min(finite))), vmax=vmax)
            im = ax.imshow(mat, aspect="auto", origin="upper", cmap=cmap, norm=norm)
        else:
            im = ax.imshow(
                mat, aspect="auto", origin="upper", cmap=cmap,
                vmin=vmin, vmax=vmax, interpolation="nearest",
            )

        panel_title = title
        if init_map is not None:
            init_mu = _global_mean(init_map)
            if init_mu is not None:
                panel_title = f"{title}\n(init µ={init_mu:.3g})"
        ax.set_title(panel_title, fontsize=8, pad=3)
        ax.set_xticks(np.arange(len(wc_labels)))
        ax.set_xticklabels(wc_labels, fontsize=6)
        ax.set_yticks(np.arange(len(length_labels)))
        if c == 0:
            ax.set_yticklabels(length_labels, fontsize=6)
        else:
            ax.set_yticklabels([])
        if r == n_rows - 1:
            ax.set_xlabel("# words", fontsize=7)
        for li in range(mat.shape[0]):
            for wi in range(mat.shape[1]):
                v = mat[li, wi]
                if not np.isfinite(v):
                    continue
                if log_scale and v > 0:
                    txt = f"{v:.0f}" if v >= 100 else f"{v:.2g}"
                    mid = np.sqrt(max(vmin, 1e-12) * vmax)
                    color = "white" if v >= mid else "black"
                else:
                    txt = f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}"
                    color = "white" if v >= (vmin + 0.65 * (vmax - vmin)) else "black"
                ax.text(wi, li, txt, ha="center", va="center", fontsize=4.8, color=color)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=5)

    for idx in range(n_metrics, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis("off")

    finalize_grid_figure(
        fig,
        top=0.94,
        bottom=0.04,
        left=0.06,
        hspace=0.55,
        wspace=0.35,
        suptitle=(
            f"Sweep metrics (final; mean over seeds {min(run_seeds)}-{max(run_seeds)}; "
            f"{spec.comparison_name})"
        ),
    )
    out_dir = Path("experiments/comparisons") / spec.comparison_name / "weights"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / outfile
    save_figure(fig, out_path, dpi=160)
    print(f"wrote {out_path}")
    return out_path


if __name__ == "__main__":
    plot_pow2_sweep_metric_board(seeds=(1, 2, 3, 4, 5))