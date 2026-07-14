"""Unified word-count x length metric heatmaps / 3D scatters for a pow2 sweep."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LogNorm, Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

from vocab_diagrams import build_minimized_vocabulary_automaton
from viz.compare.pow2_sweep_spec import POW2_SWEEP_SPEC_H100, Pow2SweepSpec
from viz.compare.sweep_output import sweep_data_dir
from viz.plot_layout import finalize_grid_figure, save_figure

# Per-cell observations: (seed, value). seed=-1 when only a cell mean is available.
CellObs = tuple[int, float]
CellMap = dict[tuple[int, object], list[CellObs]]


def _norm_length(length: object) -> object:
    if isinstance(length, str) and length.isdigit():
        return int(length)
    return length


def _cell_key(n_words: object, length: object) -> tuple[int, object]:
    return int(n_words), _norm_length(length)


def _obs_values(obs: list[CellObs]) -> list[float]:
    return [v for _, v in obs]


def _mean_grid(
    values_by_cell: CellMap,
    *,
    word_counts: tuple[int, ...],
    lengths: tuple[object, ...],
) -> np.ndarray:
    mat = np.full((len(lengths), len(word_counts)), np.nan, dtype=float)
    for li, length in enumerate(lengths):
        for wi, n_words in enumerate(word_counts):
            vals = _obs_values(values_by_cell.get(_cell_key(n_words, length), []))
            if vals:
                mat[li, wi] = float(np.mean(vals))
    return mat


def _collect_from_seed_panels(
    panels: list[dict[str, Any]],
    *,
    seeds: set[int] | None,
    getter: Callable[[dict[str, Any]], float | None],
) -> CellMap:
    out: CellMap = {}
    for panel in panels:
        if "error" in panel:
            continue
        seed = int(panel.get("seed", -1))
        if seeds is not None and seed not in seeds:
            continue
        if "n_words" not in panel:
            continue
        val = getter(panel)
        if val is None or not np.isfinite(val):
            continue
        key = _cell_key(panel["n_words"], panel["length"])
        out.setdefault(key, []).append((seed, float(val)))
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
) -> tuple[CellMap, CellMap]:
    init_map: CellMap = {}
    final_map: CellMap = {}
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
                    init_map.setdefault(cell, []).append((seed, float(init_blob[key])))
                except (TypeError, ValueError):
                    pass
            if key in final_blob:
                try:
                    final_map.setdefault(cell, []).append((seed, float(final_blob[key])))
                except (TypeError, ValueError):
                    pass
    return init_map, final_map


def _global_mean(values_by_cell: CellMap) -> float | None:
    vals = [v for lst in values_by_cell.values() for _, v in lst]
    if not vals:
        return None
    return float(np.mean(vals))


def _spectrum_metric_maps(
    panels: list[dict[str, Any]],
    *,
    metric: str,
) -> CellMap:
    out: CellMap = {}
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
        # spectra panels are already mean-over-seeds
        out.setdefault(key, []).append((-1, val))
    return out


def _metric_catalog(
    *,
    geometry_panels: list[dict[str, Any]],
    training_panels: list[dict[str, Any]],
    spectra_panels: list[dict[str, Any]],
    weight_panels: list[dict[str, Any]],
    seeds: set[int] | None,
) -> list[tuple[
    str,
    CellMap,
    CellMap | None,
    str,
    bool,
    tuple[float, float] | None,
]]:
    specs: list[tuple[
        str,
        CellMap,
        CellMap | None,
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


def _flatten_points(
    values_by_cell: CellMap,
    *,
    word_counts: tuple[int, ...],
    lengths: tuple[object, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return x=log2(#words), y=length-index, z=metric, seed for each observation."""
    length_index = {_norm_length(L): i for i, L in enumerate(lengths)}
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    seeds: list[float] = []
    for (n_words, length), obs in values_by_cell.items():
        if n_words not in word_counts:
            continue
        li = length_index.get(_norm_length(length))
        if li is None:
            continue
        x = float(np.log2(max(n_words, 1)))
        for seed, val in obs:
            if not np.isfinite(val):
                continue
            xs.append(x)
            ys.append(float(li))
            zs.append(float(val))
            seeds.append(float(seed))
    if not xs:
        empty = np.empty(0, dtype=float)
        return empty, empty, empty, empty
    return (
        np.asarray(xs, dtype=float),
        np.asarray(ys, dtype=float),
        np.asarray(zs, dtype=float),
        np.asarray(seeds, dtype=float),
    )


def _fit_plane(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray | None, float]:
    """Least-squares plane z = a + b x + c y. Returns (coef, R^2) or (None, nan)."""
    if x.size < 3:
        return None, float("nan")
    A = np.column_stack([np.ones(x.size), x, y])
    try:
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    except np.linalg.LinAlgError:
        return None, float("nan")
    pred = A @ coef
    ss_res = float(np.sum((z - pred) ** 2))
    ss_tot = float(np.sum((z - np.mean(z)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return coef, r2


def _seed_jitter(seeds: np.ndarray, *, scale: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic per-seed offset so overlapping runs are visible."""
    jx = np.zeros_like(seeds, dtype=float)
    jy = np.zeros_like(seeds, dtype=float)
    for i, s in enumerate(seeds):
        if s < 0:
            continue
        rng = np.random.default_rng(int(s) * 1_000_003 + 17)
        jx[i], jy[i] = rng.uniform(-scale, scale, size=2)
    return jx, jy


def _dfa_state_lookup(spec: Pow2SweepSpec) -> dict[tuple[int, object], int]:
    """Minimized vocabulary DFA state count for each (n_words, length) cell."""
    out: dict[tuple[int, object], int] = {}
    for n_words, length in spec.iter_cells():
        words = spec.build_vocab(n_words, length)
        automaton = build_minimized_vocabulary_automaton(words)
        out[_cell_key(n_words, length)] = int(automaton.dfa._n)
    return out


def _load_metric_sources(spec: Pow2SweepSpec) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any],
]:
    data_dir = sweep_data_dir(spec.comparison_name)
    geometry = json.loads((data_dir / "sweep_geometry.json").read_text(encoding="utf-8"))
    training = json.loads((data_dir / "sweep_training.json").read_text(encoding="utf-8"))
    spectra = json.loads((data_dir / "sweep_spectra.json").read_text(encoding="utf-8"))
    weights = json.loads((data_dir / "sweep_weight_metrics.json").read_text(encoding="utf-8"))
    return geometry, training, spectra, weights


# Drop definitional duplicates and concept-twins of kept panels.
# Keep one representative per story: loop/corpus top-2, effective dim,
# training (best err + iters), letter-columnar W_xh, local W_hh, feedforward balance.
_REDUNDANT_METRIC_TITLES = frozenset({
    # Exact / near-duplicates of kept geometry / spectra panels
    "planarity top-2",              # == loop top-2 var frac
    "PC1+2 % variance (loop)",      # == loop top-2 (spectra %)
    "PCs to 90% variance (loop)",   # == loop dims to 90%
    "PC1 variance frac",            # nested in corpus PCA / top-2 story
    "PC1 % variance (loop)",        # nested in loop top-2
    # Weaker twins of the kept dimensionality story
    "loop dims to 90%",             # ≈ loop effective dim
    "corpus dims to 90%",           # ≈ corpus effective dim
    "corpus mean |r|",              # ≈ inverse of corpus top-2 / effective dim
    # Closed-loop shape family (weak DFA signal; word spread covers "spread")
    "polygon score",
    "polygon order m*",
    "circularity",
    "turn regularity",
    # Training: best err dominates a single demo snapshot
    "word error (demo %)",
    # Weight twins / near-nulls
    r"$W_{xh}$ input entropy",      # inverse twin of W_xh top-1 mass
    r"$W_{hh}$ within/between |w|",  # stays ~1; not informative
    "mean input-drive fraction",    # twin of input / recurrent Frobenius
})


def _filtered_metric_catalog(**kwargs) -> list[tuple[
    str,
    CellMap,
    CellMap | None,
    str,
    bool,
    tuple[float, float] | None,
]]:
    return [s for s in _metric_catalog(**kwargs) if s[0] not in _REDUNDANT_METRIC_TITLES]


def plot_pow2_sweep_metric_board(
    *,
    outfile: str = "sweep_all_metrics.png",
    seeds: tuple[int, ...] | None = None,
    spec: Pow2SweepSpec = POW2_SWEEP_SPEC_H100,
    n_cols: int = 4,
) -> Path:
    geometry, training, spectra, weights = _load_metric_sources(spec)

    run_seeds = seeds if seeds is not None else spec.default_seeds
    seed_set = set(run_seeds)
    word_counts = tuple(int(w) for w in geometry["word_counts"])
    lengths = tuple(_norm_length(L) for L in geometry["lengths"])

    catalog = _filtered_metric_catalog(
        geometry_panels=geometry["panels"],
        training_panels=training["panels"],
        spectra_panels=spectra["panels"],
        weight_panels=weights["panels"],
        seeds=seed_set,
    )
    n_metrics = len(catalog)
    if n_metrics > 20:
        n_cols = max(n_cols, 5)
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.05 * n_cols + 0.4, 1.7 * n_rows + 0.6),
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
        ax.set_title(panel_title, fontsize=7, pad=2)
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
                    if v >= 100:
                        txt = f"{v:.0e}".replace("e+0", "e").replace("e+", "e")
                    else:
                        txt = f"{v:.2g}"
                    mid = np.sqrt(max(vmin, 1e-12) * vmax)
                    color = "white" if v >= mid else "black"
                else:
                    if abs(v) >= 100:
                        txt = f"{v:.0e}".replace("e+0", "e").replace("e+", "e")
                    elif abs(v) >= 10:
                        txt = f"{v:.1f}"
                    else:
                        txt = f"{v:.2f}"
                    color = "white" if v >= (vmin + 0.65 * (vmax - vmin)) else "black"
                ax.text(wi, li, txt, ha="center", va="center", fontsize=3.6, color=color)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=5)

    for idx in range(n_metrics, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis("off")

    finalize_grid_figure(
        fig,
        top=0.93,
        bottom=0.05,
        left=0.07,
        hspace=0.65,
        wspace=0.45,
        suptitle=(
            f"Sweep metrics (final; mean over seeds {min(run_seeds)}-{max(run_seeds)}; "
            f"{spec.comparison_name})"
        ),
    )
    out_dir = Path("experiments/comparisons") / spec.comparison_name / "weights"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / outfile
    save_figure(fig, out_path, dpi=150)
    print(f"wrote {out_path}")
    return out_path


def plot_pow2_sweep_metric_scatter3d(
    *,
    outfile: str = "sweep_all_metrics_scatter3d.png",
    seeds: tuple[int, ...] | None = None,
    spec: Pow2SweepSpec = POW2_SWEEP_SPEC_H100,
    n_cols: int = 4,
) -> Path:
    """Grid of 3D scatters: each seed is a point; plane = OLS fit on log2(#words), length."""
    geometry, training, spectra, weights = _load_metric_sources(spec)

    # Default: every seed present in geometry/training JSON (weights may be a subset).
    if seeds is None:
        run_seeds = tuple(int(s) for s in geometry.get("seeds", spec.default_seeds))
    else:
        run_seeds = seeds
    seed_set = set(run_seeds)
    word_counts = tuple(int(w) for w in geometry["word_counts"])
    lengths = tuple(_norm_length(L) for L in geometry["lengths"])
    length_labels = [spec.length_label(L) for L in lengths]
    log2_ticks = [float(np.log2(w)) for w in word_counts]

    catalog = _filtered_metric_catalog(
        geometry_panels=geometry["panels"],
        training_panels=training["panels"],
        spectra_panels=spectra["panels"],
        weight_panels=weights["panels"],
        seeds=seed_set,
    )
    n_metrics = len(catalog)
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig = plt.figure(figsize=(2.55 * n_cols + 0.6, 2.2 * n_rows + 0.75))
    dfa_states = _dfa_state_lookup(spec)
    dfa_cmap = plt.get_cmap("turbo")
    dfa_levels = sorted({int(v) for v in dfa_states.values() if v > 0})
    if len(dfa_levels) >= 2:
        # Midpoints between consecutive levels + open ends → one color band per count.
        mids = [
            0.5 * (dfa_levels[i] + dfa_levels[i + 1])
            for i in range(len(dfa_levels) - 1)
        ]
        boundaries = [dfa_levels[0] - 0.5, *mids, dfa_levels[-1] + 0.5]
        dfa_norm = BoundaryNorm(boundaries, ncolors=dfa_cmap.N, clip=True)
    else:
        dfa_norm = Normalize(
            vmin=float(dfa_levels[0]) if dfa_levels else 0.0,
            vmax=float(dfa_levels[0] + 1) if dfa_levels else 1.0,
        )

    for idx, (title, final_map, init_map, _cmap, log_scale, lim) in enumerate(catalog):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection="3d")
        x, y, z, seed_ids = _flatten_points(
            final_map, word_counts=word_counts, lengths=lengths,
        )
        if x.size == 0:
            ax.set_title(title, fontsize=8)
            ax.set_axis_off()
            continue

        use_log_z = bool(log_scale and np.all(z > 0))
        z_plot = np.log10(np.clip(z, 1e-12, None)) if use_log_z else z
        jx, jy = _seed_jitter(seed_ids)
        color_vals = np.array([
            float(dfa_states.get(
                _cell_key(int(round(2.0 ** float(xi))), lengths[int(round(float(yi)))]),
                np.nan,
            ))
            for xi, yi in zip(x, y)
        ], dtype=float)

        ax.scatter(
            x + jx, y + jy, z_plot,
            c=color_vals, cmap=dfa_cmap, norm=dfa_norm,
            s=11, depthshade=True, linewidths=0, alpha=0.85,
        )

        coef, r2 = _fit_plane(x, y, z_plot)
        if coef is not None:
            xg = np.linspace(min(log2_ticks), max(log2_ticks), 12)
            yg = np.linspace(0, len(lengths) - 1, 12)
            Xg, Yg = np.meshgrid(xg, yg)
            Zg = coef[0] + coef[1] * Xg + coef[2] * Yg
            ax.plot_surface(
                Xg, Yg, Zg, alpha=0.22, color="#4C78A8",
                linewidth=0, antialiased=True,
            )

        bits = [title]
        if init_map is not None:
            init_mu = _global_mean(init_map)
            if init_mu is not None:
                bits.append(f"init µ={init_mu:.3g}")
        if np.isfinite(r2):
            bits.append(f"$R^2$={r2:.2f}")
        # Two-line title avoids horizontal bleed between panels.
        if len(bits) == 1:
            ax.set_title(bits[0], fontsize=6.5, pad=3)
        else:
            ax.set_title(f"{bits[0]}\n" + " · ".join(bits[1:]), fontsize=6, pad=3)

        ax.set_xticks(log2_ticks)
        ax.set_xticklabels([str(w) for w in word_counts], fontsize=5)
        ax.set_yticks(np.arange(len(length_labels)))
        ax.set_yticklabels(length_labels, fontsize=5)
        ax.tick_params(axis="z", labelsize=5)
        ax.set_xlabel("# words", fontsize=6, labelpad=-1)
        ax.set_ylabel("length", fontsize=6, labelpad=-1)
        ax.set_zlabel("log10" if use_log_z else "value", fontsize=6, labelpad=-1)
        if lim is not None and not use_log_z:
            ax.set_zlim(lim)
        ax.view_init(elev=22, azim=-58)

    for idx in range(n_metrics, n_rows * n_cols):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        ax.axis("off")

    finalize_grid_figure(
        fig,
        top=0.93,
        bottom=0.05,
        left=0.03,
        right=0.92,
        hspace=0.42,
        wspace=0.12,
        suptitle=(
            f"Sweep metrics 3D (one point per seed; color = min-DFA #states; "
            f"OLS plane on log2(#words), length; seeds {min(run_seeds)}-{max(run_seeds)}; "
            f"{spec.comparison_name})"
        ),
    )
    sm = plt.cm.ScalarMappable(cmap=dfa_cmap, norm=dfa_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, fraction=0.012, pad=0.01, shrink=0.6)
    cbar.set_label("DFA states", fontsize=8)
    if len(dfa_levels) <= 12:
        cbar.set_ticks(dfa_levels)
    else:
        step = max(1, len(dfa_levels) // 8)
        cbar.set_ticks(dfa_levels[::step])
    cbar.ax.tick_params(labelsize=6)

    out_dir = Path("experiments/comparisons") / spec.comparison_name / "weights"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / outfile
    save_figure(fig, out_path, dpi=150)
    print(f"wrote {out_path}")
    return out_path


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Least-squares y = a + b x. Returns (coef, R^2) or (None, nan)."""
    if x.size < 2:
        return None, float("nan")
    A = np.column_stack([np.ones(x.size), x])
    try:
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    except np.linalg.LinAlgError:
        return None, float("nan")
    pred = A @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return coef, r2


def _r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return 1.0
    return 1.0 - ss_res / ss_tot


def _adjusted_r2(r2: float, n: int, n_params: int) -> float:
    if n <= n_params + 1 or not np.isfinite(r2):
        return float("nan")
    return 1.0 - (1.0 - r2) * (n - 1) / (n - n_params - 1)


def _safe_logistic(x: np.ndarray, lo: float, hi: float, k: float, x0: float) -> np.ndarray:
    z = np.clip(k * (x - x0), -40.0, 40.0)
    return lo + (hi - lo) / (1.0 + np.exp(-z))


def _safe_exp_asymp(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a + b * np.exp(-np.clip(c * x, -40.0, 40.0))


def _safe_hyperbola(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a + b * x / (np.abs(c) + x)


def _fit_trend(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_grid: int = 80,
) -> tuple[np.ndarray | None, np.ndarray | None, float, str]:
    """Fit linear / sigmoid / exp-asymptote / hyperbola; pick best adjusted R².

    Curve parameters are estimated from per-x medians (outlier-resistant);
    reported ``R²`` is still computed on all points. Returns
    ``(x_grid, y_hat, r2, model_name)``.
    """
    from scipy.optimize import curve_fit

    if x.size < 4:
        coef, r2 = _fit_line(x, y)
        if coef is None:
            return None, None, float("nan"), "none"
        xg = np.linspace(float(np.min(x)), float(np.max(x)), n_grid)
        return xg, coef[0] + coef[1] * xg, r2, "linear"

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_min, x_max = float(np.min(x)), float(np.max(x))
    # Median y at each distinct DFA count — dampens seed/outlier leverage.
    x_u = np.unique(x)
    y_med = np.array([float(np.median(y[x == xu])) for xu in x_u], dtype=float)
    y_min, y_max = float(np.min(y_med)), float(np.max(y_med))
    y_span = max(y_max - y_min, 1e-9)
    x_mid = float(np.median(x_u))
    if np.std(x_u) > 0 and np.std(y_med) > 0:
        corr = float(np.corrcoef(x_u, y_med)[0, 1])
    else:
        corr = 0.0

    candidates: list[tuple[float, float, int, str, Callable[[np.ndarray], np.ndarray]]] = []

    def _push(name: str, n_params: int, pred_fn: Callable[[np.ndarray], np.ndarray]) -> None:
        pred_all = np.asarray(pred_fn(x), dtype=float)
        r2 = _r2_score(y, pred_all)
        if not np.isfinite(r2):
            return
        candidates.append((_adjusted_r2(r2, x.size, n_params), r2, n_params, name, pred_fn))

    coef, _ = _fit_line(x_u, y_med)
    if coef is not None:
        _push("linear", 2, (lambda xx, a=float(coef[0]), b=float(coef[1]): a + b * xx))

    # Soft slope bound: transition across the x-range in a few e-folds, not a cliff
    # unless the medians really demand it (still allow moderately steep).
    k_lim = 18.0 / max(x_max - x_min, 1.0)
    try:
        k0 = (3.0 / max(x_max - x_min, 1.0)) * (1.0 if corr >= 0 else -1.0)
        p0 = (y_min, y_max, k0, x_mid)
        bounds = (
            (y_min - 2.0 * y_span, y_min - 2.0 * y_span, -k_lim, x_min - (x_max - x_min)),
            (y_max + 2.0 * y_span, y_max + 2.0 * y_span, k_lim, x_max + (x_max - x_min)),
        )
        popt, _ = curve_fit(
            _safe_logistic, x_u, y_med, p0=p0, bounds=bounds, maxfev=8000,
        )
        _push("sigmoid", 4, (lambda xx, p=tuple(float(v) for v in popt): _safe_logistic(xx, *p)))
    except (RuntimeError, ValueError, TypeError):
        pass

    try:
        b0 = (y_min - y_max) if corr < 0 else (y_max - y_min)
        a0 = y_max if corr < 0 else y_min
        c0 = 2.0 / max(x_max - x_min, 1.0)
        p0 = (a0, b0, c0)
        bounds = (
            (y_min - 3.0 * y_span, -8.0 * y_span, -20.0),
            (y_max + 3.0 * y_span, 8.0 * y_span, 20.0),
        )
        popt, _ = curve_fit(
            _safe_exp_asymp, x_u, y_med, p0=p0, bounds=bounds, maxfev=8000,
        )
        _push("exp", 3, (lambda xx, p=tuple(float(v) for v in popt): _safe_exp_asymp(xx, *p)))
    except (RuntimeError, ValueError, TypeError):
        pass

    try:
        b0 = (y_max - y_min) if corr >= 0 else (y_min - y_max)
        p0 = (y_min if corr >= 0 else y_max, b0, max(x_mid, 1.0))
        bounds = (
            (y_min - 3.0 * y_span, -8.0 * y_span, 1e-3),
            (y_max + 3.0 * y_span, 8.0 * y_span, max(5.0 * (x_max - x_min), 1.0)),
        )
        popt, _ = curve_fit(
            _safe_hyperbola, x_u, y_med, p0=p0, bounds=bounds, maxfev=8000,
        )
        _push("hyperbola", 3, (lambda xx, p=tuple(float(v) for v in popt): _safe_hyperbola(xx, *p)))
    except (RuntimeError, ValueError, TypeError):
        pass

    if not candidates:
        return None, None, float("nan"), "none"

    candidates.sort(key=lambda t: (
        -t[0] if np.isfinite(t[0]) else 1e9,
        t[2],
        -t[1],
    ))
    _adj, r2, _npar, name, pred_fn = candidates[0]
    if not np.isfinite(r2):
        return None, None, float("nan"), "none"
    xg = np.linspace(x_min, x_max, n_grid)
    yg = np.asarray(pred_fn(xg), dtype=float)
    return xg, yg, float(r2), name


def _points_vs_dfa(
    values_by_cell: CellMap,
    *,
    dfa_states: dict[tuple[int, object], int],
    word_counts: tuple[int, ...],
    lengths: tuple[object, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (dfa_states, metric, seed, length_idx, n_words) across conditions."""
    length_index = {_norm_length(L): i for i, L in enumerate(lengths)}
    wc_set = set(word_counts)
    xs: list[float] = []
    ys: list[float] = []
    seeds: list[float] = []
    length_idxs: list[float] = []
    n_words_list: list[float] = []
    for (n_words, length), obs in values_by_cell.items():
        li = length_index.get(_norm_length(length))
        if n_words not in wc_set or li is None:
            continue
        dfa = dfa_states.get(_cell_key(n_words, length))
        if dfa is None:
            continue
        for seed, val in obs:
            if not np.isfinite(val):
                continue
            xs.append(float(dfa))
            ys.append(float(val))
            seeds.append(float(seed))
            length_idxs.append(float(li))
            n_words_list.append(float(n_words))
    if not xs:
        empty = np.empty(0, dtype=float)
        return empty, empty, empty, empty, empty
    return (
        np.asarray(xs, dtype=float),
        np.asarray(ys, dtype=float),
        np.asarray(seeds, dtype=float),
        np.asarray(length_idxs, dtype=float),
        np.asarray(n_words_list, dtype=float),
    )


def _prepare_dfa_scatter_panels(
    *,
    spec: Pow2SweepSpec,
    seeds: tuple[int, ...] | None,
    min_r2: float,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[object, ...],
    list[tuple[
        str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        bool, tuple[float, float] | None, np.ndarray | None, np.ndarray | None,
    ]],
]:
    geometry, training, spectra, weights = _load_metric_sources(spec)
    if seeds is None:
        run_seeds = tuple(int(s) for s in geometry.get("seeds", spec.default_seeds))
    else:
        run_seeds = seeds
    seed_set = set(run_seeds)
    word_counts = tuple(int(w) for w in geometry["word_counts"])
    lengths = tuple(_norm_length(L) for L in geometry["lengths"])
    dfa_states = _dfa_state_lookup(spec)

    catalog = _metric_catalog(
        geometry_panels=geometry["panels"],
        training_panels=training["panels"],
        spectra_panels=spectra["panels"],
        weight_panels=weights["panels"],
        seeds=seed_set,
    )

    prepared: list[tuple[
        str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        bool, tuple[float, float] | None, np.ndarray | None, np.ndarray | None,
    ]] = []
    for title, final_map, init_map, _cmap, log_scale, lim in catalog:
        if title in _REDUNDANT_METRIC_TITLES:
            continue
        dfa_x, metric_y, seed_ids, length_idx, n_words_arr = _points_vs_dfa(
            final_map,
            dfa_states=dfa_states,
            word_counts=word_counts,
            lengths=lengths,
        )
        if dfa_x.size == 0:
            continue
        use_log_y = bool(log_scale and np.all(metric_y > 0))
        y_plot = np.log10(np.clip(metric_y, 1e-12, None)) if use_log_y else metric_y
        # Keep the same panel set as the linear screen; draw the best flexible curve.
        _coef, r2_lin = _fit_line(dfa_x, y_plot)
        if not np.isfinite(r2_lin) or r2_lin < min_r2:
            continue
        x_fit, y_fit, r2, _model = _fit_trend(dfa_x, y_plot)
        if x_fit is None or not np.isfinite(r2):
            continue
        bits = [title]
        if init_map is not None:
            init_mu = _global_mean(init_map)
            if init_mu is not None:
                bits.append(f"init µ={init_mu:.3g}")
        bits.append(f"$R^2$={r2:.2f}")
        prepared.append((
            " | ".join(bits), dfa_x, y_plot, seed_ids, length_idx, n_words_arr,
            use_log_y, lim, x_fit, y_fit,
        ))

    if not prepared:
        raise ValueError(f"no metrics with R^2 >= {min_r2}")
    return run_seeds, word_counts, lengths, prepared


def plot_pow2_sweep_metric_scatter2d(
    *,
    outfile: str = "sweep_all_metrics_scatter2d.png",
    seeds: tuple[int, ...] | None = None,
    spec: Pow2SweepSpec = POW2_SWEEP_SPEC_H100,
    min_r2: float = 0.1,
    wrap: int = 4,
) -> Path:
    """Compact metrics-vs-DFA scatters: color = #words, marker = word length."""
    run_seeds, word_counts, lengths, prepared = _prepare_dfa_scatter_panels(
        spec=spec, seeds=seeds, min_r2=min_r2,
    )
    # Prefer the strongest DFA-linked panels for a paper-sized figure.
    _PAPER_PRIORITY = (
        "loop top-2 var frac",
        "corpus top-2 var frac",
        "loop effective dim",
        "corpus effective dim",
        "iters to 3% word err",
        "input / recurrent Frobenius",
        r"$W_{hh}$ adjacent |corr|",
        r"$W_{xh}$ top-1 mass",
    )
    ranked: list[tuple[int, tuple]] = []
    for panel in prepared:
        short = panel[0].split(" | ")[0]
        for cut in (" (loop)", " (corpus)", " (init"):
            if cut in short:
                short = short.split(cut)[0]
        try:
            rank = _PAPER_PRIORITY.index(short)
        except ValueError:
            rank = 100 + len(ranked)
        ranked.append((rank, panel))
    ranked.sort(key=lambda t: t[0])
    # Keep priority hits first; cap at 8 for a paper-sized figure.
    core = [p for r, p in ranked if r < 100][:8]
    if core:
        prepared = core

    n_metrics = len(prepared)
    n_wrap = max(1, min(wrap, n_metrics))
    n_rows = int(np.ceil(n_metrics / n_wrap))
    n_cols = n_wrap
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.15 * n_cols + 0.55, 1.55 * n_rows + 0.55),
        squeeze=False,
        sharex=True,
    )

    words_cmap = plt.get_cmap("YlOrRd")
    words_levels = list(word_counts)
    if len(words_levels) >= 2:
        mids = [
            0.5 * (words_levels[i] + words_levels[i + 1])
            for i in range(len(words_levels) - 1)
        ]
        words_norm = BoundaryNorm(
            [words_levels[0] * 0.5, *mids, words_levels[-1] * 1.5],
            ncolors=words_cmap.N,
            clip=True,
        )
    else:
        words_norm = Normalize(vmin=1.0, vmax=float(words_levels[0]))

    length_markers = ("o", "s", "^", "D", "v", "P", "X")
    marker_by_length_idx = {
        i: length_markers[i % len(length_markers)] for i in range(len(lengths))
    }

    for mi, (panel_title, x, y_plot, seed_ids, length_idx, n_words_arr, use_log_y, lim, x_fit, y_fit) in enumerate(prepared):
        row, col = divmod(mi, n_wrap)
        ax = axes[row][col]
        jx, _ = _seed_jitter(seed_ids, scale=0.12)

        for li in range(len(lengths)):
            mask = length_idx == float(li)
            if not np.any(mask):
                continue
            ax.scatter(
                x[mask] + jx[mask], y_plot[mask],
                c=n_words_arr[mask], cmap=words_cmap, norm=words_norm,
                marker=marker_by_length_idx[li],
                s=11, alpha=0.70, linewidths=0.2, edgecolors="white",
                zorder=2,
            )
        if x_fit is not None and y_fit is not None:
            ax.plot(x_fit, y_fit, color="#111111", lw=1.15, zorder=3)

        short = panel_title.split(" | ")[0]
        for cut in (" (loop)", " (corpus)", " (init"):
            if cut in short:
                short = short.split(cut)[0]
        r2_bits = [b for b in panel_title.split(" | ") if b.startswith("$R")]
        title = short if not r2_bits else f"{short}\n{r2_bits[-1]}"
        ax.set_title(title, fontsize=7, pad=2)
        ax.tick_params(labelsize=6)
        if row == n_rows - 1:
            ax.set_xlabel("DFA states", fontsize=7)
        if col == 0:
            ax.set_ylabel("log10" if use_log_y else "value", fontsize=7)
        if lim is not None and not use_log_y:
            ax.set_ylim(lim)
        ax.grid(True, alpha=0.25, linewidth=0.5)

    used_in_last = n_metrics - (n_rows - 1) * n_wrap
    for col in range(used_in_last, n_cols):
        axes[n_rows - 1][col].axis("off")

    marker_handles = [
        plt.Line2D(
            [0], [0], linestyle="None",
            marker=marker_by_length_idx[i], color="0.3",
            markersize=5.5, label=spec.length_label(L),
        )
        for i, L in enumerate(lengths)
    ]
    fig.legend(
        handles=marker_handles, title="word length",
        loc="lower center", ncol=min(len(lengths), 7),
        fontsize=6, title_fontsize=7, frameon=False,
        bbox_to_anchor=(0.42, 0.005),
    )

    finalize_grid_figure(
        fig,
        top=0.90,
        bottom=0.16,
        left=0.07,
        right=0.90,
        hspace=0.45,
        wspace=0.28,
        suptitle=(
            f"Metrics vs minimized DFA size "
            f"(color = #words; marker = word length; "
            f"curve = best of linear/sigmoid/exp/hyperbola; "
            f"seeds {min(run_seeds)}–{max(run_seeds)})"
        ),
        suptitle_fontsize=9,
    )

    sm = plt.cm.ScalarMappable(cmap=words_cmap, norm=words_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_ticks(words_levels)
    cbar.set_ticklabels([str(w) for w in words_levels])
    cbar.set_label("# words", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    out_dir = Path("experiments/comparisons") / spec.comparison_name / "weights"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / outfile
    save_figure(fig, out_path, dpi=150)
    print(f"wrote {out_path}")
    return out_path


if __name__ == "__main__":
    plot_pow2_sweep_metric_board()
    plot_pow2_sweep_metric_scatter2d()
    plot_pow2_sweep_metric_scatter3d()