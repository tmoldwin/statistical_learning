"""Shape readout for closed-loop trajectories: polygon order, corners, word spread.

Metrics operate on the mean closed-loop trajectory projected into the
teacher-forced PCA 2D plane.  The rollout winds around the shape many times
(roughly once per word), so shape is read out from the angular profile
r(theta) of loop points around their centroid, which collapses all windings
onto one outline:

- polygon template scores: the shape is whitened (elongation removed) and
  r(theta) is correlated against the radial profile of an ideal regular
  m-gon, maximized over rotation.  Correlation is amplitude-invariant, so
  hexagons (whose radius only varies ~7%) compete fairly with triangles
  (~100%).  High score at m=4 reads out square, m=3 triangle ...
- corner count: circular peaks of r(theta) (polygon vertices stick out).
- circularity: isoperimetric ratio 4*pi*A / P^2 of the collapsed outline
  (circle 1.0, square ~0.785, equilateral triangle ~0.605).
- word spread: how far individual word trajectories sit from the mean loop
  (normalized by loop diameter), plus within-word consistency across
  repeated traversals of the same word.
- state space: participation-ratio effective dimensionality and mean |r|
  for the teacher-forced corpus cloud and the closed-loop rollout in ℝᴴ.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from experiment import comparison_dir, seeds_for_task
from viz.compare._data import TaskVizContext, load_task_viz_context
from viz.compare.geometry import _bbox_aspect, _path_diameter
from viz.compare.spec import ComparisonSpec
from viz.compare.state_space_metrics import paired_state_space_metrics
from viz.dimred import fit_pca_2d_with_evr
from visualize import (
    _closed_loop_summary_seed,
    _one_vocab_cycle_steps,
    _square_data_limits,
    _trajectory_seed_letters,
    corpus_segments,
    rnn_closed_loop_rollout,
)

POLYGON_ORDERS = tuple(range(3, 9))

_ANGLE_BINS = 90
_CORNER_MIN_PROMINENCE_FRAC = 0.04
_ROLLOUT_CYCLES = 6
_ROLLOUT_SEED = 0


# ---------------------------------------------------------------------------
# loop shape metrics (2D)
# ---------------------------------------------------------------------------

def densify_polyline(path: np.ndarray, n: int = 4096) -> np.ndarray | None:
    """Uniform arc-length resampling of an open 2D polyline.

    The RNN visits a handful of discrete states; the traced shape lives in
    the segments between them, so points must be spread along the path
    before any angular statistics.
    """
    p = np.asarray(path, dtype=float)[:, :2]
    if p.ndim != 2 or len(p) < 3:
        return None
    seg_len = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(s[-1])
    if total <= 1e-12:
        return None
    t = np.linspace(0.0, total, n)
    return np.column_stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])])


def radial_profile(points: np.ndarray, n_bins: int = _ANGLE_BINS) -> tuple[np.ndarray, np.ndarray] | None:
    """Median radius per angular bin around the centroid (circularly interpolated).

    Collapses a multi-winding trajectory onto a single shape outline.
    Returns (theta_centers, r_profile) or None if the shape is degenerate.
    """
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or len(p) < 8:
        return None
    centered = p[:, :2] - p[:, :2].mean(axis=0)
    r = np.linalg.norm(centered, axis=1)
    if float(r.max()) <= 1e-12:
        return None
    theta = np.arctan2(centered[:, 1], centered[:, 0])
    bins = ((theta + np.pi) / (2.0 * np.pi) * n_bins).astype(int) % n_bins

    profile = np.full(n_bins, np.nan)
    for b in range(n_bins):
        vals = r[bins == b]
        if len(vals):
            profile[b] = float(np.median(vals))

    filled = np.isfinite(profile)
    if filled.sum() < max(8, n_bins // 4):
        return None
    if not filled.all():
        idx = np.arange(n_bins)
        profile = np.interp(idx, idx[filled], profile[filled], period=n_bins)

    profile = _circular_smooth(profile, window=max(3, n_bins // 30))
    theta_centers = -np.pi + (np.arange(n_bins) + 0.5) * (2.0 * np.pi / n_bins)
    return theta_centers, profile


def _circular_smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window < 2:
        return values
    kernel = np.ones(window) / window
    n = len(values)
    tiled = np.concatenate([values, values, values])
    return np.convolve(tiled, kernel, mode="same")[n : 2 * n]


def whiten_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rescale principal axes to equal variance (removes elongation).

    A stretched hexagon becomes a regular-ish hexagon, so the polygon
    readout is not confused by the k=2 (ellipse) harmonic.  Elongation
    itself is still reported separately via bbox_aspect.

    Returns (whitened, mean, vh, scale) so whitened points can be mapped
    back via ``xy = (whitened * scale) @ vh + mean``.
    """
    p = np.asarray(points, dtype=float)[:, :2]
    mean = p.mean(axis=0)
    centered = p - mean
    _, s, vh = np.linalg.svd(centered, full_matrices=False)
    s = np.maximum(s, 1e-12 * s.max() if s.max() > 0 else 1.0)
    scale = s / s.max()
    whitened = (centered @ vh.T) / scale
    return whitened, mean, vh, scale


def _regular_mgon_profile(theta: np.ndarray, m: int) -> np.ndarray:
    """Radial profile of a regular m-gon with unit circumradius."""
    local = np.mod(theta, 2.0 * np.pi / m) - np.pi / m
    return np.cos(np.pi / m) / np.cos(local)


def polygon_template_scores(
    theta: np.ndarray,
    r_profile: np.ndarray,
    orders: tuple[int, ...] = POLYGON_ORDERS,
) -> dict[int, float]:
    """Pearson correlation of r(theta) with an ideal regular m-gon profile,
    maximized over rotation.  Amplitude-invariant, so weakly-modulated
    hexagons compete fairly with strongly-modulated triangles.  Clipped at 0.
    """
    n = len(r_profile)
    sig = r_profile - r_profile.mean()
    sig_norm = float(np.linalg.norm(sig))
    scores: dict[int, float] = {}
    for m in orders:
        if sig_norm <= 1e-12:
            scores[m] = float("nan")
            continue
        best = -1.0
        shifts_per_period = max(1, int(round(n / m)))
        for shift in range(shifts_per_period):
            tmpl = _regular_mgon_profile(theta + shift * (2.0 * np.pi / n), m)
            tmpl = tmpl - tmpl.mean()
            tmpl_norm = float(np.linalg.norm(tmpl))
            if tmpl_norm <= 1e-12:
                continue
            best = max(best, float(np.dot(sig, tmpl)) / (sig_norm * tmpl_norm))
        scores[m] = max(0.0, best)
    return scores


def best_polygon_order(scores: dict[int, float]) -> tuple[int, float]:
    finite = {m: v for m, v in scores.items() if np.isfinite(v)}
    if not finite:
        return 0, float("nan")
    best = max(finite, key=lambda m: finite[m])
    return best, finite[best]


def count_corners(
    theta: np.ndarray,
    r_profile: np.ndarray,
    *,
    min_prominence_frac: float = _CORNER_MIN_PROMINENCE_FRAC,
) -> tuple[int, np.ndarray]:
    """Corners = circular peaks of r(theta): polygon vertices stick out.

    Returns (n_corners, corner_theta_r) with peak angular positions.
    """
    n = len(r_profile)
    r_mean = float(r_profile.mean())
    if r_mean <= 1e-12:
        return 0, np.empty((0, 2))
    tiled = np.concatenate([r_profile, r_profile, r_profile])
    peaks, _ = find_peaks(
        tiled,
        prominence=min_prominence_frac * r_mean,
        distance=max(2, n // 16),
    )
    peaks = np.unique(peaks[(peaks >= n) & (peaks < 2 * n)] - n)
    if not len(peaks):
        return 0, np.empty((0, 2))
    return int(len(peaks)), np.column_stack([theta[peaks], r_profile[peaks]])


def outline_circularity(theta: np.ndarray, r_profile: np.ndarray) -> float:
    """Isoperimetric ratio 4*pi*A / P^2 of the collapsed outline."""
    x = r_profile * np.cos(theta)
    y = r_profile * np.sin(theta)
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    pts = np.column_stack([x, y])
    perim = float(np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1).sum())
    if perim <= 1e-12:
        return float("nan")
    return 4.0 * np.pi * area / (perim * perim)


def loop_shape_metrics(path_2d: np.ndarray) -> dict[str, Any]:
    """All shape-readout metrics for one (possibly multi-winding) 2D loop."""
    dense = densify_polyline(path_2d)
    if dense is None:
        return {"error": "degenerate loop"}
    prof = radial_profile(dense)
    if prof is None:
        return {"error": "degenerate loop"}
    theta, r_profile = prof
    centroid = dense.mean(axis=0)

    # Polygon + corner readout on the whitened shape (elongation removed).
    whitened, w_mean, w_vh, w_scale = whiten_points(dense)
    white_prof = radial_profile(whitened)
    if white_prof is None:
        return {"error": "degenerate loop"}
    w_theta, w_r = white_prof
    spectrum = polygon_template_scores(w_theta, w_r)
    order, score = best_polygon_order(spectrum)
    n_corners, corner_polar = count_corners(w_theta, w_r)
    if len(corner_polar):
        w_centroid = whitened.mean(axis=0)
        corner_w = w_centroid + np.column_stack([
            corner_polar[:, 1] * np.cos(corner_polar[:, 0]),
            corner_polar[:, 1] * np.sin(corner_polar[:, 0]),
        ])
        corner_xy = (corner_w * w_scale) @ w_vh + w_mean
    else:
        corner_xy = np.empty((0, 2))
    outline_xy = centroid + np.column_stack([
        r_profile * np.cos(theta),
        r_profile * np.sin(theta),
    ])

    return {
        "polygon_order": order,
        "polygon_score": score,
        "polygon_spectrum": {int(m): float(v) for m, v in spectrum.items()},
        "n_corners": n_corners,
        "corner_xy": corner_xy,
        "circularity": outline_circularity(theta, r_profile),
        "bbox_aspect": _bbox_aspect(outline_xy),
        "outline_xy": outline_xy,
    }


# ---------------------------------------------------------------------------
# word spread metrics
# ---------------------------------------------------------------------------

def labeled_word_trajectories(ctx: TaskVizContext) -> list[tuple[str, np.ndarray]]:
    """(word, hidden-state trajectory) for each word occurrence in the window."""
    segments = corpus_segments(ctx.text, list(ctx.words), spaced=ctx.spaced)
    out: list[tuple[str, np.ndarray]] = []
    for start, end, word in segments:
        traj = np.asarray(ctx.hidden_states[start : end + 1], dtype=float)
        if len(traj) >= 2:
            out.append((word.strip(), traj))
    return out


def _mean_min_distance(path: np.ndarray, ref: np.ndarray) -> float:
    return float(np.mean([np.min(np.linalg.norm(ref - p, axis=1)) for p in path]))


def word_spread_metrics(
    labeled_trajs: list[tuple[str, np.ndarray]],
    mean_loop: np.ndarray,
) -> dict[str, Any]:
    """Spread of word trajectories around the mean loop + within-word consistency.

    All distances normalized by loop diameter, so 0 = words hug the shape
    exactly and values ~1 mean word paths wander as far as the shape is wide.
    """
    diam = _path_diameter(mean_loop)
    if not labeled_trajs or diam <= 1e-12:
        return {
            "word_spread_over_diameter": float("nan"),
            "within_word_spread_over_diameter": float("nan"),
            "per_word_spread": {},
        }

    per_word: dict[str, list[float]] = {}
    for word, traj in labeled_trajs:
        per_word.setdefault(word, []).append(_mean_min_distance(traj, mean_loop) / diam)
    per_word_mean = {w: float(np.mean(v)) for w, v in per_word.items()}
    spread = float(np.mean([d for v in per_word.values() for d in v]))

    # Within-word: mean pairwise distance between aligned traversals of the same word.
    by_word: dict[str, list[np.ndarray]] = {}
    for word, traj in labeled_trajs:
        by_word.setdefault(word, []).append(traj)
    within_vals: list[float] = []
    for trajs in by_word.values():
        if len(trajs) < 2:
            continue
        for i in range(len(trajs)):
            for j in range(i + 1, len(trajs)):
                n = min(len(trajs[i]), len(trajs[j]))
                if n < 2:
                    continue
                within_vals.append(
                    float(np.mean(np.linalg.norm(trajs[i][:n] - trajs[j][:n], axis=1))) / diam
                )
    within = float(np.mean(within_vals)) if within_vals else float("nan")

    return {
        "word_spread_over_diameter": spread,
        "within_word_spread_over_diameter": within,
        "per_word_spread": per_word_mean,
    }


# ---------------------------------------------------------------------------
# per-panel computation
# ---------------------------------------------------------------------------

def closed_loop_rollout_hidden(
    ctx: TaskVizContext,
    *,
    n_cycles: int = _ROLLOUT_CYCLES,
    rollout_seed: int = _ROLLOUT_SEED,
) -> np.ndarray | None:
    """Multi-cycle closed-loop rollout in full hidden space."""
    vocab_words = list(ctx.words)
    seed_letters = _trajectory_seed_letters(ctx.model, vocab_words)
    summary_seed = _closed_loop_summary_seed(vocab_words, seed_letters, spaced=ctx.spaced)
    steps = n_cycles * _one_vocab_cycle_steps(vocab_words, spaced=ctx.spaced)
    hidden, _ = rnn_closed_loop_rollout(
        ctx.model,
        seed_text=summary_seed,
        steps=steps,
        rng=np.random.default_rng(rollout_seed),
    )
    if len(hidden) < 8:
        return None
    return np.asarray(hidden, dtype=float)


def closed_loop_rollout_pc(
    ctx: TaskVizContext,
    *,
    n_cycles: int = _ROLLOUT_CYCLES,
    rollout_seed: int = _ROLLOUT_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Multi-cycle closed-loop rollout projected into teacher-forced PCA 2D.

    Returns (rollout_pc, pca_mean, pca_components) or None if too short.
    """
    hidden = closed_loop_rollout_hidden(
        ctx, n_cycles=n_cycles, rollout_seed=rollout_seed,
    )
    if hidden is None:
        return None
    _, pca_mean, pca_components, _ = fit_pca_2d_with_evr(ctx.hidden_states)
    return (hidden - pca_mean) @ pca_components.T, pca_mean, pca_components


def compute_shape_panel(ctx: TaskVizContext) -> dict[str, Any]:
    """Shape + spread metrics for one task/seed, with plot-ready arrays."""
    rollout_hidden = closed_loop_rollout_hidden(ctx)
    if rollout_hidden is None:
        return {"task": ctx.task, "seed": ctx.seed, "error": "closed-loop path too short"}
    _, pca_mean, pca_components, _ = fit_pca_2d_with_evr(ctx.hidden_states)
    rollout_pc = (rollout_hidden - pca_mean) @ pca_components.T

    shape = loop_shape_metrics(rollout_pc)
    if shape.get("error"):
        return {"task": ctx.task, "seed": ctx.seed, "error": shape["error"]}

    labeled = labeled_word_trajectories(ctx)
    labeled_pc = [(w, (t - pca_mean) @ pca_components.T) for w, t in labeled]
    spread = word_spread_metrics(labeled_pc, shape["outline_xy"])
    state_space = paired_state_space_metrics(ctx.hidden_states, rollout_hidden)

    return {
        "task": ctx.task,
        "seed": ctx.seed,
        "shape": shape,
        "spread": spread,
        "state_space": state_space,
        "rollout_pc": rollout_pc,
        "word_trajs_pc": labeled_pc,
    }


def _panel_json(panel: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe subset of a shape panel."""
    if panel.get("error"):
        return {k: panel[k] for k in ("task", "seed", "error")}
    shape = panel["shape"]
    return {
        "task": panel["task"],
        "seed": panel["seed"],
        "shape": {
            "polygon_order": shape["polygon_order"],
            "polygon_score": shape["polygon_score"],
            "polygon_spectrum": shape["polygon_spectrum"],
            "n_corners": shape["n_corners"],
            "circularity": shape["circularity"],
            "bbox_aspect": shape["bbox_aspect"],
        },
        "spread": {
            "word_spread_over_diameter": panel["spread"]["word_spread_over_diameter"],
            "within_word_spread_over_diameter": panel["spread"]["within_word_spread_over_diameter"],
            "per_word_spread": panel["spread"]["per_word_spread"],
        },
        "state_space": panel["state_space"],
    }


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------

_SHAPE_NAMES = {3: "triangle", 4: "square", 5: "pentagon", 6: "hexagon", 7: "7-gon", 8: "octagon"}


def _fmt(v: float, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}" if np.isfinite(v) else "-"


def _plot_shape_panel(ax_traj, ax_spec, panel: dict[str, Any]) -> None:
    shape = panel["shape"]
    spread = panel["spread"]
    state_space = panel["state_space"]
    corpus = state_space["corpus"]
    loop = state_space["loop"]
    rollout_pc = panel["rollout_pc"]
    word_trajs = panel["word_trajs_pc"]
    outline = shape["outline_xy"]
    outline_closed = np.vstack([outline, outline[:1]])

    words = sorted({w for w, _ in word_trajs})
    cmap = plt.get_cmap("tab20", max(len(words), 1))
    word_color = {w: cmap(i) for i, w in enumerate(words)}

    ax_traj.plot(rollout_pc[:, 0], rollout_pc[:, 1], "-", color="#9db8dd",
                 linewidth=0.55, alpha=0.55, zorder=1)
    for w, traj in word_trajs:
        ax_traj.plot(traj[:, 0], traj[:, 1], "-", color=word_color[w],
                     linewidth=0.8, alpha=0.6, zorder=2)
    ax_traj.plot(outline_closed[:, 0], outline_closed[:, 1], "-", color="#1a3a7a",
                 linewidth=1.8, alpha=0.95, zorder=3)
    corner_xy = shape["corner_xy"]
    if len(corner_xy):
        ax_traj.scatter(corner_xy[:, 0], corner_xy[:, 1], s=40, marker="o",
                        facecolors="none", edgecolors="#e07000", linewidths=1.6, zorder=4)

    all_xy = [rollout_pc] + [t for _, t in word_trajs]
    xlim, ylim = _square_data_limits(*all_xy, padding_frac=0.12)
    ax_traj.set_xlim(xlim)
    ax_traj.set_ylim(ylim)
    ax_traj.set_aspect("equal", adjustable="box")
    ax_traj.tick_params(labelsize=5, length=2)
    ax_traj.grid(True, linestyle=":", alpha=0.25)

    m = shape["polygon_order"]
    if shape["polygon_score"] < 0.5:
        name = "no clear polygon"
    else:
        name = _SHAPE_NAMES.get(m, f"{m}-gon")
    annot = (
        f"{name}  m*={m} ({_fmt(shape['polygon_score'])})\n"
        f"corners {shape['n_corners']}  circ {_fmt(shape['circularity'])}  "
        f"asp {_fmt(shape['bbox_aspect'])}\n"
        f"spread {_fmt(spread['word_spread_over_diameter'])}  "
        f"within-word {_fmt(spread['within_word_spread_over_diameter'])}\n"
        f"dim c {_fmt(corpus['effective_dim'])}  |r| c {_fmt(corpus['mean_abs_corr'])}  "
        f"dim l {_fmt(loop['effective_dim'])}  |r| l {_fmt(loop['mean_abs_corr'])}"
    )
    ax_traj.text(
        0.02, 0.98, annot, transform=ax_traj.transAxes, fontsize=5.5,
        va="top", ha="left", family="monospace",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.5},
    )

    spectrum = shape["polygon_spectrum"]
    ms = sorted(spectrum)
    vals = [spectrum[k] for k in ms]
    colors = ["#e07000" if k == m else "#88a4cc" for k in ms]
    ax_spec.bar(ms, vals, color=colors, width=0.7)
    ax_spec.set_ylim(0.0, 1.0)
    ax_spec.set_xticks(ms)
    ax_spec.tick_params(labelsize=5, length=2)
    ax_spec.grid(True, axis="y", linestyle=":", alpha=0.3)


def plot_shape_quantification(
    spec: ComparisonSpec,
    *,
    seeds: tuple[int, ...] | None = None,
    outfile: str = "shape_quantification.png",
) -> Path | None:
    """Trajectories + shape readout grid; rows = tasks (2 rows each: loop panel
    and polygon symmetry spectrum), columns = seeds.  Skips missing checkpoints."""
    requested = seeds if seeds is not None else spec.seeds
    available: dict[str, list[int]] = {}
    for task in spec.tasks:
        have = seeds_for_task(task, spec.model_type)
        task_seeds = [s for s in requested if s in have]
        if task_seeds:
            available[task] = task_seeds
    if not available:
        print(f"no checkpoints found for any of {spec.tasks} at seeds {requested}")
        return None

    tasks = [t for t in spec.tasks if t in available]
    cols = sorted({s for v in available.values() for s in v})
    n_rows, n_cols = len(tasks), len(cols)

    fig = plt.figure(figsize=(2.1 * n_cols + 0.7, 3.0 * n_rows + 0.5))
    gs = fig.add_gridspec(
        2 * n_rows, n_cols,
        height_ratios=[v for _ in range(n_rows) for v in (2.4, 0.8)],
        hspace=0.35, wspace=0.25,
    )

    panels_json: list[dict[str, Any]] = []
    for row, task in enumerate(tasks):
        for col, run_seed in enumerate(cols):
            ax_traj = fig.add_subplot(gs[2 * row, col])
            ax_spec = fig.add_subplot(gs[2 * row + 1, col])
            if row == 0:
                ax_traj.set_title(f"s{run_seed}", fontsize=8, fontweight="bold", pad=3)
            if col == 0:
                ax_traj.set_ylabel(spec.label_for(task), fontsize=8, fontweight="bold")
                ax_spec.set_ylabel("template corr", fontsize=6)
            if row == n_rows - 1:
                ax_spec.set_xlabel("polygon order m", fontsize=6)

            if run_seed not in available[task]:
                ax_traj.axis("off")
                ax_spec.axis("off")
                ax_traj.text(0.5, 0.5, "no ckpt", transform=ax_traj.transAxes,
                             fontsize=6, ha="center", va="center", color="#888888")
                continue

            ctx = load_task_viz_context(task, model_type=spec.model_type, seed=run_seed)
            panel = compute_shape_panel(ctx)
            panels_json.append(_panel_json(panel))
            if panel.get("error"):
                ax_traj.axis("off")
                ax_spec.axis("off")
                ax_traj.text(0.5, 0.5, panel["error"], transform=ax_traj.transAxes,
                             fontsize=5, ha="center", va="center", color="#888888")
                continue
            _plot_shape_panel(ax_traj, ax_spec, panel)

    fig.suptitle(
        f"{spec.display_title}: shape quantification (PCA 2D, {spec.model_type})\n"
        "closed-loop rollout (light blue) + word trajectories (colored) + "
        "radial outline (navy) | o = detected corners | m* = polygon order readout",
        fontsize=9, y=0.995,
    )
    out_dir = comparison_dir(spec.name, "trajectories")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / outfile
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    json_path = out_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "comparison": spec.name,
                "model_type": spec.model_type,
                "tasks": tasks,
                "seeds": cols,
                "panels": panels_json,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out_path}")
    print(f"wrote {json_path}")
    return out_path
