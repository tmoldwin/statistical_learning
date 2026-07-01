"""Quantify closed-loop trajectory geometry (full hidden space + fixed PCA 2D)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiment import comparison_dir
from viz.compare._data import TaskVizContext, load_task_viz_context
from viz.compare.spec import ComparisonSpec
from viz.dimred import fit_jpca_components, fit_pca_2d_with_evr
from visualize import (
    _closed_loop_summary_seed,
    _embed_trajectories_for_text,
    _one_vocab_cycle_steps,
    _same_length_average_trajectory,
    _square_data_limits,
    _trajectory_seed_letters,
    rnn_closed_loop_rollout,
)

_GEOMETRY_TRIALS = 8
_ROLLOUT_SEED = 0

_SUMMARY_METRICS: tuple[tuple[str, str, str], ...] = (
    ("full_space", "gap_over_diameter", "closure / diam (ℝᴴ)"),
    ("full_space", "planarity_top2", "planarity top-2 (ℝᴴ)"),
    ("full_space", "turn_regularity", "turn regularity (ℝᴴ)"),
    ("pca_2d", "closure_gap_over_diameter", "closure / diam (PCA)"),
    ("pca_2d", "bbox_aspect", "bbox aspect (PCA)"),
    ("jpca", "omega", "jPCA ω"),
)


def _path_diameter(path: np.ndarray) -> float:
    if len(path) < 2:
        return 0.0
    return float(np.linalg.norm(path.max(axis=0) - path.min(axis=0)))


def _closure_metrics(path: np.ndarray) -> dict[str, float]:
    if len(path) < 2:
        return {"gap": float("nan"), "gap_over_diameter": float("nan"), "diameter": 0.0}
    gap = float(np.linalg.norm(path[-1] - path[0]))
    diam = _path_diameter(path)
    ratio = gap / diam if diam > 1e-12 else float("nan")
    return {"gap": gap, "gap_over_diameter": ratio, "diameter": diam}


def _planarity_top2(points: np.ndarray) -> float:
    """Fraction of variance in the top two principal components of ``points``."""
    points = np.asarray(points, dtype=float)
    if points.shape[0] < 3:
        return float("nan")
    centered = points - points.mean(axis=0)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    var = s * s
    total = float(var.sum())
    if total <= 1e-12:
        return float("nan")
    n = min(2, len(var))
    return float(var[:n].sum() / total)


def _mean_min_point_distance(path_a: np.ndarray, path_b: np.ndarray) -> float:
    if len(path_a) == 0 or len(path_b) == 0:
        return float("nan")
    dists = [float(np.min(np.linalg.norm(path_b - p, axis=1))) for p in path_a]
    return float(np.mean(dists))


def _word_bundle_dispersion(
    word_paths: list[np.ndarray],
    mean_loop: np.ndarray,
) -> dict[str, float]:
    if not word_paths or len(mean_loop) < 2:
        return {"mean_min_distance": float("nan"), "mean_min_distance_over_diameter": float("nan")}
    raw = float(np.mean([_mean_min_point_distance(w, mean_loop) for w in word_paths]))
    diam = _path_diameter(mean_loop)
    norm = raw / diam if diam > 1e-12 else float("nan")
    return {"mean_min_distance": raw, "mean_min_distance_over_diameter": norm}


def _bbox_aspect(path: np.ndarray) -> float:
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return float("nan")
    xr = float(path[:, 0].max() - path[:, 0].min())
    yr = float(path[:, 1].max() - path[:, 1].min()) if path.shape[1] > 1 else 0.0
    short = min(xr, yr)
    long = max(xr, yr)
    if short <= 1e-12:
        return float("nan")
    return long / short


def _turning_angle_stats(path: np.ndarray) -> dict[str, float]:
    """Interior turn angles; high regularity ≈ equal corners (square, triangle, …)."""
    path = np.asarray(path, dtype=float)
    if len(path) < 4:
        return {
            "turn_angle_std": float("nan"),
            "turn_regularity": float("nan"),
            "n_corners": 0.0,
        }
    seg = path[1:] - path[:-1]
    seg_len = np.linalg.norm(seg, axis=1, keepdims=True)
    seg_len = np.maximum(seg_len, 1e-12)
    unit = seg / seg_len
    dots = np.sum(unit[:-1] * unit[1:], axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    turns = np.arccos(dots)
    std = float(np.std(turns))
    mean_abs = float(np.mean(np.abs(turns)))
    regularity = 1.0 / (1.0 + std) if np.isfinite(std) else float("nan")
    corner_thresh = float(np.percentile(turns, 75))
    n_corners = float(np.sum(turns >= corner_thresh))
    return {
        "turn_angle_std": std,
        "turn_regularity": regularity,
        "n_corners": n_corners,
        "mean_turn_angle": mean_abs,
    }


def _mean_closed_loop_and_pca(
    ctx: TaskVizContext,
    *,
    rollout_seed: int = _ROLLOUT_SEED,
    n_trials: int = _GEOMETRY_TRIALS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Mean closed loop in ℝᴴ and teacher-forced PCA 2D; None if path too short."""
    vocab_words = list(ctx.words)
    seed_letters = _trajectory_seed_letters(ctx.model, vocab_words)
    summary_seed = _closed_loop_summary_seed(vocab_words, seed_letters, spaced=ctx.spaced)
    summary_steps = _one_vocab_cycle_steps(vocab_words, spaced=ctx.spaced)
    mean_loop = _mean_closed_loop_hidden(
        ctx.model,
        summary_seed=summary_seed,
        steps=summary_steps,
        rollout_seed=rollout_seed,
        n_trials=n_trials,
    )
    if mean_loop is None or len(mean_loop) < 2:
        return None
    _, pca_mean, pca_components, _ = fit_pca_2d_with_evr(ctx.hidden_states)
    mean_loop_pc = (mean_loop - pca_mean) @ pca_components.T
    return mean_loop, mean_loop_pc, pca_mean, pca_components


def _mean_closed_loop_hidden(
    model: dict,
    *,
    summary_seed: str,
    steps: int,
    rollout_seed: int,
    n_trials: int,
) -> np.ndarray | None:
    trials: list[np.ndarray] = []
    for trial in range(max(1, n_trials)):
        rng = np.random.default_rng(int(rollout_seed) + trial)
        hidden, _ = rnn_closed_loop_rollout(
            model, seed_text=summary_seed, steps=steps, rng=rng,
        )
        if len(hidden) >= 2:
            trials.append(hidden)
    return _same_length_average_trajectory(trials)


def compute_panel_geometry(
    ctx: TaskVizContext,
    *,
    rollout_seed: int = _ROLLOUT_SEED,
    n_trials: int = _GEOMETRY_TRIALS,
) -> dict[str, Any]:
    """Geometry for one task/seed: full ℝ^H and fixed teacher-forced PCA 2D."""
    vocab_words = list(ctx.words)
    seed_letters = _trajectory_seed_letters(ctx.model, vocab_words)
    summary_seed = _closed_loop_summary_seed(vocab_words, seed_letters, spaced=ctx.spaced)
    summary_steps = _one_vocab_cycle_steps(vocab_words, spaced=ctx.spaced)

    word_trajs = _embed_trajectories_for_text(
        ctx.text, ctx.hidden_states, spaced=ctx.spaced, words=vocab_words,
    )
    word_trajs = [np.asarray(t, dtype=float) for t in word_trajs if len(t) >= 2]

    mean_loop = _mean_closed_loop_hidden(
        ctx.model,
        summary_seed=summary_seed,
        steps=summary_steps,
        rollout_seed=rollout_seed,
        n_trials=n_trials,
    )
    if mean_loop is None or len(mean_loop) < 2:
        return {
            "task": ctx.task,
            "seed": ctx.seed,
            "summary_seed": summary_seed,
            "summary_steps": summary_steps,
            "error": "closed-loop path too short",
        }

    _, pca_mean, pca_components, pca_evr = fit_pca_2d_with_evr(ctx.hidden_states)
    mean_loop_pc = (mean_loop - pca_mean) @ pca_components.T
    word_trajs_pc = [(w - pca_mean) @ pca_components.T for w in word_trajs]

    jpca_omega: float | None = None
    try:
        _, _, rates = fit_jpca_components(word_trajs, num_jpcs=2)
        if len(rates):
            jpca_omega = float(rates[0])
    except ValueError:
        pass

    full_closure = _closure_metrics(mean_loop)
    pc_closure = _closure_metrics(mean_loop_pc)
    full_turns = _turning_angle_stats(mean_loop)
    pc_turns = _turning_angle_stats(mean_loop_pc)

    return {
        "task": ctx.task,
        "seed": ctx.seed,
        "summary_seed": summary_seed,
        "summary_steps": summary_steps,
        "n_geometry_trials": n_trials,
        "full_space": {
            **full_closure,
            "planarity_top2": _planarity_top2(mean_loop),
            **_word_bundle_dispersion(word_trajs, mean_loop),
            **full_turns,
            "hidden_size": int(mean_loop.shape[1]),
        },
        "pca_2d": {
            "pc1_variance_frac": float(pca_evr[0]) if len(pca_evr) > 0 else float("nan"),
            "pc2_variance_frac": float(pca_evr[1]) if len(pca_evr) > 1 else float("nan"),
            "closure_gap": pc_closure["gap"],
            "closure_gap_over_diameter": pc_closure["gap_over_diameter"],
            "diameter": pc_closure["diameter"],
            "bbox_aspect": _bbox_aspect(mean_loop_pc),
            **_word_bundle_dispersion(word_trajs_pc, mean_loop_pc),
            **pc_turns,
        },
        "jpca": {
            "omega": jpca_omega,
        },
    }


def _metric_value(panel: dict[str, Any], section: str, key: str) -> float:
    if panel.get("error"):
        return float("nan")
    block = panel.get(section)
    if not isinstance(block, dict):
        return float("nan")
    val = block.get(key)
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def _values_by_condition(
    panels: list[dict[str, Any]],
    tasks: tuple[str, ...],
    seeds: tuple[int, ...],
    section: str,
    key: str,
) -> list[tuple[list[float], list[int]]]:
    """Per condition: finite metric values and their seed ids."""
    lookup = {(p.get("task"), p.get("seed")): p for p in panels}
    out: list[tuple[list[float], list[int]]] = []
    for task in tasks:
        vals: list[float] = []
        seed_ids: list[int] = []
        for seed in seeds:
            panel = lookup.get((task, seed))
            v = _metric_value(panel, section, key) if panel is not None else float("nan")
            if np.isfinite(v):
                vals.append(v)
                seed_ids.append(int(seed))
        out.append((vals, seed_ids))
    return out


def _plot_metric_by_condition(
    ax,
    groups: list[tuple[list[float], list[int]]],
    condition_labels: list[str],
    *,
    ylabel: str,
    seed_colors: dict[int, tuple],
) -> None:
    positions = np.arange(len(groups))
    box_data = [g[0] for g in groups]
    ax.boxplot(
        box_data,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
        boxprops={"facecolor": "#e8e8e8", "edgecolor": "#666666", "linewidth": 1.0},
        whiskerprops={"color": "#666666", "linewidth": 1.0},
        capprops={"color": "#666666", "linewidth": 1.0},
    )
    jitter_rng = np.random.default_rng(0)
    for i, (vals, seed_ids) in enumerate(groups):
        if not vals:
            continue
        jitter = jitter_rng.uniform(-0.1, 0.1, size=len(vals))
        for xoff, y, seed in zip(jitter, vals, seed_ids):
            ax.scatter(
                i + xoff, y,
                s=36, alpha=0.9, zorder=3,
                color=seed_colors.get(seed, "C0"),
                edgecolors="white", linewidths=0.6,
            )
    ax.set_xticks(positions)
    ax.set_xticklabels(condition_labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)


def _fmt_metric(v: float, *, decimals: int = 2) -> str:
    if not np.isfinite(v):
        return "—"
    return f"{v:.{decimals}f}"


def _panel_metric_oneline(panel: dict[str, Any]) -> str:
    if panel.get("error"):
        return str(panel["error"])
    return (
        f"ℝᴴ {_fmt_metric(_metric_value(panel, 'full_space', 'gap_over_diameter'))}  "
        f"plan {_fmt_metric(_metric_value(panel, 'full_space', 'planarity_top2'))}  "
        f"turn {_fmt_metric(_metric_value(panel, 'full_space', 'turn_regularity'))}\n"
        f"PCA {_fmt_metric(_metric_value(panel, 'pca_2d', 'closure_gap_over_diameter'))}  "
        f"asp {_fmt_metric(_metric_value(panel, 'pca_2d', 'bbox_aspect'))}  "
        f"ω {_fmt_metric(_metric_value(panel, 'jpca', 'omega'), decimals=0)}"
    )


def _panel_metric_annotation(panel: dict[str, Any]) -> str:
    return _panel_metric_oneline(panel)


def _plot_mean_loop_2d(ax, path_pc: np.ndarray) -> None:
    path_pc = np.asarray(path_pc, dtype=float)
    if len(path_pc) < 2:
        return
    ax.plot(path_pc[:, 0], path_pc[:, 1], "-", color="#2255aa", linewidth=1.1, alpha=0.92)
    ax.scatter(path_pc[0, 0], path_pc[0, 1], s=14, color="#22aa22", zorder=3)
    ax.scatter(path_pc[-1, 0], path_pc[-1, 1], s=14, color="#cc3333", zorder=3)
    xlim, ylim = _square_data_limits(path_pc, padding_frac=0.15)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=5, length=2)
    ax.grid(True, linestyle=":", alpha=0.25)


def plot_geometry_examples(
    spec: ComparisonSpec,
    panels: list[dict[str, Any]],
    *,
    seeds: tuple[int, ...] | None = None,
    outfile: str = "geometry_examples.png",
) -> Path:
    """Grid of mean closed loops (PCA 2D) with per-seed metrics annotated."""
    run_seeds = seeds if seeds is not None else spec.seeds
    tasks = list(spec.tasks)
    panel_lookup = {(p.get("task"), p.get("seed")): p for p in panels}
    n_rows = len(tasks)
    n_cols = len(run_seeds)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(0.95 * n_cols + 0.4, 2.35 * n_rows + 0.6),
        squeeze=False,
        constrained_layout=True,
    )

    for row_idx, task in enumerate(tasks):
        for col_idx, run_seed in enumerate(run_seeds):
            ax = axes[row_idx, col_idx]
            panel = panel_lookup.get((task, run_seed), {"error": "missing"})

            if row_idx == 0:
                ax.set_title(f"s{run_seed}", fontsize=8, fontweight="bold", pad=3)
            if col_idx == 0:
                ax.set_ylabel(spec.label_for(task), fontsize=8, fontweight="bold")

            if panel.get("error"):
                ax.axis("off")
                ax.set_xlabel(panel["error"], fontsize=5, labelpad=2)
                continue

            try:
                ctx = load_task_viz_context(
                    task, model_type=spec.model_type, seed=run_seed,
                )
                paths = _mean_closed_loop_and_pca(ctx)
            except FileNotFoundError:
                ax.axis("off")
                ax.set_xlabel("no ckpt", fontsize=5, labelpad=2)
                continue

            if paths is None:
                ax.axis("off")
                ax.set_xlabel("short path", fontsize=5, labelpad=2)
                continue

            _mean_loop, mean_loop_pc, _, _ = paths
            _plot_mean_loop_2d(ax, mean_loop_pc)
            ax.set_xlabel(_panel_metric_oneline(panel), fontsize=5, labelpad=3, linespacing=1.15)

    fig.suptitle(
        f"{spec.display_title}: closed-loop examples + metrics (PCA 2D, {spec.model_type})",
        fontsize=10,
    )
    out_dir = comparison_dir(spec.name, "trajectories")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / outfile
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_trajectory_geometry_summary(
    spec: ComparisonSpec,
    panels: list[dict[str, Any]],
    *,
    seeds: tuple[int, ...] | None = None,
    outfile: str = "geometry_summary.png",
) -> Path:
    """Box-and-whisker by condition; points = individual seeds."""
    run_seeds = seeds if seeds is not None else spec.seeds
    tasks = tuple(spec.tasks)
    condition_labels = [spec.label_for(t) for t in tasks]

    seed_colors = {
        int(s): plt.cm.tab20(i % 20)
        for i, s in enumerate(run_seeds)
    }

    fig, axes = plt.subplots(
        2, 3,
        figsize=(12.0, 6.8),
        constrained_layout=True,
        squeeze=False,
    )
    for ax, (section, key, label) in zip(axes.ravel(), _SUMMARY_METRICS):
        groups = _values_by_condition(panels, tasks, run_seeds, section, key)
        _plot_metric_by_condition(
            ax, groups, condition_labels, ylabel=label, seed_colors=seed_colors,
        )

    fig.suptitle(
        f"{spec.display_title}: closed-loop geometry ({spec.model_type}, n={len(run_seeds)} seeds)",
        fontsize=11,
    )
    out_dir = comparison_dir(spec.name, "trajectories")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / outfile
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_trajectory_geometry(
    spec: ComparisonSpec,
    *,
    seeds: tuple[int, ...] | None = None,
    outfile: str = "trajectory_geometry.json",
) -> Path:
    """Write per-panel geometry summary for all tasks × seeds in a comparison spec."""
    run_seeds = seeds if seeds is not None else spec.seeds
    panels: list[dict[str, Any]] = []

    for task in spec.tasks:
        for run_seed in run_seeds:
            try:
                ctx = load_task_viz_context(
                    task, model_type=spec.model_type, seed=run_seed,
                )
            except FileNotFoundError:
                panels.append({
                    "task": task,
                    "seed": run_seed,
                    "error": "missing checkpoint",
                })
                continue
            panels.append(compute_panel_geometry(ctx))

    out_dir = comparison_dir(spec.name, "trajectories")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / outfile
    payload = {
        "comparison": spec.name,
        "model_type": spec.model_type,
        "seeds": list(run_seeds),
        "tasks": list(spec.tasks),
        "panels": panels,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary_path = plot_trajectory_geometry_summary(spec, panels, seeds=run_seeds)
    examples_path = plot_geometry_examples(spec, panels, seeds=run_seeds)
    print(f"wrote {summary_path}")
    print(f"wrote {examples_path}")
    return out_path


def replot_trajectory_geometry_summary(
    spec: ComparisonSpec,
    *,
    seeds: tuple[int, ...] | None = None,
    json_file: str = "trajectory_geometry.json",
    outfile: str = "geometry_summary.png",
) -> Path:
    """Rebuild summary figure from an existing ``trajectory_geometry.json``."""
    run_seeds = seeds if seeds is not None else spec.seeds
    json_path = comparison_dir(spec.name, "trajectories") / json_file
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    plot_trajectory_geometry_summary(
        spec, payload["panels"], seeds=run_seeds, outfile=outfile,
    )
    return plot_geometry_examples(
        spec, payload["panels"], seeds=run_seeds,
    )
