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
    _trajectory_seed_letters,
    rnn_closed_loop_rollout,
)

_GEOMETRY_TRIALS = 8
_ROLLOUT_SEED = 0

_SUMMARY_METRICS: tuple[tuple[str, str, str], ...] = (
    ("full_space", "gap_over_diameter", "closure / diam (ℝᴴ)"),
    ("full_space", "planarity_top2", "planarity top-2 (ℝᴴ)"),
    ("full_space", "mean_min_distance_over_diameter", "word dispersion (ℝᴴ)"),
    ("pca_2d", "closure_gap_over_diameter", "closure / diam (PCA)"),
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

    return {
        "task": ctx.task,
        "seed": ctx.seed,
        "summary_seed": summary_seed,
        "summary_steps": summary_steps,
        "n_geometry_trials": n_trials,
        "full_space": {
            **_closure_metrics(mean_loop),
            "planarity_top2": _planarity_top2(mean_loop),
            **_word_bundle_dispersion(word_trajs, mean_loop),
            "hidden_size": int(mean_loop.shape[1]),
        },
        "pca_2d": {
            "pc1_variance_frac": float(pca_evr[0]) if len(pca_evr) > 0 else float("nan"),
            "pc2_variance_frac": float(pca_evr[1]) if len(pca_evr) > 1 else float("nan"),
            "closure_gap": pc_closure["gap"],
            "closure_gap_over_diameter": pc_closure["gap_over_diameter"],
            "diameter": pc_closure["diameter"],
            **_word_bundle_dispersion(word_trajs_pc, mean_loop_pc),
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
    n_metrics = len(_SUMMARY_METRICS)

    seed_colors = {
        int(s): plt.cm.tab10(i % 10)
        for i, s in enumerate(run_seeds)
    }

    fig, axes = plt.subplots(
        2, 3,
        figsize=(11.5, 6.5),
        constrained_layout=True,
        squeeze=False,
    )
    for ax, (section, key, label) in zip(axes.ravel()[:n_metrics], _SUMMARY_METRICS):
        groups = _values_by_condition(panels, tasks, run_seeds, section, key)
        _plot_metric_by_condition(
            ax, groups, condition_labels, ylabel=label, seed_colors=seed_colors,
        )
    for ax in axes.ravel()[n_metrics:]:
        ax.set_visible(False)

    # seed legend on the spare panel or below
    legend_ax = axes.ravel()[n_metrics]
    legend_ax.set_visible(True)
    legend_ax.axis("off")
    for seed in run_seeds:
        legend_ax.scatter([], [], color=seed_colors[int(seed)], s=40, label=f"seed {seed}")
    legend_ax.legend(loc="center", fontsize=9, title="seeds", frameon=False)

    fig.suptitle(
        f"{spec.display_title}: closed-loop geometry ({spec.model_type})",
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
    fig_path = plot_trajectory_geometry_summary(spec, panels, seeds=run_seeds)
    print(f"wrote {fig_path}")
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
    return plot_trajectory_geometry_summary(
        spec, payload["panels"], seeds=run_seeds, outfile=outfile,
    )
