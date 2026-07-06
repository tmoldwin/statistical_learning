"""Heatmap summaries for powers-of-2 word-count × length sweep."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from experiment import comparison_dir, checkpoint_path, TASKS
from viz.compare.geometry import _metric_value, compute_panel_geometry
from viz.compare._data import load_task_viz_context
from viz.compare.sweep_heatmap import (
    _aggregate_matrix,
    _aggregate_training_matrix,
    _format_heatmap_cell,
    _iter_to_word_error_threshold,
    _plot_heatmap_panel,
    _word_error_threshold,
)
from viz.compare.word_uniformity import measure_output_word_uniformity
from viz.plot_layout import finalize_grid_figure, save_figure
from vocab_sweep_pow2 import (
    POW2_DEFAULT_SEEDS,
    POW2_LENGTHS,
    POW2_WORD_COUNTS,
    iter_pow2_sweep_cells,
    task_name,
)

POW2_SWEEP_COMPARISON_NAME = "word_count_pow2_sweep_ns"

_HEATMAP_METRICS: tuple[tuple[str, str, str], ...] = (
    ("shape", "polygon_score", "polygon score"),
    ("shape", "polygon_order", "polygon order m*"),
    ("shape", "circularity", "circularity"),
    ("state_space", "loop_effective_dim", "eff dim (loop)"),
    ("state_space", "corpus_mean_abs_corr", "mean |r| (corpus)"),
    ("full_space", "planarity_top2", "top-2 PC var"),
    ("full_space", "turn_regularity", "turn regularity"),
    ("jpca", "omega", "jPCA rate"),
)

_TRAINING_HEATMAP_METRICS: tuple[tuple[str, str, str, bool], ...] = (
    ("demo_word_error_pct", "word error (demo %)", "YlOrRd_r", False),
    ("iter_to_threshold", "iters to 3% word err (log)", "YlOrRd_r", True),
    ("uniform_tv_distance", "TV dist from uniform", "YlGn_r", False),
)


def _training_panel_from_checkpoint(
    task: str,
    *,
    n_words: int,
    length: int,
    seed: int,
    model_type: str = "rnn",
) -> dict[str, Any]:
    ckpt = checkpoint_path(task, model_type, seed=seed)
    if not ckpt.is_file():
        return {
            "task": task,
            "n_words": n_words,
            "length": length,
            "seed": seed,
            "error": "missing checkpoint",
        }
    data = np.load(ckpt, allow_pickle=True)
    threshold = _word_error_threshold(task)
    demo_err = float(data["demo_word_error_frac"])
    best_err = float(data["best_metric_word_error_frac"])
    total_iters = int(data["loss_iterations"].shape[0])
    iter_to_threshold = _iter_to_word_error_threshold(
        data["metric_iterations"],
        data["metric_word_error_frac"],
        threshold=threshold,
    )
    if not np.isfinite(iter_to_threshold):
        iter_to_threshold = float(total_iters)

    panel: dict[str, Any] = {
        "task": task,
        "n_words": n_words,
        "length": length,
        "seed": seed,
        "demo_word_error_frac": demo_err,
        "demo_word_error_pct": 100.0 * demo_err,
        "best_metric_word_error_frac": best_err,
        "best_metric_word_error_pct": 100.0 * best_err,
        "best_metric_iter": int(data["best_metric_iter"]),
        "iter_to_threshold": iter_to_threshold,
        "reached_word_error_target": bool(best_err <= threshold),
        "target_word_error_frac": threshold,
        "total_iters": total_iters,
    }

    try:
        ctx = load_task_viz_context(task, model_type=model_type, seed=seed)
        uniformity = measure_output_word_uniformity(
            ctx.model, ctx.words, task=task, seed=seed,
        )
        panel.update(uniformity)
    except (FileNotFoundError, KeyError) as exc:
        panel["uniformity_error"] = str(exc)

    return panel


def write_pow2_sweep_training_metrics(
    *,
    seeds: tuple[int, ...] | None = None,
    model_type: str = "rnn",
    outfile: str = "sweep_training.json",
) -> Path:
    run_seeds = seeds if seeds is not None else POW2_DEFAULT_SEEDS
    panels: list[dict[str, Any]] = []

    for n_words, length in iter_pow2_sweep_cells():
        task = task_name(n_words, length)
        for run_seed in run_seeds:
            panels.append(_training_panel_from_checkpoint(
                task,
                n_words=n_words,
                length=length,
                seed=run_seed,
                model_type=model_type,
            ))
            print(f"  training {task} seed {run_seed}", flush=True)

    out_dir = comparison_dir(POW2_SWEEP_COMPARISON_NAME, "trajectories")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / outfile
    payload = {
        "comparison": POW2_SWEEP_COMPARISON_NAME,
        "model_type": model_type,
        "word_counts": list(POW2_WORD_COUNTS),
        "lengths": list(POW2_LENGTHS),
        "seeds": list(run_seeds),
        "uniformity_note": (
            "uniform_tv_distance / uniform_kl_divergence / uniform_l2_distance "
            "compare closed-loop word frequencies to training uniform 1/K"
        ),
        "panels": panels,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def write_pow2_sweep_geometry(
    *,
    seeds: tuple[int, ...] | None = None,
    model_type: str = "rnn",
    outfile: str = "sweep_geometry.json",
) -> Path:
    run_seeds = seeds if seeds is not None else POW2_DEFAULT_SEEDS
    panels: list[dict[str, Any]] = []

    for n_words, length in iter_pow2_sweep_cells():
        task = task_name(n_words, length)
        for run_seed in run_seeds:
            try:
                ctx = load_task_viz_context(task, model_type=model_type, seed=run_seed)
            except FileNotFoundError:
                panels.append({
                    "task": task,
                    "n_words": n_words,
                    "length": length,
                    "seed": run_seed,
                    "error": "missing checkpoint",
                })
                continue
            except KeyError:
                panels.append({
                    "task": task,
                    "n_words": n_words,
                    "length": length,
                    "seed": run_seed,
                    "error": "checkpoint vocab mismatch",
                })
                continue
            panel = compute_panel_geometry(ctx)
            panel["n_words"] = n_words
            panel["length"] = length
            panels.append(panel)
            print(f"  geometry {task} seed {run_seed}", flush=True)

    out_dir = comparison_dir(POW2_SWEEP_COMPARISON_NAME, "trajectories")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / outfile
    payload = {
        "comparison": POW2_SWEEP_COMPARISON_NAME,
        "model_type": model_type,
        "word_counts": list(POW2_WORD_COUNTS),
        "lengths": list(POW2_LENGTHS),
        "seeds": list(run_seeds),
        "panels": panels,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def plot_pow2_sweep_heatmaps(
    panels: list[dict[str, Any]],
    *,
    training_panels: list[dict[str, Any]] | None = None,
    seeds: tuple[int, ...] | None = None,
    word_counts: tuple[int, ...] = POW2_WORD_COUNTS,
    lengths: tuple[int, ...] = POW2_LENGTHS,
    outfile: str = "sweep_heatmaps.png",
) -> Path:
    run_seeds = seeds if seeds is not None else POW2_DEFAULT_SEEDS
    n_training = len(_TRAINING_HEATMAP_METRICS) if training_panels else 0
    n_metrics = len(_HEATMAP_METRICS) + n_training
    n_cols = 4
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.4 * n_cols, 2.8 * n_rows),
        squeeze=False,
    )

    panel_idx = 0
    for section, key, title in _HEATMAP_METRICS:
        ax = axes.ravel()[panel_idx]
        panel_idx += 1
        means, _sems = _aggregate_matrix(
            panels,
            word_counts=word_counts,
            lengths=lengths,
            seeds=run_seeds,
            section=section,
            key=key,
        )
        _plot_heatmap_panel(
            ax, means,
            title=title,
            cmap="YlOrRd",
            word_counts=word_counts,
            lengths=lengths,
            value_key=key,
        )

    if training_panels is not None:
        for key, title, cmap, log_scale in _TRAINING_HEATMAP_METRICS:
            ax = axes.ravel()[panel_idx]
            panel_idx += 1
            means, _sems = _aggregate_training_matrix(
                training_panels,
                word_counts=word_counts,
                lengths=lengths,
                seeds=run_seeds,
                key=key,
            )
            _plot_heatmap_panel(
                ax, means,
                title=title,
                cmap=cmap,
                word_counts=word_counts,
                lengths=lengths,
                value_key=key,
                log_scale=log_scale,
            )

    for ax in axes.ravel()[panel_idx:]:
        ax.axis("off")

    finalize_grid_figure(
        fig,
        suptitle=f"Powers-of-2 word-count × length sweep (RNN, n={len(run_seeds)} seeds)",
        top=0.92,
        hspace=0.55,
        wspace=0.45,
    )
    out_dir = comparison_dir(POW2_SWEEP_COMPARISON_NAME, "trajectories")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / outfile
    save_figure(fig, out_path, dpi=160)
    return out_path


def replot_pow2_sweep_heatmaps(
    *,
    geometry_file: str = "sweep_geometry.json",
    training_file: str = "sweep_training.json",
    outfile: str = "sweep_heatmaps.png",
) -> Path:
    out_dir = comparison_dir(POW2_SWEEP_COMPARISON_NAME, "trajectories")
    geom_payload = json.loads((out_dir / geometry_file).read_text(encoding="utf-8"))
    train_path = out_dir / training_file
    training_panels = None
    if train_path.is_file():
        train_payload = json.loads(train_path.read_text(encoding="utf-8"))
        training_panels = train_payload["panels"]
    return plot_pow2_sweep_heatmaps(
        geom_payload["panels"],
        training_panels=training_panels,
        seeds=tuple(geom_payload["seeds"]),
        outfile=outfile,
    )


def run_pow2_sweep_plots(
    *,
    seeds: tuple[int, ...] | None = None,
    json_file: str = "sweep_geometry.json",
    geometry: bool = True,
    training: bool = True,
) -> tuple[Path, ...]:
    outputs: list[Path] = []
    training_panels: list[dict[str, Any]] | None = None
    run_seeds = seeds if seeds is not None else POW2_DEFAULT_SEEDS

    if training:
        train_json = write_pow2_sweep_training_metrics(seeds=run_seeds)
        train_payload = json.loads(train_json.read_text(encoding="utf-8"))
        training_panels = train_payload["panels"]
        outputs.append(train_json)
        print(f"wrote {train_json}")

    if geometry:
        json_path = write_pow2_sweep_geometry(seeds=run_seeds, outfile=json_file)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        heatmap_path = plot_pow2_sweep_heatmaps(
            payload["panels"],
            training_panels=training_panels,
            seeds=tuple(payload["seeds"]),
        )
        print(f"wrote {json_path}")
        print(f"wrote {heatmap_path}")
        outputs.extend([json_path, heatmap_path])
    elif training_panels is not None:
        out_dir = comparison_dir(POW2_SWEEP_COMPARISON_NAME, "trajectories")
        geom_path = out_dir / json_file
        if geom_path.is_file():
            payload = json.loads(geom_path.read_text(encoding="utf-8"))
            heatmap_path = plot_pow2_sweep_heatmaps(
                payload["panels"],
                training_panels=training_panels,
                seeds=tuple(payload["seeds"]),
            )
            print(f"wrote {heatmap_path}")
            outputs.append(heatmap_path)
    return tuple(outputs)
