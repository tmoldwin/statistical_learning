"""Heatmap summaries for word-count × length sweep experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from experiment import checkpoint_path, TASKS
from viz.compare.sweep_output import sweep_data_dir, sweep_figures_dir
from viz.compare.geometry import _metric_value, compute_panel_geometry
from viz.compare._data import load_task_viz_context
from viz.plot_layout import finalize_grid_figure, save_figure
from vocab_sweep import (
    SWEEP_DEFAULT_SEEDS,
    SWEEP_LENGTHS,
    SWEEP_WORD_COUNTS,
    iter_sweep_cells,
    task_name,
)

SWEEP_COMPARISON_NAME = "word_length_sweep_ns"

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
)

def _word_error_threshold(task: str) -> float:
    return float(TASKS[task].get("target_word_error_frac", 0.03))


def _iter_to_word_error_threshold(
    metric_iterations: np.ndarray,
    metric_word_error_frac: np.ndarray,
    *,
    threshold: float,
) -> float:
    """First eval iteration at or below ``threshold``; NaN if never reached."""
    for iteration, word_err in zip(metric_iterations, metric_word_error_frac):
        if float(word_err) <= threshold:
            return float(iteration)
    return float("nan")


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
    return {
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


def write_sweep_training_metrics(
    *,
    seeds: tuple[int, ...] | None = None,
    model_type: str = "rnn",
    outfile: str = "sweep_training.json",
) -> Path:
    """Read word-error / convergence scalars from sweep checkpoints."""
    run_seeds = seeds if seeds is not None else SWEEP_DEFAULT_SEEDS
    panels: list[dict[str, Any]] = []

    for n_words, length in iter_sweep_cells():
        task = task_name(n_words, length)
        for run_seed in run_seeds:
            panels.append(_training_panel_from_checkpoint(
                task,
                n_words=n_words,
                length=length,
                seed=run_seed,
                model_type=model_type,
            ))

    out_path = sweep_data_dir(SWEEP_COMPARISON_NAME) / outfile
    payload = {
        "comparison": SWEEP_COMPARISON_NAME,
        "model_type": model_type,
        "word_counts": list(SWEEP_WORD_COUNTS),
        "lengths": list(SWEEP_LENGTHS),
        "seeds": list(run_seeds),
        "panels": panels,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def write_sweep_geometry(
    *,
    seeds: tuple[int, ...] | None = None,
    model_type: str = "rnn",
    outfile: str = "sweep_geometry.json",
) -> Path:
    """Compute geometry panels for every sweep cell × seed."""
    run_seeds = seeds if seeds is not None else SWEEP_DEFAULT_SEEDS
    panels: list[dict[str, Any]] = []

    for n_words, length in iter_sweep_cells():
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
            print(f"  {task} seed {run_seed}", flush=True)

    out_path = sweep_data_dir(SWEEP_COMPARISON_NAME) / outfile
    payload = {
        "comparison": SWEEP_COMPARISON_NAME,
        "model_type": model_type,
        "word_counts": list(SWEEP_WORD_COUNTS),
        "lengths": list(SWEEP_LENGTHS),
        "seeds": list(run_seeds),
        "panels": panels,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _aggregate_matrix(
    panels: list[dict[str, Any]],
    *,
    word_counts: tuple[int, ...],
    lengths: tuple[int, ...],
    seeds: tuple[int, ...],
    section: str,
    key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, sem) arrays shaped (len(lengths), len(word_counts))."""
    lookup = {
        (p.get("n_words"), p.get("length"), p.get("seed")): p
        for p in panels
    }
    means = np.full((len(lengths), len(word_counts)), np.nan)
    sems = np.full_like(means, np.nan)
    for li, length in enumerate(lengths):
        for wi, n_words in enumerate(word_counts):
            vals: list[float] = []
            for seed in seeds:
                panel = lookup.get((n_words, length, seed))
                if panel is None:
                    continue
                v = _metric_value(panel, section, key)
                if np.isfinite(v):
                    vals.append(v)
            if vals:
                arr = np.asarray(vals, dtype=float)
                means[li, wi] = float(np.mean(arr))
                sems[li, wi] = (
                    float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
                )
    return means, sems


def _aggregate_training_matrix(
    panels: list[dict[str, Any]],
    *,
    word_counts: tuple[int, ...],
    lengths: tuple[int, ...],
    seeds: tuple[int, ...],
    key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, sem) for a top-level training scalar."""
    lookup = {
        (p.get("n_words"), p.get("length"), p.get("seed")): p
        for p in panels
    }
    means = np.full((len(lengths), len(word_counts)), np.nan)
    sems = np.full_like(means, np.nan)
    for li, length in enumerate(lengths):
        for wi, n_words in enumerate(word_counts):
            vals: list[float] = []
            for seed in seeds:
                panel = lookup.get((n_words, length, seed))
                if panel is None or panel.get("error"):
                    continue
                v = panel.get(key)
                if v is not None and np.isfinite(v):
                    vals.append(float(v))
            if vals:
                arr = np.asarray(vals, dtype=float)
                means[li, wi] = float(np.mean(arr))
                sems[li, wi] = (
                    float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
                )
    return means, sems


def _format_heatmap_cell(key: str, val: float) -> str:
    if key == "iter_to_threshold":
        if val >= 10_000:
            return f"{val / 1000:.1f}k"
        return f"{int(round(val))}"
    return f"{val:.2f}"


def _length_tick_label(length: object) -> str:
    if isinstance(length, (int, np.integer)):
        return f"{int(length)}-letter"
    return str(length)


def _plot_heatmap_panel(
    ax: plt.Axes,
    means: np.ndarray,
    *,
    title: str,
    cmap: str,
    word_counts: tuple[int, ...],
    lengths: tuple[int, ...],
    value_key: str,
    log_scale: bool = False,
) -> None:
    finite = means[np.isfinite(means)]
    vmin = float(np.min(finite)) if len(finite) else 0.0
    vmax = float(np.max(finite)) if len(finite) else 1.0
    if np.isclose(vmin, vmax):
        vmin -= 0.5
        vmax += 0.5
    norm = None
    if log_scale:
        vmin = max(vmin, 1.0)
        norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin * 1.01))
    im = ax.imshow(
        means,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=None if norm else vmin,
        vmax=None if norm else vmax,
        norm=norm,
    )
    ax.set_xticks(range(len(word_counts)))
    ax.set_xticklabels([str(n) for n in word_counts], fontsize=8)
    ax.set_yticks(range(len(lengths)))
    ax.set_yticklabels([_length_tick_label(L) for L in lengths], fontsize=8)
    ax.set_xlabel("# words", fontsize=8)
    ax.set_title(title, fontsize=9)
    for i in range(len(lengths)):
        for j in range(len(word_counts)):
            val = means[i, j]
            if np.isfinite(val):
                if log_scale and norm is not None:
                    mid = float(norm(vmin) + 0.55 * (norm(vmax) - norm(vmin)))
                    dark = norm(val) > mid
                else:
                    dark = val > (vmin + 0.55 * (vmax - vmin))
                ax.text(
                    j, i, _format_heatmap_cell(value_key, val),
                    ha="center", va="center", fontsize=6.5,
                    color="white" if dark else "black",
                )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_sweep_heatmaps(
    panels: list[dict[str, Any]],
    *,
    training_panels: list[dict[str, Any]] | None = None,
    seeds: tuple[int, ...] | None = None,
    word_counts: tuple[int, ...] = SWEEP_WORD_COUNTS,
    lengths: tuple[int, ...] = SWEEP_LENGTHS,
    outfile: str = "sweep_heatmaps.png",
) -> Path:
    """Grid of heatmaps: rows = letter length, cols = word count, one panel per metric."""
    run_seeds = seeds if seeds is not None else SWEEP_DEFAULT_SEEDS
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
        suptitle=f"Word-count × length sweep (RNN, n={len(run_seeds)} seeds)",
        top=0.92,
        hspace=0.55,
        wspace=0.45,
    )
    out_path = sweep_figures_dir(SWEEP_COMPARISON_NAME) / outfile
    save_figure(fig, out_path, dpi=160)
    return out_path


def plot_sweep_training_heatmaps(
    panels: list[dict[str, Any]],
    *,
    seeds: tuple[int, ...] | None = None,
    word_counts: tuple[int, ...] = SWEEP_WORD_COUNTS,
    lengths: tuple[int, ...] = SWEEP_LENGTHS,
    outfile: str = "sweep_heatmaps.png",
) -> Path:
    """Deprecated alias — training panels are included in ``plot_sweep_heatmaps``."""
    return plot_sweep_heatmaps(
        [],
        training_panels=panels,
        seeds=seeds,
        word_counts=word_counts,
        lengths=lengths,
        outfile=outfile,
    )


def replot_sweep_heatmaps(
    *,
    geometry_file: str = "sweep_geometry.json",
    training_file: str = "sweep_training.json",
    outfile: str = "sweep_heatmaps.png",
) -> Path:
    """Regenerate combined heatmaps from existing JSON summaries."""
    data_dir = sweep_data_dir(SWEEP_COMPARISON_NAME)
    geom_payload = json.loads((data_dir / geometry_file).read_text(encoding="utf-8"))
    train_path = data_dir / training_file
    training_panels = None
    if train_path.is_file():
        train_payload = json.loads(train_path.read_text(encoding="utf-8"))
        training_panels = train_payload["panels"]
    return plot_sweep_heatmaps(
        geom_payload["panels"],
        training_panels=training_panels,
        seeds=tuple(geom_payload["seeds"]),
        outfile=outfile,
    )


def run_sweep_training_plots(
    *,
    seeds: tuple[int, ...] | None = None,
    json_file: str = "sweep_training.json",
) -> Path:
    json_path = write_sweep_training_metrics(seeds=seeds, outfile=json_file)
    print(f"wrote {json_path}")
    return json_path


def run_sweep_plots(
    *,
    seeds: tuple[int, ...] | None = None,
    json_file: str = "sweep_geometry.json",
    geometry: bool = True,
    training: bool = True,
) -> tuple[Path, ...]:
    outputs: list[Path] = []
    training_panels: list[dict[str, Any]] | None = None
    run_seeds = seeds if seeds is not None else SWEEP_DEFAULT_SEEDS

    if training:
        train_json = write_sweep_training_metrics(seeds=run_seeds)
        train_payload = json.loads(train_json.read_text(encoding="utf-8"))
        training_panels = train_payload["panels"]
        outputs.append(train_json)
        print(f"wrote {train_json}")

    if geometry:
        json_path = write_sweep_geometry(seeds=run_seeds, outfile=json_file)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        heatmap_path = plot_sweep_heatmaps(
            payload["panels"],
            training_panels=training_panels,
            seeds=tuple(payload["seeds"]),
        )
        print(f"wrote {json_path}")
        print(f"wrote {heatmap_path}")
        outputs.extend([json_path, heatmap_path])
    elif training_panels is not None:
        geom_path = sweep_data_dir(SWEEP_COMPARISON_NAME) / json_file
        if geom_path.is_file():
            payload = json.loads(geom_path.read_text(encoding="utf-8"))
            heatmap_path = plot_sweep_heatmaps(
                payload["panels"],
                training_panels=training_panels,
                seeds=tuple(payload["seeds"]),
            )
            print(f"wrote {heatmap_path}")
            outputs.append(heatmap_path)
    return tuple(outputs)
