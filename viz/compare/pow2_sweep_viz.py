"""Learning curves and closed-loop trajectory grids for the pow2 sweep."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiment import TASKS, checkpoint_path, comparison_dir
from viz.compare._data import load_task_viz_context
from viz.compare.sweep_output import sweep_figures_dir
from viz.compare.trajectories import _plot_task_closed_loop_panel
from viz.dimred import embed_dim_label, embed_save_path
from viz.plot_layout import finalize_grid_figure, hide_x_tick_labels, save_figure
from visualize import load_model_for_viz, plot_learning_curve_on_axes
from vocab_sweep_pow2 import (
    POW2_DEFAULT_SEEDS,
    POW2_LENGTHS,
    POW2_WORD_COUNTS,
    iter_pow2_sweep_cells,
    length_label,
    task_name,
)

POW2_SWEEP_COMPARISON_NAME = "word_count_pow2_sweep_ns"


def _grid_axes(n_rows: int, n_cols: int, *, is_3d: bool = False):
    if is_3d:
        fig = plt.figure(figsize=(2.6 * n_cols, 2.2 * n_rows))
        axes = np.empty((n_rows, n_cols), dtype=object)
        for li in range(n_rows):
            for wi in range(n_cols):
                axes[li, wi] = fig.add_subplot(
                    n_rows, n_cols, li * n_cols + wi + 1, projection="3d",
                )
        return fig, axes
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.6 * n_cols, 2.0 * n_rows),
        squeeze=False,
    )
    return fig, axes


def plot_pow2_sweep_learning_curves(
    *,
    seeds: tuple[int, ...] = (1,),
    model_type: str = "rnn",
) -> list[Path]:
    """Grid of training curves: rows = length, cols = word count (one file per seed)."""
    n_rows = len(POW2_LENGTHS)
    n_cols = len(POW2_WORD_COUNTS)
    out_dir = comparison_dir(POW2_SWEEP_COMPARISON_NAME, "learning_curves")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for run_seed in seeds:
        fig, axes = _grid_axes(n_rows, n_cols)
        for li, length in enumerate(POW2_LENGTHS):
            for wi, n_words in enumerate(POW2_WORD_COUNTS):
                ax = axes[li, wi]
                task = task_name(n_words, length)
                ckpt = checkpoint_path(task, model_type, seed=run_seed)
                if not ckpt.is_file():
                    ax.axis("off")
                    ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=7)
                    continue
                model = load_model_for_viz(str(ckpt), model_type)
                seq_len = int(TASKS[task].get("sequence_length", 1))
                ok = plot_learning_curve_on_axes(
                    ax,
                    model,
                    compact=True,
                    smoothed=True,
                    show_legend=False,
                    show_ylabel=(wi == 0),
                    show_metric_ylabel=False,
                    sequence_length=seq_len,
                )
                if not ok:
                    ax.text(0.5, 0.5, "no history", ha="center", va="center", fontsize=7)
                if li == 0:
                    ax.set_title(f"{n_words}w", fontsize=8)
                if wi == 0:
                    ax.set_ylabel(length_label(length), fontsize=7)
                if li < n_rows - 1:
                    hide_x_tick_labels(ax)
                ax.tick_params(labelsize=6)

        suffix = f"_seed{run_seed}" if len(seeds) > 1 else ""
        finalize_grid_figure(
            fig,
            suptitle=f"Pow2 sweep training curves ({model_type}, seed {run_seed})",
            top=0.94,
            bottom=0.06,
            hspace=0.45,
            wspace=0.28,
        )
        out_path = out_dir / f"overview{suffix}.png"
        save_figure(fig, out_path, dpi=160)
        paths.append(out_path)
    return paths


def plot_pow2_sweep_closed_loop(
    *,
    seeds: tuple[int, ...] = (1,),
    dimensions: int = 2,
    embed_method: str = "pca",
    model_type: str = "rnn",
) -> list[Path]:
    """Grid of closed-loop trajectory PCA/jPCA panels per sweep cell."""
    if dimensions not in (2, 3):
        raise ValueError("dimensions must be 2 or 3")
    is_3d = dimensions == 3
    n_rows = len(POW2_LENGTHS)
    n_cols = len(POW2_WORD_COUNTS)
    out_dir = sweep_figures_dir(POW2_SWEEP_COMPARISON_NAME)
    paths: list[Path] = []

    for run_seed in seeds:
        fig, axes = _grid_axes(n_rows, n_cols, is_3d=is_3d)
        for li, length in enumerate(POW2_LENGTHS):
            for wi, n_words in enumerate(POW2_WORD_COUNTS):
                ax = axes[li, wi]
                task = task_name(n_words, length)
                try:
                    ctx = load_task_viz_context(task, model_type=model_type, seed=run_seed)
                except (FileNotFoundError, KeyError, ValueError):
                    ax.set_visible(False)
                    continue
                _plot_task_closed_loop_panel(
                    ax, ctx, is_3d=is_3d, rollout_seed=0,
                    embed_method=embed_method, minimal_axes=True,
                )
                if li == 0:
                    ax.set_title(f"{n_words}w", fontsize=8)
                if wi == 0 and not is_3d:
                    ax.set_ylabel(length_label(length), fontsize=7)

        base = f"sweep_closed_loop_{dimensions}d"
        if len(seeds) > 1:
            base = f"{base}_seed{run_seed}"
        outfile = embed_save_path(f"{base}.png", embed_method)
        dim_lbl = embed_dim_label(embed_method)
        finalize_grid_figure(
            fig,
            suptitle=(
                f"Pow2 sweep closed-loop trajectories "
                f"({dimensions}D {dim_lbl}, {model_type}, seed {run_seed})"
            ),
            top=0.94,
            bottom=0.06,
            hspace=0.38,
            wspace=0.22,
        )
        out_path = out_dir / outfile
        save_figure(fig, out_path, dpi=160)
        paths.append(out_path)
    return paths


def run_pow2_sweep_learning_curve_plots(
    *,
    seeds: tuple[int, ...] | None = None,
) -> list[Path]:
    run_seeds = seeds if seeds is not None else (1,)
    paths = plot_pow2_sweep_learning_curves(seeds=run_seeds)
    for p in paths:
        print(f"wrote {p}")
    return paths


def run_pow2_sweep_closed_loop_plots(
    *,
    seeds: tuple[int, ...] | None = None,
    dimensions: tuple[int, ...] = (2,),
    embed_methods: tuple[str, ...] = ("pca",),
) -> list[Path]:
    """Closed-loop grids. Default is PCA only; pass embed_methods=(\"jpca\",) if needed."""
    run_seeds = seeds if seeds is not None else (1,)
    paths: list[Path] = []
    for dims in dimensions:
        for method in embed_methods:
            paths.extend(plot_pow2_sweep_closed_loop(
                seeds=run_seeds, dimensions=dims, embed_method=method,
            ))
    for p in paths:
        print(f"wrote {p}")
    return paths
