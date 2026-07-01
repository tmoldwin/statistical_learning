"""Cross-task closed-loop trajectory comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from experiment import comparison_dir
from viz.compare._data import load_task_viz_context
from viz.compare.spec import ComparisonSpec
from viz.dimred import (
    embed_axis_labels_2d,
    embed_axis_labels_3d,
    embed_dim_label,
    embed_save_path,
    fit_embed_2d_with_evr,
    fit_embed_3d_with_evr,
)
from visualize import (
    _add_trajectory_word_legend,
    _apply_cube_limits_3d,
    _closed_loop_summary_seed,
    _cube_data_limits,
    _embed_trajectories_for_text,
    _one_vocab_cycle_steps,
    _plot_trajectory_closed_loop_panel,
    _square_data_limits,
    _trajectory_seed_letters,
    _vocab_word_colors,
)


def _closed_loop_panel_limits(
    ax,
    limit_arrays: list,
    *,
    is_3d: bool,
    xlabel: str,
    ylabel: str,
    zlabel: str | None = None,
) -> None:
    if not limit_arrays:
        return
    if is_3d:
        xlim, ylim, zlim = _cube_data_limits(*limit_arrays, padding_frac=0.12)
        _apply_cube_limits_3d(ax, xlim, ylim, zlim)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        if zlabel is not None:
            ax.set_zlabel(zlabel, fontsize=8)
    else:
        xlim, ylim = _square_data_limits(*limit_arrays, padding_frac=0.12)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.35)


def _plot_task_closed_loop_panel(
    ax, ctx, *, is_3d: bool, rollout_seed: int, embed_method: str,
    average_trials: int = 1,
) -> list:
    trajs = _embed_trajectories_for_text(
        ctx.text, ctx.hidden_states, spaced=ctx.spaced, words=ctx.words,
    )
    if is_3d:
        _projected, mean, components, evr = fit_embed_3d_with_evr(
            ctx.hidden_states, method=embed_method, trajectories=trajs,
        )
        xlabel, ylabel, zlabel = embed_axis_labels_3d(evr, embed_method)
    else:
        _projected, mean, components, evr = fit_embed_2d_with_evr(
            ctx.hidden_states, method=embed_method, trajectories=trajs,
        )
        xlabel, ylabel = embed_axis_labels_2d(evr, embed_method)
        zlabel = None

    vocab_words = list(ctx.words)
    seed_letters = _trajectory_seed_letters(ctx.model, vocab_words)
    summary_seed = _closed_loop_summary_seed(vocab_words, seed_letters, spaced=ctx.spaced)
    summary_steps = _one_vocab_cycle_steps(vocab_words, spaced=ctx.spaced)
    word_colors = _vocab_word_colors(vocab_words)
    limit_arrays: list = []

    _plot_trajectory_closed_loop_panel(
        ax, ctx.model, [summary_seed], summary_steps, rollout_seed,
        mean, components, limit_arrays,
        vocab_words=vocab_words, word_colors=word_colors,
        spaced=ctx.spaced, is_3d=is_3d,
        average_trials=average_trials,
    )
    _closed_loop_panel_limits(
        ax, limit_arrays, is_3d=is_3d, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
    )
    ax.set_title(f"start '{summary_seed}' · {summary_steps} steps", fontsize=9)
    return limit_arrays


def plot_closed_loop_trajectories(
    spec: ComparisonSpec,
    *,
    dimensions: int = 3,
    seeds: tuple[int, ...] | None = None,
    outfile: str | None = None,
    embed_method: str = "pca",
) -> Path:
    """Grid: rows = tasks (conditions), columns = RNG seeds."""
    if dimensions not in (2, 3):
        raise ValueError("dimensions must be 2 or 3")

    run_seeds = seeds if seeds is not None else spec.seeds
    is_3d = dimensions == 3
    tasks = list(spec.tasks)
    n_task_rows = len(tasks)
    n_seed_cols = len(run_seeds)

    out_dir = comparison_dir(spec.name, "trajectories")
    base_name = outfile or ("closed_loop_3d.png" if is_3d else "closed_loop_2d.png")
    out_path = out_dir / embed_save_path(base_name, embed_method)

    fig = plt.figure(figsize=(3.8 * n_seed_cols + 0.7, 4.0 * n_task_rows + 0.5))
    gs = fig.add_gridspec(
        1 + n_task_rows,
        1 + n_seed_cols,
        height_ratios=[0.07] + [1.0] * n_task_rows,
        width_ratios=[0.09] + [1.0] * n_seed_cols,
        hspace=0.32,
        wspace=0.22,
    )

    for col_idx, run_seed in enumerate(run_seeds):
        ax_head = fig.add_subplot(gs[0, col_idx + 1])
        ax_head.axis("off")
        ax_head.set_title(f"seed {run_seed}", fontsize=11, fontweight="bold", pad=6)

    merged_word_colors: dict[str, tuple] = {}

    for row_idx, task in enumerate(tasks):
        ax_row = fig.add_subplot(gs[row_idx + 1, 0])
        ax_row.axis("off")
        ax_row.text(
            0.5, 0.5, spec.label_for(task),
            ha="center", va="center", fontsize=10, fontweight="bold", rotation=90,
        )

        for col_idx, run_seed in enumerate(run_seeds):
            if is_3d:
                ax = fig.add_subplot(gs[row_idx + 1, col_idx + 1], projection="3d")
            else:
                ax = fig.add_subplot(gs[row_idx + 1, col_idx + 1])

            try:
                ctx = load_task_viz_context(
                    task, model_type=spec.model_type, seed=run_seed,
                )
            except FileNotFoundError:
                ax.set_visible(False)
                ax.text(
                    0.5, 0.5, f"no model\nseed {run_seed}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9,
                )
                continue

            _plot_task_closed_loop_panel(
                ax, ctx, is_3d=is_3d, rollout_seed=0, embed_method=embed_method,
                average_trials=1,
            )
            merged_word_colors.update(_vocab_word_colors(ctx.words))

    if merged_word_colors:
        _add_trajectory_word_legend(fig, merged_word_colors)

    fig.suptitle(
        f"{spec.display_title} ({spec.model_type}, {dimensions}D {embed_dim_label(embed_method)})",
        fontsize=12,
        y=0.98,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
