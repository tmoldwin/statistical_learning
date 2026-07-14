"""Cross-task closed-loop trajectory comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
    _one_vocab_cycle_steps,
    _plot_trajectory_closed_loop_panel,
    _square_data_limits,
    _states_after_first_word,
    _trajectory_seed_letters,
    _vocab_word_colors,
)


def plot_closed_loop_run_seed_row(
    task: str,
    *,
    seeds: tuple[int, ...] = (1, 2, 3, 5, 7, 8, 11, 13, 17, 19, 23, 29),
    out_path: Path | None = None,
    model_type: str = "rnn",
    embed_method: str = "pca",
    rollout_seed: int = 0,
    text_chars: int = 100,
    ncols: int = 4,
    panel_inches: float = 1.35,
) -> Path:
    """Grid of closed-loop trajectories for the same task across training seeds.

    Panel physical size is fixed (``panel_inches``) so the grid stays readable
    without huge or tiny cells as seed count changes.
    """
    from experiment import TASKS, plots_dir
    from viz.plot_layout import finalize_grid_figure, save_figure

    cap = min(int(TASKS[task].get("viz_length", 80)), text_chars)
    n = len(seeds)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))
    panel = float(panel_inches)
    fig_w = panel * ncols + 0.55
    fig_h = panel * nrows + 0.45
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    xlabel, ylabel = "PC1", "PC2"
    for idx, seed in enumerate(seeds):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        try:
            ctx = load_task_viz_context(
                task, model_type=model_type, seed=seed, text_chars=cap,
            )
        except Exception as exc:  # noqa: BLE001
            ax.set_visible(False)
            ax.text(0.5, 0.5, f"missing\n{exc.__class__.__name__}", ha="center", va="center", fontsize=6)
            continue
        _plot_task_closed_loop_panel(
            ax, ctx, is_3d=False, rollout_seed=rollout_seed,
            embed_method=embed_method, average_trials=1,
            minimal_axes=True, annotate=False, annotate_fontsize=4.5,
            linewidth=1.05, arrow_mutation_scale=7.0,
        )
        ax.set_title(f"seed {seed}", fontsize=7, pad=1)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    for row in range(nrows):
        for col in range(ncols):
            ax = axes[row, col]
            if not ax.get_visible():
                continue
            show_x = row == nrows - 1
            show_y = col == 0
            ax.tick_params(
                labelsize=4.5, left=show_y, bottom=show_x,
                labelleft=show_y, labelbottom=show_x, length=2, pad=1,
            )
            ax.set_ylabel(ylabel if show_y else "", fontsize=6.5, labelpad=1)
            ax.set_xlabel(xlabel if (show_x and col == ncols // 2) else "", fontsize=6.5, labelpad=1)
            ax.grid(True, linestyle=":", alpha=0.28)

    finalize_grid_figure(
        fig,
        suptitle="Closed-loop trajectories · same condition · panels = training seed",
        top=0.94,
        bottom=0.08,
        left=0.06,
        hspace=0.22,
        wspace=0.12,
    )

    if out_path is None:
        out_path = Path(plots_dir(task, model_type)) / "trajectories" / "closed_loop_run_seed_row.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, dpi=150)
    print(f"wrote {out_path}")
    return out_path


def _closed_loop_panel_limits(
    ax,
    limit_arrays: list,
    *,
    is_3d: bool,
    xlabel: str,
    ylabel: str,
    zlabel: str | None = None,
    minimal_axes: bool = False,
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
        if minimal_axes:
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        else:
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.35)


def _plot_task_closed_loop_panel(
    ax, ctx, *, is_3d: bool, rollout_seed: int, embed_method: str,
    average_trials: int = 1,
    minimal_axes: bool = False,
    annotate: bool = True,
    annotate_fontsize: float = 8.0,
    linewidth: float = 1.6,
    arrow_mutation_scale: float = 12.0,
) -> list:
    fit_states, trajs = _states_after_first_word(
        ctx.text, ctx.hidden_states, spaced=ctx.spaced, words=ctx.words,
    )
    if is_3d:
        _projected, mean, components, evr = fit_embed_3d_with_evr(
            fit_states, method=embed_method, trajectories=trajs,
        )
        xlabel, ylabel, zlabel = embed_axis_labels_3d(evr, embed_method)
    else:
        _projected, mean, components, evr = fit_embed_2d_with_evr(
            fit_states, method=embed_method, trajectories=trajs,
        )
        xlabel, ylabel = embed_axis_labels_2d(evr, embed_method)
        zlabel = None

    vocab_words = list(ctx.words)
    seed_letters = _trajectory_seed_letters(ctx.model, vocab_words)
    summary_seed = _closed_loop_summary_seed(vocab_words, seed_letters, spaced=ctx.spaced)
    # Extra word length so after dropping the seed-primed first word we still
    # cover roughly one full vocabulary cycle of complete words.
    summary_steps = _one_vocab_cycle_steps(vocab_words, spaced=ctx.spaced)
    if vocab_words:
        summary_steps += max(len(w) for w in vocab_words)
    word_colors = _vocab_word_colors(vocab_words)
    limit_arrays: list = []

    _plot_trajectory_closed_loop_panel(
        ax, ctx.model, [summary_seed], summary_steps, rollout_seed,
        mean, components, limit_arrays,
        vocab_words=vocab_words, word_colors=word_colors,
        spaced=ctx.spaced, is_3d=is_3d,
        average_trials=average_trials,
        annotate=annotate,
        annotate_fontsize=annotate_fontsize,
        linewidth=linewidth,
        arrow_mutation_scale=arrow_mutation_scale,
    )
    _closed_loop_panel_limits(
        ax, limit_arrays, is_3d=is_3d, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
        minimal_axes=minimal_axes and not is_3d,
    )
    if not minimal_axes:
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

    minimal = not is_3d and n_seed_cols > 6

    panel_w = 2.2 if n_seed_cols <= 8 else (1.55 if n_seed_cols <= 12 else 1.25)
    row_h = 4.0 if is_3d else (2.35 if n_seed_cols <= 8 else 1.35)
    if n_seed_cols <= 8:
        head_fs, row_fs, hspace = 9.0, 8.0, 0.18
    elif n_seed_cols <= 12:
        head_fs, row_fs, hspace = 7.0, 7.0, 0.28
    else:
        head_fs, row_fs, hspace = 6.0, 6.0, 0.34

    figsize = (panel_w * n_seed_cols + 0.4, row_h * n_task_rows + 0.5)
    if is_3d:
        fig = plt.figure(figsize=figsize)
        axes = np.empty((n_task_rows, n_seed_cols), dtype=object)
        for row_idx in range(n_task_rows):
            for col_idx in range(n_seed_cols):
                axes[row_idx, col_idx] = fig.add_subplot(
                    n_task_rows, n_seed_cols, row_idx * n_seed_cols + col_idx + 1,
                    projection="3d",
                )
        fig.subplots_adjust(hspace=hspace, wspace=0.10)
    else:
        fig, axes = plt.subplots(
            n_task_rows,
            n_seed_cols,
            figsize=figsize,
            squeeze=False,
            gridspec_kw={"hspace": hspace, "wspace": 0.10},
        )

    merged_word_colors: dict[str, tuple] = {}

    for row_idx, task in enumerate(tasks):
        for col_idx, run_seed in enumerate(run_seeds):
            ax = axes[row_idx, col_idx]

            try:
                ctx = load_task_viz_context(
                    task, model_type=spec.model_type, seed=run_seed,
                )
            except FileNotFoundError:
                ax.set_visible(False)
                msg = f"no model\nseed {run_seed}"
                if is_3d:
                    ax.text2D(
                        0.5, 0.5, msg,
                        transform=ax.transAxes, ha="center", va="center", fontsize=9,
                    )
                else:
                    ax.text(
                        0.5, 0.5, msg,
                        ha="center", va="center", transform=ax.transAxes, fontsize=9,
                    )
                continue

            _plot_task_closed_loop_panel(
                ax, ctx, is_3d=is_3d, rollout_seed=0, embed_method=embed_method,
                average_trials=1, minimal_axes=minimal,
            )
            merged_word_colors.update(_vocab_word_colors(ctx.words))
            if row_idx == 0:
                ax.set_title(f"s{run_seed}", fontsize=head_fs, fontweight="bold", pad=2)
            if col_idx == 0:
                ax.set_ylabel(spec.label_for(task), fontsize=row_fs, fontweight="bold")

    if merged_word_colors:
        _add_trajectory_word_legend(fig, merged_word_colors)

    fig.suptitle(
        f"{spec.display_title} ({spec.model_type}, {dimensions}D {embed_dim_label(embed_method)})",
        fontsize=12,
        y=0.98,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
