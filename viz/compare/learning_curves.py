"""Side-by-side learning-curve grid for multiple tasks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiment import checkpoint_path, comparison_dir
from viz.compare._data import load_task_viz_context
from viz.compare.geometry import _plot_metric_by_condition
from viz.compare.spec import ComparisonSpec
from viz.dimred import fit_embed_2d_with_evr
from visualize import (
    _closed_loop_summary_seed,
    _embed_trajectories_for_text,
    _one_vocab_cycle_steps,
    _plot_letter_seed_closed_loop_on_axis,
    _square_data_limits,
    _trajectory_seed_letters,
    _vocab_word_colors,
    load_model_for_viz,
    plot_learning_curve_on_axes,
)

_CLOSED_LOOP_INSET_TRIALS = 24

_SUCCESS_WORD_ERR_PCT = 15.0


def _checkpoint_training_stats(
    task: str,
    model_type: str,
    run_seed: int,
) -> dict[str, float] | None:
    ckpt = checkpoint_path(task, model_type, seed=run_seed)
    if not ckpt.is_file():
        return None
    data = np.load(ckpt)
    if "metric_word_error_frac" not in data.files:
        return None
    final_we = float(data["metric_word_error_frac"][-1]) * 100.0
    best_we = float(data["best_metric_word_error_frac"]) * 100.0
    final_ce = float(data["loss_smooth"][-1]) if "loss_smooth" in data.files else float("nan")
    final_iter = float(data["loss_iterations"][-1]) if "loss_iterations" in data.files else float("nan")
    return {
        "final_word_err_pct": final_we,
        "best_word_err_pct": best_we,
        "final_ce": final_ce,
        "final_iter": final_iter,
    }


def plot_learning_summary(
    spec: ComparisonSpec,
    *,
    seeds: tuple[int, ...] | None = None,
) -> Path:
    """Box plots of final training outcomes by condition; points = seeds."""
    run_seeds = seeds if seeds is not None else spec.seeds
    tasks = tuple(spec.tasks)
    condition_labels = [spec.label_for(t) for t in tasks]
    seed_colors = {int(s): plt.cm.tab20(i % 20) for i, s in enumerate(run_seeds)}

    metric_specs = (
        ("final_word_err_pct", "final % invalid words"),
        ("best_word_err_pct", "best % invalid words"),
        ("final_ce", "final cross-entropy"),
        ("final_iter", "training iterations"),
    )
    groups_by_key: dict[str, list[tuple[list[float], list[int]]]] = {
        key: [] for key, _ in metric_specs
    }

    for task in tasks:
        for key, _ in metric_specs:
            vals: list[float] = []
            seed_ids: list[int] = []
            for run_seed in run_seeds:
                stats = _checkpoint_training_stats(task, spec.model_type, run_seed)
                if stats is None:
                    continue
                vals.append(float(stats[key]))
                seed_ids.append(int(run_seed))
            groups_by_key[key].append((vals, seed_ids))

    fig, axes = plt.subplots(
        2, 2, figsize=(11.0, 7.0), constrained_layout=True, squeeze=False,
    )
    for ax, (key, ylabel) in zip(axes.ravel(), metric_specs):
        _plot_metric_by_condition(
            ax, groups_by_key[key], condition_labels, ylabel=ylabel, seed_colors=seed_colors,
        )
        if key == "final_word_err_pct":
            ax.axhline(_SUCCESS_WORD_ERR_PCT, color="#cc3333", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.grid(True, axis="y", linestyle=":", alpha=0.35)

    fig.suptitle(
        f"{spec.display_title}: training summary ({spec.model_type}, n={len(run_seeds)} seeds)",
        fontsize=11,
    )
    out_dir = comparison_dir(spec.name, "learning_curves")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _add_mean_pca_trajectory_inset(ax, task: str, model_type: str, run_seed: int) -> None:
    """Upper-right inset: mean closed-loop trajectory in the model's PCA plane."""
    try:
        ctx = load_task_viz_context(task, model_type=model_type, seed=run_seed)
        trajs = _embed_trajectories_for_text(
            ctx.text, ctx.hidden_states, spaced=ctx.spaced, words=ctx.words,
        )
        _projected, mean, components, _evr = fit_embed_2d_with_evr(
            ctx.hidden_states, method="pca", trajectories=trajs,
        )
        vocab_words = list(ctx.words)
        seed_letters = _trajectory_seed_letters(ctx.model, vocab_words)
        summary_seed = _closed_loop_summary_seed(vocab_words, seed_letters, spaced=ctx.spaced)
        summary_steps = _one_vocab_cycle_steps(vocab_words, spaced=ctx.spaced)
        word_colors = _vocab_word_colors(vocab_words)

        inset = ax.inset_axes([0.56, 0.52, 0.43, 0.46])
        inset.set_facecolor((1.0, 1.0, 1.0, 0.85))
        limit_arrays: list[np.ndarray] = []
        _plot_letter_seed_closed_loop_on_axis(
            inset, ctx.model,
            seed_letters=[summary_seed],
            steps=summary_steps,
            closed_loop_seed=0,
            mean=mean,
            components=components,
            limit_arrays=limit_arrays,
            vocab_words=vocab_words,
            word_colors=word_colors,
            spaced=ctx.spaced,
            annotate=False,
            is_3d=False,
            unique_word_labels=True,
            average_trials=_CLOSED_LOOP_INSET_TRIALS,
        )
        if limit_arrays:
            xlim, ylim = _square_data_limits(*limit_arrays, padding_frac=0.10)
            inset.set_xlim(xlim)
            inset.set_ylim(ylim)
            inset.set_aspect("equal", adjustable="box")
        inset.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in inset.spines.values():
            spine.set_linewidth(0.6)
            spine.set_alpha(0.5)
    except Exception as exc:  # inset failure should not break the grid
        print(f"warn: no PCA inset for {task} seed {run_seed}: {exc}")


def plot_learning_curves(
    spec: ComparisonSpec,
    *,
    truncate_to_plateau: bool = False,
    seeds: tuple[int, ...] | None = None,
) -> Path:
    """Grid: rows = tasks (conditions), columns = RNG seeds."""
    run_seeds = seeds if seeds is not None else spec.seeds
    tasks = list(spec.tasks)
    out_dir = comparison_dir(spec.name, "learning_curves")
    out_path = out_dir / "overview.png"

    if not any(
        checkpoint_path(t, spec.model_type, seed=s).is_file()
        for t in tasks
        for s in run_seeds
    ):
        raise FileNotFoundError(f"no {spec.model_type} checkpoints for comparison {spec.name!r}")

    ncols = len(run_seeds)
    nrows = len(tasks)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.6 * ncols, 3.0 * nrows),
        constrained_layout=True,
        squeeze=False,
    )

    for row_idx, task in enumerate(tasks):
        for col_idx, run_seed in enumerate(run_seeds):
            ax = axes[row_idx, col_idx]
            show_ylabel = col_idx == 0
            show_metric_ylabel = col_idx == ncols - 1
            show_legend = row_idx == 0 and col_idx == 0

            ckpt = checkpoint_path(task, spec.model_type, seed=run_seed)
            if not ckpt.is_file():
                ax.set_visible(False)
                ax.text(
                    0.5, 0.5, f"missing\nseed {run_seed}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8,
                )
                continue
            model = load_model_for_viz(str(ckpt), spec.model_type)
            title = f"seed {run_seed}" if row_idx == 0 else ""
            if row_idx == 0:
                ax.set_title(title, fontsize=9, fontweight="bold")
            if not plot_learning_curve_on_axes(
                ax,
                model,
                title=spec.label_for(task) if col_idx == 0 else "",
                compact=True,
                smoothed=True,
                truncate_to_plateau=truncate_to_plateau,
                show_legend=show_legend,
                show_ylabel=show_ylabel,
                show_metric_ylabel=show_metric_ylabel,
            ):
                ax.text(
                    0.5, 0.5, f"no loss history",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8,
                )
                continue
            if show_legend:
                legend = ax.get_legend()
                if legend is not None:
                    legend.set_loc("center right")
            _add_mean_pca_trajectory_inset(ax, task, spec.model_type, run_seed)

    fig.suptitle(
        f"{spec.display_title}: training ({spec.model_type})",
        fontsize=12,
        y=1.02,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
