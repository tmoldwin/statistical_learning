"""Side-by-side learning-curve grid for multiple tasks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiment import checkpoint_path, comparison_dir
from viz.compare.spec import ComparisonSpec
from visualize import load_model_for_viz, plot_learning_curve_on_axes


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

    fig.suptitle(
        f"{spec.display_title}: training ({spec.model_type})",
        fontsize=12,
        y=1.02,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
