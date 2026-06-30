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
    seed: int | None = None,
) -> Path:
    tasks = list(spec.tasks)
    out_dir = comparison_dir(spec.name, "learning_curves")
    out_path = out_dir / "overview.png"

    if not any(checkpoint_path(t, spec.model_type, seed=seed).is_file() for t in tasks):
        raise FileNotFoundError(f"no {spec.model_type} checkpoints for comparison {spec.name!r}")

    n = len(tasks)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.2 * ncols, 3.2 * nrows),
        constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    for idx, (ax, task) in enumerate(zip(axes_flat, tasks, strict=False)):
        col = idx % ncols
        show_ylabel = col == 0
        show_metric_ylabel = col == ncols - 1
        show_legend = idx == 0

        ckpt = checkpoint_path(task, spec.model_type, seed=seed)
        if not ckpt.is_file():
            ax.set_visible(False)
            ax.text(
                0.5, 0.5, f"missing {spec.model_type} model\n{task}",
                ha="center", va="center", transform=ax.transAxes, fontsize=8,
            )
            continue
        model = load_model_for_viz(str(ckpt), spec.model_type)
        title = f"({spec.label_for(task)})"
        if not plot_learning_curve_on_axes(
            ax,
            model,
            title=title,
            compact=True,
            smoothed=True,
            truncate_to_plateau=truncate_to_plateau,
            show_legend=show_legend,
            show_ylabel=show_ylabel,
            show_metric_ylabel=show_metric_ylabel,
        ):
            ax.text(
                0.5, 0.5, f"no loss history\n{task}",
                ha="center", va="center", transform=ax.transAxes, fontsize=8,
            )

    for ax in axes_flat[len(tasks):]:
        ax.set_visible(False)

    fig.suptitle(
        f"{spec.display_title}: training learning curves ({spec.model_type})",
        fontsize=12,
        y=1.02,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
