"""Side-by-side learning-curve grid for multiple tasks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiment import DEFAULT_SEED, checkpoint_path, comparison_dir
from viz.compare.spec import ComparisonSpec
from visualize import load_model_for_viz, plot_learning_curve_on_axes


def plot_learning_curves(
    spec: ComparisonSpec,
    *,
    truncate_to_plateau: bool = False,
    seeds: tuple[int, ...] | None = None,
) -> Path:
    run_seeds = seeds if seeds is not None else spec.seeds
    tasks = list(spec.tasks)
    out_dir = comparison_dir(spec.name, "learning_curves")
    if len(run_seeds) > 1:
        seed_tag = "_".join(str(s) for s in run_seeds)
        out_path = out_dir / f"overview_seeds_{seed_tag}.png"
    elif run_seeds != (DEFAULT_SEED,):
        out_path = out_dir / f"overview_seed{run_seeds[0]}.png"
    else:
        out_path = out_dir / "overview.png"

    if not any(
        checkpoint_path(t, spec.model_type, seed=s).is_file()
        for t in tasks
        for s in run_seeds
    ):
        raise FileNotFoundError(f"no {spec.model_type} checkpoints for comparison {spec.name!r}")

    ncols = min(4, max(1, len(tasks)))
    panels_per_seed = int(np.ceil(len(tasks) / ncols))
    nrows = len(run_seeds) * panels_per_seed

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.2 * ncols, 3.2 * nrows),
        constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()
    panel_idx = 0

    for run_seed in run_seeds:
        for task_idx, task in enumerate(tasks):
            ax = axes_flat[panel_idx]
            panel_idx += 1
            col = task_idx % ncols
            show_ylabel = col == 0
            show_metric_ylabel = col == ncols - 1
            show_legend = panel_idx == 1

            ckpt = checkpoint_path(task, spec.model_type, seed=run_seed)
            if not ckpt.is_file():
                ax.set_visible(False)
                ax.text(
                    0.5, 0.5, f"missing model\n{task}\nseed {run_seed}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8,
                )
                continue
            model = load_model_for_viz(str(ckpt), spec.model_type)
            title = spec.label_for(task)
            if len(run_seeds) > 1:
                title = f"seed {run_seed} · {title}"
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

    for ax in axes_flat[panel_idx:]:
        ax.set_visible(False)

    seed_note = (
        f"seeds {', '.join(str(s) for s in run_seeds)}"
        if len(run_seeds) > 1
        else f"seed {run_seeds[0]}"
    )
    fig.suptitle(
        f"{spec.display_title}: training learning curves ({spec.model_type} · {seed_note})",
        fontsize=12,
        y=1.02,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
