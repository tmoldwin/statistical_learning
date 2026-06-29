"""Overview figure: training learning curves for the micro curriculum."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment import (
    MICRO_CURRICULUM,
    MICRO_CURRICULUM_INIT_SEEDS,
    MICRO_CURRICULUM_REGIME_LABELS,
    MODEL_TYPES,
    micro_curriculum_viz_dir,
    model_path,
    spaced_experiment_name,
)
from task import REGIMES
from visualize import load_model_for_viz, plot_learning_curve_on_axes

REGIME_LABELS = MICRO_CURRICULUM_REGIME_LABELS

DEFAULT_INIT_SEED = MICRO_CURRICULUM_INIT_SEEDS[0]


def _checkpoint_path(exp: str, model_type: str) -> Path | None:
    path = model_path(exp, model_type)
    if path.is_file():
        return path
    seeded = model_path(exp, model_type, seed=DEFAULT_INIT_SEED)
    if seeded.is_file():
        return seeded
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-word-space",
        action="store_true",
        help="use unspaced micro-curriculum experiments (output under _ns)",
    )
    parser.add_argument(
        "--model-type",
        default="rnn",
        choices=list(MODEL_TYPES),
        help="which model checkpoints to plot (default: rnn)",
    )
    args = parser.parse_args()

    spaced = not args.no_word_space
    regimes = list(MICRO_CURRICULUM)
    exps = (
        [spaced_experiment_name(r) for r in regimes]
        if spaced
        else regimes
    )
    out_dir = micro_curriculum_viz_dir(spaced=spaced, model_type=args.model_type, kind="learning_curves")
    out_path = out_dir / "curriculum_overview.png"
    if not any(_checkpoint_path(exp, args.model_type) for exp in exps):
        print(f"skip {out_path}: no {args.model_type} checkpoints for micro curriculum")
        return

    n = len(exps)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.2 * ncols, 3.2 * nrows),
        constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    for idx, (ax, regime, exp) in enumerate(zip(axes_flat, regimes, exps, strict=False)):
        col = idx % ncols
        show_ylabel = col == 0
        show_metric_ylabel = col == ncols - 1
        show_legend = idx == 0

        ckpt = _checkpoint_path(exp, args.model_type)
        if ckpt is None:
            ax.set_visible(False)
            ax.text(
                0.5, 0.5, f"missing {args.model_type} model\n{exp}",
                ha="center", va="center", transform=ax.transAxes, fontsize=8,
            )
            continue
        model = load_model_for_viz(str(ckpt), args.model_type)
        words = REGIMES[regime]
        tag = REGIME_LABELS.get(regime, regime)
        title = f"{', '.join(words)}\n({tag})"
        if not plot_learning_curve_on_axes(
            ax,
            model,
            title=title,
            compact=True,
            smoothed=True,
            truncate_to_plateau=True,
            show_legend=show_legend,
            show_ylabel=show_ylabel,
            show_metric_ylabel=show_metric_ylabel,
        ):
            ax.text(
                0.5, 0.5, f"no loss history\n{exp}",
                ha="center", va="center", transform=ax.transAxes, fontsize=8,
            )

    for ax in axes_flat[len(exps):]:
        ax.set_visible(False)

    spacing = "spaced" if spaced else "unspaced"
    fig.suptitle(
        f"Micro curriculum: training learning curves ({spacing}, {args.model_type})",
        fontsize=12,
        y=1.02,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
