#!/usr/bin/env python3
"""Paper comparison: closed-loop trajectory grids (rows = condition, cols = seed)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment import TASKS, comparison_dir
from viz.compare._data import load_task_viz_context
from viz.compare.trajectories import _plot_task_closed_loop_panel
from viz.plot_layout import finalize_grid_figure, save_figure

SEEDS = (1, 2, 3)


def _load_ctx(task: str, seed: int):
    cap = min(int(TASKS[task].get("viz_length", 80)), 100)
    return load_task_viz_context(task, model_type="rnn", seed=seed, text_chars=cap)


def plot_task_seed_grid(
    *,
    rows: list[tuple[str, str]],
    seeds: tuple[int, ...] = SEEDS,
    out_path: Path,
    suptitle: str,
    annotate_words: int = 8,
) -> Path:
    """rows: list of (row_label, task_name)."""
    n_rows = len(rows)
    n_cols = len(seeds)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.1 * n_cols + 1.4, 3.4 * n_rows + 0.8),
        squeeze=False,
    )
    for ri, (row_label, task) in enumerate(rows):
        for ci, seed in enumerate(seeds):
            ax = axes[ri, ci]
            try:
                ctx = _load_ctx(task, seed)
            except Exception as exc:  # noqa: BLE001
                ax.set_visible(False)
                ax.text(0.5, 0.5, f"missing\n{exc.__class__.__name__}", ha="center", va="center", fontsize=8)
                continue
            n_words = len(ctx.words) if ctx.words else 99
            _plot_task_closed_loop_panel(
                ax, ctx, is_3d=False, rollout_seed=0,
                embed_method="pca", minimal_axes=True,
                annotate=n_words <= annotate_words,
                annotate_fontsize=6.5 if n_words <= 8 else 5.0,
            )
            if ri == 0:
                ax.set_title(f"seed {seed}", fontsize=10, fontweight="bold", pad=10)
        axes[ri, 0].set_ylabel(row_label, fontsize=11, fontweight="bold", labelpad=12)

    finalize_grid_figure(
        fig,
        suptitle=suptitle,
        top=0.92,
        bottom=0.04,
        left=0.12,
        hspace=0.28,
        wspace=0.16,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, dpi=160)
    print(f"wrote {out_path}")
    return out_path


def main() -> None:
    out_dir = comparison_dir("paper_seed_grids", "trajectories")
    # Fixed 16 words: rows = word length, cols = seeds
    plot_task_seed_grid(
        rows=[
            ("3-letter", "sixteen_word_ns"),
            ("4-letter", "sixteen_word_four_letter_ns"),
            ("5-letter", "sixteen_word_five_letter_ns"),
        ],
        out_path=out_dir / "closed_loop_by_length_16words.png",
        suptitle="Closed-loop trajectories · 16 words fixed (rows = length, cols = seed)",
    )
    # Fixed 4-letter words: rows = vocabulary size, cols = seeds
    plot_task_seed_grid(
        rows=[
            ("8 words", "eight_word_four_letter_ns"),
            ("16 words", "sixteen_word_four_letter_ns"),
        ],
        out_path=out_dir / "closed_loop_by_wordcount_4letter.png",
        suptitle="Closed-loop trajectories · 4-letter words fixed (rows = vocab size, cols = seed)",
    )


if __name__ == "__main__":
    main()
