"""Learning curves and closed-loop trajectory grids for the pow2 sweep."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiment import TASKS, checkpoint_path, comparison_dir, experiment_regime
from task import REGIMES, corpus_for_experiment
from vocab_diagrams import corpus_region_snippet
from viz.compare._data import load_task_viz_context, TaskVizContext
from viz.compare.learning_curves import _draw_overview_text_line, _overview_fontsize
from viz.compare.sweep_output import sweep_figures_dir
from viz.compare.trajectories import _plot_task_closed_loop_panel
from viz.dimred import embed_dim_label, embed_save_path
from viz.plot_layout import finalize_grid_figure, hide_x_tick_labels, save_figure
from visualize import load_model_for_viz, plot_learning_curve_on_axes
from viz.compare.pow2_sweep_spec import POW2_SWEEP_SPEC_NS, Pow2SweepSpec
from vocab_sweep_pow2 import (
    POW2_DEFAULT_SEEDS,
    POW2_LENGTHS,
    POW2_SEED_COMPARISON_SEEDS,
    POW2_WORD_COUNTS,
    iter_pow2_sweep_cells,
    length_label,
    task_name,
)

POW2_SWEEP_COMPARISON_NAME = POW2_SWEEP_SPEC_NS.comparison_name
_DEMO_SNIPPET_CAP = 56
_VOCAB_LABEL_MAX = 30
_SEED_CMP_COL_FONTSIZE = 8
_SEED_CMP_ROW_FONTSIZE = 7
_SEED_CMP_ROW_LABELPAD = 8


def _demo_snippet_len(task: str) -> int:
    cfg = TASKS[task]
    target = int(cfg.get("demo_snippet_len", cfg.get("viz_length", _DEMO_SNIPPET_CAP)))
    return min(target, _DEMO_SNIPPET_CAP)


def _format_vocab_label(words: list[str]) -> str:
    joined = ", ".join(words)
    if len(joined) <= _VOCAB_LABEL_MAX:
        return joined
    if len(words) <= 4:
        return joined[: _VOCAB_LABEL_MAX - 1] + "…"
    preview = ", ".join(words[:3])
    return f"{len(words)}w: {preview}…"


def _length_slug(length: int | str) -> str:
    return "lmix" if length == "mixed" else f"l{length}"


def _plot_demo_sequence_cell(
    ax,
    *,
    task: str,
    run_seed: int,
    ncols: int,
    show_vocab: bool = True,
    show_sample_label: bool = True,
) -> None:
    ax.set_axis_off()
    words = REGIMES[experiment_regime(task)]
    vocab = set(words)
    spaced = bool(TASKS[task].get("word_space", False))
    snippet_len = _demo_snippet_len(task)

    corpus = corpus_for_experiment(task, seed=run_seed)
    snippet = corpus_region_snippet(corpus, snippet_len, "middle")
    if not snippet:
        ax.text(0.5, 0.5, "no corpus", ha="center", va="center", fontsize=6, color="0.5")
        return

    vocab_label = _format_vocab_label(words)
    fs = _overview_fontsize(ncols, max(len(snippet), len(vocab_label) if show_vocab else 0))
    label_fs = max(fs - 0.4, 3.6)
    y_text = 0.52

    if show_vocab:
        ax.text(
            0.0, 0.92, vocab_label,
            transform=ax.transAxes, fontsize=label_fs, va="top", color="0.35",
            fontfamily="monospace",
        )
        y_text = 0.52
    if show_sample_label:
        ax.text(
            0.0, 0.72, "sample",
            transform=ax.transAxes, fontsize=label_fs, va="top", color="0.40",
        )
    elif not show_vocab:
        y_text = 0.55

    _draw_overview_text_line(
        ax, snippet, y_text, vocab=vocab, spaced=spaced, fontsize=fs,
    )


def plot_pow2_sweep_demo_sequences(
    *,
    seeds: tuple[int, ...] = (1,),
    spec: Pow2SweepSpec = POW2_SWEEP_SPEC_NS,
) -> list[Path]:
    """Grid of vocabulary labels + corpus demo snippets per sweep cell."""
    n_rows = len(spec.lengths)
    n_cols = len(spec.word_counts)
    out_dir = comparison_dir(spec.comparison_name, "sequences")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for run_seed in seeds:
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(1.85 * n_cols, 1.55 * n_rows),
            squeeze=False,
        )
        for li, length in enumerate(spec.lengths):
            for wi, n_words in enumerate(spec.word_counts):
                ax = axes[li, wi]
                task = spec.task_name(n_words, length)
                _plot_demo_sequence_cell(ax, task=task, run_seed=run_seed, ncols=n_cols)
                if li == 0:
                    ax.text(
                        0.5, 1.06, f"{n_words}w",
                        transform=ax.transAxes, ha="center", va="bottom", fontsize=8,
                    )
                if wi == 0:
                    ax.text(
                        -0.08, 0.5, spec.length_label(length),
                        transform=ax.transAxes, ha="right", va="center", fontsize=7,
                        rotation=90,
                    )

        suffix = f"_seed{run_seed}" if len(seeds) > 1 else ""
        finalize_grid_figure(
            fig,
            suptitle=(
                f"Pow2 sweep demo sequences (corpus middle snippet, seed {run_seed}; "
                "green = in-vocab word)"
            ),
            top=0.94,
            bottom=0.05,
            hspace=0.55,
            wspace=0.30,
        )
        out_path = out_dir / f"demo_sequences{suffix}.png"
        save_figure(fig, out_path, dpi=150)
        paths.append(out_path)
    return paths


def _seed_comparison_grid_axes(n_rows: int, n_cols: int):
    """Compact cells — readable but not poster-sized."""
    cell_w, cell_h = 1.45, 1.55
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(cell_w * n_cols + 0.9, cell_h * n_rows + 0.5),
        squeeze=False,
    )
    return fig, axes


def _annotate_for_word_count(n_words: int) -> bool:
    return n_words <= 8


def _load_closed_loop_ctx(task: str, *, model_type: str, run_seed: int) -> TaskVizContext:
    """Shorter teacher-forced window — enough for PCA, faster on large vocabs."""
    cap = min(int(TASKS[task].get("viz_length", 80)), 100)
    return load_task_viz_context(task, model_type=model_type, seed=run_seed, text_chars=cap)


def _plot_closed_loop_seed_comparison_grid(
    *,
    seeds: tuple[int, ...],
    row_specs: tuple[tuple[int, int | str], ...],
    row_label_fn,
    out_path: Path,
    suptitle: str,
    dimensions: int = 2,
    embed_method: str = "pca",
    model_type: str = "rnn",
    spec: Pow2SweepSpec = POW2_SWEEP_SPEC_NS,
) -> Path:
    if dimensions not in (2, 3):
        raise ValueError("dimensions must be 2 or 3")
    is_3d = dimensions == 3
    n_rows = len(row_specs)
    n_cols = len(seeds)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if is_3d:
        fig = plt.figure(figsize=(1.55 * n_cols + 0.6, 1.7 * n_rows + 0.4))
        axes = np.empty((n_rows, n_cols), dtype=object)
        for ri in range(n_rows):
            for ci in range(n_cols):
                axes[ri, ci] = fig.add_subplot(
                    n_rows, n_cols, ri * n_cols + ci + 1, projection="3d",
                )
    else:
        fig, axes = _seed_comparison_grid_axes(n_rows, n_cols)

    for ri, (n_words, length) in enumerate(row_specs):
        task = spec.task_name(n_words, length)
        for si, run_seed in enumerate(seeds):
            ax = axes[ri, si]
            try:
                ctx = _load_closed_loop_ctx(task, model_type=model_type, run_seed=run_seed)
            except (FileNotFoundError, KeyError, ValueError):
                ax.set_visible(False)
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=6)
                continue
            _plot_task_closed_loop_panel(
                ax, ctx, is_3d=is_3d, rollout_seed=0,
                embed_method=embed_method, minimal_axes=True,
                annotate=_annotate_for_word_count(n_words),
                annotate_fontsize=5.5 if n_words <= 4 else 4.5,
            )
            if ri == 0:
                ax.set_title(f"seed {run_seed}", fontsize=_SEED_CMP_COL_FONTSIZE, pad=4)
        if not is_3d:
            axes[ri, 0].set_ylabel(
                row_label_fn(n_words, length),
                fontsize=_SEED_CMP_ROW_FONTSIZE,
                labelpad=_SEED_CMP_ROW_LABELPAD,
                fontweight="bold",
            )
            axes[ri, 0].yaxis.get_label().set_fontweight("bold")

    for ax in axes[0]:
        if ax.get_title():
            ax.title.set_fontweight("bold")

    finalize_grid_figure(
        fig,
        suptitle=suptitle,
        suptitle_fontsize=10,
        top=0.92,
        bottom=0.04,
        left=0.10,
        hspace=0.28,
        wspace=0.14,
    )
    save_figure(fig, out_path, dpi=150)
    return out_path


def plot_pow2_sweep_closed_loop_seed_comparison(
    *,
    seeds: tuple[int, ...] | None = None,
    dimensions: int = 2,
    embed_method: str = "pca",
    model_type: str = "rnn",
    spec: Pow2SweepSpec = POW2_SWEEP_SPEC_NS,
) -> list[Path]:
    """Two figure families under seed_comparison/: by_length and by_word_count."""
    if dimensions not in (2, 3):
        raise ValueError("dimensions must be 2 or 3")
    run_seeds = seeds if seeds is not None else spec.seed_comparison_seeds
    base = comparison_dir(spec.comparison_name, "seed_comparison")
    by_length_dir = base / "by_length"
    by_word_count_dir = base / "by_word_count"
    paths: list[Path] = []
    dim_lbl = embed_dim_label(embed_method)
    dim_tag = f"closed_loop_{dimensions}d"

    for length in spec.lengths:
        row_specs = tuple((n_words, length) for n_words in spec.word_counts)
        slug = _length_slug(length)
        out_path = by_length_dir / embed_save_path(f"{dim_tag}_{slug}.png", embed_method)
        paths.append(_plot_closed_loop_seed_comparison_grid(
            seeds=run_seeds,
            row_specs=row_specs,
            row_label_fn=lambda n_words, _length: f"{n_words}w",
            out_path=out_path,
            suptitle=(
                f"Pow2 sweep by length · {spec.length_label(length)} "
                f"(rows = word count, cols = seed; {dimensions}D {dim_lbl}, {model_type})"
            ),
            dimensions=dimensions,
            embed_method=embed_method,
            model_type=model_type,
            spec=spec,
        ))

    for n_words in spec.word_counts:
        row_specs = tuple((n_words, length) for length in spec.lengths)
        out_path = by_word_count_dir / embed_save_path(f"{dim_tag}_w{n_words}.png", embed_method)
        paths.append(_plot_closed_loop_seed_comparison_grid(
            seeds=run_seeds,
            row_specs=row_specs,
            row_label_fn=lambda _n_words, length: spec.length_label(length),
            out_path=out_path,
            suptitle=(
                f"Pow2 sweep by word count · {n_words}w "
                f"(rows = length, cols = seed; {dimensions}D {dim_lbl}, {model_type})"
            ),
            dimensions=dimensions,
            embed_method=embed_method,
            model_type=model_type,
            spec=spec,
        ))

    return paths


def _grid_axes(n_rows: int, n_cols: int, *, is_3d: bool = False):
    if is_3d:
        fig = plt.figure(figsize=(1.55 * n_cols, 1.4 * n_rows))
        axes = np.empty((n_rows, n_cols), dtype=object)
        for li in range(n_rows):
            for wi in range(n_cols):
                axes[li, wi] = fig.add_subplot(
                    n_rows, n_cols, li * n_cols + wi + 1, projection="3d",
                )
        return fig, axes
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(1.55 * n_cols, 1.25 * n_rows),
        squeeze=False,
    )
    return fig, axes


def plot_pow2_sweep_learning_curves(
    *,
    seeds: tuple[int, ...] = (1,),
    model_type: str = "rnn",
    spec: Pow2SweepSpec = POW2_SWEEP_SPEC_NS,
) -> list[Path]:
    """Grid of training curves: rows = length, cols = word count (one file per seed)."""
    n_rows = len(spec.lengths)
    n_cols = len(spec.word_counts)
    out_dir = comparison_dir(spec.comparison_name, "learning_curves")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for run_seed in seeds:
        fig, axes = _grid_axes(n_rows, n_cols)
        for li, length in enumerate(spec.lengths):
            for wi, n_words in enumerate(spec.word_counts):
                ax = axes[li, wi]
                task = spec.task_name(n_words, length)
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
                    ax.set_ylabel(spec.length_label(length), fontsize=7)
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
        save_figure(fig, out_path, dpi=150)
        paths.append(out_path)
    return paths


def plot_pow2_sweep_closed_loop(
    *,
    seeds: tuple[int, ...] = (1,),
    dimensions: int = 2,
    embed_method: str = "pca",
    model_type: str = "rnn",
    spec: Pow2SweepSpec = POW2_SWEEP_SPEC_NS,
) -> list[Path]:
    """Grid of closed-loop trajectory PCA/jPCA panels per sweep cell."""
    if dimensions not in (2, 3):
        raise ValueError("dimensions must be 2 or 3")
    is_3d = dimensions == 3
    n_rows = len(spec.lengths)
    n_cols = len(spec.word_counts)
    out_dir = sweep_figures_dir(spec.comparison_name)
    paths: list[Path] = []

    for run_seed in seeds:
        fig, axes = _grid_axes(n_rows, n_cols, is_3d=is_3d)
        for li, length in enumerate(spec.lengths):
            for wi, n_words in enumerate(spec.word_counts):
                ax = axes[li, wi]
                task = spec.task_name(n_words, length)
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
                    ax.set_ylabel(spec.length_label(length), fontsize=7)

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
        save_figure(fig, out_path, dpi=150)
        paths.append(out_path)
    return paths


def run_pow2_sweep_seed_comparison_plots(
    *,
    seeds: tuple[int, ...] | None = None,
    model_type: str = "rnn",
    dimensions: int = 2,
    embed_method: str = "pca",
    spec: Pow2SweepSpec = POW2_SWEEP_SPEC_NS,
) -> list[Path]:
    run_seeds = seeds if seeds is not None else spec.seed_comparison_seeds
    paths = plot_pow2_sweep_closed_loop_seed_comparison(
        seeds=run_seeds,
        dimensions=dimensions,
        embed_method=embed_method,
        model_type=model_type,
        spec=spec,
    )
    for p in paths:
        print(f"wrote {p}")
    return paths


def run_pow2_sweep_demo_sequence_plots(
    *,
    seeds: tuple[int, ...] | None = None,
    spec: Pow2SweepSpec = POW2_SWEEP_SPEC_NS,
) -> list[Path]:
    run_seeds = seeds if seeds is not None else (1,)
    paths = plot_pow2_sweep_demo_sequences(seeds=run_seeds, spec=spec)
    for p in paths:
        print(f"wrote {p}")
    return paths


def run_pow2_sweep_learning_curve_plots(
    *,
    seeds: tuple[int, ...] | None = None,
    spec: Pow2SweepSpec = POW2_SWEEP_SPEC_NS,
) -> list[Path]:
    run_seeds = seeds if seeds is not None else (1,)
    paths = plot_pow2_sweep_learning_curves(seeds=run_seeds, spec=spec)
    for p in paths:
        print(f"wrote {p}")
    return paths


def run_pow2_sweep_closed_loop_plots(
    *,
    seeds: tuple[int, ...] | None = None,
    dimensions: tuple[int, ...] = (2,),
    embed_methods: tuple[str, ...] = ("pca",),
    spec: Pow2SweepSpec = POW2_SWEEP_SPEC_NS,
) -> list[Path]:
    """Closed-loop grids. Default is PCA only; pass embed_methods=("jpca",) if needed."""
    run_seeds = seeds if seeds is not None else (1,)
    paths: list[Path] = []
    for dims in dimensions:
        for method in embed_methods:
            paths.extend(plot_pow2_sweep_closed_loop(
                seeds=run_seeds, dimensions=dims, embed_method=method, spec=spec,
            ))
    for p in paths:
        print(f"wrote {p}")
    return paths
