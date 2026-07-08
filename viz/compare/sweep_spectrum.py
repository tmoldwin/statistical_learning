"""PC variance spectra for word-count × length sweep (closed-loop hidden states)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from viz.compare.sweep_output import sweep_data_dir, sweep_figures_dir
from viz.compare._data import load_task_viz_context
from viz.compare.geometry import _GEOMETRY_TRIALS, _ROLLOUT_SEED, _mean_closed_loop_hidden
from viz.compare.state_space_metrics import pc_variance_percent
from viz.plot_layout import finalize_grid_figure, save_figure
from visualize import (
    _closed_loop_summary_seed,
    _one_vocab_cycle_steps,
    _trajectory_seed_letters,
)
from vocab_sweep import (
    SWEEP_DEFAULT_SEEDS,
    SWEEP_LENGTHS,
    SWEEP_WORD_COUNTS,
    iter_sweep_cells,
    task_name,
)

SWEEP_COMPARISON_NAME = "word_length_sweep_ns"
_DEFAULT_MAX_PCS = 20
_LENGTH_COLORS = ("#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B")


def _pad_spectrum(spectrum: np.ndarray | list[float], max_pcs: int) -> np.ndarray:
    """Pad/truncate to ``max_pcs`` (trajectories have at most T-1 nonzero PCs)."""
    arr = np.asarray(spectrum, dtype=float)
    out = np.zeros(max_pcs, dtype=float)
    n = min(len(arr), max_pcs)
    if n:
        out[:n] = arr[:n]
    return out


def _loop_pc_spectrum(ctx) -> np.ndarray:
    """% variance per PC on the trial-averaged closed loop in ℝᴴ."""
    vocab_words = list(ctx.words)
    seed_letters = _trajectory_seed_letters(ctx.model, vocab_words)
    summary_seed = _closed_loop_summary_seed(vocab_words, seed_letters, spaced=ctx.spaced)
    summary_steps = _one_vocab_cycle_steps(vocab_words, spaced=ctx.spaced)
    if vocab_words:
        summary_steps += max(len(w) for w in vocab_words)
    mean_loop = _mean_closed_loop_hidden(
        ctx.model,
        summary_seed=summary_seed,
        steps=summary_steps,
        rollout_seed=_ROLLOUT_SEED,
        n_trials=_GEOMETRY_TRIALS,
        vocab_words=vocab_words,
        spaced=ctx.spaced,
    )
    if mean_loop is None or len(mean_loop) < 3:
        return np.empty(0)
    return pc_variance_percent(mean_loop)


def write_sweep_spectra(
    *,
    seeds: tuple[int, ...] | None = None,
    model_type: str = "rnn",
    max_pcs: int = _DEFAULT_MAX_PCS,
    outfile: str = "sweep_spectra.json",
) -> Path:
    run_seeds = seeds if seeds is not None else SWEEP_DEFAULT_SEEDS
    panels: list[dict[str, Any]] = []

    for n_words, length in iter_sweep_cells():
        task = task_name(n_words, length)
        seed_spectra: list[list[float]] = []
        for run_seed in run_seeds:
            try:
                ctx = load_task_viz_context(task, model_type=model_type, seed=run_seed)
            except (FileNotFoundError, KeyError):
                continue
            spectrum = _loop_pc_spectrum(ctx)
            if len(spectrum):
                seed_spectra.append(_pad_spectrum(spectrum, max_pcs).tolist())
            print(f"  {task} seed {run_seed}", flush=True)

        mean_spectrum: list[float] = []
        if seed_spectra:
            arr = np.asarray(seed_spectra, dtype=float)
            mean_spectrum = np.mean(arr, axis=0).tolist()

        panels.append({
            "task": task,
            "n_words": n_words,
            "length": length,
            "n_seeds": len(seed_spectra),
            "max_pcs": max_pcs,
            "spectrum_pct": mean_spectrum,
        })

    out_path = sweep_data_dir(SWEEP_COMPARISON_NAME) / outfile
    payload = {
        "comparison": SWEEP_COMPARISON_NAME,
        "model_type": model_type,
        "word_counts": list(SWEEP_WORD_COUNTS),
        "lengths": list(SWEEP_LENGTHS),
        "seeds": list(run_seeds),
        "max_pcs": max_pcs,
        "panels": panels,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _spectrum_display_end(spectrum: np.ndarray, *, floor_pct: float = 0.5) -> int:
    """Last PC index (1-based count) with at least ``floor_pct`` variance."""
    y = np.asarray(spectrum, dtype=float)
    if len(y) == 0:
        return 1
    above = np.where(y >= floor_pct)[0]
    if len(above) == 0:
        return min(3, len(y))
    return int(above[-1]) + 1


def plot_sweep_spectra(
    panels: list[dict[str, Any]],
    *,
    word_counts: tuple[int, ...] = SWEEP_WORD_COUNTS,
    lengths: tuple[int, ...] = SWEEP_LENGTHS,
    max_pcs: int = _DEFAULT_MAX_PCS,
    outfile: str = "sweep_pc_spectra.png",
) -> Path:
    """One panel per word count; overlapping PC scree curves for each letter length."""
    lookup = {(p["n_words"], p["length"]): p for p in panels}
    n_word_panels = len(word_counts)
    n_cols = min(2, n_word_panels)
    n_rows = int(np.ceil(n_word_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.2 * n_cols, 3.8 * n_rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    pc_x = np.arange(1, max_pcs + 1, dtype=float)

    global_xmax = 3
    global_ymax = 1.0
    for n_words in word_counts:
        for length in lengths:
            spectrum = lookup.get((n_words, length), {}).get("spectrum_pct") or []
            if spectrum:
                y = np.asarray(spectrum, dtype=float)
                global_xmax = max(global_xmax, _spectrum_display_end(y))
                global_ymax = max(global_ymax, float(np.max(y)))
    global_ymax *= 1.05

    for pi, n_words in enumerate(word_counts):
        ax = axes.ravel()[pi]
        for li, length in enumerate(lengths):
            panel = lookup.get((n_words, length), {})
            spectrum = panel.get("spectrum_pct") or []
            if not spectrum:
                continue
            y = np.asarray(spectrum, dtype=float)
            n_plot = min(len(y), global_xmax)
            color = _LENGTH_COLORS[li % len(_LENGTH_COLORS)]
            ax.plot(
                pc_x[:n_plot],
                y[:n_plot],
                color=color,
                linewidth=1.8,
                alpha=0.85,
                marker="o",
                markersize=3.5,
                label=f"{length}-letter",
            )
        ax.set_title(f"{n_words} words", fontsize=10)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.85)
        row, col = divmod(pi, n_cols)
        if row == n_rows - 1:
            ax.set_xlabel("PC #", fontsize=9)
        else:
            ax.tick_params(labelbottom=False)
        if col == 0:
            ax.set_ylabel("% variance", fontsize=9)

    for ax in axes.ravel()[:n_word_panels]:
        ax.set_xlim(0.8, global_xmax + 0.35)
        ax.set_ylim(0, global_ymax)
        ax.set_xticks(range(1, global_xmax + 1))

    for ax in axes.ravel()[n_word_panels:]:
        ax.axis("off")

    finalize_grid_figure(
        fig,
        suptitle="Closed-loop PC variance spectra (mean over seeds)",
        top=0.92,
        hspace=0.38,
        wspace=0.28,
    )
    out_path = sweep_figures_dir(SWEEP_COMPARISON_NAME) / outfile
    save_figure(fig, out_path, dpi=160)
    return out_path


def replot_sweep_spectra(
    *,
    spectra_file: str = "sweep_spectra.json",
    outfile: str = "sweep_pc_spectra.png",
) -> Path:
    payload = json.loads(
        (sweep_data_dir(SWEEP_COMPARISON_NAME) / spectra_file).read_text(encoding="utf-8"),
    )
    return plot_sweep_spectra(
        payload["panels"],
        max_pcs=int(payload.get("max_pcs", _DEFAULT_MAX_PCS)),
        outfile=outfile,
    )


def run_sweep_spectrum_plots(
    *,
    seeds: tuple[int, ...] | None = None,
    max_pcs: int = _DEFAULT_MAX_PCS,
    recompute: bool = True,
    spectra_file: str = "sweep_spectra.json",
) -> tuple[Path, Path]:
    json_path = sweep_data_dir(SWEEP_COMPARISON_NAME) / spectra_file
    if recompute or not json_path.is_file():
        json_path = write_sweep_spectra(seeds=seeds, max_pcs=max_pcs, outfile=spectra_file)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    fig_path = plot_sweep_spectra(
        payload["panels"],
        max_pcs=int(payload.get("max_pcs", max_pcs)),
    )
    print(f"wrote {json_path}")
    print(f"wrote {fig_path}")
    return json_path, fig_path
