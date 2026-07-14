"""PC variance spectra for the powers-of-2 word-count × length sweep."""

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
from viz.compare.pow2_sweep_spec import POW2_SWEEP_SPEC_NS, Pow2SweepSpec
from vocab_sweep_pow2 import (
    POW2_DEFAULT_SEEDS,
    POW2_LENGTHS,
    POW2_WORD_COUNTS,
    iter_pow2_sweep_cells,
    length_label,
    task_name,
)

POW2_SWEEP_COMPARISON_NAME = POW2_SWEEP_SPEC_NS.comparison_name
_DEFAULT_MAX_PCS = 20
_LENGTH_COLORS = ("#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B", "#EECA3B", "#B279A2")


def _pad_spectrum(spectrum: np.ndarray | list[float], max_pcs: int) -> np.ndarray:
    """Pad/truncate to ``max_pcs`` (trajectories have at most T-1 nonzero PCs)."""
    arr = np.asarray(spectrum, dtype=float)
    out = np.zeros(max_pcs, dtype=float)
    n = min(len(arr), max_pcs)
    if n:
        out[:n] = arr[:n]
    return out


def _loop_pc_spectrum(ctx) -> np.ndarray:
    """% variance per PC on the trial-averaged closed loop in R^H."""
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


def write_pow2_sweep_spectra(
    *,
    seeds: tuple[int, ...] | None = None,
    model_type: str = "rnn",
    max_pcs: int = _DEFAULT_MAX_PCS,
    outfile: str = "sweep_spectra.json",
    spec: Pow2SweepSpec = POW2_SWEEP_SPEC_NS,
) -> Path:
    run_seeds = seeds if seeds is not None else spec.default_seeds
    panels: list[dict[str, Any]] = []

    for n_words, length in spec.iter_cells():
        task = spec.task_name(n_words, length)
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

    out_path = sweep_data_dir(spec.comparison_name) / outfile
    payload = {
        "comparison": spec.comparison_name,
        "model_type": model_type,
        "word_counts": list(spec.word_counts),
        "lengths": list(spec.lengths),
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


def plot_pow2_sweep_spectra(
    panels: list[dict[str, Any]],
    *,
    word_counts: tuple[int, ...] | None = None,
    lengths: tuple[object, ...] | None = None,
    max_pcs: int = _DEFAULT_MAX_PCS,
    outfile: str = "sweep_pc_spectra.png",
    spec: Pow2SweepSpec = POW2_SWEEP_SPEC_NS,
) -> Path:
    """One panel per word count; overlapping PC scree curves for each letter length."""
    word_counts = spec.word_counts if word_counts is None else word_counts
    lengths = spec.lengths if lengths is None else lengths
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
                label=spec.length_label(length),
            )
        ax.set_title(f"{n_words} words", fontsize=10)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
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
    out_path = sweep_figures_dir(spec.comparison_name) / outfile
    save_figure(fig, out_path, dpi=160)
    return out_path



_WORD_MARKERS = ("o", "s", "^", "D", "v", "P", "X")
_MIXED_LENGTH_COLOR = "#111111"  # mixed kept off the letter-length gradient


def _length_plot_color(length: object, lengths: tuple[object, ...]) -> str:
    """Sequential gradient for letter lengths 1..N; mixed gets its own hue."""
    if length == "mixed" or length == "mix":
        return _MIXED_LENGTH_COLOR
    letter_lengths = [L for L in lengths if isinstance(L, int) or (isinstance(L, str) and str(L).isdigit())]
    letter_lengths = sorted(int(L) for L in letter_lengths)
    if not letter_lengths:
        return _LENGTH_COLORS[0]
    L = int(length)
    if len(letter_lengths) == 1:
        t = 0.0
    else:
        t = (L - letter_lengths[0]) / (letter_lengths[-1] - letter_lengths[0])
    # Skip very pale ends of Greens for readability on white.
    cmap = plt.cm.Greens
    rgba = cmap(0.30 + 0.65 * float(np.clip(t, 0.0, 1.0)))
    return "#{:02x}{:02x}{:02x}".format(
        int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255),
    )


def plot_pow2_sweep_spectra_overlay(
    panels: list[dict[str, Any]],
    *,
    word_counts: tuple[int, ...] | None = None,
    lengths: tuple[object, ...] | None = None,
    max_pcs: int = _DEFAULT_MAX_PCS,
    outfile: str = "sweep_pc_spectra_overlay.png",
    cumulative: bool = True,
    spec: Pow2SweepSpec = POW2_SWEEP_SPEC_NS,
) -> Path:
    """Single panel: color = word length, marker = # words."""
    word_counts = spec.word_counts if word_counts is None else word_counts
    lengths = spec.lengths if lengths is None else lengths
    lookup = {(p["n_words"], p["length"]): p for p in panels}

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    pc_x = np.arange(1, max_pcs + 1, dtype=float)

    global_xmax = 3
    for n_words in word_counts:
        for length in lengths:
            spectrum = lookup.get((n_words, length), {}).get("spectrum_pct") or []
            if not spectrum:
                continue
            y = np.asarray(spectrum, dtype=float)
            if cumulative:
                y = np.cumsum(y)
                # stop once we are essentially flat at ~100
                n_plot = int(np.searchsorted(y, 99.5, side="left")) + 1
                n_plot = max(n_plot, _spectrum_display_end(np.asarray(spectrum, dtype=float)))
            else:
                n_plot = _spectrum_display_end(y)
            global_xmax = max(global_xmax, min(n_plot, len(y), max_pcs))

    # Length color legend handles
    length_handles = []
    for length in lengths:
        color = _length_plot_color(length, lengths)
        h, = ax.plot([], [], color=color, linewidth=2.0, label=spec.length_label(length))
        length_handles.append(h)

    # Word-count marker legend handles
    marker_handles = []
    for wi, n_words in enumerate(word_counts):
        marker = _WORD_MARKERS[wi % len(_WORD_MARKERS)]
        h, = ax.plot(
            [], [], color="0.35", marker=marker, linestyle="None",
            markersize=7, label=f"{n_words} words",
        )
        marker_handles.append(h)

    y_min_seen = float("inf")
    y_max_seen = float("-inf")
    for length in lengths:
        color = _length_plot_color(length, lengths)
        for wi, n_words in enumerate(word_counts):
            spectrum = lookup.get((n_words, length), {}).get("spectrum_pct") or []
            if not spectrum:
                continue
            raw = np.asarray(spectrum, dtype=float)
            y = np.cumsum(raw) if cumulative else raw
            n_plot = min(len(y), global_xmax)
            y_plot = y[:n_plot]
            y_min_seen = min(y_min_seen, float(np.min(y_plot)))
            y_max_seen = max(y_max_seen, float(np.max(y_plot)))
            marker = _WORD_MARKERS[wi % len(_WORD_MARKERS)]
            ax.plot(
                pc_x[:n_plot],
                y_plot,
                color=color,
                linewidth=1.5,
                alpha=0.9,
                marker=marker,
                markersize=5.5,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.4,
            )

    ylabel = "cumulative % variance" if cumulative else "% variance"
    ax.set_xlabel("PC #", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlim(0.8, global_xmax + 0.35)
    # Pad around the full plotted envelope so PC1 mins stay visible.
    if np.isfinite(y_min_seen) and np.isfinite(y_max_seen):
        span = max(y_max_seen - y_min_seen, 1.0)
        pad = 0.04 * span
        ymin = max(0.0, y_min_seen - pad)
        ymax = (min(101.5, max(y_max_seen + pad, 100.0)) if cumulative
                else y_max_seen + pad)
        ax.set_ylim(ymin, ymax)
    elif cumulative:
        ax.set_ylim(0, 105)
    ax.set_xticks(range(1, global_xmax + 1))
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)

    leg1 = ax.legend(
        handles=length_handles, title="word length",
        bbox_to_anchor=(1.02, 1.0), loc="upper left",
        fontsize=8, title_fontsize=8, frameon=False, borderaxespad=0.0,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=marker_handles, title="# words",
        bbox_to_anchor=(1.02, 0.52), loc="upper left",
        fontsize=8, title_fontsize=8, frameon=False, borderaxespad=0.0,
    )

    fig.suptitle(
        "Closed-loop PC variance explained (mean over seeds)\n"
        "color = word length, marker = # words",
        fontsize=11,
        y=0.98,
    )
    fig.subplots_adjust(top=0.86, bottom=0.12, left=0.10, right=0.78)

    out_path = sweep_figures_dir(spec.comparison_name) / outfile
    save_figure(fig, out_path, dpi=160)
    print(f"wrote {out_path}")
    return out_path


def replot_pow2_sweep_spectra(
    *,
    spectra_file: str = "sweep_spectra.json",
    outfile: str = "sweep_pc_spectra.png",
    spec: Pow2SweepSpec = POW2_SWEEP_SPEC_NS,
) -> Path:
    payload = json.loads(
        (sweep_data_dir(spec.comparison_name) / spectra_file).read_text(encoding="utf-8"),
    )
    return plot_pow2_sweep_spectra(
        payload["panels"],
        max_pcs=int(payload.get("max_pcs", _DEFAULT_MAX_PCS)),
        outfile=outfile,
        spec=spec,
    )


def run_pow2_sweep_spectrum_plots(
    *,
    seeds: tuple[int, ...] | None = None,
    max_pcs: int = _DEFAULT_MAX_PCS,
    recompute: bool = True,
    spectra_file: str = "sweep_spectra.json",
    spec: Pow2SweepSpec = POW2_SWEEP_SPEC_NS,
) -> tuple[Path, Path, Path]:
    json_path = sweep_data_dir(spec.comparison_name) / spectra_file
    if recompute or not json_path.is_file():
        json_path = write_pow2_sweep_spectra(
            seeds=seeds, max_pcs=max_pcs, outfile=spectra_file, spec=spec,
        )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    fig_path = plot_pow2_sweep_spectra(
        payload["panels"],
        max_pcs=int(payload.get("max_pcs", max_pcs)),
        spec=spec,
    )
    overlay_path = plot_pow2_sweep_spectra_overlay(
        payload["panels"],
        max_pcs=int(payload.get("max_pcs", max_pcs)),
        spec=spec,
    )
    print(f"wrote {json_path}")
    print(f"wrote {fig_path}")
    print(f"wrote {overlay_path}")
    return json_path, fig_path, overlay_path
