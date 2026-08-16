"""Canonical figure entrypoint for the Dale fixed-letters grid sweep.

Reuses the mixed-DFA plot suite (via ``set_active_sweep``) so the fixed-letters
grid gets the same learning-curve, weight-matrix, and weight-graph-motif
figures without a parallel plotting pipeline.

Subcommands
-----------
curves        learning-curve overview + 4x6 word-error grid
weight-grids  small-multiple W_xh (input) and W_hh (hidden) heatmaps
motifs        weight-graph / E-I motif board (+ curated paper variant)
all           everything above

All figures land in ``comparisons/fixed_letters_grid_ns/trajectories/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import vocab_fixed_letters_grid as sweep_mod
import viz.compare.mixed_dfa_viz as mdv
from experiment import checkpoint_path
from viz.compare.sweep_output import sweep_data_dir, sweep_figures_dir
from viz.plot_layout import finalize_grid_figure, save_figure

mdv.set_active_sweep(sweep_mod)

COMPARISON = sweep_mod.COMPARISON_NAME
MODEL = "rnn_dale"
SEED = 1
WORD_LENS = sweep_mod.WORD_LENS
N_WORDS_LEVELS = sweep_mod.N_WORDS_LEVELS


def _manifest() -> list[dict]:
    path = sweep_mod.write_run_manifest(sweep_data_dir(COMPARISON) / "run_manifest.json")
    return json.loads(path.read_text(encoding="utf-8"))["runs"]


def _curves(task: str) -> dict | None:
    return mdv._overview_learning_curves_from_checkpoint(task, model_type=MODEL, seed=SEED)


def summarize(runs: list[dict]) -> None:
    print("run  L  nW  DFA  lastWE  bestWE  lastIter", flush=True)
    n_ok = 0
    for run in runs:
        ckpt = checkpoint_path(str(run["task"]), MODEL, seed=SEED)
        if not ckpt.is_file():
            print(f"r{int(run['run_id']):02d} MISSING", flush=True)
            continue
        d = np.load(ckpt, allow_pickle=True)
        last_we = float(d["metric_word_error_frac"][-1])
        best_we = float(d["best_metric_word_error_frac"])
        n_ok += int(best_we <= 0.03)
        print(
            f"r{int(run['run_id']):02d}  {int(run['word_length'])}  "
            f"{int(run['n_words']):2d}  {int(run['n_dfa_states']):3d}  "
            f"{last_we:6.3f}  {best_we:6.3f}  "
            f"{int(d['metric_iterations'][-1]):7d}",
            flush=True,
        )
    print(f"best-WE <= 3%: {n_ok} / {len(runs)}", flush=True)


def plot_learning_curves(runs: list[dict]) -> Path:
    """CE + word-error overlay, one line per run, colored by DFA size."""
    loaded: list[tuple[dict, dict]] = []
    for run in runs:
        curves = _curves(str(run["task"]))
        if curves is not None:
            loaded.append((run, curves))
    if not loaded:
        raise SystemExit("no learning curves found in checkpoints")

    dfas = [float(r["n_dfa_states"]) for r, _ in loaded]
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=min(dfas), vmax=max(dfas))

    fig = plt.figure(figsize=(9.6, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.05], wspace=0.28)
    ax_ce = fig.add_subplot(gs[0, 0])
    ax_we = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])
    for run, curves in sorted(loaded, key=lambda t: float(t[0]["n_dfa_states"])):
        color = cmap(norm(float(run["n_dfa_states"])))
        ax_ce.plot(curves["ce_iters"], curves["ce"], color=color, lw=1.0, alpha=0.75)
        ax_we.plot(curves["we_iters"], curves["word_err"], color=color, lw=1.0, alpha=0.75)
    ax_ce.set_xlabel("iteration", fontsize=8)
    ax_ce.set_ylabel("val CE / char", fontsize=8)
    ax_ce.set_title("cross-entropy learning", fontsize=9, pad=4)
    ax_ce.grid(True, alpha=0.25)
    ax_ce.tick_params(labelsize=7)
    ax_we.axhline(0.03, color="0.45", ls="--", lw=0.9, zorder=1)
    ax_we.set_xlabel("iteration", fontsize=8)
    ax_we.set_ylabel("word error frac", fontsize=8)
    ax_we.set_title("word-error learning", fontsize=9, pad=4)
    ax_we.set_ylim(0.0, 1.05)
    ax_we.grid(True, alpha=0.25)
    ax_we.tick_params(labelsize=7)
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax)
    cbar.set_label("DFA states", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    finalize_grid_figure(
        fig,
        suptitle="Dale fixed-letter grid: learning curves (H=200, seed 1)",
        bottom=0.16,
        left=0.07,
        right=0.94,
        top=0.84,
        wspace=0.28,
    )
    out = sweep_figures_dir(COMPARISON) / "learning_curves_dale.png"
    save_figure(fig, out)
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out


def plot_we_grid(runs: list[dict]) -> Path:
    """4x6 word-error grid: rows = word length, cols = n_words; reps overlaid."""
    by_cell: dict[tuple[int, int], list[tuple[dict, dict]]] = {}
    for run in runs:
        curves = _curves(str(run["task"]))
        if curves is None:
            continue
        key = (int(run["word_length"]), int(run["n_words"]))
        by_cell.setdefault(key, []).append((run, curves))

    nrows, ncols = len(WORD_LENS), len(N_WORDS_LEVELS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.4, 7.2), sharex=True, sharey=True)
    cmap = plt.get_cmap("tab10")
    for ri, length in enumerate(WORD_LENS):
        for ci, n_words in enumerate(N_WORDS_LEVELS):
            ax = axes[ri, ci]
            cell = by_cell.get((length, n_words), [])
            for k, (run, curves) in enumerate(cell):
                ax.plot(
                    curves["we_iters"], curves["word_err"],
                    color=cmap(k % 10), lw=1.1, alpha=0.9,
                    label=f"r{int(run['run_id']):02d} DFA={int(run['n_dfa_states'])}",
                )
            ax.axhline(0.03, color="0.55", ls="--", lw=0.7, zorder=1)
            ax.set_ylim(0.0, 1.05)
            ax.grid(True, alpha=0.22)
            ax.tick_params(labelsize=6)
            if ri == 0:
                ax.set_title(f"{n_words} words", fontsize=8, pad=3)
            if ci == 0:
                ax.set_ylabel(f"L={length}\nword err", fontsize=7)
            if ri == nrows - 1:
                ax.set_xlabel("iter", fontsize=7)
            if cell:
                ax.legend(fontsize=5, loc="upper right", frameon=False)
    finalize_grid_figure(
        fig,
        suptitle="Dale fixed-letter grid: word-error vs iteration (reps overlaid)",
        bottom=0.06,
        left=0.07,
        right=0.99,
        top=0.90,
        wspace=0.12,
        hspace=0.22,
    )
    out = sweep_figures_dir(COMPARISON) / "learning_curves_dale_grid.png"
    save_figure(fig, out)
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out


def cmd_curves(runs: list[dict]) -> None:
    plot_learning_curves(runs)
    plot_we_grid(runs)


def cmd_weight_grids(_runs: list[dict]) -> None:
    paths = mdv.plot_mixed_dfa_weight_matrix_grids(seed=SEED, model_type=MODEL)
    for p in paths:
        print(f"wrote {p}", flush=True)


def cmd_motifs(_runs: list[dict]) -> None:
    out = mdv.plot_mixed_dfa_weight_graph_metrics_vs_dfa(
        seed=SEED, recompute=True, model_type=MODEL,
    )
    print(f"wrote {out}", flush=True)
    out_paper = mdv.plot_mixed_dfa_weight_graph_metrics_paper(
        seed=SEED, recompute=False, model_type=MODEL,
    )
    print(f"wrote {out_paper}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "command",
        choices=["curves", "weight-grids", "motifs", "all"],
        help="which figure set to generate",
    )
    args = p.parse_args()

    runs = _manifest()
    summarize(runs)

    if args.command in ("curves", "all"):
        cmd_curves(runs)
    if args.command in ("weight-grids", "all"):
        cmd_weight_grids(runs)
    if args.command in ("motifs", "all"):
        cmd_motifs(runs)


if __name__ == "__main__":
    main()
