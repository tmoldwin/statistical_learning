"""Entrywise spikiness / ergodicity of W_xh vs W_hh vs DFA size."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiment import checkpoint_path
from viz.compare.pow2_sweep_metric_board import _fit_trend
from viz.compare.sweep_output import sweep_data_dir, sweep_figures_dir
from viz.plot_layout import finalize_grid_figure, save_figure
from viz.weight_structure import (
    compute_weight_motif_metrics,
    compute_weight_spikiness_pair,
    compute_weight_structure_metrics,
)


def collect_weight_spikiness_panels(
    *,
    comparison: str,
    runs: Iterable[dict[str, Any]],
    seed: int = 1,
    model_type: str = "rnn",
    dfa_from_entry: Callable[[dict[str, Any]], int] | None = None,
) -> Path:
    """Load checkpoints and write structure + entrywise spikiness JSON."""
    from visualize import load_model_for_viz, weights_for_plot

    panels: list[dict[str, Any]] = []
    for entry in runs:
        task = str(entry["task"])
        ckpt = checkpoint_path(task, model_type, seed=seed)
        if dfa_from_entry is not None:
            n_dfa = int(dfa_from_entry(entry))
        else:
            n_dfa = int(entry["n_dfa_states"])
        if not ckpt.is_file():
            panels.append({
                "task": task, "seed": seed, "n_dfa_states": n_dfa,
                "error": f"missing {ckpt}",
            })
            continue
        print(f"spikiness {task} seed {seed} dfa={n_dfa}", flush=True)
        try:
            model = load_model_for_viz(str(ckpt), model_type)
            w_in, w_rec, w_out, _dale = weights_for_plot(model)
            motif = compute_weight_motif_metrics(w_in, w_rec)
            panels.append({
                "task": task,
                "seed": seed,
                "run_id": int(entry.get("run_id", -1)),
                "n_words": int(entry.get("n_words", 0)),
                "n_letters": int(entry.get("n_letters", w_in.shape[1])),
                "n_dfa_states": n_dfa,
                "structure": compute_weight_structure_metrics(w_in, w_rec, w_out),
                "motif": {
                    "xh_top1_mass": motif["xh_top1_mass"],
                    "input_tuning_entropy": motif["input_tuning_entropy"],
                },
                "spikiness": compute_weight_spikiness_pair(w_in, w_rec),
            })
        except Exception as exc:  # noqa: BLE001
            panels.append({
                "task": task, "seed": seed, "n_dfa_states": n_dfa,
                "error": str(exc),
            })

    out = sweep_data_dir(comparison) / "weight_spikiness_vs_dfa.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "comparison": comparison,
        "seed": seed,
        "model_type": model_type,
        "panels": panels,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return out


def plot_weight_spikiness_vs_dfa(
    *,
    comparison: str,
    outfile: str = "weight_spikiness_vs_dfa.png",
    title: str | None = None,
    seed: int = 1,
) -> Path:
    """Two-row board: norms + entrywise spikiness of W_xh vs W_hh."""
    path = sweep_data_dir(comparison) / "weight_spikiness_vs_dfa.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    panels = [p for p in payload.get("panels", []) if "error" not in p and "spikiness" in p]
    if not panels:
        raise FileNotFoundError(f"no spikiness panels in {path}")

    dfa = np.asarray([float(p["n_dfa_states"]) for p in panels], dtype=float)
    win = np.asarray([float(p["structure"]["input_frobenius"]) for p in panels])
    whh = np.asarray([float(p["structure"]["recurrent_frobenius"]) for p in panels])
    ratio = np.asarray([float(p["structure"]["input_over_recurrent_norm"]) for p in panels])
    letter_top1 = np.asarray([float(p["motif"]["xh_top1_mass"]) for p in panels])

    xh_pr = np.asarray([float(p["spikiness"]["xh"]["participation_frac"]) for p in panels])
    hh_pr = np.asarray([float(p["spikiness"]["hh"]["participation_frac"]) for p in panels])
    xh_top = np.asarray([float(p["spikiness"]["xh"]["top1_mass"]) for p in panels])
    hh_top = np.asarray([float(p["spikiness"]["hh"]["top1_mass"]) for p in panels])
    xh_gini = np.asarray([float(p["spikiness"]["xh"]["gini"]) for p in panels])
    hh_gini = np.asarray([float(p["spikiness"]["hh"]["gini"]) for p in panels])

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.6))

    def _scatter_fit(ax, x, y, *, color: str, label: str) -> float:
        ax.scatter(x, y, s=18, alpha=0.85, color=color, label=label,
                   linewidths=0.2, edgecolors="white", zorder=2)
        x_fit, y_fit, r2, _ = _fit_trend(x, y)
        if x_fit is not None and y_fit is not None and np.isfinite(r2):
            ax.plot(x_fit, y_fit, color=color, lw=1.2, alpha=0.85, zorder=3)
            return float(r2)
        return float("nan")

    # Row 0: norms + letter specialization
    ax = axes[0, 0]
    r_in = _scatter_fit(ax, dfa, win, color="#1f77b4", label=r"$W_{xh}$")
    r_hh = _scatter_fit(ax, dfa, whh, color="#d62728", label=r"$W_{hh}$")
    ax.set_ylabel("Frobenius norm", fontsize=8)
    ax.set_title(
        f"matrix norms\n$R^2$ xh={r_in:.2f}  hh={r_hh:.2f}",
        fontsize=8, pad=3,
    )
    ax.legend(fontsize=6.5, loc="best", framealpha=0.9)

    ax = axes[0, 1]
    r_ratio = _scatter_fit(ax, dfa, ratio, color="#2ca02c", label="ratio")
    ax.set_ylabel(r"$||W_{xh}||_F / ||W_{hh}||_F$", fontsize=8)
    ax.set_title(f"input / recurrent\n$R^2$={r_ratio:.2f}", fontsize=8, pad=3)

    ax = axes[0, 2]
    r_let = _scatter_fit(ax, dfa, letter_top1, color="#1f77b4", label="letter top-1")
    ax.set_ylabel("mean row top-1 letter mass", fontsize=8)
    ax.set_title(
        f"$W_{{xh}}$ letter specialization\n$R^2$={r_let:.2f}",
        fontsize=8, pad=3,
    )

    # Row 1: entrywise spikiness (comparable via participation_frac)
    ax = axes[1, 0]
    r1 = _scatter_fit(ax, dfa, xh_pr, color="#1f77b4", label=r"$W_{xh}$")
    r2 = _scatter_fit(ax, dfa, hh_pr, color="#d62728", label=r"$W_{hh}$")
    ax.set_ylabel("participation frac  (PR / n)", fontsize=8)
    ax.set_title(
        f"entry ergodicity  (higher = more spread)\n$R^2$ xh={r1:.2f}  hh={r2:.2f}",
        fontsize=8, pad=3,
    )
    ax.legend(fontsize=6.5, loc="best", framealpha=0.9)

    ax = axes[1, 1]
    r1 = _scatter_fit(ax, dfa, xh_top, color="#1f77b4", label=r"$W_{xh}$")
    r2 = _scatter_fit(ax, dfa, hh_top, color="#d62728", label=r"$W_{hh}$")
    ax.set_ylabel("entry top-1 mass", fontsize=8)
    ax.set_title(
        f"entry spikiness  (higher = more peaked)\n$R^2$ xh={r1:.2f}  hh={r2:.2f}",
        fontsize=8, pad=3,
    )
    ax.legend(fontsize=6.5, loc="best", framealpha=0.9)

    ax = axes[1, 2]
    r1 = _scatter_fit(ax, dfa, xh_gini, color="#1f77b4", label=r"$W_{xh}$")
    r2 = _scatter_fit(ax, dfa, hh_gini, color="#d62728", label=r"$W_{hh}$")
    ax.set_ylabel("Gini of |w|", fontsize=8)
    ax.set_title(
        f"inequality  (higher = spikier)\n$R^2$ xh={r1:.2f}  hh={r2:.2f}",
        fontsize=8, pad=3,
    )
    ax.legend(fontsize=6.5, loc="best", framealpha=0.9)

    for ax in axes.ravel():
        ax.set_xlabel("DFA states", fontsize=7.5)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=6.5)

    finalize_grid_figure(
        fig,
        suptitle=title or f"Weight spikiness / ergodicity vs DFA  ({comparison}, seed {seed})",
        top=0.86, bottom=0.10, left=0.08, right=0.98, hspace=0.42, wspace=0.32,
    )
    out = sweep_figures_dir(comparison) / outfile
    save_figure(fig, out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out
