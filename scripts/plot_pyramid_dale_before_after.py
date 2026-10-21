"""Before/after pyramids for selective units of the trained pyramid Dale RNN."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rnn.rnn_dyn import reconstruct_init_weights
from visualize import (
    _draw_letter_pyramid_on_axes,
    _mean_activation_by_current_letter,
    forward_pass,
    load_model,
)
from vocab_pyramid import EDGES, LETTERS, PYRAMID_NODE_XY

TASK = "pyramid_letters"
SEED = 42
MODEL = REPO / "experiments" / TASK / "rnn_dale" / "model.npz"
TEXT = REPO / "experiments" / TASK / "input.txt"
OUT = REPO / "experiments" / TASK / "rnn_dale" / "plots" / "unit_selectivity" / "neuron_pyramids_before_after.png"
HEATMAP_OUT = REPO / "experiments" / TASK / "rnn_dale" / "plots" / "unit_selectivity" / "letter_heatmap_before_after.png"
SI_MIN = 0.5
N_MAX = 12


def _selectivity(mean_act: np.ndarray) -> np.ndarray:
    """Peak-vs-rest SI in [0, 1]. mean_act is (n_units, n_letters)."""
    shifted = mean_act - mean_act.min(axis=1, keepdims=True)
    r_max = shifted.max(axis=1)
    is_max = np.abs(shifted - r_max[:, None]) <= 1e-15
    n_below = shifted.shape[1] - is_max.sum(axis=1)
    rest = np.where(is_max, 0.0, shifted).sum(axis=1) / np.maximum(n_below, 1)
    si = (r_max - rest) / (r_max + rest + 1e-12)
    si = np.where((r_max > 1e-12) & (n_below > 0), si, 0.0)
    return np.clip(si, 0.0, 1.0)


def _init_model(trained: dict) -> dict:
    hidden = int(trained["hidden_size"])
    vocab = int(trained["vocab_size"])
    w_ih, w_hh, w_ho = reconstruct_init_weights(
        hidden, vocab, SEED, dale_law=True, e_fraction=float(trained["e_fraction"]),
    )
    model = dict(trained)
    model["weights_input_to_hidden"] = w_ih
    model["weights_hidden_to_hidden"] = w_hh
    model["weights_hidden_to_output"] = w_ho
    model["bias_hidden"] = np.zeros_like(trained["bias_hidden"])
    model["bias_output"] = np.zeros_like(trained["bias_output"])
    model["timestep_noise_std"] = 0.0
    return model


def _selective_units(signs: np.ndarray, si: np.ndarray) -> list[int]:
    chosen: list[int] = []
    for mask in (signs > 0, signs < 0):
        pool = np.flatnonzero(mask & (si >= SI_MIN))
        if pool.size == 0:
            pool = np.flatnonzero(mask)
        order = pool[np.argsort(-si[pool])]
        chosen.extend(int(u) for u in order[: N_MAX // 2])
    chosen.sort(key=lambda u: -float(si[u]))
    return chosen


def _paint(ax, row: np.ndarray, cmap) -> None:
    peak = float(np.max(row))
    faces = {}
    for j, letter in enumerate(LETTERS):
        level = float(row[j]) / peak if peak > 0.0 else 0.0
        rgba = cmap(level)
        faces[letter] = (float(rgba[0]), float(rgba[1]), float(rgba[2]), 1.0)
    _draw_letter_pyramid_on_axes(ax, edges=EDGES, pos=PYRAMID_NODE_XY, node_face_rgba=faces)
    for letter, (x, y) in PYRAMID_NODE_XY.items():
        ax.text(x, y, letter, ha="center", va="center", fontsize=7, color="#111111", zorder=4)


def _preferred_order(mean_act: np.ndarray) -> np.ndarray:
    """Group by the letter with the highest mean firing, then by that firing."""
    preferred = np.argmax(mean_act, axis=1)
    preferred_rate = mean_act[np.arange(mean_act.shape[0]), preferred]
    return np.lexsort((-preferred_rate, preferred))


def plot_letter_heatmaps(
    mean_before: np.ndarray,
    mean_after: np.ndarray,
    signs: np.ndarray,
    save_path: Path,
) -> None:
    """Mean firing to each letter. Same after-learning row order in both columns."""
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.6), sharex=True)
    panels = (
        (0, "Excitatory", signs > 0),
        (1, "Inhibitory", signs < 0),
    )
    for row, name, mask in panels:
        order = _preferred_order(mean_after[mask])
        pair = (
            mean_before[mask][order],
            mean_after[mask][order],
        )
        for col, (block, stage) in enumerate(zip(pair, ("before", "after"))):
            ax = axes[row, col]
            panel_peak = float(np.max(block)) if block.size else 1.0
            im = ax.imshow(
                block,
                aspect="auto",
                cmap="Reds",
                vmin=0.0,
                vmax=panel_peak if panel_peak > 0.0 else 1.0,
                interpolation="nearest",
            )
            ax.set_xticks(range(len(LETTERS)))
            ax.set_xticklabels(list(LETTERS))
            ax.set_yticks([])
            ax.set_title(f"{name} · {stage}  ·  n={int(mask.sum())}", fontsize=11)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cbar.set_label("mean firing", fontsize=8)
            cbar.ax.tick_params(labelsize=7)
        axes[row, 0].set_ylabel("neurons, grouped by preferred letter", fontsize=9)
    fig.suptitle("Mean firing to each letter · before and after learning", fontsize=12)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {save_path}")


def main() -> None:
    trained = load_model(str(MODEL))
    trained["timestep_noise_std"] = 0.0
    initial = _init_model(trained)
    text = TEXT.read_text(encoding="utf-8")
    before = forward_pass(initial, text)[0]
    after = forward_pass(trained, text)[0]
    mean_before = _mean_activation_by_current_letter(text, before, LETTERS)
    mean_after = _mean_activation_by_current_letter(text, after, LETTERS)
    si_after = _selectivity(mean_after)
    signs = np.asarray(trained["dale_sign"], dtype=float)
    units = _selective_units(signs, si_after)
    print("selective", [(u, "E" if signs[u] > 0 else "I", f"{si_after[u]:.2f}") for u in units])

    cmap = plt.get_cmap("Reds")
    ncols = len(units)
    fig, axes = plt.subplots(2, ncols, figsize=(1.55 * ncols + 1.1, 3.7), squeeze=False)
    for col, unit in enumerate(units):
        kind = "E" if signs[unit] > 0 else "I"
        _paint(axes[0, col], mean_before[unit], cmap)
        _paint(axes[1, col], mean_after[unit], cmap)
        axes[0, col].set_title(
            f"h{unit} {kind}\nSI {si_after[unit]:.2f}",
            fontsize=8,
            pad=2,
        )
    fig.subplots_adjust(left=0.06, right=0.90, wspace=0.15, hspace=0.08)
    fig.text(0.005, 0.72, "before", rotation=90, va="center", ha="left", fontsize=9)
    fig.text(0.005, 0.30, "after", rotation=90, va="center", ha="left", fontsize=9)
    cax = fig.add_axes([0.92, 0.22, 0.012, 0.56])
    sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("fraction of this pyramid's max", fontsize=8)
    fig.suptitle("Selective neurons · before and after learning", fontsize=12, y=1.02)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT}")
    plot_letter_heatmaps(mean_before, mean_after, signs, HEATMAP_OUT)


if __name__ == "__main__":
    main()
