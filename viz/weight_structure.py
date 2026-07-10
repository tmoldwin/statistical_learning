"""Init-vs-final weight structure analysis (feedforward vs recurrent balance)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

from rnn.rnn_dyn import reconstruct_init_weights
from viz.plot_layout import finalize_grid_figure, save_figure


def _frobenius_norm(w: np.ndarray) -> float:
    return float(np.linalg.norm(w, "fro"))


def _spectral_radius(w: np.ndarray) -> float:
    eigs = np.linalg.eigvals(w)
    return float(np.max(np.abs(eigs)))


def _per_unit_input_drive_fraction(w_in: np.ndarray, w_rec: np.ndarray) -> np.ndarray:
    """Per hidden unit: mean |input| / (mean |input| + mean |recurrent in|)."""
    mean_in = np.mean(np.abs(w_in), axis=1)
    mean_rec = np.mean(np.abs(w_rec), axis=1)
    denom = mean_in + mean_rec
    return np.where(denom > 0, mean_in / denom, 0.5)


def compute_weight_structure_metrics(
    w_in: np.ndarray,
    w_rec: np.ndarray,
    w_out: np.ndarray | None = None,
) -> dict[str, float]:
    """Scalar summaries of input vs recurrent dominance."""
    mean_in = np.mean(np.abs(w_in), axis=1)
    mean_rec_in = np.mean(np.abs(w_rec), axis=1)
    drive_frac = _per_unit_input_drive_fraction(w_in, w_rec)
    out: dict[str, float] = {
        "input_frobenius": _frobenius_norm(w_in),
        "recurrent_frobenius": _frobenius_norm(w_rec),
        "input_over_recurrent_norm": _frobenius_norm(w_in) / max(_frobenius_norm(w_rec), 1e-12),
        "mean_input_drive_fraction": float(np.mean(drive_frac)),
        "median_input_drive_fraction": float(np.median(drive_frac)),
        "spectral_radius_hh": _spectral_radius(w_rec),
        "mean_abs_input_per_unit": float(np.mean(mean_in)),
        "mean_abs_recurrent_in_per_unit": float(np.mean(mean_rec_in)),
    }
    if w_out is not None:
        out["readout_frobenius"] = _frobenius_norm(w_out)
    return out


def init_weights_for_model(model: dict, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hidden_size = int(model["hidden_size"])
    vocab_size = int(model["vocab_size"])
    dale_law = bool(model.get("dale_law", False))
    e_fraction = float(model.get("e_fraction", 0.8))
    return reconstruct_init_weights(
        hidden_size, vocab_size, seed, dale_law=dale_law, e_fraction=e_fraction,
    )


def plot_weight_structure_init_vs_final(
    model: dict,
    save_path: str | Path,
    *,
    seed: int,
    chars: list[str] | None = None,
) -> dict[str, Any]:
    """Compare random init vs learned weights: heatmaps, deltas, and feedforward metrics."""
    from visualize import weights_for_plot

    save_path = Path(save_path)
    w_in_f, w_rec_f, w_out_f, dale_sign = weights_for_plot(model)
    w_in_i, w_rec_i, w_out_i = init_weights_for_model(model, seed)
    if dale_sign is not None and len(dale_sign) == w_in_f.shape[0]:
        from rnn.rnn_dyn import dale_signs_ordered, permute_hidden_by_dale

        if not dale_signs_ordered(np.asarray(dale_sign)):
            w_in_i, w_rec_i, w_out_i, _, _ = permute_hidden_by_dale(
                w_in_i, w_rec_i, w_out_i, np.zeros(w_in_i.shape[0]), np.asarray(dale_sign),
            )

    chars = chars or list(model["chars"])
    hidden_size, vocab_size = w_in_f.shape
    metrics_init = compute_weight_structure_metrics(w_in_i, w_rec_i, w_out_i)
    metrics_final = compute_weight_structure_metrics(w_in_f, w_rec_f, w_out_f)
    motif_final = compute_weight_motif_metrics(w_in_f, w_rec_f)
    summary = {"seed": seed, "init": metrics_init, "final": metrics_final, "motif_final": motif_final}

    json_path = save_path.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {json_path}")

    vmax_xh = max(
        symmetric_abs_vmax(w_in_i, w_in_f),
        1e-9,
    )
    vmax_hh = max(
        symmetric_abs_vmax(w_rec_i, w_rec_f),
        1e-9,
    )
    cmap = plt.cm.RdBu_r

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    finalize_grid_figure(
        fig,
        suptitle="Weight structure: random init vs after learning",
        top=0.92,
        hspace=0.42,
        wspace=0.28,
    )

    panels = [
        (axes[0, 0], w_in_i.T, "Init $W_{xh}$ (input)", vmax_xh, "hidden unit", "input char"),
        (axes[0, 1], w_in_f.T, "Final $W_{xh}$ (input)", vmax_xh, "hidden unit", "input char"),
        (axes[0, 2], (w_in_f - w_in_i).T, r"$\Delta W_{xh}$ (final $-$ init)", vmax_xh, "hidden unit", "input char"),
        (axes[1, 0], w_rec_i, "Init $W_{hh}$ (recurrent)", vmax_hh, "source h", "target h"),
        (axes[1, 1], w_rec_f, "Final $W_{hh}$ (recurrent)", vmax_hh, "source h", "target h"),
        (axes[1, 2], w_rec_f - w_rec_i, r"$\Delta W_{hh}$ (final $-$ init)", vmax_hh, "source h", "target h"),
    ]
    last_im = None
    for ax, data, title, vmax, xlabel, ylabel in panels:
        last_im = ax.imshow(
            data, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
            interpolation="nearest", origin="lower",
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)

    save_figure(fig, save_path)
    print(f"wrote {save_path}")

    _plot_weight_structure_bars(metrics_init, metrics_final, save_path.with_name("weight_structure_metrics.png"))
    _plot_input_drive_histogram(w_in_i, w_rec_i, w_in_f, w_rec_f, save_path.with_name("weight_input_drive_fraction.png"))
    _plot_weight_motif_summary(motif_final, save_path.with_name("weight_motif_summary.png"))
    plot_weight_clustered_heatmaps(
        w_in_f, w_rec_f, chars,
        save_path.parent,
        basename="weights",
    )

    return summary


def symmetric_abs_vmax(*arrays: np.ndarray, pct: float = 99.0) -> float:
    finite = np.concatenate([np.abs(a).ravel() for a in arrays])
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1e-3
    return max(float(np.percentile(finite, pct)), 1e-4)


def _plot_weight_structure_bars(
    metrics_init: dict[str, float],
    metrics_final: dict[str, float],
    save_path: Path,
) -> None:
    labels = [
        "input / recurrent\nFrobenius norm",
        "mean input-drive\nfraction per unit",
        "mean |input|\nper unit",
        "mean |recurrent in|\nper unit",
    ]
    keys = [
        "input_over_recurrent_norm",
        "mean_input_drive_fraction",
        "mean_abs_input_per_unit",
        "mean_abs_recurrent_in_per_unit",
    ]
    x = np.arange(len(labels))
    width = 0.35
    init_vals = [metrics_init[k] for k in keys]
    final_vals = [metrics_final[k] for k in keys]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - width / 2, init_vals, width, label="init", color="#9ecae1", edgecolor="0.3")
    ax.bar(x + width / 2, final_vals, width, label="final", color="#3182bd", edgecolor="0.3")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("value", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_title(
        "Feedforward balance metrics (higher input/recurrent ratio & input-drive fraction "
        "→ more input-driven dynamics)",
        fontsize=9,
        pad=8,
    )
    finalize_grid_figure(fig, bottom=0.18)
    save_figure(fig, save_path)
    print(f"wrote {save_path}")


def _plot_input_drive_histogram(
    w_in_i: np.ndarray,
    w_rec_i: np.ndarray,
    w_in_f: np.ndarray,
    w_rec_f: np.ndarray,
    save_path: Path,
) -> None:
    frac_i = _per_unit_input_drive_fraction(w_in_i, w_rec_i)
    frac_f = _per_unit_input_drive_fraction(w_in_f, w_rec_f)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bins = np.linspace(0, 1, 26)
    ax.hist(frac_i, bins=bins, alpha=0.55, label="init", color="#9ecae1", edgecolor="0.4")
    ax.hist(frac_f, bins=bins, alpha=0.55, label="final", color="#3182bd", edgecolor="0.4")
    ax.axvline(np.mean(frac_i), color="#6baed6", ls="--", lw=1.2, label=f"init mean={np.mean(frac_i):.2f}")
    ax.axvline(np.mean(frac_f), color="#08519c", ls="--", lw=1.2, label=f"final mean={np.mean(frac_f):.2f}")
    ax.set_xlabel("per-unit input-drive fraction  (|input| / (|input| + |recurrent in|))", fontsize=9)
    ax.set_ylabel("# hidden units", fontsize=9)
    ax.set_title("Distribution of input vs recurrent drive per hidden unit", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    finalize_grid_figure(fig, bottom=0.14)
    save_figure(fig, save_path)
    print(f"wrote {save_path}")


def _cluster_unit_order(w_in: np.ndarray, w_rec: np.ndarray) -> np.ndarray:
    """Hierarchical clustering order over hidden units (rows of W_xh / W_hh)."""
    feats = np.hstack([w_in, w_rec, w_rec.T])
    dist = pdist(feats, metric="correlation")
    dist = np.nan_to_num(dist, nan=1.0)
    link = hierarchy.linkage(dist, method="average")
    return hierarchy.leaves_list(link)


def _block_labels_from_clusters(n_units: int, n_blocks: int = 4) -> np.ndarray:
    order = np.arange(n_units)
    edges = np.linspace(0, n_units, n_blocks + 1, dtype=int)
    labels = np.zeros(n_units, dtype=int)
    for b in range(n_blocks):
        labels[order[edges[b]:edges[b + 1]]] = b
    return labels


def compute_weight_motif_metrics(
    w_in: np.ndarray,
    w_rec: np.ndarray,
    *,
    n_blocks: int = 4,
) -> dict[str, float]:
    """Block coupling, cluster cohesion, and input-tuning entropy on learned weights."""
    n_units = w_rec.shape[0]
    order = _cluster_unit_order(w_in, w_rec)
    w_rec_ord = w_rec[np.ix_(order, order)]
    abs_rec = np.abs(w_rec_ord)
    total = float(np.sum(abs_rec))
    if total < 1e-12:
        block_coupling = 0.0
    else:
        block_size = max(n_units // n_blocks, 1)
        off_diag_mass = 0.0
        for i in range(n_blocks):
            for j in range(n_blocks):
                if i == j:
                    continue
                r0, r1 = i * block_size, min((i + 1) * block_size, n_units)
                c0, c1 = j * block_size, min((j + 1) * block_size, n_units)
                off_diag_mass += float(np.sum(abs_rec[r0:r1, c0:c1]))
        block_coupling = off_diag_mass / total

    w_in_ord = w_in[order]
    within_corrs: list[float] = []
    for b in range(n_blocks):
        r0 = b * block_size
        r1 = min((b + 1) * block_size, n_units)
        block = w_in_ord[r0:r1]
        if block.shape[0] < 2:
            continue
        c = np.corrcoef(block)
        tri = c[np.triu_indices(c.shape[0], k=1)]
        within_corrs.extend(tri[np.isfinite(tri)].tolist())
    cluster_cohesion = float(np.mean(within_corrs)) if within_corrs else 0.0

    row_abs = np.abs(w_in_ord)
    row_sums = row_abs.sum(axis=1, keepdims=True)
    probs = np.where(row_sums > 0, row_abs / row_sums, 1.0 / max(w_in.shape[1], 1))
    ent = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    max_ent = np.log(max(w_in.shape[1], 2))
    input_tuning_entropy = float(np.mean(ent / max_ent))

    return {
        "block_coupling_hh": block_coupling,
        "cluster_cohesion_xh": cluster_cohesion,
        "input_tuning_entropy": input_tuning_entropy,
        "n_units": float(n_units),
        "n_blocks": float(n_blocks),
    }


def _plot_weight_motif_summary(motif: dict[str, float], save_path: Path) -> None:
    labels = [
        "block coupling\n(off-diagonal |W_hh|)",
        "cluster cohesion\n(within-block corr)",
        "input tuning entropy\n(normalized)",
    ]
    keys = ["block_coupling_hh", "cluster_cohesion_xh", "input_tuning_entropy"]
    vals = [motif[k] for k in keys]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    colors = ["#e6550d", "#31a354", "#756bb1"]
    ax.bar(np.arange(len(labels)), vals, color=colors, edgecolor="0.3", width=0.55)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("value", fontsize=9)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_title("Recurrent motif structure (clustered units)", fontsize=10, pad=8)
    finalize_grid_figure(fig, bottom=0.22)
    save_figure(fig, save_path)
    print(f"wrote {save_path}")


def plot_weight_clustered_heatmaps(
    w_in: np.ndarray,
    w_rec: np.ndarray,
    chars: list[str],
    out_dir: str | Path,
    *,
    basename: str = "weights",
) -> None:
    """Save hierarchically clustered W_hh and W_xh heatmaps."""
    out_dir = Path(out_dir)
    order = _cluster_unit_order(w_in, w_rec)
    w_rec_ord = w_rec[np.ix_(order, order)]
    w_in_ord = w_in[order]
    char_labels = [c if c not in (" ", "\n", "\t") else repr(c)[1:-1] for c in chars]
    vmax_hh = symmetric_abs_vmax(w_rec_ord)
    vmax_xh = symmetric_abs_vmax(w_in_ord)
    cmap = plt.cm.RdBu_r

    panels = [
        ("hh_clustered", w_rec_ord, vmax_hh, r"Clustered $W_{hh}$", "source h", "target h"),
        ("xh_clustered", w_in_ord.T, vmax_xh, r"Clustered $W_{xh}$", "hidden unit", "input char"),
    ]
    for suffix, data, vmax, title, xlabel, ylabel in panels:
        fig_h = max(4.5, min(10, 0.10 * max(data.shape)))
        fig_w = max(5.0, min(12, 0.10 * max(data.shape)))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(
            data, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
            interpolation="nearest", origin="lower",
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        if suffix == "xh_clustered":
            ax.set_yticks(np.arange(len(char_labels)))
            ax.set_yticklabels(char_labels, fontsize=7)
            if len(char_labels) > 10:
                for lbl in ax.get_yticklabels():
                    lbl.set_fontsize(6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        finalize_grid_figure(fig, bottom=0.12 if suffix == "xh_clustered" else 0.10)
        save_path = out_dir / f"{basename}_{suffix}.png"
        save_figure(fig, save_path)
        print(f"wrote {save_path}")
