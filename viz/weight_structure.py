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
    """Compare random init vs learned weights (clustered 2x2) plus motif side plots."""
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
    metrics_init = compute_weight_structure_metrics(w_in_i, w_rec_i, w_out_i)
    metrics_final = compute_weight_structure_metrics(w_in_f, w_rec_f, w_out_f)
    motif_final = compute_weight_motif_metrics(w_in_f, w_rec_f)
    summary = {"seed": seed, "init": metrics_init, "final": metrics_final, "motif_final": motif_final}

    json_path = save_path.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {json_path}")

    # Paper-facing panel: 2x2 init/final, hierarchically clustered, no deltas.
    plot_weight_init_vs_final_clustered(
        w_in_i, w_rec_i, w_in_f, w_rec_f, chars, save_path,
    )

    _plot_weight_structure_bars(metrics_init, metrics_final, save_path.with_name("weight_structure_metrics.png"))
    _plot_input_drive_histogram(w_in_i, w_rec_i, w_in_f, w_rec_f, save_path.with_name("weight_input_drive_fraction.png"))
    _plot_weight_motif_summary(motif_final, save_path.with_name("weight_motif_summary.png"))

    return summary


def plot_weight_matrices_by_seed(
    exp: str,
    save_path: str | Path,
    *,
    seeds: tuple[int, ...] = (1, 2, 3, 5, 7),
    model_type: str = "rnn",
) -> dict[str, Any]:
    """Four-row grid across seeds: init/final W_xh and W_hh (each stage clustered).

    Also writes motif metrics (init + final) across seeds next to the figure.
    """
    from experiment import checkpoint_path
    from visualize import load_model_for_viz, weights_for_plot

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    n_seeds = len(seeds)
    cmap = plt.cm.RdBu_r
    row_specs = (
        ("init_xh", r"Init $W_{xh}$", "input char", True),
        ("final_xh", r"Final $W_{xh}$", "input char", True),
        ("init_hh", r"Init $W_{hh}$", "target h", False),
        ("final_hh", r"Final $W_{hh}$", "target h", False),
    )
    n_rows = len(row_specs)

    fig = plt.figure(figsize=(1.4 * n_seeds + 1.35, 1.55 * n_rows + 0.55))
    gs = fig.add_gridspec(
        n_rows, n_seeds + 1,
        width_ratios=[1.0] * n_seeds + [0.06],
        wspace=0.18,
        hspace=0.32,
        left=0.08,
        right=0.97,
        top=0.90,
        bottom=0.08,
    )
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(n_seeds)] for r in range(n_rows)])
    cax_by_row = [fig.add_subplot(gs[r, n_seeds]) for r in range(n_rows)]
    motif_by_seed: dict[str, Any] = {"seeds": list(seeds), "init": {}, "final": {}}
    char_labels: list[str] | None = None
    panel_data: dict[tuple[int, int], np.ndarray] = {}

    for col, seed in enumerate(seeds):
        model = load_model_for_viz(str(checkpoint_path(exp, model_type, seed=seed)), model_type)
        w_in_f, w_rec_f, _w_out_f, dale_sign = weights_for_plot(model)
        w_in_i, w_rec_i, _w_out_i = init_weights_for_model(model, seed)
        if dale_sign is not None and len(dale_sign) == w_in_f.shape[0]:
            from rnn.rnn_dyn import dale_signs_ordered, permute_hidden_by_dale

            if not dale_signs_ordered(np.asarray(dale_sign)):
                w_in_i, w_rec_i, _w_out_i, _, _ = permute_hidden_by_dale(
                    w_in_i, w_rec_i, _w_out_i, np.zeros(w_in_i.shape[0]), np.asarray(dale_sign),
                )

        if char_labels is None:
            chars = list(model["chars"])
            char_labels = [c if c not in (" ", "\n", "\t") else repr(c)[1:-1] for c in chars]

        order_i = _cluster_unit_order(w_in_i, w_rec_i)
        order_f = _cluster_unit_order(w_in_f, w_rec_f)
        panels = {
            "init_xh": w_in_i[order_i].T,
            "final_xh": w_in_f[order_f].T,
            "init_hh": w_rec_i[np.ix_(order_i, order_i)],
            "final_hh": w_rec_f[np.ix_(order_f, order_f)],
        }
        motif_by_seed["init"][str(seed)] = compute_weight_motif_metrics(w_in_i, w_rec_i)
        motif_by_seed["final"][str(seed)] = compute_weight_motif_metrics(w_in_f, w_rec_f)
        for row, (key, *_rest) in enumerate(row_specs):
            panel_data[(row, col)] = panels[key]

    row_vmax = []
    for row in range(n_rows):
        vmax = max(
            (symmetric_abs_vmax(panel_data[(row, col)]) for col in range(n_seeds)),
            default=1e-9,
        )
        row_vmax.append(max(float(vmax), 1e-9))

    last_im_by_row: list[Any] = [None] * n_rows
    for col, seed in enumerate(seeds):
        for row, (_key, ylabel, _tick_name, is_xh) in enumerate(row_specs):
            ax = axes[row, col]
            data = panel_data[(row, col)]
            vmax = row_vmax[row]
            im = ax.imshow(
                data, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
                interpolation="nearest", origin="lower",
            )
            last_im_by_row[row] = im
            if row == 0:
                ax.set_title(f"seed {seed}", fontsize=8, pad=2)
            ny, nx = data.shape
            if row == n_rows - 1:
                if col == 0:
                    ax.set_xlabel("hidden unit" if is_xh else "source h", fontsize=6)
                ax.set_xticks([0, max(nx - 1, 0)])
                ax.tick_params(axis="x", labelsize=5)
            else:
                ax.set_xticks([])
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=7)
                if is_xh and char_labels is not None:
                    ax.set_yticks(np.arange(len(char_labels)))
                    ax.set_yticklabels(char_labels, fontsize=4.5)
                else:
                    ax.set_yticks([0, max(ny - 1, 0)])
                    ax.tick_params(axis="y", labelsize=5)
            else:
                ax.set_ylabel("")
                ax.set_yticks([0, max(ny - 1, 0)])
                ax.tick_params(axis="y", labelleft=False)

    for row, im in enumerate(last_im_by_row):
        if im is None:
            continue
        cbar = fig.colorbar(im, cax=cax_by_row[row])
        cbar.ax.tick_params(labelsize=4)

    fig.suptitle(
        r"Init vs final $W_{xh}$ and $W_{hh}$ across seeds (each stage clustered)",
        fontsize=11,
        y=0.98,
    )
    save_figure(fig, save_path)
    print(f"wrote {save_path}")

    json_path = save_path.with_suffix(".json")
    json_path.write_text(json.dumps(motif_by_seed, indent=2), encoding="utf-8")
    print(f"wrote {json_path}")
    return motif_by_seed


def plot_weight_init_vs_final_clustered(
    w_in_i: np.ndarray,
    w_rec_i: np.ndarray,
    w_in_f: np.ndarray,
    w_rec_f: np.ndarray,
    chars: list[str],
    save_path: str | Path,
) -> None:
    """2x2 init vs final heatmaps; each stage clustered on its own weights."""
    save_path = Path(save_path)
    order_i = _cluster_unit_order(w_in_i, w_rec_i)
    order_f = _cluster_unit_order(w_in_f, w_rec_f)
    w_in_i_c = w_in_i[order_i]
    w_in_f_c = w_in_f[order_f]
    w_rec_i_c = w_rec_i[np.ix_(order_i, order_i)]
    w_rec_f_c = w_rec_f[np.ix_(order_f, order_f)]

    cmap = plt.cm.RdBu_r
    char_labels = [c if c not in (" ", "\n", "\t") else repr(c)[1:-1] for c in chars]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    finalize_grid_figure(
        fig,
        suptitle="Weights: random init vs after learning (each stage clustered)",
        top=0.90,
        hspace=0.42,
        wspace=0.45,
    )
    panels = [
        (axes[0, 0], w_in_i_c.T, "Init $W_{xh}$", "hidden unit", "input char", True),
        (axes[0, 1], w_in_f_c.T, "Final $W_{xh}$", "hidden unit", "input char", True),
        (axes[1, 0], w_rec_i_c, "Init $W_{hh}$", "source h", "target h", False),
        (axes[1, 1], w_rec_f_c, "Final $W_{hh}$", "source h", "target h", False),
    ]
    for ax, data, title, xlabel, ylabel, is_xh in panels:
        vmax = max(symmetric_abs_vmax(data), 1e-9)
        im = ax.imshow(
            data, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
            interpolation="nearest", origin="lower",
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        if is_xh:
            ax.set_yticks(np.arange(len(char_labels)))
            ax.set_yticklabels(char_labels, fontsize=7)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
        cbar.ax.tick_params(labelsize=7)
    save_figure(fig, save_path)
    print(f"wrote {save_path}")


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
    """Clusteriness / motif scalars for W_xh and W_hh after hierarchical unit order."""
    n_units = w_rec.shape[0]
    order = _cluster_unit_order(w_in, w_rec)
    w_rec_ord = w_rec[np.ix_(order, order)]
    w_in_ord = w_in[order]
    abs_rec = np.abs(w_rec_ord)
    total = float(np.sum(abs_rec))
    block_size = max(n_units // n_blocks, 1)

    within_mass = 0.0
    between_mass = 0.0
    within_vals: list[float] = []
    between_vals: list[float] = []
    for i in range(n_blocks):
        r0, r1 = i * block_size, min((i + 1) * block_size, n_units)
        for j in range(n_blocks):
            c0, c1 = j * block_size, min((j + 1) * block_size, n_units)
            block = abs_rec[r0:r1, c0:c1]
            mass = float(np.sum(block))
            if i == j:
                within_mass += mass
                within_vals.extend(block.ravel().tolist())
            else:
                between_mass += mass
                between_vals.extend(block.ravel().tolist())
    if total < 1e-12:
        block_coupling = 0.0
        within_block_frac = 0.0
        within_between_ratio = 0.0
    else:
        block_coupling = between_mass / total
        within_block_frac = within_mass / total
        mean_within = float(np.mean(within_vals)) if within_vals else 0.0
        mean_between = float(np.mean(between_vals)) if between_vals else 1e-12
        within_between_ratio = mean_within / max(mean_between, 1e-12)

    # Local clumpiness along the cluster order: adjacent-row |corr| of W_hh.
    adj_corrs: list[float] = []
    for i in range(n_units - 1):
        a, b = w_rec_ord[i], w_rec_ord[i + 1]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if np.isfinite(r):
            adj_corrs.append(abs(r))
    hh_adjacent_corr = float(np.mean(adj_corrs)) if adj_corrs else 0.0

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
    # Columnar specialization: fraction of |W_xh| mass on the strongest input char.
    top1_mass = float(np.mean(np.max(row_abs, axis=1) / np.maximum(row_abs.sum(axis=1), 1e-12)))

    return {
        "block_coupling_hh": block_coupling,
        "hh_within_block_frac": within_block_frac,
        "hh_within_between_ratio": within_between_ratio,
        "hh_adjacent_corr": hh_adjacent_corr,
        "cluster_cohesion_xh": cluster_cohesion,
        "xh_top1_mass": top1_mass,
        "input_tuning_entropy": input_tuning_entropy,
        "n_units": float(n_units),
        "n_blocks": float(n_blocks),
    }


def compute_weight_rank_metrics(w_rec: np.ndarray) -> dict[str, float]:
    """Matrix rank / pseudo-rank scalars for W_hh (singular spectrum)."""
    w = np.asarray(w_rec, dtype=float)
    try:
        s = np.linalg.svd(w, compute_uv=False)
    except np.linalg.LinAlgError:
        return {
            "stable_rank": float("nan"),
            "effective_rank": float("nan"),
            "numerical_rank_1e3": float("nan"),
            "participation_ratio": float("nan"),
            "spectral_gap_sv": float("nan"),
            "nuclear_norm": float("nan"),
            "op_norm": float("nan"),
            "frobenius": float("nan"),
        }
    s = np.asarray(s, dtype=float)
    s_pos = s[s > 0]
    fro2 = float(np.sum(s ** 2))
    op = float(s[0]) if s.size else float("nan")
    if s_pos.size == 0 or fro2 < 1e-30:
        return {
            "stable_rank": float("nan"),
            "effective_rank": float("nan"),
            "numerical_rank_1e3": float("nan"),
            "participation_ratio": float("nan"),
            "spectral_gap_sv": float("nan"),
            "nuclear_norm": float("nan"),
            "op_norm": op,
            "frobenius": float(np.sqrt(fro2)),
        }
    p = (s_pos ** 2) / fro2
    # Roy & Vetterli effective rank; participation ratio = 1 / sum p^2.
    effective_rank = float(np.exp(-np.sum(p * np.log(p + 1e-30))))
    participation = float(1.0 / np.sum(p ** 2))
    stable_rank = fro2 / max(op ** 2, 1e-30)
    thr = op * 1e-3
    numerical = float(np.sum(s >= thr))
    gap = float(s[0] / s[1]) if s.size >= 2 and s[1] > 1e-30 else float("nan")
    return {
        "stable_rank": float(stable_rank),
        "effective_rank": effective_rank,
        "numerical_rank_1e3": numerical,
        "participation_ratio": participation,
        "spectral_gap_sv": gap,
        "nuclear_norm": float(np.sum(s_pos)),
        "op_norm": op,
        "frobenius": float(np.sqrt(fro2)),
    }


def compute_weight_directionality_metrics(
    w_in: np.ndarray,
    w_rec: np.ndarray,
) -> dict[str, float]:
    """Feedforward / ordering proxies after letter-cluster unit order."""
    order = _cluster_unit_order(w_in, w_rec)
    w_ord = np.asarray(w_rec, dtype=float)[np.ix_(order, order)]
    abs_w = np.abs(w_ord)
    n = abs_w.shape[0]
    iu = np.triu_indices(n, k=1)
    il = np.tril_indices(n, k=1)
    upper = float(abs_w[iu].sum())
    lower = float(abs_w[il].sum())
    off = upper + lower
    upper_frac = upper / max(off, 1e-12)
    antisym = w_ord - w_ord.T
    sym = w_ord + w_ord.T
    asymmetry = _frobenius_norm(antisym) / max(_frobenius_norm(sym), 1e-12)
    try:
        rho = float(np.max(np.abs(np.linalg.eigvals(w_ord))))
    except np.linalg.LinAlgError:
        rho = float("nan")
    try:
        rho_abs = float(np.max(np.abs(np.linalg.eigvals(abs_w))))
    except np.linalg.LinAlgError:
        rho_abs = float("nan")
    if not np.isfinite(rho_abs) or rho_abs < 1e-12:
        walk_ratio_2 = float("nan")
    else:
        an = abs_w / rho_abs
        walk_ratio_2 = _frobenius_norm(an @ an) / max(_frobenius_norm(an), 1e-12)

    # Mean shortest path on strong-|W| undirected graph (legacy q=0.75 cut).
    mean_path = float("nan")
    pos = abs_w[abs_w > 0]
    if pos.size >= 2:
        thr = float(np.quantile(pos, 0.75))
        adj = abs_w >= thr
        np.fill_diagonal(adj, False)
        adj = np.logical_or(adj, adj.T)
        lens: list[float] = []
        for src in range(n):
            dist = np.full(n, -1, dtype=int)
            dist[src] = 0
            queue = [src]
            qi = 0
            while qi < len(queue):
                u = queue[qi]
                qi += 1
                for v in np.where(adj[u])[0]:
                    if dist[v] < 0:
                        dist[v] = dist[u] + 1
                        queue.append(int(v))
            lens.extend(float(d) for d in dist if d > 0)
        if lens:
            mean_path = float(np.mean(lens))

    return {
        "hh_upper_frac": float(upper_frac),
        "hh_asymmetry": float(asymmetry),
        "hh_walk_ratio_2": float(walk_ratio_2),
        "hh_mean_path_q75": mean_path,
        "spectral_radius_hh": float(rho) if np.isfinite(rho) else float("nan"),
        "spectral_radius_abs_hh": float(rho_abs) if np.isfinite(rho_abs) else float("nan"),
    }


def _thresholded_digraph(
    w_rec: np.ndarray,
    *,
    mode: str = "mean",
    q: float = 0.75,
):
    """Build a digraph from strong |W_hh| edges.

    ``mode="mean"`` (default): keep |W_ij| ≥ mean(off-diagonal |W|), so density
    can vary with weight concentration. ``mode="quantile"``: keep ≥ ``q`` quantile.
    """
    import networkx as nx

    abs_w = np.abs(np.asarray(w_rec, dtype=float))
    np.fill_diagonal(abs_w, 0.0)
    pos = abs_w[abs_w > 0]
    n = abs_w.shape[0]
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    if pos.size == 0:
        return g, float("nan")
    if mode == "quantile":
        thr = float(np.quantile(pos, q))
    else:
        thr = float(np.mean(pos))
    src, dst = np.where(abs_w >= thr)
    for i, j in zip(src.tolist(), dst.tolist()):
        g.add_edge(i, j, weight=float(abs_w[i, j]))
    return g, thr


def _digraph_motif_rates_from_adj(E: np.ndarray) -> dict[str, float]:
    """3-node motif rates on a boolean directed adjacency (no self-loops)."""
    n = E.shape[0]
    n_dir = int(np.sum(E))
    n_rec = int(np.sum(E & E.T)) // 2
    recip = (2.0 * n_rec) / max(n_dir, 1)
    if n < 3 or n_dir < 2:
        return {
            "motif_feedforward_rate": float("nan"),
            "motif_cycle_rate": float("nan"),
            "motif_reciprocal_frac": float(recip),
            "motif_n_triples": 0.0,
        }
    ff = 0
    cyc = 0
    triples = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                edges = (
                    int(E[i, j]) + int(E[j, i])
                    + int(E[i, k]) + int(E[k, i])
                    + int(E[j, k]) + int(E[k, j])
                )
                if edges < 2:
                    continue
                triples += 1
                if (
                    (E[i, j] and E[j, k] and E[k, i])
                    or (E[i, k] and E[k, j] and E[j, i])
                ):
                    cyc += 1
                ff_hit = False
                for a, b, c in (
                    (i, j, k), (i, k, j), (j, i, k),
                    (j, k, i), (k, i, j), (k, j, i),
                ):
                    if E[a, b] and E[b, c] and E[a, c]:
                        if not (E[b, a] or E[c, b] or E[c, a]):
                            ff_hit = True
                            break
                if ff_hit:
                    ff += 1
    return {
        "motif_feedforward_rate": float(ff / triples) if triples else float("nan"),
        "motif_cycle_rate": float(cyc / triples) if triples else float("nan"),
        "motif_reciprocal_frac": float(recip),
        "motif_n_triples": float(triples),
    }


def compute_weight_digraph_motifs(
    w_rec: np.ndarray,
    *,
    mode: str = "mean",
    q: float = 0.75,
) -> dict[str, float]:
    """3-node digraph motifs + key triad-census fractions on thresholded |W_hh|."""
    import networkx as nx

    g, thr = _thresholded_digraph(w_rec, mode=mode, q=q)
    n = g.number_of_nodes()
    E = np.zeros((n, n), dtype=bool)
    for u, v in g.edges():
        E[int(u), int(v)] = True
    out = _digraph_motif_rates_from_adj(E)
    out["motif_threshold"] = float(thr) if np.isfinite(thr) else float("nan")

    try:
        census = nx.triadic_census(g)
        total = float(sum(census.values())) or 1.0
        # Holland–Leinhardt labels of interest for feedforward / cycle structure.
        out["triad_030T_frac"] = float(census.get("030T", 0) / total)  # transitive
        out["triad_030C_frac"] = float(census.get("030C", 0) / total)  # 3-cycle
        out["triad_120D_frac"] = float(census.get("120D", 0) / total)
        out["triad_120U_frac"] = float(census.get("120U", 0) / total)
        out["triad_120C_frac"] = float(census.get("120C", 0) / total)
        out["triad_210_frac"] = float(census.get("210", 0) / total)
        out["triad_300_frac"] = float(census.get("300", 0) / total)  # complete mutual
        # Among non-empty connected triads, share that are pure transitive vs cyclic.
        connected = (
            census.get("030T", 0) + census.get("030C", 0)
            + census.get("120D", 0) + census.get("120U", 0) + census.get("120C", 0)
            + census.get("210", 0) + census.get("300", 0)
            + census.get("021D", 0) + census.get("021U", 0) + census.get("021C", 0)
            + census.get("111D", 0) + census.get("111U", 0) + census.get("201", 0)
        )
        out["triad_connected"] = float(connected)
        out["triad_ff_over_cycle"] = (
            float(census.get("030T", 0) / max(census.get("030C", 0), 1))
        )
    except Exception:
        for k in (
            "triad_030T_frac", "triad_030C_frac", "triad_120D_frac",
            "triad_120U_frac", "triad_120C_frac", "triad_210_frac",
            "triad_300_frac", "triad_connected", "triad_ff_over_cycle",
        ):
            out[k] = float("nan")
    return out


def _triangle_node_xy() -> dict[int, tuple[float, float]]:
    """Canonical 3-node layout: 0=top, 1=bottom-left, 2=bottom-right."""
    return {
        0: (0.50, 0.82),
        1: (0.18, 0.18),
        2: (0.82, 0.18),
    }


# Directed edges (i→j) for motif schematics on the metric board.
_MOTIF_SCHEMA_EDGES: dict[str, tuple[tuple[int, int], ...]] = {
    # Transitive feedforward: i→j→k and i→k.
    "motif_feedforward_rate": ((0, 1), (1, 2), (0, 2)),
    "motif_cycle_rate": ((0, 1), (1, 2), (2, 0)),
    # Reciprocal dyad (third node unused).
    "motif_reciprocal_frac": ((0, 1), (1, 0)),
    # Holland–Leinhardt (A=0, B=1, C=2).
    "triad_030T_frac": ((0, 1), (2, 1), (0, 2)),  # A→B←C, A→C
    "triad_030C_frac": ((1, 0), (2, 1), (0, 2)),  # A←B←C, A→C
    "triad_120D_frac": ((1, 0), (1, 2), (0, 2), (2, 0)),  # A←B→C, A↔C
    "triad_120U_frac": ((0, 1), (2, 1), (0, 2), (2, 0)),  # A→B←C, A↔C
    "triad_120C_frac": ((0, 1), (1, 2), (0, 2), (2, 0)),  # A→B→C, A↔C
    "triad_210_frac": ((0, 1), (1, 2), (2, 1), (0, 2), (2, 0)),
    "triad_300_frac": ((0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)),
}


def _draw_motif_nodes_edges(
    ax,
    pos: dict[int, tuple[float, float]],
    edges: tuple[tuple[int, int], ...],
    *,
    color: str,
    node_r: float,
    rad: float = 0.0,
) -> None:
    from matplotlib.patches import Circle, FancyArrowPatch

    edge_set = set(edges)
    drawn_pairs: set[tuple[int, int]] = set()
    for i, j in edges:
        if (i, j) in drawn_pairs:
            continue
        x0, y0 = pos[i]
        x1, y1 = pos[j]
        mutual = (j, i) in edge_set
        if mutual:
            # Two curved arrows for reciprocity.
            for (a, b), sign in (((i, j), 1.0), ((j, i), -1.0)):
                xa, ya = pos[a]
                xb, yb = pos[b]
                arr = FancyArrowPatch(
                    (xa, ya), (xb, yb),
                    arrowstyle="-|>",
                    mutation_scale=7,
                    lw=1.0,
                    color=color,
                    connectionstyle=f"arc3,rad={0.22 * sign}",
                    shrinkA=6,
                    shrinkB=6,
                )
                ax.add_patch(arr)
            drawn_pairs.add((i, j))
            drawn_pairs.add((j, i))
        else:
            use_rad = rad
            arr = FancyArrowPatch(
                (x0, y0), (x1, y1),
                arrowstyle="-|>",
                mutation_scale=7,
                lw=1.0,
                color=color,
                connectionstyle=f"arc3,rad={use_rad}",
                shrinkA=6,
                shrinkB=6,
            )
            ax.add_patch(arr)
            drawn_pairs.add((i, j))

    for idx, (x, y) in pos.items():
        circ = Circle((x, y), node_r, facecolor="white", edgecolor=color, lw=1.0, zorder=3)
        ax.add_patch(circ)


def _signed_threshold_adj(
    w_rec: np.ndarray,
    *,
    mode: str = "quantile",
    q: float = 0.75,
) -> tuple[np.ndarray, float]:
    """Signed adjacency: S_ij ∈ {-1,0,+1} for |W_ij| at/above threshold."""
    w = np.asarray(w_rec, dtype=float)
    abs_w = np.abs(w)
    np.fill_diagonal(abs_w, 0.0)
    pos = abs_w[abs_w > 0]
    if pos.size == 0:
        return np.zeros_like(w), float("nan")
    thr = float(np.quantile(pos, q)) if mode == "quantile" else float(np.mean(pos))
    s = np.zeros_like(w)
    mask = abs_w >= thr
    s[mask] = np.sign(w[mask])
    return s, thr


def compute_weight_signed_motifs(
    w_rec: np.ndarray,
    *,
    mode: str = "quantile",
    q: float = 0.75,
) -> dict[str, float]:
    """Signed digraph motifs on thresholded W_hh (keep edge signs).

    Blue(+) / red(−) structure: edge polarity, signed reciprocal dyads,
    balanced vs unbalanced directed 3-cycles (sign product), and signed
    feedforward triples (all+/all−/mixed).
    """
    s, thr = _signed_threshold_adj(w_rec, mode=mode, q=q)
    n = s.shape[0]
    out: dict[str, float] = {
        "signed_threshold": float(thr) if np.isfinite(thr) else float("nan"),
    }
    edges = [(i, j, int(s[i, j])) for i in range(n) for j in range(n) if s[i, j] != 0]
    n_edge = len(edges)
    if n_edge == 0:
        for k in (
            "signed_pos_edge_frac", "signed_neg_edge_frac",
            "signed_recip_pp_frac", "signed_recip_pm_frac", "signed_recip_mm_frac",
            "signed_cycle_balanced_frac", "signed_cycle_unbalanced_frac",
            "signed_ff_all_pos_rate", "signed_ff_all_neg_rate", "signed_ff_mixed_rate",
            "signed_undir_tri_balanced_frac",
        ):
            out[k] = float("nan")
        return out

    n_pos = sum(1 for _, _, sg in edges if sg > 0)
    out["signed_pos_edge_frac"] = float(n_pos / n_edge)
    out["signed_neg_edge_frac"] = float(1.0 - out["signed_pos_edge_frac"])

    # Reciprocal dyads (unordered pairs).
    pp = pm = mm = 0
    n_rec = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = s[i, j], s[j, i]
            if a == 0 or b == 0:
                continue
            n_rec += 1
            if a > 0 and b > 0:
                pp += 1
            elif a < 0 and b < 0:
                mm += 1
            else:
                pm += 1
    if n_rec:
        out["signed_recip_pp_frac"] = float(pp / n_rec)
        out["signed_recip_pm_frac"] = float(pm / n_rec)
        out["signed_recip_mm_frac"] = float(mm / n_rec)
    else:
        out["signed_recip_pp_frac"] = float("nan")
        out["signed_recip_pm_frac"] = float("nan")
        out["signed_recip_mm_frac"] = float("nan")

    # Directed 3-cycles and feedforward triples among connected triples.
    bal = unbal = 0
    ff_pos = ff_neg = ff_mix = 0
    ff_n = 0
    undir_bal = undir_unbal = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Undirected structural-balance triangle on strong |W| edges.
                e_ij = 1 if (s[i, j] or s[j, i]) else 0
                e_ik = 1 if (s[i, k] or s[k, i]) else 0
                e_jk = 1 if (s[j, k] or s[k, j]) else 0
                if e_ij + e_ik + e_jk == 3:
                    def _u_sign(a: int, b: int) -> int:
                        sab, sba = s[a, b], s[b, a]
                        if sab != 0 and sba == 0:
                            return int(sab)
                        if sba != 0 and sab == 0:
                            return int(sba)
                        if sab != 0 and sba != 0:
                            return 1 if (sab > 0 and sba > 0) else -1
                        return 0

                    p = _u_sign(i, j) * _u_sign(i, k) * _u_sign(j, k)
                    if p > 0:
                        undir_bal += 1
                    elif p < 0:
                        undir_unbal += 1

                # Directed cycles (both orientations).
                for a, b, c in ((i, j, k), (i, k, j)):
                    if s[a, b] and s[b, c] and s[c, a]:
                        prod = int(s[a, b] * s[b, c] * s[c, a])
                        if prod > 0:
                            bal += 1
                        else:
                            unbal += 1

                # Feedforward transitive (no back-edges), track signs.
                for a, b, c in (
                    (i, j, k), (i, k, j), (j, i, k),
                    (j, k, i), (k, i, j), (k, j, i),
                ):
                    if not (s[a, b] and s[b, c] and s[a, c]):
                        continue
                    if s[b, a] or s[c, b] or s[c, a]:
                        continue
                    ff_n += 1
                    signs = (s[a, b], s[b, c], s[a, c])
                    if all(x > 0 for x in signs):
                        ff_pos += 1
                    elif all(x < 0 for x in signs):
                        ff_neg += 1
                    else:
                        ff_mix += 1
                    break  # count each unordered triple at most once as FF

    n_cyc = bal + unbal
    out["signed_cycle_balanced_frac"] = float(bal / n_cyc) if n_cyc else float("nan")
    out["signed_cycle_unbalanced_frac"] = float(unbal / n_cyc) if n_cyc else float("nan")
    out["signed_ff_all_pos_rate"] = float(ff_pos / ff_n) if ff_n else float("nan")
    out["signed_ff_all_neg_rate"] = float(ff_neg / ff_n) if ff_n else float("nan")
    out["signed_ff_mixed_rate"] = float(ff_mix / ff_n) if ff_n else float("nan")
    n_utri = undir_bal + undir_unbal
    out["signed_undir_tri_balanced_frac"] = (
        float(undir_bal / n_utri) if n_utri else float("nan")
    )
    return out


# Signed motif schematics: (src, dst, sign) with sign ∈ {+1, −1}.
_SIGNED_MOTIF_SCHEMAS: dict[str, tuple[tuple[int, int, int], ...]] = {
    "signed_pos_edge_frac": ((0, 1, +1),),
    "signed_neg_edge_frac": ((0, 1, -1),),
    "signed_recip_pp_frac": ((0, 1, +1), (1, 0, +1)),
    "signed_recip_pm_frac": ((0, 1, +1), (1, 0, -1)),
    "signed_recip_mm_frac": ((0, 1, -1), (1, 0, -1)),
    "signed_cycle_balanced_frac": ((0, 1, +1), (1, 2, +1), (2, 0, +1)),  # +++ cycle
    "signed_cycle_unbalanced_frac": ((0, 1, +1), (1, 2, +1), (2, 0, -1)),  # ++− cycle
    "signed_ff_all_pos_rate": ((0, 1, +1), (1, 2, +1), (0, 2, +1)),
    "signed_ff_all_neg_rate": ((0, 1, -1), (1, 2, -1), (0, 2, -1)),
    "signed_ff_mixed_rate": ((0, 1, +1), (1, 2, +1), (0, 2, -1)),
    "signed_undir_tri_balanced_frac": ((0, 1, +1), (1, 2, +1), (0, 2, +1)),
}

_SIGNED_POS_COLOR = "#2166ac"
_SIGNED_NEG_COLOR = "#b2182b"


def draw_digraph_motif_schema(ax, key: str, *, color: str = "#1a1a1a") -> bool:
    """Draw a tiny digraph motif schematic on ``ax``. Returns False if unknown."""
    from matplotlib.patches import Circle, FancyArrowPatch

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")

    signed = _SIGNED_MOTIF_SCHEMAS.get(key)
    if signed is not None:
        # Dyad-only schemas use two nodes; triangles use three.
        nodes = {i for e in signed for i in e[:2]}
        if nodes <= {0, 1} and max(nodes) <= 1:
            pos = {0: (0.32, 0.50), 1: (0.68, 0.50)}
            _draw_signed_motif_edges(ax, pos, signed, node_r=0.07, rad=0.28)
        else:
            pos = _triangle_node_xy()
            _draw_signed_motif_edges(ax, pos, signed, node_r=0.065)
        return True

    if key == "triad_ff_over_cycle":
        # Side-by-side 030T | 030C.
        for x0, edges in (
            (0.0, _MOTIF_SCHEMA_EDGES["triad_030T_frac"]),
            (0.52, _MOTIF_SCHEMA_EDGES["triad_030C_frac"]),
        ):
            pos = {
                0: (x0 + 0.24, 0.78),
                1: (x0 + 0.08, 0.22),
                2: (x0 + 0.40, 0.22),
            }
            _draw_motif_nodes_edges(ax, pos, edges, color=color, node_r=0.055)
        ax.text(0.50, 0.48, "/", fontsize=9, ha="center", va="center", color=color)
        return True

    edges = _MOTIF_SCHEMA_EDGES.get(key)
    if edges is None:
        return False

    if key == "motif_reciprocal_frac":
        pos = {0: (0.32, 0.50), 1: (0.68, 0.50)}
        _draw_motif_nodes_edges(ax, pos, edges, color=color, node_r=0.07, rad=0.25)
        return True

    pos = _triangle_node_xy()
    _draw_motif_nodes_edges(ax, pos, edges, color=color, node_r=0.065)
    return True


def _draw_signed_motif_edges(
    ax,
    pos: dict[int, tuple[float, float]],
    edges: tuple[tuple[int, int, int], ...],
    *,
    node_r: float,
    rad: float = 0.0,
) -> None:
    from matplotlib.patches import Circle, FancyArrowPatch

    # Group by unordered pair to curve mutuals.
    by_pair: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for i, j, sg in edges:
        key = (min(i, j), max(i, j))
        by_pair.setdefault(key, []).append((i, j, sg))

    for pair_edges in by_pair.values():
        mutual = len(pair_edges) == 2
        for idx, (i, j, sg) in enumerate(pair_edges):
            col = _SIGNED_POS_COLOR if sg > 0 else _SIGNED_NEG_COLOR
            curve = rad
            if mutual:
                curve = 0.22 if idx == 0 else -0.22
            elif i > j and rad == 0.0:
                curve = 0.0
            arr = FancyArrowPatch(
                pos[i], pos[j],
                arrowstyle="-|>",
                mutation_scale=7,
                lw=1.15,
                color=col,
                connectionstyle=f"arc3,rad={curve}",
                shrinkA=6,
                shrinkB=6,
            )
            ax.add_patch(arr)

    for idx, (x, y) in pos.items():
        circ = Circle((x, y), node_r, facecolor="white", edgecolor="#333333", lw=1.0, zorder=3)
        ax.add_patch(circ)


def compute_weight_graph_metrics(
    w_rec: np.ndarray,
    *,
    mode: str = "mean",
    q: float = 0.75,
) -> dict[str, float]:
    """Standard network metrics on a thresholded |W_hh| digraph.

    Default threshold: off-diagonal |W| ≥ mean(|W|) so edge density can vary.
    Path / diameter / modularity use the undirected projection (largest CC).
    SCC stats stay directed.
    """
    import networkx as nx
    from networkx.algorithms import community as nx_comm

    g, thr = _thresholded_digraph(w_rec, mode=mode, q=q)
    n = g.number_of_nodes()
    m = g.number_of_edges()
    out: dict[str, float] = {
        "threshold_mode": 0.0 if mode == "mean" else float(q),
        "threshold_value": float(thr),
        "n_nodes": float(n),
        "n_edges": float(m),
        "density": float(nx.density(g)) if n > 1 else float("nan"),
        "mean_out_degree": float(m / n) if n else float("nan"),
    }

    try:
        out["reciprocity"] = float(nx.reciprocity(g)) if m else float("nan")
    except Exception:
        out["reciprocity"] = float("nan")

    try:
        out["avg_clustering"] = float(nx.average_clustering(g)) if n > 2 else float("nan")
    except Exception:
        out["avg_clustering"] = float("nan")

    try:
        out["degree_assortativity"] = float(
            nx.degree_assortativity_coefficient(g)
        ) if m > 1 else float("nan")
    except Exception:
        out["degree_assortativity"] = float("nan")

    # Strongly connected components (directed).
    try:
        sccs = list(nx.strongly_connected_components(g))
        out["n_scc"] = float(len(sccs))
        largest_scc = max((len(c) for c in sccs), default=0)
        out["largest_scc_frac"] = float(largest_scc / n) if n else float("nan")
    except Exception:
        out["n_scc"] = float("nan")
        out["largest_scc_frac"] = float("nan")

    # Undirected projection for classical path / community metrics.
    u = g.to_undirected()
    if u.number_of_edges() == 0:
        out["avg_shortest_path"] = float("nan")
        out["diameter"] = float("nan")
        out["modularity"] = float("nan")
        out["n_communities"] = float("nan")
        out["transitivity"] = float("nan")
        out["mean_betweenness"] = float("nan")
        out["max_betweenness"] = float("nan")
        out["mean_closeness"] = float("nan")
        out["out_degree_cv"] = float("nan")
        out["condensation_height"] = float("nan")
        out["n_condensation_nodes"] = float("nan")
        out["largest_cc_frac"] = float("nan")
        return out

    try:
        out["transitivity"] = float(nx.transitivity(u))
    except Exception:
        out["transitivity"] = float("nan")

    # Largest connected component for paths.
    try:
        largest_cc_nodes = max(nx.connected_components(u), key=len)
        u_cc = u.subgraph(largest_cc_nodes).copy()
        if u_cc.number_of_nodes() >= 2 and nx.is_connected(u_cc):
            out["avg_shortest_path"] = float(nx.average_shortest_path_length(u_cc))
            out["diameter"] = float(nx.diameter(u_cc))
            out["largest_cc_frac"] = float(u_cc.number_of_nodes() / n)
        else:
            out["avg_shortest_path"] = float("nan")
            out["diameter"] = float("nan")
            out["largest_cc_frac"] = float(u_cc.number_of_nodes() / n) if n else float("nan")
    except Exception:
        out["avg_shortest_path"] = float("nan")
        out["diameter"] = float("nan")
        out["largest_cc_frac"] = float("nan")

    try:
        communities = list(nx_comm.greedy_modularity_communities(u))
        out["n_communities"] = float(len(communities))
        out["modularity"] = float(nx_comm.modularity(u, communities))
    except Exception:
        out["n_communities"] = float("nan")
        out["modularity"] = float("nan")

    try:
        bc = nx.betweenness_centrality(u)
        vals = list(bc.values())
        out["mean_betweenness"] = float(np.mean(vals)) if vals else float("nan")
        out["max_betweenness"] = float(np.max(vals)) if vals else float("nan")
    except Exception:
        out["mean_betweenness"] = float("nan")
        out["max_betweenness"] = float("nan")

    try:
        cc = nx.closeness_centrality(u)
        out["mean_closeness"] = float(np.mean(list(cc.values()))) if cc else float("nan")
    except Exception:
        out["mean_closeness"] = float("nan")

    # Degree heterogeneity on the digraph.
    try:
        out_degs = np.asarray([d for _, d in g.out_degree()], dtype=float)
        if out_degs.size and float(np.mean(out_degs)) > 0:
            out["out_degree_cv"] = float(np.std(out_degs) / np.mean(out_degs))
        else:
            out["out_degree_cv"] = float("nan")
    except Exception:
        out["out_degree_cv"] = float("nan")

    # Condensation DAG height = longest path among SCCs (graph-theoretic "rank").
    try:
        cond = nx.condensation(g)
        if cond.number_of_nodes() >= 1 and nx.is_directed_acyclic_graph(cond):
            out["condensation_height"] = float(nx.dag_longest_path_length(cond))
            out["n_condensation_nodes"] = float(cond.number_of_nodes())
        else:
            out["condensation_height"] = float("nan")
            out["n_condensation_nodes"] = float(cond.number_of_nodes())
    except Exception:
        out["condensation_height"] = float("nan")
        out["n_condensation_nodes"] = float("nan")

    return out


def compute_weight_layeredness_metrics(
    w_in: np.ndarray,
    w_rec: np.ndarray,
) -> dict[str, float]:
    """Legacy layeredness bag: directionality + digraph motifs (quantile q=0.75)."""
    direction = compute_weight_directionality_metrics(w_in, w_rec)
    motifs = compute_weight_digraph_motifs(w_rec, mode="quantile", q=0.75)
    return {**direction, **motifs}


def collect_weight_metrics_by_seed(
    exp: str,
    *,
    seeds: tuple[int, ...] | None = None,
    model_type: str = "rnn",
) -> dict[str, Any]:
    """Init/final structure + motif metrics for every available seed."""
    from experiment import checkpoint_path, seeds_for_task
    from visualize import load_model_for_viz, weights_for_plot

    if seeds is None:
        seeds = tuple(sorted(seeds_for_task(exp, model_type)))
    payload: dict[str, Any] = {
        "exp": exp,
        "seeds": list(seeds),
        "init": {},
        "final": {},
        "motif_init": {},
        "motif_final": {},
        "distributions": {
            "input_drive_frac": {"init": [], "final": []},
            "hh_adjacent_corr": {"init": [], "final": []},
            "hh_abs_within": {"init": [], "final": []},
            "hh_abs_between": {"init": [], "final": []},
            "xh_within_corr": {"init": [], "final": []},
            "unit_mean_abs_input": {"init": [], "final": []},
        },
    }
    for seed in seeds:
        model = load_model_for_viz(str(checkpoint_path(exp, model_type, seed=seed)), model_type)
        w_in_f, w_rec_f, w_out_f, dale_sign = weights_for_plot(model)
        w_in_i, w_rec_i, w_out_i = init_weights_for_model(model, seed)
        if dale_sign is not None and len(dale_sign) == w_in_f.shape[0]:
            from rnn.rnn_dyn import dale_signs_ordered, permute_hidden_by_dale

            if not dale_signs_ordered(np.asarray(dale_sign)):
                w_in_i, w_rec_i, w_out_i, _, _ = permute_hidden_by_dale(
                    w_in_i, w_rec_i, w_out_i, np.zeros(w_in_i.shape[0]), np.asarray(dale_sign),
                )
        payload["init"][str(seed)] = compute_weight_structure_metrics(w_in_i, w_rec_i, w_out_i)
        payload["final"][str(seed)] = compute_weight_structure_metrics(w_in_f, w_rec_f, w_out_f)
        payload["motif_init"][str(seed)] = compute_weight_motif_metrics(w_in_i, w_rec_i)
        payload["motif_final"][str(seed)] = compute_weight_motif_metrics(w_in_f, w_rec_f)
        for stage, w_in, w_rec in (("init", w_in_i, w_rec_i), ("final", w_in_f, w_rec_f)):
            dist = _weight_distribution_samples(w_in, w_rec)
            for key, vals in dist.items():
                payload["distributions"][key][stage].extend(vals)
    return payload


def _weight_distribution_samples(
    w_in: np.ndarray,
    w_rec: np.ndarray,
    *,
    n_blocks: int = 4,
) -> dict[str, list[float]]:
    """Pooled sample lists for init/final histogram overlays."""
    n_units = w_rec.shape[0]
    order = _cluster_unit_order(w_in, w_rec)
    w_rec_ord = w_rec[np.ix_(order, order)]
    w_in_ord = w_in[order]
    abs_rec = np.abs(w_rec_ord)
    block_size = max(n_units // n_blocks, 1)

    within_vals: list[float] = []
    between_vals: list[float] = []
    for i in range(n_blocks):
        r0, r1 = i * block_size, min((i + 1) * block_size, n_units)
        for j in range(n_blocks):
            c0, c1 = j * block_size, min((j + 1) * block_size, n_units)
            block = abs_rec[r0:r1, c0:c1].ravel()
            if i == j:
                within_vals.extend(block.tolist())
            else:
                between_vals.extend(block.tolist())

    adj_corrs: list[float] = []
    for i in range(n_units - 1):
        a, b = w_rec_ord[i], w_rec_ord[i + 1]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if np.isfinite(r):
            adj_corrs.append(abs(r))

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

    return {
        "input_drive_frac": _per_unit_input_drive_fraction(w_in, w_rec).tolist(),
        "hh_adjacent_corr": adj_corrs,
        "hh_abs_within": within_vals,
        "hh_abs_between": between_vals,
        "xh_within_corr": within_corrs,
        "unit_mean_abs_input": np.mean(np.abs(w_in), axis=1).tolist(),
    }


def plot_weight_metrics_compact_by_seed(
    payload: dict[str, Any],
    save_path: str | Path,
    *,
    max_per_row: int = 5,
) -> None:
    """Summary bars (≤5/row) plus init/final distribution histograms (≤5/row)."""
    save_path = Path(save_path)
    seeds = [str(s) for s in payload["seeds"]]
    n = len(seeds)
    bar_panels = [
        ("motif", "cluster_cohesion_xh", "$W_{xh}$ cluster\ncohesion", (0, 1.05)),
        ("motif", "hh_adjacent_corr", "$W_{hh}$ adjacent\n|corr|", (0, 1.05)),
        ("motif", "hh_within_between_ratio", "$W_{hh}$ within/\nbetween |w|", None),
        ("structure", "input_over_recurrent_norm", "input / recurrent\nFrobenius ratio", None),
        ("structure", "mean_input_drive_fraction", "mean input-drive\nfraction", (0, 1.05)),
    ]
    hist_panels = [
        ("input_drive_frac", "per-unit input-drive\nfraction", (0.0, 1.0), False),
        ("xh_within_corr", "$W_{xh}$ within-block\npairwise corr", (-1.0, 1.0), False),
        ("hh_adjacent_corr", "$W_{hh}$ adjacent\n|corr| samples", (0.0, 1.0), False),
        ("hh_abs_within", "$|W_{hh}|$ within\nblocks", None, True),
        ("hh_abs_between", "$|W_{hh}|$ between\nblocks", None, True),
    ]

    usable_bars = []
    for family, key, title, ylim in bar_panels:
        src = payload["init"] if family == "structure" else payload["motif_init"]
        if seeds and seeds[0] in src and key in src[seeds[0]]:
            usable_bars.append((family, key, title, ylim))

    n_bar = len(usable_bars)
    n_hist = len(hist_panels)
    n_cols = max_per_row
    n_bar_rows = int(np.ceil(n_bar / n_cols))
    n_hist_rows = int(np.ceil(n_hist / n_cols))
    n_rows = n_bar_rows + n_hist_rows
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(1.45 * n_cols + 0.3, 1.4 * n_rows + 0.4),
        squeeze=False,
    )

    x = np.arange(2)
    for idx, (family, key, title, ylim) in enumerate(usable_bars):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        if family == "structure":
            init_vals = np.array([payload["init"][s][key] for s in seeds], dtype=float)
            final_vals = np.array([payload["final"][s][key] for s in seeds], dtype=float)
        else:
            init_vals = np.array([payload["motif_init"][s][key] for s in seeds], dtype=float)
            final_vals = np.array([payload["motif_final"][s][key] for s in seeds], dtype=float)
        means = [float(np.mean(init_vals)), float(np.mean(final_vals))]
        stds = [float(np.std(init_vals)), float(np.std(final_vals))]
        ax.bar(
            x, means, yerr=stds, width=0.62, color=["#9ecae1", "#e6550d"],
            edgecolor="0.35", linewidth=0.6, capsize=2.5,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(["init", "final"], fontsize=8)
        ax.set_xlabel("stage", fontsize=8)
        ax.set_ylabel(title, fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", alpha=0.25, linewidth=0.4)
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            ymax = max(means[0] + stds[0], means[1] + stds[1], 1e-6)
            ax.set_ylim(0, ymax * 1.18)
    for idx in range(n_bar, n_bar_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis("off")

    dists = payload.get("distributions", {})
    for idx, (key, title, xlim, log_x) in enumerate(hist_panels):
        r, c = divmod(idx, n_cols)
        r += n_bar_rows
        ax = axes[r][c]
        if key not in dists:
            ax.axis("off")
            continue
        init_vals = np.asarray(dists[key]["init"], dtype=float)
        final_vals = np.asarray(dists[key]["final"], dtype=float)
        init_vals = init_vals[np.isfinite(init_vals)]
        final_vals = final_vals[np.isfinite(final_vals)]
        if init_vals.size == 0 and final_vals.size == 0:
            ax.axis("off")
            continue
        if log_x:
            init_plot = init_vals[init_vals > 0]
            final_plot = final_vals[final_vals > 0]
            if init_plot.size == 0 and final_plot.size == 0:
                ax.axis("off")
                continue
            lo = float(np.percentile(np.concatenate([init_plot, final_plot]), 1))
            hi = float(np.percentile(np.concatenate([init_plot, final_plot]), 99))
            bins = np.geomspace(max(lo, 1e-6), max(hi, 1e-5), 28)
            ax.hist(init_plot, bins=bins, alpha=0.55, color="#9ecae1", label="init", density=True)
            ax.hist(final_plot, bins=bins, alpha=0.55, color="#e6550d", label="final", density=True)
            ax.set_xscale("log")
            ax.set_xlabel("|weight| (log)", fontsize=8)
        else:
            if xlim is not None:
                bins = np.linspace(xlim[0], xlim[1], 28)
            else:
                both = np.concatenate([init_vals, final_vals])
                bins = np.linspace(float(np.min(both)), float(np.max(both)), 28)
            ax.hist(init_vals, bins=bins, alpha=0.55, color="#9ecae1", label="init", density=True)
            ax.hist(final_vals, bins=bins, alpha=0.55, color="#e6550d", label="final", density=True)
            ax.set_xlabel("value", fontsize=8)
            if xlim is not None:
                ax.set_xlim(*xlim)
        ax.set_ylabel(f"{title}\n(density)", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.25, linewidth=0.4)
        if idx == 0:
            ax.legend(fontsize=7, frameon=False, loc="upper right")
    for idx in range(n_hist, n_hist_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[n_bar_rows + r][c].axis("off")

    fig.suptitle(
        f"Weight metrics + distributions across seeds (n={n}; histograms pooled over units/edges)",
        fontsize=11, y=0.99,
    )
    fig.subplots_adjust(
        left=0.08, right=0.98, top=0.90, bottom=0.08,
        wspace=0.55, hspace=0.55,
    )
    save_figure(fig, save_path)
    print(f"wrote {save_path}")


def _plot_weight_motif_summary(motif: dict[str, float], save_path: Path) -> None:
    labels = [
        "W_xh cohesion",
        "W_xh top-1 mass",
        "W_hh within-block",
        "W_hh adj |corr|",
        "input entropy",
    ]
    keys = [
        "cluster_cohesion_xh",
        "xh_top1_mass",
        "hh_within_block_frac",
        "hh_adjacent_corr",
        "input_tuning_entropy",
    ]
    vals = [motif[k] for k in keys]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    colors = ["#31a354", "#74c476", "#e6550d", "#fd8d3c", "#756bb1"]
    ax.bar(np.arange(len(labels)), vals, color=colors, edgecolor="0.3", width=0.62)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("value", fontsize=8)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_title("Clusteriness summary (final weights, clustered units)", fontsize=9, pad=6)
    finalize_grid_figure(fig, bottom=0.18)
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
