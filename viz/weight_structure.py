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
    """4×N grid: rows = init/final W_xh, init/final W_hh; columns = seeds.

    Each panel is clustered on its own stage and has its own color scale.
    Also writes motif metrics (init + final) across seeds next to the figure.
    """
    from experiment import checkpoint_path
    from visualize import load_model_for_viz, weights_for_plot

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    n_seeds = len(seeds)
    cmap = plt.cm.RdBu_r
    row_specs = [
        ("init_xh", r"Init $W_{xh}$"),
        ("final_xh", r"Final $W_{xh}$"),
        ("init_hh", r"Init $W_{hh}$"),
        ("final_hh", r"Final $W_{hh}$"),
    ]

    fig, axes = plt.subplots(
        4, n_seeds,
        figsize=(2.15 * n_seeds + 1.4, 9.6),
        squeeze=False,
    )
    motif_by_seed: dict[str, Any] = {"seeds": list(seeds), "init": {}, "final": {}}

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

        for row, (key, row_label) in enumerate(row_specs):
            ax = axes[row, col]
            data = panels[key]
            vmax = max(symmetric_abs_vmax(data), 1e-9)
            im = ax.imshow(
                data, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
                interpolation="nearest", origin="lower",
            )
            is_xh = key.endswith("_xh")
            x_name = "hidden unit (clustered)" if is_xh else "source h (clustered)"
            y_name = "input char" if is_xh else "target h (clustered)"
            if row == 0:
                ax.set_title(f"seed {seed}", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{row_label}\n{y_name}", fontsize=7)
            else:
                ax.set_ylabel("")
            # Show x label only on the last row of each matrix family.
            if row in (1, 3):
                ax.set_xlabel(x_name, fontsize=6)
            else:
                ax.set_xlabel("")
            ny, nx = data.shape
            ax.set_xticks([0, max(nx - 1, 0)])
            ax.set_yticks([0, max(ny - 1, 0)])
            if row not in (1, 3):
                ax.tick_params(axis="x", labelbottom=False)
            else:
                ax.tick_params(axis="x", labelsize=5)
            if col == 0:
                ax.tick_params(axis="y", labelsize=5)
            else:
                ax.tick_params(axis="y", labelleft=False)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, shrink=0.82)
            cbar.ax.tick_params(labelsize=5)
            cbar.set_label("weight", fontsize=5)

    finalize_grid_figure(
        fig,
        suptitle=f"{exp} · clustered weight matrices across seeds",
        top=0.93,
        hspace=0.42,
        wspace=0.42,
        bottom=0.06,
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

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.2))
    finalize_grid_figure(
        fig,
        suptitle="Weights: random init vs after learning (each stage clustered)",
        top=0.92,
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
        figsize=(2.45 * n_cols + 0.5, 2.45 * n_rows + 0.7),
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
