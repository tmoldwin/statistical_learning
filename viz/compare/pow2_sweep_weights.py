"""Weight-structure metrics across the pow2 (word-count x length) sweep."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiment import checkpoint_path
from viz.compare.pow2_sweep_spec import POW2_SWEEP_SPEC_H100, Pow2SweepSpec
from viz.compare.sweep_output import sweep_data_dir, sweep_figures_dir
from viz.plot_layout import finalize_grid_figure, save_figure
from viz.weight_structure import (
    compute_weight_motif_metrics,
    compute_weight_structure_metrics,
    init_weights_for_model,
)


# (family, key, title, shared vmin/vmax across init+final or None)
_WEIGHT_HEATMAP_METRICS: tuple[tuple[str, str, str, tuple[float, float] | None], ...] = (
    ("motif", "cluster_cohesion_xh", r"$W_{xh}$ cluster cohesion", (0.0, 1.0)),
    ("motif", "hh_adjacent_corr", r"$W_{hh}$ adjacent |corr|", (0.0, 1.0)),
    ("motif", "hh_within_between_ratio", r"$W_{hh}$ within/between |w|", None),
    ("structure", "input_over_recurrent_norm", "input / recurrent Frobenius", None),
    ("structure", "mean_input_drive_fraction", "mean input-drive fraction", (0.0, 1.0)),
)


def _metrics_for_checkpoint(
    task: str,
    *,
    seed: int,
    model_type: str = "rnn",
) -> dict[str, Any] | None:
    from visualize import load_model_for_viz, weights_for_plot

    ckpt = checkpoint_path(task, model_type, seed=seed)
    if not ckpt.is_file():
        return None
    model = load_model_for_viz(str(ckpt), model_type)
    w_in_f, w_rec_f, w_out_f, dale_sign = weights_for_plot(model)
    w_in_i, w_rec_i, w_out_i = init_weights_for_model(model, seed)
    if dale_sign is not None and len(dale_sign) == w_in_f.shape[0]:
        from rnn.rnn_dyn import dale_signs_ordered, permute_hidden_by_dale

        if not dale_signs_ordered(np.asarray(dale_sign)):
            w_in_i, w_rec_i, w_out_i, _, _ = permute_hidden_by_dale(
                w_in_i, w_rec_i, w_out_i, np.zeros(w_in_i.shape[0]), np.asarray(dale_sign),
            )
    return {
        "init": compute_weight_structure_metrics(w_in_i, w_rec_i, w_out_i),
        "final": compute_weight_structure_metrics(w_in_f, w_rec_f, w_out_f),
        "motif_init": compute_weight_motif_metrics(w_in_i, w_rec_i),
        "motif_final": compute_weight_motif_metrics(w_in_f, w_rec_f),
    }


def write_pow2_sweep_weight_metrics(
    *,
    seeds: tuple[int, ...] | None = None,
    model_type: str = "rnn",
    outfile: str = "sweep_weight_metrics.json",
    spec: Pow2SweepSpec = POW2_SWEEP_SPEC_H100,
) -> Path:
    """Collect init/final weight metrics for every cell x seed; write JSON."""
    run_seeds = seeds if seeds is not None else spec.default_seeds
    panels: list[dict[str, Any]] = []
    n_cells = len(list(spec.iter_cells()))
    done = 0
    for n_words, length in spec.iter_cells():
        task = spec.task_name(n_words, length)
        seed_rows: list[dict[str, Any]] = []
        for seed in run_seeds:
            metrics = _metrics_for_checkpoint(task, seed=seed, model_type=model_type)
            if metrics is None:
                seed_rows.append({"seed": seed, "error": "missing checkpoint"})
                continue
            seed_rows.append({"seed": seed, **metrics})
        panels.append({
            "task": task,
            "n_words": n_words,
            "length": length if isinstance(length, int) else str(length),
            "seeds": seed_rows,
        })
        done += 1
        print(f"  weights {done}/{n_cells} {task} ({sum(1 for r in seed_rows if 'error' not in r)}/{len(run_seeds)} seeds)")

    out_path = sweep_data_dir(spec.comparison_name) / outfile
    payload = {
        "comparison": spec.comparison_name,
        "model_type": model_type,
        "word_counts": list(spec.word_counts),
        "lengths": [L if isinstance(L, int) else str(L) for L in spec.lengths],
        "seeds": list(run_seeds),
        "panels": panels,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    return out_path


def _mean_metric_matrix(
    panels: list[dict[str, Any]],
    *,
    word_counts: tuple[int, ...],
    lengths: tuple[object, ...],
    family: str,
    key: str,
    stage: str,
) -> np.ndarray:
    """Rows = lengths, cols = word counts; mean over seeds (NaN if none)."""
    by_cell: dict[tuple[int, object], list[float]] = {}
    for panel in panels:
        n_words = int(panel["n_words"])
        length = panel["length"]
        if isinstance(length, str) and length.isdigit():
            length = int(length)
        vals: list[float] = []
        for row in panel.get("seeds", []):
            if "error" in row:
                continue
            if family == "structure":
                blob = row[stage]  # init / final
            else:
                blob = row[f"motif_{stage}"]
            if key not in blob:
                continue
            vals.append(float(blob[key]))
        by_cell[(n_words, length)] = vals

    mat = np.full((len(lengths), len(word_counts)), np.nan, dtype=float)
    for li, length in enumerate(lengths):
        length_key: object = length if isinstance(length, int) else str(length)
        for wi, n_words in enumerate(word_counts):
            vals = by_cell.get((n_words, length_key), [])
            # tolerate length stored as int when panel has int
            if not vals and length == "mixed":
                vals = by_cell.get((n_words, "mixed"), [])
            elif not vals:
                vals = by_cell.get((n_words, length), [])
            if vals:
                mat[li, wi] = float(np.mean(vals))
    return mat


def plot_pow2_sweep_weight_metric_heatmaps(
    payload: dict[str, Any],
    *,
    outfile: str = "sweep_weight_metrics.png",
    spec: Pow2SweepSpec = POW2_SWEEP_SPEC_H100,
) -> Path:
    """One row per metric; columns = init | final (word-count x length heatmaps)."""
    word_counts = tuple(int(w) for w in payload["word_counts"])
    lengths_raw = payload["lengths"]
    lengths: tuple[object, ...] = tuple(
        int(L) if isinstance(L, int) or (isinstance(L, str) and L.isdigit()) else L
        for L in lengths_raw
    )
    panels = payload["panels"]
    n_metrics = len(_WEIGHT_HEATMAP_METRICS)
    fig, axes = plt.subplots(
        n_metrics, 2,
        figsize=(2.4 * len(word_counts) + 1.5, 1.55 * n_metrics + 1.2),
        squeeze=False,
    )
    length_labels = [spec.length_label(L) for L in lengths]
    wc_labels = [str(w) for w in word_counts]

    for ri, (family, key, title, fixed_lim) in enumerate(_WEIGHT_HEATMAP_METRICS):
        mats = []
        for stage in ("init", "final"):
            mats.append(_mean_metric_matrix(
                panels,
                word_counts=word_counts,
                lengths=lengths,
                family=family,
                key=key,
                stage=stage,
            ))
        if fixed_lim is not None:
            vmin, vmax = fixed_lim
        else:
            both = np.concatenate([m[np.isfinite(m)] for m in mats if np.any(np.isfinite(m))])
            if both.size == 0:
                vmin, vmax = 0.0, 1.0
            else:
                vmin = float(np.nanmin(both))
                vmax = float(np.nanmax(both))
                if vmax <= vmin:
                    vmax = vmin + 1e-6

        for ci, (stage, mat) in enumerate(zip(("init", "final"), mats)):
            ax = axes[ri, ci]
            im = ax.imshow(
                mat, aspect="auto", origin="upper",
                cmap="YlOrRd", vmin=vmin, vmax=vmax,
                interpolation="nearest",
            )
            ax.set_xticks(np.arange(len(wc_labels)))
            ax.set_xticklabels(wc_labels, fontsize=7)
            ax.set_yticks(np.arange(len(length_labels)))
            if ci == 0:
                ax.set_yticklabels(length_labels, fontsize=7)
                ax.set_ylabel(title, fontsize=8)
            else:
                ax.set_yticklabels([])
            if ri == 0:
                ax.set_title(stage, fontsize=9, fontweight="medium")
            if ri == n_metrics - 1:
                ax.set_xlabel("# words", fontsize=8)
            for li in range(mat.shape[0]):
                for wi in range(mat.shape[1]):
                    v = mat[li, wi]
                    if not np.isfinite(v):
                        continue
                    ax.text(
                        wi, li, f"{v:.2f}",
                        ha="center", va="center", fontsize=5.5,
                        color="black" if v < (vmin + vmax) * 0.65 else "white",
                    )
            cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
            cbar.ax.tick_params(labelsize=6)

    n_seeds = len(payload.get("seeds", []))
    finalize_grid_figure(
        fig,
        top=0.93,
        bottom=0.06,
        left=0.14,
        hspace=0.35,
        wspace=0.18,
        suptitle=(
            f"Weight metrics init vs final across word-count x length "
            f"(mean over {n_seeds} seeds; {spec.comparison_name})"
        ),
    )
    out_dir = sweep_figures_dir(spec.comparison_name)
    # put under weights/ sibling if preferred, but trajectories dir is the sweep figures bucket
    weights_dir = out_dir.parent / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    out_path = weights_dir / outfile
    save_figure(fig, out_path)
    print(f"wrote {out_path}")
    return out_path


def run_pow2_sweep_weight_metric_plots(
    *,
    seeds: tuple[int, ...] | None = None,
    recompute: bool = True,
    outfile_json: str = "sweep_weight_metrics.json",
    outfile_fig: str = "sweep_weight_metrics.png",
    spec: Pow2SweepSpec = POW2_SWEEP_SPEC_H100,
) -> Path:
    json_path = sweep_data_dir(spec.comparison_name) / outfile_json
    if recompute or not json_path.is_file():
        write_pow2_sweep_weight_metrics(seeds=seeds, outfile=outfile_json, spec=spec)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    return plot_pow2_sweep_weight_metric_heatmaps(payload, outfile=outfile_fig, spec=spec)


def _pick_dfa_span_examples(
    spec: Pow2SweepSpec,
    *,
    n_final: int = 5,
    seed: int = 1,
    model_type: str = "rnn",
) -> list[tuple[int, int, object, str]]:
    """Select distinct DFA-state levels with available checkpoints (sorted ascending)."""
    from viz.compare.pow2_sweep_metric_board import _dfa_state_lookup

    dfa = _dfa_state_lookup(spec)
    by_dfa: dict[int, list[tuple[int, object, str]]] = {}
    for (n_words, length), n_states in dfa.items():
        task = spec.task_name(n_words, length)
        if not checkpoint_path(task, model_type, seed=seed).is_file():
            continue
        by_dfa.setdefault(int(n_states), []).append((int(n_words), length, task))

    levels = sorted(by_dfa)
    if not levels:
        raise FileNotFoundError(f"no weight checkpoints for {spec.comparison_name}")

    # Prefer mid-length fixed words at a level (avoid always picking mixed when both exist).
    def prefer(cands: list[tuple[int, object, str]]) -> tuple[int, object, str]:
        fixed = [c for c in cands if c[1] != "mixed" and c[1] != "mix"]
        pool = fixed or cands
        pool.sort(key=lambda t: (abs(t[0] - 15), t[0], str(t[1])))
        return pool[0]

    if len(levels) <= n_final:
        chosen_levels = levels
    else:
        idxs = np.linspace(0, len(levels) - 1, n_final)
        chosen_levels = []
        for i in idxs:
            lv = levels[int(round(float(i)))]
            if lv not in chosen_levels:
                chosen_levels.append(lv)
        # Fill if rounding collided.
        for lv in levels:
            if len(chosen_levels) >= n_final:
                break
            if lv not in chosen_levels:
                chosen_levels.append(lv)
        chosen_levels = sorted(chosen_levels)[:n_final]

    return [(lv, *prefer(by_dfa[lv])) for lv in chosen_levels]


def plot_pow2_sweep_weight_matrices_by_dfa(
    *,
    outfile: str = "sweep_weight_matrices_by_dfa.png",
    seed: int = 1,
    n_final: int = 5,
    model_type: str = "rnn",
    spec: Pow2SweepSpec = POW2_SWEEP_SPEC_H100,
) -> Path:
    """One init $W_{xh}$/$W_{hh}$ plus clustered finals spanning #DFA states."""
    from visualize import load_model_for_viz, weights_for_plot
    from viz.weight_structure import _cluster_unit_order, symmetric_abs_vmax

    examples = _pick_dfa_span_examples(
        spec, n_final=n_final, seed=seed, model_type=model_type,
    )
    # Init from the hardest example (same H / alphabet scale as a high-DFA final).
    _dfa_hi, _nw_hi, _L_hi, task_init = examples[-1]
    model0 = load_model_for_viz(str(checkpoint_path(task_init, model_type, seed=seed)), model_type)
    w_in_i, w_rec_i, _w_out_i = init_weights_for_model(model0, seed)
    dale0 = model0.get("dale_signs")
    if dale0 is not None and len(dale0) == w_in_i.shape[0]:
        from rnn.rnn_dyn import dale_signs_ordered, permute_hidden_by_dale

        if not dale_signs_ordered(np.asarray(dale0)):
            w_in_i, w_rec_i, _w_out_i, _, _ = permute_hidden_by_dale(
                w_in_i, w_rec_i, _w_out_i, np.zeros(w_in_i.shape[0]), np.asarray(dale0),
            )
    order_i = _cluster_unit_order(w_in_i, w_rec_i)
    init_xh = w_in_i[order_i].T
    init_hh = w_rec_i[np.ix_(order_i, order_i)]

    finals: list[tuple[str, np.ndarray, np.ndarray]] = []
    for n_dfa, n_words, length, task in examples:
        model = load_model_for_viz(str(checkpoint_path(task, model_type, seed=seed)), model_type)
        w_in_f, w_rec_f, _w_out_f, dale_sign = weights_for_plot(model)
        order_f = _cluster_unit_order(w_in_f, w_rec_f)
        label = f"DFA={n_dfa}\n{n_words}w {spec.length_label(length)}"
        finals.append((label, w_in_f[order_f].T, w_rec_f[np.ix_(order_f, order_f)]))

    n_cols = 1 + len(finals)
    fig = plt.figure(figsize=(1.45 * n_cols + 0.55, 3.45))
    gs = fig.add_gridspec(
        2, n_cols,
        wspace=0.20,
        hspace=0.32,
        left=0.07,
        right=0.99,
        top=0.86,
        bottom=0.11,
    )
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(2)])
    cmap = plt.cm.RdBu_r

    xh_panels = [init_xh] + [xh for _lab, xh, _hh in finals]
    hh_panels = [init_hh] + [hh for _lab, _xh, hh in finals]

    col_titles = ["Init"] + [lab for lab, *_ in finals]
    row_panels = (xh_panels, hh_panels)
    row_ylabels = (r"$W_{xh}$", r"$W_{hh}$")

    for row in range(2):
        for col in range(n_cols):
            ax = axes[row, col]
            data = row_panels[row][col]
            vmax = max(symmetric_abs_vmax(data), 1e-9)
            ax.imshow(
                data, aspect="auto", cmap=cmap,
                vmin=-vmax, vmax=vmax,
                interpolation="nearest", origin="lower",
            )
            if row == 0:
                ax.set_title(f"{col_titles[col]}\n±{vmax:.2g}", fontsize=7, pad=2)
            else:
                ax.set_title(f"±{vmax:.2g}", fontsize=6.5, pad=1, color="0.3")
            ny, nx = data.shape
            if row == 1:
                ax.set_xticks([0, max(nx - 1, 0)])
                ax.tick_params(axis="x", labelsize=5)
            else:
                ax.set_xticks([])
            if col == 0:
                ax.set_ylabel(row_ylabels[row], fontsize=9)
            else:
                ax.set_ylabel("")
            ax.set_yticks([0, max(ny - 1, 0)])
            ax.tick_params(axis="y", labelsize=5, labelleft=(col == 0))

    axes[1, 0].set_xlabel("hidden unit", fontsize=6)
    if n_cols > 1:
        axes[1, -1].set_xlabel("source h", fontsize=6)

    fig.suptitle(
        r"Weight matrices: one init, then learned by #DFA states "
        rf"(seed {seed}; each panel clustered + scaled)",
        fontsize=10,
        y=0.98,
    )
    out_dir = Path("experiments/comparisons") / spec.comparison_name / "weights"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / outfile
    save_figure(fig, out_path, dpi=150)
    print(f"wrote {out_path}")
    return out_path


if __name__ == "__main__":
    plot_pow2_sweep_weight_matrices_by_dfa()
