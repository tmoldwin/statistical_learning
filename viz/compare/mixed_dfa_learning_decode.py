"""Learning-time decoding for mixed-vocab runs, binned like Figure 12."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiment import checkpoint_path
from vocab_mixed_dfa import COMPARISON_NAME, iter_runs
from viz.compare.decoding import (
    DECODE_FEATURE_COLORS,
    DECODING_FEATURES,
    chance_corrected,
    compute_panel_decoding,
    feature_display_name,
)
from viz.compare.mixed_dfa_viz import (
    _dfa_bin_title,
    _dfa_quantile_edges,
    _dfa_states,
    _load_panels,
    _sanitize,
    _subset_for_dfa_bin,
)
from viz.compare.sweep_output import sweep_decoding_dir
from viz.plot_layout import finalize_grid_figure, hide_x_tick_labels, save_figure

_FEATURE_ORDER = DECODING_FEATURES
_PC_KS: tuple[int | None, ...] = tuple([*range(1, 11), None])
_NEU_KS: tuple[int | None, ...] = tuple([*range(1, 11), None])
_NEURON_TRIALS = 10
_EARLY_FRAC = 0.25


def _basis_label(kind: str, k: int | None) -> tuple[str, str]:
    if kind == "pca":
        if k is None:
            return "full", "full H"
        if k == 1:
            return "pc1", "1 PC"
        return f"pc{k}", f"{k} PCs"
    if k is None:
        return "neu_full", "full H"
    if k == 1:
        return "neu1", "1 neuron"
    return f"neu{k}", f"{k} neurons"


def _all_basis_keys() -> tuple[str, ...]:
    return tuple(_basis_label("pca", k)[0] for k in _PC_KS) + tuple(
        _basis_label("neuron", k)[0] for k in _NEU_KS
    )


def _early_snaps(snaps: list[Path], *, early_frac: float = _EARLY_FRAC) -> list[Path]:
    if not snaps:
        return []
    iters = [int(s.stem.split("_")[1]) for s in snaps]
    stop = float(max(iters))
    keep = [s for s, it in zip(snaps, iters) if it / stop <= float(early_frac)]
    return keep if keep else snaps[:1]


def _interp(progress: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    order = np.argsort(progress)
    xp = np.asarray(progress, dtype=float)[order]
    yp = np.asarray(values, dtype=float)[order]
    mask = np.isfinite(xp) & np.isfinite(yp)
    if int(mask.sum()) < 2:
        return np.full(grid.shape, np.nan, dtype=float)
    return np.interp(grid, xp[mask], yp[mask], left=np.nan, right=np.nan)


def _aggregate(
    seed_rows: list[list[dict[str, Any]]],
    *,
    early_xlim: float,
    n_grid: int = 41,
) -> dict[str, Any]:
    grid = np.linspace(0.0, float(early_xlim), int(n_grid))
    basis_keys = _all_basis_keys()
    feat_out: dict[str, Any] = {}
    for feat in _FEATURE_ORDER:
        feat_out[feat] = {}
        for bkey in basis_keys:
            mats = []
            for rows in seed_rows:
                prog = np.asarray([r["progress"] for r in rows], dtype=float)
                vals = np.asarray(
                    [
                        float(r.get("features", {}).get(feat, {}).get(bkey, float("nan")))
                        for r in rows
                    ],
                    dtype=float,
                )
                mats.append(_interp(prog, vals, grid))
            mat = np.vstack(mats)
            feat_out[feat][f"{bkey}_mean"] = np.nanmean(mat, axis=0).tolist()
            feat_out[feat][f"{bkey}_std"] = np.nanstd(mat, axis=0).tolist()
    we_mats = []
    for rows in seed_rows:
        prog = np.asarray([r["progress"] for r in rows], dtype=float)
        vals = np.asarray([r["word_err"] for r in rows], dtype=float)
        we_mats.append(_interp(prog, vals, grid))
    we_mat = np.vstack(we_mats)
    return {
        "progress_grid": grid.tolist(),
        "word_err_mean": np.nanmean(we_mat, axis=0).tolist(),
        "word_err_std": np.nanstd(we_mat, axis=0).tolist(),
        "features": feat_out,
        "n_runs": len(seed_rows),
        "basis_keys": list(basis_keys),
    }


def _decode_one_run(task: str, *, seed: int) -> tuple[list[dict[str, Any]], int]:
    from experiment import TASKS
    from rnn.learning_snaps import list_learning_snaps
    from viz.compare._data import load_task_viz_context

    ckpt = checkpoint_path(task, "rnn", seed=seed)
    all_snaps = list_learning_snaps(ckpt)
    if not all_snaps:
        raise FileNotFoundError(f"no learning snaps for {task} seed {seed}")
    stop_iter = max(int(s.stem.split("_")[1]) for s in all_snaps)
    snaps = _early_snaps(all_snaps)

    cfg = TASKS[task]
    text_chars = min(int(cfg.get("metric_rollout_len", cfg.get("viz_length", 50))), 500)
    max_k = 10

    rows: list[dict[str, Any]] = []
    for snap in snaps:
        print(f"  {task} seed {seed} decode {snap.name}", flush=True)
        meta = np.load(snap, allow_pickle=True)
        iteration = (
            int(meta["learning_snap_iteration"])
            if "learning_snap_iteration" in meta.files
            else int(snap.stem.split("_")[1])
        )
        word_err = (
            float(meta["learning_snap_word_err"])
            if "learning_snap_word_err" in meta.files
            else float("nan")
        )
        ctx = load_task_viz_context(
            task, model_type="rnn", seed=seed, text_chars=text_chars, checkpoint=snap,
        )
        panel = compute_panel_decoding(
            ctx,
            max_k=max_k,
            neuron_sampling="random",
            n_random_trials=_NEURON_TRIALS,
        )
        feat_out: dict[str, Any] = {}
        for feat in _FEATURE_ORDER:
            blob = panel.get("features", {}).get(feat, {})
            if blob.get("error"):
                feat_out[feat] = {"error": blob["error"]}
                continue
            chance = float(blob.get("chance", float("nan")))
            by_k = blob.get("by_k") or []
            by_k_neu = blob.get("by_k_neurons") or []
            vals: dict[str, float] = {}
            for k in _PC_KS:
                key, _ = _basis_label("pca", k)
                if k is None:
                    full = blob.get("full_hidden")
                    y = blob.get("full_hidden_cc")
                    if y is None and full is not None and np.isfinite(full) and np.isfinite(chance):
                        y = chance_corrected(float(full), chance)
                    vals[key] = float(y) if y is not None and np.isfinite(y) else float("nan")
                else:
                    raw = by_k[k - 1] if k <= len(by_k) else None
                    vals[key] = (
                        float(chance_corrected(float(raw), chance))
                        if raw is not None and np.isfinite(raw) and np.isfinite(chance)
                        else float("nan")
                    )
            for k in _NEU_KS:
                key, _ = _basis_label("neuron", k)
                if k is None:
                    full = blob.get("full_hidden_neurons", blob.get("full_hidden"))
                    vals[key] = (
                        float(chance_corrected(float(full), chance))
                        if full is not None and np.isfinite(full) and np.isfinite(chance)
                        else float("nan")
                    )
                else:
                    raw = by_k_neu[k - 1] if k <= len(by_k_neu) else None
                    vals[key] = (
                        float(chance_corrected(float(raw), chance))
                        if raw is not None and np.isfinite(raw) and np.isfinite(chance)
                        else float("nan")
                    )
            feat_out[feat] = {"chance": chance, **vals}
        rows.append({
            "iteration": iteration,
            "word_err": word_err,
            "snap": snap.name,
            "features": feat_out,
            "progress": float(iteration) / float(max(stop_iter, 1)),
        })
    rows.sort(key=lambda r: int(r["iteration"]))
    return rows, int(stop_iter)


def collect_learning_decode_by_dfa(
    *,
    seed: int = 1,
    recompute: bool = False,
    early_xlim: float = 0.2,
) -> Path:
    out = sweep_decoding_dir(COMPARISON_NAME) / "learning_decode_by_dfa.json"
    if out.is_file() and not recompute:
        return out

    panels_payload = _load_panels()
    panel_by_run = {
        int(p["run_id"]): p
        for p in panels_payload.get("panels", [])
        if "error" not in p and "run_id" in p
    }

    run_payloads: list[dict[str, Any]] = []
    for entry in iter_runs():
        rid = int(entry["run_id"])
        task = str(entry["task"])
        panel = panel_by_run.get(rid)
        n_dfa = int(panel["n_dfa_states"]) if panel else _dfa_states(list(entry["words"]))
        print(f"learning-decode collect {task} seed {seed}  dfa={n_dfa}", flush=True)
        rows, stop_iter = _decode_one_run(task, seed=seed)
        run_payloads.append({
            "run_id": rid,
            "task": task,
            "seed": int(seed),
            "n_dfa_states": n_dfa,
            "n_words": int(entry["n_words"]),
            "stop_iter": int(stop_iter),
            "snaps": rows,
        })

    dfa_vals = np.asarray([r["n_dfa_states"] for r in run_payloads], dtype=float)
    edges = _dfa_quantile_edges(dfa_vals, n_bins=4)
    bins: list[dict[str, Any]] = []
    for bi in range(len(edges) - 1):
        subset = _subset_for_dfa_bin(run_payloads, edges=edges, bin_index=bi)
        agg = _aggregate([r["snaps"] for r in subset], early_xlim=early_xlim) if subset else None
        bins.append({
            "bin_index": bi,
            "title": _dfa_bin_title(edges, bi),
            "lo": float(edges[bi]),
            "hi": float(edges[bi + 1]),
            "n_runs": len(subset),
            "run_ids": [int(r["run_id"]) for r in subset],
            "aggregated": agg,
        })

    payload = {
        "seed": int(seed),
        "early_xlim": float(early_xlim),
        "pc_ks": [k if k is not None else "full" for k in _PC_KS],
        "neuron_ks": [k if k is not None else "full" for k in _NEU_KS],
        "n_neuron_trials": _NEURON_TRIALS,
        "edges": [float(x) for x in edges],
        "runs": run_payloads,
        "bins": bins,
    }
    out.write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return out


def plot_learning_decode_by_dfa_bins(
    *,
    json_path: Path | None = None,
    outfile: str = "learning_decode_by_dfa.png",
    early_xlim: float = 0.2,
) -> Path:
    """Columns = Fig-12 DFA bins; rows = PCA 1-10+full then random neurons 1-10+full."""
    path = json_path or (sweep_decoding_dir(COMPARISON_NAME) / "learning_decode_by_dfa.json")
    if not path.is_file():
        path = collect_learning_decode_by_dfa(early_xlim=early_xlim)
    payload = json.loads(path.read_text(encoding="utf-8"))
    bins = [b for b in payload.get("bins", []) if b.get("aggregated")]
    if not bins:
        raise FileNotFoundError(f"no binned learning curves in {path}")

    blocks = (
        ("PCA", [_basis_label("pca", k) for k in _PC_KS]),
        ("random neurons", [_basis_label("neuron", k) for k in _NEU_KS]),
    )
    n_basis = sum(len(specs) for _, specs in blocks)
    n_bins = len(bins)
    n_rows = n_basis + len(blocks)
    fig = plt.figure(figsize=(2.45 * n_bins + 0.9, 1.15 * n_rows + 0.9))
    height_ratios: list[float] = []
    for _, specs in blocks:
        height_ratios.append(0.28)
        height_ratios.extend([1.0] * len(specs))
    gs = fig.add_gridspec(n_rows, n_bins, height_ratios=height_ratios)
    axes = np.empty((n_rows, n_bins), dtype=object)
    for r in range(n_rows):
        for c in range(n_bins):
            axes[r, c] = fig.add_subplot(gs[r, c])

    word_err_line = None
    seed = int(payload.get("seed", 1))
    x0, x1 = 0.0, float(early_xlim)
    row_i = 0
    first_data_ax = None

    for block_name, specs in blocks:
        for bi in range(n_bins):
            ax = axes[row_i, bi]
            ax.set_axis_off()
            if bi == 0:
                ax.text(
                    0.0, 0.5, block_name,
                    transform=ax.transAxes, ha="left", va="center",
                    fontsize=8, fontweight="bold",
                )
        row_i += 1

        for ki, (bkey, blabel) in enumerate(specs):
            for bi, blob in enumerate(bins):
                ax = axes[row_i, bi]
                if first_data_ax is None:
                    first_data_ax = ax
                agg = blob["aggregated"]
                progress_all = np.asarray(agg["progress_grid"], dtype=float)
                we_mean_all = np.asarray(agg["word_err_mean"], dtype=float)
                we_std_all = np.asarray(agg["word_err_std"], dtype=float)
                zoom = (progress_all >= x0) & (progress_all <= x1 + 1e-9)
                progress = progress_all[zoom]
                we_mean = we_mean_all[zoom]
                we_std = we_std_all[zoom]
                for feat in _FEATURE_ORDER:
                    color = DECODE_FEATURE_COLORS[feat]
                    feat_blob = agg.get("features", {}).get(feat, {})
                    y_mean = np.asarray(feat_blob.get(f"{bkey}_mean", []), dtype=float)
                    y_std = np.asarray(feat_blob.get(f"{bkey}_std", []), dtype=float)
                    if y_mean.size == 0:
                        continue
                    y_mean = y_mean[zoom]
                    if y_std.size == zoom.size:
                        y_std = y_std[zoom]
                    show_label = first_data_ax is ax and feat == _FEATURE_ORDER[0]
                    # label all features once on first data panel
                    ax.plot(
                        progress, y_mean, color=color, lw=1.2, marker="o", ms=1.8,
                        label=feature_display_name(feat) if (row_i == 1 and bi == 0) else None,
                    )
                    del show_label
                    if y_std.size == y_mean.size and np.any(np.isfinite(y_std)):
                        ax.fill_between(
                            progress, y_mean - y_std, y_mean + y_std,
                            color=color, alpha=0.12, linewidth=0,
                        )
                if row_i == 1:
                    ax.set_title(f"{blob['title']}  (n={blob['n_runs']})", fontsize=7.5, pad=2)
                if row_i == n_rows - 1:
                    ax.set_xlabel("progress", fontsize=6.5)
                else:
                    hide_x_tick_labels(ax)
                ax.set_xlim(x0, x1)
                ax.set_ylim(-0.05, 1.05)
                ax.axhline(0.0, color="0.75", lw=0.5, ls=":")
                ax.grid(True, alpha=0.22)
                ax.tick_params(labelsize=5)
                if bi == 0:
                    ax.set_ylabel(blabel, fontsize=6, labelpad=2)
                if np.any(np.isfinite(we_mean)):
                    ax2 = ax.twinx()
                    (line,) = ax2.plot(
                        progress, we_mean, color="0.45", lw=0.8, ls="--", alpha=0.75,
                    )
                    if word_err_line is None:
                        word_err_line = line
                    if np.any(np.isfinite(we_std)):
                        ax2.fill_between(
                            progress, we_mean - we_std, we_mean + we_std,
                            color="0.45", alpha=0.08, linewidth=0,
                        )
                    ax2.set_ylim(-0.02, 1.05)
                    ax2.tick_params(labelsize=4.5, colors="0.45")
                    if bi == n_bins - 1 and row_i == n_rows - 1:
                        ax2.set_ylabel("word err", fontsize=5.5, color="0.45")
                    else:
                        ax2.set_yticklabels([])
            row_i += 1

    handles, labels = ([], [])
    if first_data_ax is not None:
        handles, labels = first_data_ax.get_legend_handles_labels()
    if word_err_line is not None:
        handles = [*handles, word_err_line]
        labels = [*labels, "word err"]
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.002),
        ncol=max(1, len(labels)), fontsize=6, frameon=False,
        columnspacing=0.9, handletextpad=0.35,
    )
    n_runs = sum(int(b.get("n_runs", 0)) for b in bins)
    title = (
        "Early readout over learning by DFA bin "
        + "(seed {}, {} mixed runs; progress 0-{}; PCA + random neurons)".format(
            seed, n_runs, early_xlim,
        )
    )
    finalize_grid_figure(
        fig,
        # Match Figure 12 DFA bins; early window only.
        # Title set via format string above.
        suptitle=title,
        top=0.965,
        bottom=0.035,
        left=0.07,
        right=0.95,
        wspace=0.18,
        hspace=0.35,
    )
    out = sweep_decoding_dir(COMPARISON_NAME) / outfile
    save_figure(fig, out)
    plt.close(fig)
    return out
