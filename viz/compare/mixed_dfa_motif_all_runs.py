"""All-runs mixed DFA motif fold board (used by scripts/r43_motif_figs.py)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from viz.plot_layout import finalize_grid_figure, save_figure
from viz.weight_structure import compute_weight_colored_hl_motif_counts

EPS = 0.5


def _snap_census(snap_path: Path) -> dict:
    d = np.load(snap_path, allow_pickle=True)
    out = compute_weight_colored_hl_motif_counts(
        d["weights_hidden_to_hidden"], d["dale_law"], mode="mean",
    )
    it = int(d["learning_snap_iteration"]) if "learning_snap_iteration" in d.files else -1
    we = float(d["learning_snap_word_err"]) if "learning_snap_word_err" in d.files else float("nan")
    return {"it": it, "we": we, **out}


def _subsample_snaps(snaps: list[Path], max_snaps: int) -> list[Path]:
    if len(snaps) <= max_snaps:
        return snaps
    idx = np.linspace(0, len(snaps) - 1, max_snaps, dtype=int)
    return [snaps[int(i)] for i in idx]


def collect_all_runs_motif_cache(
    cache_path: Path,
    *,
    min_start: int,
    max_snaps_per_run: int,
    model: str,
    seed: int,
) -> Path:
    import vocab_mixed_dfa as vocab
    from experiment import checkpoint_path
    from rnn.learning_snaps import list_learning_snaps

    manifest_path = cache_path.parent.parent / "data" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dfa_by_run = {int(r["run_id"]): int(r["n_dfa_states"]) for r in manifest["runs"]}

    fold_rows: list[dict] = []
    run_series: list[dict] = []
    for entry in vocab.iter_runs():
        run_id = int(entry["run_id"])
        n_dfa = int(dfa_by_run[run_id])
        ckpt = checkpoint_path(entry["task"], model, seed=seed)
        snaps = list_learning_snaps(ckpt)
        if len(snaps) < 2:
            print(f"skip r{run_id:02d}: need >=2 learning snaps", flush=True)
            continue
        series = [_snap_census(p) for p in _subsample_snaps(snaps, max_snaps_per_run)]
        c0, c1 = series[0]["cnt"], series[-1]["cnt"]
        for key, v0 in c0.items():
            if not key.startswith("T|") or float(v0) < min_start:
                continue
            v1 = float(c1.get(key, 0))
            fold_rows.append({
                "run_id": run_id,
                "n_dfa_states": n_dfa,
                "key": key,
                "log_c0": float(np.log(float(v0) + EPS)),
                "log_fold": float(np.log((v1 + EPS) / (float(v0) + EPS))),
            })
        it0, it1 = float(series[0]["it"]), float(series[-1]["it"])
        denom = max(it1 - it0, 1.0)
        prog_rows = []
        for row in series:
            triad_keys = [k for k in row["cnt"] if k.startswith("T|")]
            counts = np.array([float(row["cnt"].get(k, 0)) for k in triad_keys], dtype=float)
            slog = np.log(np.maximum(counts, EPS))
            prog_rows.append({
                "progress": float((float(row["it"]) - it0) / denom),
                "edges": float(row["edges"]),
                "triples_conn": float(row["triples_conn"]),
                "sd_log_count": float(slog.std()) if len(slog) else float("nan"),
            })
        run_series.append({"run_id": run_id, "n_dfa_states": n_dfa, "progress": prog_rows})
        print(f"r{run_id:02d} DFA={n_dfa:3d}  pts={sum(1 for r in fold_rows if r['run_id']==run_id)}", flush=True)

    payload = {
        "comparison": vocab.COMPARISON_NAME,
        "model": model,
        "seed": seed,
        "min_start": min_start,
        "n_runs": len(run_series),
        "n_fold_points": len(fold_rows),
        "fold_rows": fold_rows,
        "run_series": run_series,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {cache_path}  runs={len(run_series)}  points={len(fold_rows)}", flush=True)
    return cache_path


def plot_all_runs_over_learning(
    cache_path: Path,
    out_path: Path,
    *,
    min_start: int,
    rebuild_cache: bool,
    model: str,
    seed: int,
    max_snaps_per_run: int,
    ols_fn,
) -> Path:
    if rebuild_cache or not cache_path.is_file():
        collect_all_runs_motif_cache(
            cache_path,
            min_start=min_start,
            max_snaps_per_run=max_snaps_per_run,
            model=model,
            seed=seed,
            )
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    rows = payload["fold_rows"]
    series = payload["run_series"]
    log_c0 = np.array([r["log_c0"] for r in rows], dtype=float)
    log_fold = np.array([r["log_fold"] for r in rows], dtype=float)
    beta, _, r2 = ols_fn(log_fold, log_c0)
    x_line = np.linspace(float(log_c0.min()), float(log_c0.max()), 100)
    y_line = beta[0] + beta[1] * x_line

    dfa_vals = [float(r["n_dfa_states"]) for r in series]
    dfa_cmap = plt.get_cmap("viridis")
    dfa_norm = plt.Normalize(vmin=min(dfa_vals), vmax=max(dfa_vals))
    sc_colors = [dfa_cmap(dfa_norm(r["n_dfa_states"])) for r in rows]

    grid = np.linspace(0.0, 1.0, 25)
    edge_mat, tri_mat, sd_mat = [], [], []
    for run in series:
        pr = np.array([p["progress"] for p in run["progress"]], dtype=float)
        if len(pr) < 2:
            continue
        edge_mat.append(np.interp(grid, pr, [p["edges"] for p in run["progress"]]))
        tri_mat.append(np.interp(grid, pr, [p["triples_conn"] for p in run["progress"]]))
        sd_mat.append(np.interp(grid, pr, [p["sd_log_count"] for p in run["progress"]]))
    edge_mat = np.asarray(edge_mat)
    tri_mat = np.asarray(tri_mat)
    sd_mat = np.asarray(sd_mat)
    edge_med = np.median(edge_mat, axis=0)
    tri_med = np.median(tri_mat, axis=0)
    sd_med = np.median(sd_mat, axis=0)
    edge_q25, edge_q75 = np.percentile(edge_mat, [25, 75], axis=0)
    tri_q25, tri_q75 = np.percentile(tri_mat, [25, 75], axis=0)

    fig = plt.figure(figsize=(12.0, 6.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 0.85])
    ax_sc, ax_log, ax_spr = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    ax_edge, ax_sd, ax_tri = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])

    ax_sc.scatter(log_c0, log_fold, c=sc_colors, s=14, alpha=0.55, edgecolors="none")
    ax_sc.plot(x_line, y_line, color="#c0392b", lw=1.4, label=f"pooled OLS  R2={r2:.2f}")
    ax_sc.axhline(0.0, color="0.55", lw=0.7, ls="--")
    ax_sc.set_xlabel("log start count", fontsize=8)
    ax_sc.set_ylabel("log fold (end / start)", fontsize=8)
    ax_sc.set_title(f"all runs: start vs fold  (start>={min_start}, n={len(rows)}, {payload['n_runs']} runs)", fontsize=8, pad=4)
    ax_sc.legend(fontsize=7, loc="upper right", frameon=True)
    ax_sc.tick_params(labelsize=7)
    ax_sc.grid(True, alpha=0.25)

    ax_log.plot(grid, tri_med, color="0.1", lw=1.6, label="median run")
    ax_log.fill_between(grid, tri_q25, tri_q75, color="0.75", label="middle 50% runs")
    ax_log.set_yscale("log")
    ax_log.set_xlabel("normalized training progress", fontsize=8)
    ax_log.set_ylabel("connected triad instances", fontsize=8)
    ax_log.set_title("triad mass (median across runs)", fontsize=8, pad=4)
    ax_log.legend(fontsize=6, loc="lower right", frameon=True)
    ax_log.tick_params(labelsize=7)
    ax_log.grid(True, alpha=0.25)

    ax_spr.plot(grid, sd_med, color="0.1", lw=1.5, label="median sd")
    ax_spr.set_xlabel("normalized training progress", fontsize=8)
    ax_spr.set_ylabel("sd log triad count", fontsize=8)
    ax_spr.set_title(f"spread  (sd {sd_med[0]:.2f} -> {sd_med[-1]:.2f})", fontsize=8, pad=4)
    ax_spr.legend(fontsize=6, loc="upper right", frameon=True)
    ax_spr.tick_params(labelsize=7)
    ax_spr.grid(True, alpha=0.25)

    ax_edge.plot(grid, edge_med, color="#4c78a8", lw=1.6)
    ax_edge.fill_between(grid, edge_q25, edge_q75, color="#4c78a8", alpha=0.2)
    ax_edge.set_xlabel("normalized training progress", fontsize=8)
    ax_edge.set_ylabel("count", fontsize=8)
    ax_edge.set_title(f"strong |W_hh| edges  (median x{edge_med[-1]/max(edge_med[0],1):.2f})", fontsize=8, pad=4)
    ax_edge.tick_params(labelsize=7)
    ax_edge.grid(True, alpha=0.25)

    ax_sd.plot(grid, sd_med, color="#c0392b", lw=1.6)
    ax_sd.set_xlabel("normalized training progress", fontsize=8)
    ax_sd.set_ylabel("sd log count", fontsize=8)
    ax_sd.set_title(f"triad spread  (sd {sd_med[0]:.2f} -> {sd_med[-1]:.2f})", fontsize=8, pad=4)
    ax_sd.tick_params(labelsize=7)
    ax_sd.grid(True, alpha=0.25)

    ax_tri.plot(grid, tri_med, color="#2ca02c", lw=1.6)
    ax_tri.fill_between(grid, tri_q25, tri_q75, color="#2ca02c", alpha=0.2)
    ax_tri.set_xlabel("normalized training progress", fontsize=8)
    ax_tri.set_ylabel("count", fontsize=8)
    ax_tri.set_title(f"connected triads  (median x{tri_med[-1]/max(tri_med[0],1):.2f})", fontsize=8, pad=4)
    ax_tri.tick_params(labelsize=7)
    ax_tri.grid(True, alpha=0.25)

    sm = plt.cm.ScalarMappable(cmap=dfa_cmap, norm=dfa_norm)
    cax = fig.add_axes([0.935, 0.34, 0.012, 0.52])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("DFA states", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    finalize_grid_figure(
        fig,
        suptitle="mixed DFA (all runs): motif homogenization vs DFA size",
        top=0.92, bottom=0.08, left=0.06, right=0.92, hspace=0.42, wspace=0.32,
    )
    save_figure(fig, out_path, dpi=160)
    plt.close(fig)
    print(f"wrote {out_path}")
    print(f"pooled R2={r2:.3f}  n={len(rows)}  runs={payload['n_runs']}")
    return out_path
