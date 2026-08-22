"""All-runs mixed DFA motif fold board (used by scripts/r43_motif_figs.py)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from viz.plot_layout import finalize_grid_figure, save_figure
from viz.weight_structure import (
    compute_weight_colored_hl_motif_counts,
    compute_weight_edge_signed_hl_motif_counts,
)

EPS = 0.5
TARGET_WE = 0.03
_SESSION_GAP_S = 3600.0


def _dominant_session(snaps: list[Path], *, gap_s: float = _SESSION_GAP_S) -> list[Path]:
    """Keep the largest mtime-contiguous snap group (drops stale overwrite sessions)."""
    if not snaps:
        return []
    ts = sorted((f.stat().st_mtime, f) for f in snaps)
    groups: list[list[tuple[float, Path]]] = [[ts[0]]]
    for t in ts[1:]:
        if t[0] - groups[-1][-1][0] > gap_s:
            groups.append([t])
        else:
            groups[-1].append(t)
    kept = [f for _, f in max(groups, key=len)]
    return sorted(kept, key=lambda p: int(p.stem.split("_")[1]))


def _snap_iteration(snap_path: Path) -> int:
    return int(snap_path.stem.split("_")[1])


def _dale_sign_from_ckpt(ckpt: Path) -> np.ndarray:
    """Dale signs from checkpoint, or column-sign fallback (pseudo-E/I)."""
    d = np.load(ckpt, allow_pickle=True)
    if "dale_sign" in d.files:
        sign = np.asarray(d["dale_sign"], dtype=float).ravel()
        if sign.size > 0:
            return sign
    W = np.asarray(d["weights_hidden_to_hidden"], dtype=float)
    sign = np.ones(W.shape[1], dtype=float)
    for j in range(W.shape[1]):
        col = np.delete(W[:, j], j)
        nz = col[col != 0]
        if nz.size:
            sign[j] = float(np.sign(nz[np.argmax(np.abs(nz))]))
    return sign


def _best_word_err(ckpt: Path) -> float:
    d = np.load(ckpt, allow_pickle=True)
    if "best_metric_word_error_frac" in d.files:
        return float(np.asarray(d["best_metric_word_error_frac"]).reshape(-1)[0])
    return float("nan")


def _snap_census(snap_path: Path, *, coloring: str, dale_sign: np.ndarray | None) -> dict:
    d = np.load(snap_path, allow_pickle=True)
    W = d["weights_hidden_to_hidden"]
    if coloring == "edge_sign":
        out = compute_weight_edge_signed_hl_motif_counts(W, mode="mean")
    elif coloring == "dale_node":
        if dale_sign is None:
            raise ValueError("dale_node coloring requires dale_sign")
        out = compute_weight_colored_hl_motif_counts(W, dale_sign, mode="mean")
    else:
        raise ValueError(f"unknown coloring={coloring!r}")
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
    coloring: str = "dale_node",
) -> Path:
    import vocab_mixed_dfa as vocab
    from experiment import checkpoint_path
    from rnn.learning_snaps import list_learning_snaps

    manifest_path = cache_path.parent.parent / "data" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dfa_by_run = {int(r["run_id"]): int(r["n_dfa_states"]) for r in manifest["runs"]}

    fold_rows: list[dict] = []
    run_series: list[dict] = []
    n_solved = 0
    for entry in vocab.iter_runs():
        run_id = int(entry["run_id"])
        n_dfa = int(dfa_by_run[run_id])
        ckpt = checkpoint_path(entry["task"], model, seed=seed)
        if not ckpt.is_file():
            print(f"skip r{run_id:02d}: missing checkpoint", flush=True)
            continue
        best_we = _best_word_err(ckpt)
        solved = bool(np.isfinite(best_we) and best_we <= TARGET_WE)
        if solved:
            n_solved += 1
        snaps = [
            p for p in _dominant_session(list_learning_snaps(ckpt))
            if _snap_iteration(p) > 0
        ]
        if len(snaps) < 2:
            print(f"skip r{run_id:02d}: need >=2 post-init learning snaps", flush=True)
            continue
        dale_sign = _dale_sign_from_ckpt(ckpt) if coloring == "dale_node" else None
        series = [
            _snap_census(p, coloring=coloring, dale_sign=dale_sign)
            for p in _subsample_snaps(snaps, max_snaps_per_run)
        ]
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
        run_series.append({
            "run_id": run_id,
            "n_dfa_states": n_dfa,
            "best_we": best_we,
            "solved": solved,
            "progress": prog_rows,
        })
        tag = "solved" if solved else f"WE={100.0 * best_we:.1f}%"
        print(
            f"r{run_id:02d} DFA={n_dfa:3d}  {tag}  "
            f"pts={sum(1 for r in fold_rows if r['run_id']==run_id)}",
            flush=True,
        )

    payload = {
        "comparison": vocab.COMPARISON_NAME,
        "model": model,
        "seed": seed,
        "coloring": coloring,
        "min_start": min_start,
        "target_we": TARGET_WE,
        "skip_iter0": True,
        "n_runs": len(run_series),
        "n_solved": n_solved,
        "n_fold_points": len(fold_rows),
        "fold_rows": fold_rows,
        "run_series": run_series,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"wrote {cache_path}  runs={len(run_series)}  solved={n_solved}  "
        f"points={len(fold_rows)}  coloring={coloring}",
        flush=True,
    )
    return cache_path


def _run_sort_key(run: dict) -> tuple[int, int]:
    return (int(run["n_dfa_states"]), int(run["run_id"]))


def _plot_beta_vs_dfa(
    per_run: list[dict],
    *,
    ax: plt.Axes,
    dfa_cmap,
    dfa_norm,
    target_we: float,
) -> None:
    xs, ys, cs, solved_flags = [], [], [], []
    for row in per_run:
        beta = row.get("beta")
        if beta is None or not np.isfinite(beta):
            continue
        xs.append(float(row["n_dfa_states"]))
        ys.append(float(beta))
        cs.append(dfa_cmap(dfa_norm(float(row["n_dfa_states"]))))
        solved_flags.append(bool(row.get("solved", False)))

    if not xs:
        ax.text(0.5, 0.5, "no per-run slopes", ha="center", va="center", transform=ax.transAxes)
        return

    x_arr = np.array(xs, dtype=float)
    y_arr = np.array(ys, dtype=float)
    for x, y, c, solved in zip(xs, ys, cs, solved_flags):
        if solved:
            ax.scatter(x, y, s=28, c=[c], alpha=0.9, edgecolors="0.15", linewidths=0.4, zorder=3)
        else:
            ax.scatter(
                x, y, s=34, facecolors="none", edgecolors=c, linewidths=1.2, alpha=0.95, zorder=3,
            )

    if len(x_arr) >= 3 and np.std(x_arr) > 1e-12:
        X = np.column_stack([np.ones(len(x_arr)), x_arr])
        b, *_ = np.linalg.lstsq(X, y_arr, rcond=None)
        x_line = np.linspace(x_arr.min(), x_arr.max(), 100)
        ax.plot(x_line, b[0] + b[1] * x_line, color="#c0392b", lw=1.2, zorder=2)
        ss_res = float(np.sum((y_arr - (b[0] + b[1] * x_arr)) ** 2))
        ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
        meta_r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
        ax.text(
            0.03, 0.97,
            f"meta: $\\beta$ = {b[0]:+.2f} + {b[1]:+.3f}·DFA  ($R^2$={meta_r2:.2f})",
            transform=ax.transAxes, fontsize=7, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.92),
        )

    ax.axhline(0.0, color="0.55", lw=0.7, ls="--", zorder=1)
    ax.set_xlabel("DFA states", fontsize=8)
    ax.set_ylabel(r"$\beta$ (log fold ~ log start)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.97, 0.03,
        f"filled: best WE $\\leq$ {100 * target_we:.0f}%\nopen: unsolved",
        transform=ax.transAxes, fontsize=6.5, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.92),
    )


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
    ncol: int = 6,
    beta_out_path: Path | None = None,
    coloring: str = "dale_node",
) -> Path:
    if rebuild_cache or not cache_path.is_file():
        collect_all_runs_motif_cache(
            cache_path,
            min_start=min_start,
            max_snaps_per_run=max_snaps_per_run,
            model=model,
            seed=seed,
            coloring=coloring,
        )
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    rows = payload["fold_rows"]
    series = sorted(payload["run_series"], key=_run_sort_key)
    target_we = float(payload.get("target_we", TARGET_WE))
    n_solved = int(payload.get("n_solved", sum(1 for r in series if r.get("solved"))))
    coloring_used = str(payload.get("coloring", coloring))
    color_tag = "edge-sign" if coloring_used == "edge_sign" else "Dale-node"

    by_run: dict[int, list[dict]] = {}
    for r in rows:
        by_run.setdefault(int(r["run_id"]), []).append(r)

    dfa_cmap = plt.get_cmap("viridis")
    dfa_vals = [float(r["n_dfa_states"]) for r in series]
    dfa_norm = plt.Normalize(vmin=min(dfa_vals), vmax=max(dfa_vals))

    all_log_c0 = np.array([r["log_c0"] for r in rows], dtype=float)
    all_log_fold = np.array([r["log_fold"] for r in rows], dtype=float)
    x_lo, x_hi = float(all_log_c0.min()), float(all_log_c0.max())
    y_lo, y_hi = float(all_log_fold.min()), float(all_log_fold.max())
    x_pad = 0.04 * max(x_hi - x_lo, 0.1)
    y_pad = 0.06 * max(y_hi - y_lo, 0.1)
    x_lim = (x_lo - x_pad, x_hi + x_pad)
    y_lim = (y_lo - y_pad, y_hi + y_pad)

    nrow = int(np.ceil(len(series) / ncol))
    fig_w = 2.05 * ncol + 0.55
    fig_h = 1.85 * nrow + 2.35
    fig = plt.figure(figsize=(fig_w, fig_h))
    height_ratios = [1.0] * nrow + [0.55]
    gs = fig.add_gridspec(nrow + 1, ncol, height_ratios=height_ratios, hspace=0.42, wspace=0.28)

    per_run: list[dict] = []
    for idx, run in enumerate(series):
        rid = int(run["run_id"])
        n_dfa = int(run["n_dfa_states"])
        solved = bool(run.get("solved", False))
        ax = fig.add_subplot(gs[idx // ncol, idx % ncol])
        pts = by_run.get(rid, [])
        beta_val = float("nan")
        r2 = float("nan")
        if not pts:
            ax.axis("off")
            per_run.append({
                "run_id": rid, "n_dfa_states": n_dfa, "solved": solved,
                "beta": beta_val, "r2": r2, "n": 0,
            })
            continue
        x = np.array([p["log_c0"] for p in pts], dtype=float)
        y = np.array([p["log_fold"] for p in pts], dtype=float)
        c = [dfa_cmap(dfa_norm(n_dfa))] * len(x)
        ax.scatter(x, y, s=10, c=c, alpha=0.75, edgecolors="none", zorder=3)
        if len(x) >= 3 and np.std(x) > 1e-12:
            beta, _, r2 = ols_fn(y, x)
            beta_val = float(beta[1])
            x_line = np.linspace(x_lim[0], x_lim[1], 50)
            ax.plot(x_line, beta[0] + beta[1] * x_line, color="#c0392b", lw=1.0, zorder=2)
        per_run.append({
            "run_id": rid, "n_dfa_states": n_dfa, "solved": solved,
            "beta": beta_val, "r2": r2, "n": len(x),
        })
        ax.axhline(0.0, color="0.55", lw=0.6, ls="--", zorder=1)
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        solve_tag = "" if solved else "  (unsolved)"
        ax.set_title(
            f"r{rid:02d}  DFA={n_dfa}  $\\beta$={beta_val:+.2f}  R$^2$={r2:.2f}{solve_tag}",
            fontsize=7.0, pad=2,
        )
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx // ncol == nrow - 1:
            ax.set_xlabel("log start", fontsize=6.5)
        else:
            ax.tick_params(labelbottom=False)
        if idx % ncol == 0:
            ax.set_ylabel("log fold", fontsize=6.5)
        else:
            ax.tick_params(labelleft=False)

    for j in range(len(series), nrow * ncol):
        fig.add_subplot(gs[j // ncol, j % ncol]).axis("off")

    ax_beta = fig.add_subplot(gs[nrow, :])
    _plot_beta_vs_dfa(
        per_run, ax=ax_beta, dfa_cmap=dfa_cmap, dfa_norm=dfa_norm, target_we=target_we,
    )

    sm = plt.cm.ScalarMappable(cmap=dfa_cmap, norm=dfa_norm)
    cax = fig.add_axes([0.935, 0.30, 0.012, 0.52])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("DFA states", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    r2_vals = [r["r2"] for r in per_run if np.isfinite(r["r2"])]
    med_r2 = float(np.median(r2_vals)) if r2_vals else float("nan")
    finalize_grid_figure(
        fig,
        suptitle=(
            f"mixed DFA {model} ({color_tag}): start vs fold "
            f"(DFA order, post-init, start>={min_start}, "
            f"n={len(series)} runs, {n_solved} solved, median R$^2$={med_r2:.2f})"
        ),
        top=0.93,
        bottom=0.05,
        left=0.05,
        right=0.92,
        hspace=0.42,
        wspace=0.28,
    )
    save_figure(fig, out_path, dpi=150)
    print(f"wrote {out_path}")

    beta_path = beta_out_path or out_path.with_name(
        out_path.stem.replace("_counts_raw_over_learning", "_fold_beta_vs_dfa") + out_path.suffix
    )
    fig_b, ax_b = plt.subplots(figsize=(5.2, 3.6))
    _plot_beta_vs_dfa(
        per_run, ax=ax_b, dfa_cmap=dfa_cmap, dfa_norm=dfa_norm, target_we=target_we,
    )
    sm_b = plt.cm.ScalarMappable(cmap=dfa_cmap, norm=dfa_norm)
    cbar_b = fig_b.colorbar(sm_b, ax=ax_b, pad=0.02)
    cbar_b.set_label("DFA states", fontsize=8)
    cbar_b.ax.tick_params(labelsize=7)
    finalize_grid_figure(
        fig_b,
        suptitle=(
            f"{model} motif compression slope vs DFA ({color_tag}; "
            f"n={len(series)}, {n_solved} solved, start>={min_start})"
        ),
        top=0.88,
        bottom=0.14,
        left=0.12,
        right=0.88,
    )
    save_figure(fig_b, beta_path, dpi=150)
    print(f"wrote {beta_path}")

    for row in sorted(per_run, key=lambda r: (r["n_dfa_states"], r["run_id"])):
        rid, n_dfa = row["run_id"], row["n_dfa_states"]
        beta_val, r2, n = row["beta"], row["r2"], row["n"]
        tag = "ok" if row["solved"] else "unsolved"
        print(f"  r{rid:02d} DFA={n_dfa:3d}  n={n:3d}  beta={beta_val:+.3f}  R2={r2:.3f}  {tag}")
    print(f"median per-run R2={med_r2:.3f}  runs={len(series)}  points={len(rows)}")
    return out_path
