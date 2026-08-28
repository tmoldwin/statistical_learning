"""All-runs mixed DFA motif fold board (used by scripts/r43_motif_figs.py)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from viz.plot_layout import finalize_grid_figure, save_figure
from viz.weight_structure import (
    compute_weight_colored_hl_motif_counts,
    compute_weight_edge_signed_hl_motif_counts,
    parse_edge_signed_hl_motif,
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
    motif_prefix: str = "T|",
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
            if not key.startswith(motif_prefix) or float(v0) < min_start:
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
            keys = [k for k in row["cnt"] if k.startswith(motif_prefix)]
            counts = np.array([float(row["cnt"].get(k, 0)) for k in keys], dtype=float)
            slog = np.log(np.maximum(counts, EPS))
            prog_rows.append({
                "progress": float((float(row["it"]) - it0) / denom),
                "edges": float(row["edges"]),
                "triples_conn": float(row.get("triples_conn", float("nan"))),
                "dyads_conn": float(row.get("dyads_conn", float("nan"))),
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
        "motif_prefix": motif_prefix,
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
        f"points={len(fold_rows)}  coloring={coloring}  prefix={motif_prefix}",
        flush=True,
    )
    return cache_path



def _n_edges_from_key(key: str) -> int:
    """Number of directed edges encoded in an edge-sign / Dale motif key."""
    parts = str(key).split("|")
    if len(parts) < 2:
        return 0
    return len([e for e in parts[1].split(",") if e])


def _n_inhibitory_from_key(key: str) -> int:
    """Count inhibitory (``-``) edges in an edge-sign HL motif key."""
    parts = str(key).split("|")
    if len(parts) < 3:
        return 0
    return parts[2].count("-")


def _inh_fraction_from_key(key: str) -> float:
    ne = _n_edges_from_key(key)
    if ne <= 0:
        return float("nan")
    return float(_n_inhibitory_from_key(key)) / float(ne)


def _edge_pattern_counts_from_key(key: str) -> dict[str, int]:
    """Count mutual / unidirectional excitatory and inhibitory edges in one motif key."""
    edges = parse_edge_signed_hl_motif(key)
    empty = {
        "n_mut_exc": 0,
        "n_mut_inh": 0,
        "n_mut_mixed": 0,
        "n_uni_exc": 0,
        "n_uni_inh": 0,
    }
    if edges is None:
        return empty

    by_dir: dict[tuple[int, int], int] = {}
    for i, j, s in edges:
        by_dir[(i, j)] = int(s)

    seen: set[tuple[int, int]] = set()
    out = dict(empty)
    for (i, j), s in by_dir.items():
        if (i, j) in seen:
            continue
        rev = (j, i)
        if rev in by_dir:
            s2 = by_dir[rev]
            if s == 1 and s2 == 1:
                out["n_mut_exc"] += 1
            elif s == -1 and s2 == -1:
                out["n_mut_inh"] += 1
            else:
                out["n_mut_mixed"] += 1
            seen.add((i, j))
            seen.add(rev)
        else:
            if s == 1:
                out["n_uni_exc"] += 1
            else:
                out["n_uni_inh"] += 1
            seen.add((i, j))
    return out


def _hypothesis_features_from_key(key: str) -> dict[str, float] | None:
    """Structural hypothesis features for one edge-sign motif key.

    Covers density, sign counts, reciprocity, cycles, feedback-loop sign,
    disinhibition chains, and degree concentration.
    """
    edges = parse_edge_signed_hl_motif(key)
    if edges is None:
        return None

    by_dir: dict[tuple[int, int], int] = {(i, j): int(s) for i, j, s in edges}
    nodes = sorted({n for i, j, _ in edges for n in (i, j)})
    ne = len(by_dir)
    n_inh = sum(1 for s in by_dir.values() if s == -1)
    n_exc = ne - n_inh

    pat = _edge_pattern_counts_from_key(key)
    n_recip = pat["n_mut_exc"] + pat["n_mut_inh"] + pat["n_mut_mixed"]
    n_mut_same = pat["n_mut_exc"] + pat["n_mut_inh"]

    # Directed 3-cycles and their sign products.
    n_3cycles = 0
    n_neg_3cycles = 0
    if len(nodes) == 3:
        a, b, c = nodes
        for i, j, k in ((a, b, c), (a, c, b)):
            s1 = by_dir.get((i, j))
            s2 = by_dir.get((j, k))
            s3 = by_dir.get((k, i))
            if s1 is not None and s2 is not None and s3 is not None:
                n_3cycles += 1
                if s1 * s2 * s3 < 0:
                    n_neg_3cycles += 1

    # Negative feedback loops: mixed mutual pairs (2-loops with product < 0)
    # plus 3-cycles with negative sign product.
    n_neg_loops = pat["n_mut_mixed"] + n_neg_3cycles

    # Disinhibition chains: i -> j -> k, both inhibitory, i != k.
    n_disinh = 0
    for (i, j), s1 in by_dir.items():
        if s1 != -1:
            continue
        for (j2, k), s2 in by_dir.items():
            if j2 == j and k != i and s2 == -1:
                n_disinh += 1

    out_deg: dict[int, int] = {}
    in_deg: dict[int, int] = {}
    for i, j in by_dir:
        out_deg[i] = out_deg.get(i, 0) + 1
        in_deg[j] = in_deg.get(j, 0) + 1

    return {
        "n_edges": float(ne),
        "n_inh": float(n_inh),
        "n_exc": float(n_exc),
        "net_exc": float(n_exc - n_inh),
        "n_recip": float(n_recip),
        "n_mut_mixed": float(pat["n_mut_mixed"]),
        "n_mut_same": float(n_mut_same),
        "n_3cycles": float(n_3cycles),
        "is_acyclic": float(1.0 if (n_recip == 0 and n_3cycles == 0) else 0.0),
        "n_neg_loops": float(n_neg_loops),
        "n_disinh": float(n_disinh),
        "max_out_deg": float(max(out_deg.values()) if out_deg else 0),
        "max_in_deg": float(max(in_deg.values()) if in_deg else 0),
    }


_HYPOTHESIS_SPECS: list[tuple[str, str]] = [
    ("n_edges", "H1: # directed edges (density)"),
    ("n_inh", "H2: # inhibitory edges"),
    ("n_exc", "H3: # excitatory edges"),
    ("net_exc", "H4: net excitation (exc - inh)"),
    ("n_recip", "H5: # reciprocal pairs"),
    ("n_mut_mixed", "H6: # mixed mutual pairs (neg 2-loops)"),
    ("n_mut_same", "H7: # same-sign mutual pairs"),
    ("n_3cycles", "H8: # directed 3-cycles"),
    ("is_acyclic", "H9: feedforward (no loops)"),
    ("n_neg_loops", "H10: # negative feedback loops"),
    ("n_disinh", "H11: # disinhibition chains (inh->inh)"),
    ("max_out_deg", "H12: max out-degree (broadcast)"),
    ("max_in_deg", "H13: max in-degree (convergence)"),
    ("log_start", "H14: log initial count"),
]


_EDGE_COUNT_COLORS: dict[int, str] = {
    1: "#72B7B2",
    2: "#4C78A8",
    3: "#F58518",
    4: "#54A24B",
    5: "#E45756",
    6: "#B279A2",
}

_INH_COUNT_COLORS: dict[int, str] = {
    0: "#4C78A8",
    1: "#72B7B2",
    2: "#F58518",
    3: "#E45756",
    4: "#B279A2",
    5: "#9D174D",
    6: "#6B0F1A",
}


def _run_sort_key(run: dict) -> tuple[int, int]:
    return (int(run["n_dfa_states"]), int(run["run_id"]))


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(y)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    return beta, pred, r2


def _outlined_regression_line(
    ax,
    x_line: np.ndarray,
    y_line: np.ndarray,
    color: str,
    *,
    lw: float = 1.5,
) -> None:
    ax.plot(x_line, y_line, color="white", lw=lw + 2.0, zorder=4, solid_capstyle="round")
    ax.plot(x_line, y_line, color="0.05", lw=lw + 0.5, zorder=4.5, solid_capstyle="round", alpha=0.35)
    ax.plot(x_line, y_line, color=color, lw=lw, zorder=5, solid_capstyle="round", alpha=0.98)


def _start_quintile_summary(
    log_c0: np.ndarray,
    log_fold: np.ndarray,
    *,
    n_bins: int = 5,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Bin motifs by log start; return labels and median / q25 / q75 log fold per bin."""
    order = np.argsort(log_c0)
    x_sorted = log_c0[order]
    y_sorted = log_fold[order]
    n = len(x_sorted)
    if n < n_bins:
        n_bins = max(1, n)
    splits = np.array_split(np.arange(n), n_bins)
    labels: list[str] = []
    medians, q25, q75 = [], [], []
    for i, idx in enumerate(splits):
        if len(idx) == 0:
            continue
        ys = y_sorted[idx]
        xs = x_sorted[idx]
        lo, hi = float(xs.min()), float(xs.max())
        labels.append(f"Q{i + 1}\n{lo:.1f}–{hi:.1f}")
        medians.append(float(np.median(ys)))
        q25.append(float(np.percentile(ys, 25)))
        q75.append(float(np.percentile(ys, 75)))
    return labels, np.array(medians), np.array(q25), np.array(q75)


def plot_homogenization_summary(
    log_c0: np.ndarray,
    log_fold: np.ndarray,
    n_edges: np.ndarray,
    out_path: Path,
    *,
    title: str,
    min_fit: int = 8,
    n_quintiles: int = 5,
    beta_vs_dfa: list[dict] | None = None,
    target_we: float = TARGET_WE,
) -> Path:
    """Compact homogenization summary: scatter, start-quintile compression, tier β or β vs DFA."""
    log_c0 = np.asarray(log_c0, dtype=float)
    log_fold = np.asarray(log_fold, dtype=float)
    n_edges = np.asarray(n_edges, dtype=int)
    tiers = sorted(set(int(v) for v in n_edges))
    rng = np.random.default_rng(0)
    x_disp = log_c0 + rng.normal(0.0, 0.025, size=len(log_c0))

    beta, _, r2 = _ols(log_fold, log_c0)
    beta_tiers: list[tuple[int, float, float, int]] = []
    for ne in tiers:
        mask = n_edges == ne
        if int(mask.sum()) < min_fit or float(np.std(log_c0[mask])) < 1e-9:
            continue
        b_e, _, r2_e = _ols(log_fold[mask], log_c0[mask])
        beta_tiers.append((ne, float(b_e[1]), r2_e, int(mask.sum())))

    q_labels, q_med, q25, q75 = _start_quintile_summary(log_c0, log_fold, n_bins=n_quintiles)
    xs = np.arange(len(q_med))

    third_ratio = 1.05 if beta_vs_dfa is not None else 0.95
    fig = plt.figure(figsize=(11.2, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 1.0, third_ratio], wspace=0.34)
    ax_sc = fig.add_subplot(gs[0, 0])
    ax_q = fig.add_subplot(gs[0, 1])
    ax_third = fig.add_subplot(gs[0, 2])

    for ne in tiers:
        mask = n_edges == ne
        col = _EDGE_COUNT_COLORS.get(ne, "#888888")
        alpha = 0.35 if beta_vs_dfa is not None else 0.55
        size = 10 if beta_vs_dfa is not None else 14
        ax_sc.scatter(
            x_disp[mask], log_fold[mask], s=size, c=col, alpha=alpha,
            edgecolors="none", zorder=3, label=f"{ne}e",
        )
        if beta_vs_dfa is None and int(mask.sum()) >= min_fit and float(np.std(log_c0[mask])) > 1e-9:
            b_e, _, _ = _ols(log_fold[mask], log_c0[mask])
            x0, x1 = float(log_c0[mask].min()), float(log_c0[mask].max())
            x_line = np.linspace(x0, x1, 40)
            _outlined_regression_line(ax_sc, x_line, b_e[0] + b_e[1] * x_line, col, lw=1.25)

    if len(log_c0) >= 3 and float(np.std(log_c0)) > 1e-12:
        x_line = np.linspace(float(log_c0.min()), float(log_c0.max()), 60)
        _outlined_regression_line(ax_sc, x_line, beta[0] + beta[1] * x_line, "0.12", lw=2.0)

    ax_sc.axhline(0.0, color="0.55", lw=0.7, ls="--", zorder=1)
    ax_sc.set_xlabel("log start count", fontsize=9)
    ax_sc.set_ylabel("log fold (end / start)", fontsize=9)
    ax_sc.set_title(
        rf"pooled $\beta={beta[1]:+.2f}$, $R^2={r2:.2f}$, n={len(log_c0)}",
        fontsize=9, pad=4,
    )
    ax_sc.tick_params(labelsize=8)
    ax_sc.grid(True, alpha=0.22)
    ax_sc.spines["top"].set_visible(False)
    ax_sc.spines["right"].set_visible(False)
    if beta_vs_dfa is None:
        ax_sc.legend(loc="lower left", fontsize=7, frameon=True, fancybox=False, edgecolor="0.8")

    ax_q.fill_between(xs, q25, q75, color="#4C78A8", alpha=0.22, zorder=1)
    ax_q.plot(xs, q_med, color="#4C78A8", lw=2.0, marker="o", ms=6, zorder=3)
    ax_q.axhline(0.0, color="0.55", lw=0.7, ls="--", zorder=1)
    ax_q.set_xticks(xs)
    ax_q.set_xticklabels(q_labels, fontsize=7.5)
    ax_q.set_xlabel("start quintile (log count range)", fontsize=9)
    ax_q.set_ylabel("median log fold", fontsize=9)
    ax_q.set_title("compression by initial abundance", fontsize=9, pad=4)
    ax_q.tick_params(labelsize=8)
    ax_q.grid(True, alpha=0.22)
    ax_q.spines["top"].set_visible(False)
    ax_q.spines["right"].set_visible(False)

    if beta_vs_dfa is not None:
        dfa_vals = [float(r["n_dfa_states"]) for r in beta_vs_dfa if np.isfinite(r.get("beta", float("nan")))]
        dfa_norm = plt.Normalize(vmin=min(dfa_vals), vmax=max(dfa_vals)) if dfa_vals else plt.Normalize(0, 1)
        _plot_beta_vs_dfa(
            beta_vs_dfa, ax=ax_third, dfa_cmap=plt.get_cmap("viridis"),
            dfa_norm=dfa_norm, target_we=target_we,
        )
        ax_third.set_title("per-run compression slope vs DFA", fontsize=9, pad=4)
    elif beta_tiers:
        ne_vals = [t[0] for t in beta_tiers]
        betas = [t[1] for t in beta_tiers]
        cols = [_EDGE_COUNT_COLORS.get(ne, "#888888") for ne in ne_vals]
        ax_third.bar(
            np.arange(len(betas)), betas, color=cols, edgecolor="0.25", linewidth=0.4, zorder=2,
        )
        ax_third.axhline(0.0, color="0.35", lw=0.8, zorder=1)
        ax_third.set_xticks(np.arange(len(betas)))
        ax_third.set_xticklabels([f"{ne}e" for ne in ne_vals], fontsize=8)
        for i, (_ne, b, r2_e, _n) in enumerate(beta_tiers):
            ax_third.text(
                i, b + (0.02 if b >= 0 else -0.02), f"{b:+.2f}\n$R^2$={r2_e:.2f}",
                ha="center", va="bottom" if b >= 0 else "top", fontsize=6.5, color="0.2",
            )
        ax_third.set_ylabel(r"$\beta$ (log fold ~ log start)", fontsize=9)
        ax_third.set_title(r"$\beta$ by motif size (#edges)", fontsize=9, pad=4)
    else:
        ax_third.text(
            0.5, 0.5, "insufficient data\nfor tier fits", ha="center", va="center",
            transform=ax_third.transAxes, fontsize=8,
        )
    ax_third.tick_params(labelsize=8)
    ax_third.grid(True, axis="y", alpha=0.22)
    ax_third.spines["top"].set_visible(False)
    ax_third.spines["right"].set_visible(False)

    finalize_grid_figure(
        fig,
        suptitle=title,
        top=0.88,
        bottom=0.16,
        left=0.06,
        right=0.98,
        wspace=0.34,
    )
    save_figure(fig, out_path, dpi=150)
    print(f"wrote {out_path}")
    print(f"  pooled beta={beta[1]:+.3f}  R2={r2:.3f}  n={len(log_c0)}")
    for ne, b, r2_e, n in beta_tiers:
        print(f"  {ne}e  beta={b:+.3f}  R2={r2_e:.3f}  n={n}")
    return out_path


def _subsample_stratified_indices(
    n_edges: np.ndarray,
    *,
    max_pts: int = 3500,
    seed: int = 0,
) -> np.ndarray:
    if len(n_edges) <= max_pts:
        return np.arange(len(n_edges))
    rng = np.random.default_rng(seed)
    tiers = sorted(set(int(v) for v in n_edges))
    per_tier = max(1, max_pts // len(tiers))
    idx: list[int] = []
    for ne in tiers:
        mask = np.where(n_edges == ne)[0]
        if len(mask) <= per_tier:
            idx.extend(mask.tolist())
        else:
            idx.extend(rng.choice(mask, size=per_tier, replace=False).tolist())
    return np.array(idx, dtype=int)


def _jitter_display_coords(
    log_c0: np.ndarray,
    log_fold: np.ndarray,
    *,
    x_scale: float = 0.022,
    y_scale: float = 0.075,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Display-only jitter: x noise + vertical spread for stacked log-start bins."""
    rng = np.random.default_rng(seed)
    x_disp = log_c0 + rng.normal(0.0, x_scale, size=len(log_c0))
    y_disp = np.array(log_fold, dtype=float, copy=True)
    bins = np.round(log_c0, 2)
    for val in np.unique(bins):
        mask = bins == val
        n = int(mask.sum())
        if n <= 1:
            continue
        spread = y_scale * min(1.0, np.log1p(n) / np.log1p(24))
        y_disp[mask] += rng.uniform(-spread, spread, size=n)
    return x_disp, y_disp


def _trajectory_monotonic_score(counts: np.ndarray, *, want_down: bool) -> float:
    if len(counts) < 4:
        return float("-inf")
    c0 = max(float(counts[0]), EPS)
    c1 = max(float(counts[-1]), EPS)
    fold = c1 / c0
    if want_down and fold > 0.82:
        return float("-inf")
    if not want_down and fold < 1.12:
        return float("-inf")
    diffs = np.diff(counts)
    step_frac = float(np.mean(diffs <= 0)) if want_down else float(np.mean(diffs >= 0))
    log_span = abs(float(np.log(c1 / c0)))
    log_c = np.log(np.maximum(counts, EPS))
    smooth = 1.0 / (1.0 + float(np.std(np.diff(log_c))))
    return step_frac * log_span * smooth


def _top_runs_for_key(rows: list[dict], key: str, *, n: int = 10) -> list[int]:
    pts = sorted(
        [r for r in rows if r["key"] == key],
        key=lambda p: float(p["log_c0"]),
        reverse=True,
    )
    seen: set[int] = set()
    out: list[int] = []
    for p in pts:
        rid = int(p["run_id"])
        if rid in seen:
            continue
        seen.add(rid)
        out.append(rid)
        if len(out) >= n:
            break
    return out


def _pick_demo_trajectories(
    rows: list[dict],
    *,
    motif_prefix: str,
    model: str,
    seed: int,
    coloring: str,
    n_demos: int = 4,
    min_runs: int = 8,
) -> list[dict]:
    """Pick demo motifs with the clearest monotonic learning curves."""
    from collections import defaultdict

    by_key: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        key = str(r["key"])
        if not key.startswith(motif_prefix):
            continue
        by_key[key].append(r)

    stats: list[dict] = []
    for key, pts in by_key.items():
        if len(pts) < min_runs:
            continue
        log_fold = np.array([p["log_fold"] for p in pts], dtype=float)
        stats.append({
            "key": key,
            "n_edges": _n_edges_from_key(key),
            "n_runs": len(pts),
            "med_log_fold": float(np.median(log_fold)),
        })
    if not stats:
        return []

    is_dyad = motif_prefix.startswith("D")
    slots: list[tuple[int, bool, str]] = []
    if is_dyad:
        slots = [(2, True, "2-edge down"), (2, False, "2-edge up"), (1, True, "1-edge down"), (1, False, "1-edge up")]
    else:
        slots = [
            (5, True, "5-edge down"), (4, True, "4-edge down"),
            (3, True, "3-edge down"), (2, False, "2-edge up"),
        ]

    chosen: list[dict] = []
    used: set[str] = set()

    for ne_target, want_down, label in slots[:n_demos]:
        pool = sorted(
            [s for s in stats if s["n_edges"] == ne_target and s["key"] not in used],
            key=lambda s: s["med_log_fold"],
            reverse=not want_down,
        )
        best: dict | None = None
        best_score = float("-inf")
        for cand in pool[:12]:
            for run_id in _top_runs_for_key(rows, cand["key"], n=8):
                iters, counts = _load_motif_trajectory(
                    run_id, cand["key"],
                    model=model, seed=seed, coloring=coloring, max_snaps=60,
                )
                score = _trajectory_monotonic_score(counts, want_down=want_down)
                if score > best_score:
                    best_score = score
                    best = {
                        **cand,
                        "run_id": run_id,
                        "iters": iters,
                        "counts": counts,
                        "label": label,
                        "want_down": want_down,
                        "score": score,
                    }
        if best is not None:
            used.add(best["key"])
            chosen.append(best)
    return chosen


def _tier_log_counts(
    cnt: dict,
    *,
    motif_prefix: str,
    min_start: int,
    n_edges: int,
) -> np.ndarray:
    keys = [
        k for k in cnt
        if str(k).startswith(motif_prefix)
        and _n_edges_from_key(k) == n_edges
        and float(cnt.get(k, 0.0)) >= min_start
    ]
    if not keys:
        return np.array([], dtype=float)
    return np.log10([max(float(cnt.get(k, 0.0)), EPS) for k in keys])


def _tier_counts(
    cnt: dict,
    *,
    motif_prefix: str,
    min_start: int,
    n_edges: int,
) -> np.ndarray:
    keys = [
        k for k in cnt
        if str(k).startswith(motif_prefix)
        and _n_edges_from_key(k) == n_edges
        and float(cnt.get(k, 0.0)) >= min_start
    ]
    if not keys:
        return np.array([], dtype=float)
    return np.array([max(float(cnt.get(k, 0.0)), EPS) for k in keys], dtype=float)


def _pooled_tier_counts(
    rows: list[dict],
    *,
    n_edges: int,
    which: str,
) -> np.ndarray:
    """Raw motif counts pooled across runs (start or end snapshot per run×class)."""
    vals: list[float] = []
    for r in rows:
        if _n_edges_from_key(r["key"]) != n_edges:
            continue
        log_count = float(r["log_c0"]) if which == "start" else float(r["log_c0"] + r["log_fold"])
        vals.append(max(float(np.exp(log_count) - EPS), EPS))
    return np.array(vals, dtype=float)


def _pooled_tier_log10_counts(
    rows: list[dict],
    *,
    n_edges: int,
    which: str,
) -> np.ndarray:
    """log10 motif counts pooled across runs (start or end snapshot per run×class)."""
    ln10 = np.log(10.0)
    vals: list[float] = []
    for r in rows:
        if _n_edges_from_key(r["key"]) != n_edges:
            continue
        log_count = float(r["log_c0"]) if which == "start" else float(r["log_c0"] + r["log_fold"])
        vals.append(log_count / ln10)
    return np.array(vals, dtype=float)


def _pooled_tier_log10_fold(rows: list[dict], *, n_edges: int) -> np.ndarray:
    """log10(end/start) pooled across runs for one #edges tier."""
    ln10 = np.log(10.0)
    return np.array([
        float(r["log_fold"]) / ln10
        for r in rows
        if _n_edges_from_key(r["key"]) == n_edges
    ], dtype=float)


def _draw_tier_fold_kde(
    ax,
    rows: list[dict],
    *,
    n_edges: int,
    n_runs: int | None = None,
) -> None:
    """KDE of log10 fold (end/start) — shows compression directly (peak < 0 = shrink)."""
    from scipy.stats import gaussian_kde

    vals = _pooled_tier_log10_fold(rows, n_edges=n_edges)
    col = _EDGE_COUNT_COLORS.get(n_edges, "#888888")
    run_tag = f"{n_runs} runs, " if n_runs is not None else ""
    ax.set_title(f"{n_edges}-edge fold", fontsize=7.4, pad=3.5, color=col)

    if len(vals) < 2:
        ax.text(0.5, 0.5, "insufficient\npoints", ha="center", va="center",
                transform=ax.transAxes, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    med = float(np.median(vals))
    frac_shrink = float(np.mean(vals < 0.0))
    x_lo = float(vals.min()) - 0.08
    x_hi = float(vals.max()) + 0.08
    xs = np.linspace(x_lo, x_hi, 200)
    kde = gaussian_kde(vals, bw_method=lambda k: max(float(k.scotts_factor()), 0.06))
    ax.plot(xs, kde(xs), color=col, lw=1.5, alpha=0.95,
            label=f"{run_tag}n={len(vals)}")
    ax.axvline(0.0, color="0.45", lw=0.8, ls="--", zorder=1)
    ax.axvline(med, color=col, lw=1.1, ls=":", zorder=2)
    ax.text(
        0.97, 0.97,
        rf"med={med:+.2f}  shrink={frac_shrink:.0%}",
        transform=ax.transAxes, ha="right", va="top", fontsize=6.0, color="0.25",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.6),
    )
    ax.set_xlabel(r"$\log_{10}$(end / start)", fontsize=7.0)
    if n_edges == 2:
        ax.set_ylabel("density", fontsize=7.0)
    ax.tick_params(labelsize=6.0)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=5.5, frameon=True, fancybox=False, edgecolor="0.8", loc="upper left")


def _draw_tier_kde_before_after(
    ax,
    *,
    n_edges: int,
    fold_rows: list[dict] | None = None,
    c0: dict | None = None,
    c1: dict | None = None,
    motif_prefix: str = "T|",
    min_start: int = 20,
    it0: int | None = None,
    it1: int | None = None,
    n_runs: int | None = None,
) -> None:
    """Histogram of raw motif counts for one #edges tier: initial vs final overlaid.

    Linear count axis so equal tick spacing is equal count spacing; dispersion
    (spread across motif classes within a tier) is read directly in count units.
    """
    if fold_rows is not None:
        v0 = _pooled_tier_counts(fold_rows, n_edges=n_edges, which="start")
        v1 = _pooled_tier_counts(fold_rows, n_edges=n_edges, which="end")
        run_tag = f"{n_runs} runs, " if n_runs is not None else ""
        lab0 = f"initial ({run_tag}n={len(v0)})"
        lab1 = f"final ({run_tag}n={len(v1)})"
    else:
        if c0 is None or c1 is None:
            raise ValueError("need fold_rows or c0/c1")
        v0 = _tier_counts(c0, motif_prefix=motif_prefix, min_start=min_start, n_edges=n_edges)
        v1 = _tier_counts(c1, motif_prefix=motif_prefix, min_start=min_start, n_edges=n_edges)
        lab0 = f"initial (iter {it0}, n={len(v0)})"
        lab1 = f"final (iter {it1}, n={len(v1)})"

    col = _EDGE_COUNT_COLORS.get(n_edges, "#888888")
    ax.set_title(f"{n_edges}-edge count spread", fontsize=7.4, pad=3.5, color=col)

    if len(v0) < 2 and len(v1) < 2:
        ax.text(0.5, 0.5, "insufficient\nmotif classes", ha="center", va="center",
                transform=ax.transAxes, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    all_vals = np.concatenate([v for v in (v0, v1) if len(v)])
    span = max(float(all_vals.max() - all_vals.min()), 1.0)
    pad = max(0.04 * span, 1.0)
    x_lo = max(float(min_start), float(all_vals.min()) - pad)
    x_hi = float(all_vals.max()) + pad
    nbins = int(min(18, max(8, np.sqrt(len(all_vals)))))
    bins = np.linspace(x_lo, x_hi, nbins + 1)

    if len(v0) >= 2:
        ax.hist(
            v0, bins=bins, density=True, histtype="stepfilled",
            color=col, alpha=0.30, edgecolor=col, linewidth=0.8,
            label=lab0, zorder=2,
        )
    if len(v1) >= 2:
        ax.hist(
            v1, bins=bins, density=True, histtype="step",
            color=col, linewidth=1.5, linestyle="--",
            label=lab1, zorder=3,
        )

    if len(v0) >= 2 and len(v1) >= 2:
        sd0, sd1 = float(np.std(v0)), float(np.std(v1))
        ratio = sd1 / sd0 if sd0 > 1e-9 else float("nan")
        narrow_tag = "narrower" if ratio < 0.98 else ("wider" if ratio > 1.02 else "~same")
        ax.text(
            0.97, 0.97,
            rf"$\sigma$: {sd0:.0f}$\rightarrow${sd1:.0f} ({narrow_tag})",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.0, color="0.25",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.6),
        )

    ax.set_xlabel("count", fontsize=7.0)
    if n_edges == 2:
        ax.set_ylabel("density", fontsize=7.0)
    ax.tick_params(labelsize=6.0)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=5.5, frameon=True, fancybox=False, edgecolor="0.8", loc="upper left")


def _draw_motif_distribution_hist(
    ax,
    cnt: dict,
    *,
    motif_prefix: str,
    min_start: int,
    title: str,
) -> None:
    """Histogram of log10 motif-class counts, colored by #edges tier."""
    keys = [
        k for k in cnt
        if str(k).startswith(motif_prefix) and float(cnt.get(k, 0.0)) >= min_start
    ]
    tiers = sorted({_n_edges_from_key(k) for k in keys})
    bins = np.linspace(0.0, 4.0, 22)
    for ne in tiers:
        vals = np.log10([
            max(float(cnt.get(k, 0.0)), EPS) for k in keys if _n_edges_from_key(k) == ne
        ])
        col = _EDGE_COUNT_COLORS.get(ne, "#888888")
        ax.hist(vals, bins=bins, alpha=0.55, color=col, label=f"{ne}e", density=True)
    ax.set_xlabel(r"$\log_{10}$ count", fontsize=7.0)
    ax.set_ylabel("density", fontsize=7.0)
    ax.set_title(title, fontsize=7.4, pad=3.5)
    ax.tick_params(labelsize=6.0)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if tiers:
        ax.legend(fontsize=5.8, frameon=True, fancybox=False, edgecolor="0.8", loc="upper right")


def _pick_single_run_demos(
    snaps: list[dict],
    *,
    motif_prefix: str,
    min_start: int,
    n_demos: int = 4,
) -> list[dict]:
    """Pick clearest monotonic motif trajectories from one run's census JSON."""
    iters = np.array([float(s["it"]) for s in snaps], dtype=float)
    keys = [k for k in snaps[0]["cnt"] if str(k).startswith(motif_prefix)]
    stats: list[dict] = []
    for key in keys:
        counts = np.array([float(s["cnt"].get(key, 0.0)) for s in snaps], dtype=float)
        if counts[0] < min_start:
            continue
        stats.append({
            "key": key,
            "n_edges": _n_edges_from_key(key),
            "counts": counts,
            "iters": iters,
        })
    is_dyad = motif_prefix.startswith("D")
    if is_dyad:
        slots = [(2, True, "2-edge down"), (2, False, "2-edge up"), (1, True, "1-edge down"), (1, False, "1-edge up")]
    else:
        slots = [
            (5, True, "5-edge down"), (4, True, "4-edge down"),
            (3, True, "3-edge down"), (2, False, "2-edge up"),
        ]
    chosen: list[dict] = []
    used: set[str] = set()
    for ne_target, want_down, label in slots[:n_demos]:
        pool = [s for s in stats if s["n_edges"] == ne_target and s["key"] not in used]
        if not pool:
            continue
        best = max(
            pool,
            key=lambda s: _trajectory_monotonic_score(s["counts"], want_down=want_down),
        )
        score = _trajectory_monotonic_score(best["counts"], want_down=want_down)
        if not np.isfinite(score):
            continue
        used.add(best["key"])
        chosen.append({**best, "label": label, "want_down": want_down, "score": score})
    return chosen


def _fold_arrays_from_snaps(
    snaps: list[dict],
    *,
    motif_prefix: str,
    min_start: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c0, c1 = snaps[0]["cnt"], snaps[-1]["cnt"]
    keys = [k for k in c0 if str(k).startswith(motif_prefix)]
    rows: list[tuple[float, float, int]] = []
    for key in keys:
        v0 = float(c0.get(key, 0.0))
        if v0 < min_start:
            continue
        v1 = float(c1.get(key, 0.0))
        rows.append((
            float(np.log(v0 + EPS)),
            float(np.log((v1 + EPS) / (v0 + EPS))),
            int(_n_edges_from_key(key)),
        ))
    if not rows:
        raise ValueError(f"no motifs with start>={min_start}")
    return (
        np.array([r[0] for r in rows], dtype=float),
        np.array([r[1] for r in rows], dtype=float),
        np.array([r[2] for r in rows], dtype=int),
    )


def _load_all_runs_fold_context(
    cache_path: Path,
    *,
    motif_prefix: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], dict]:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    rows = [r for r in payload["fold_rows"] if str(r["key"]).startswith(motif_prefix)]
    series = sorted(payload["run_series"], key=_run_sort_key)
    by_run: dict[int, list[dict]] = {}
    for r in rows:
        by_run.setdefault(int(r["run_id"]), []).append(r)
    per_run: list[dict] = []
    for run in series:
        rid = int(run["run_id"])
        pts = by_run.get(rid, [])
        beta_val, r2 = float("nan"), float("nan")
        if len(pts) >= 3:
            x = np.array([p["log_c0"] for p in pts], dtype=float)
            y = np.array([p["log_fold"] for p in pts], dtype=float)
            if float(np.std(x)) > 1e-12:
                b, _, r2 = _ols(y, x)
                beta_val = float(b[1])
        per_run.append({
            "run_id": rid,
            "n_dfa_states": int(run["n_dfa_states"]),
            "solved": bool(run.get("solved", False)),
            "beta": beta_val,
            "r2": r2,
            "n": len(pts),
        })
    log_c0 = np.array([r["log_c0"] for r in rows], dtype=float)
    log_fold = np.array([r["log_fold"] for r in rows], dtype=float)
    n_edges = np.array([_n_edges_from_key(r["key"]) for r in rows], dtype=int)
    return log_c0, log_fold, n_edges, per_run, payload


def pick_highest_dfa_exemplar(cache_path: Path) -> tuple[int, int]:
    """Return (run_id, n_dfa_states) for the run with the largest grammar."""
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    best = max(payload["run_series"], key=lambda r: int(r["n_dfa_states"]))
    return int(best["run_id"]), int(best["n_dfa_states"])


def lookup_run_dfa(cache_path: Path, run_id: int) -> int:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    for run in payload["run_series"]:
        if int(run["run_id"]) == int(run_id):
            return int(run["n_dfa_states"])
    raise KeyError(f"run_id {run_id} not in {cache_path}")


def _draw_tier_fold_scatter(
    ax,
    log_c0: np.ndarray,
    log_fold: np.ndarray,
    n_edges: np.ndarray,
    *,
    min_fit: int,
    title: str,
    jitter_y: bool = True,
    subsample: int | None = None,
) -> tuple[float, float, list[tuple[int, float, float, int]]]:
    tiers = sorted(set(int(v) for v in n_edges))
    beta, _, r2 = _ols(log_fold, log_c0)
    beta_tiers: list[tuple[int, float, float, int]] = []
    idx = np.arange(len(log_c0))
    if subsample is not None and len(idx) > subsample:
        idx = _subsample_stratified_indices(n_edges, max_pts=subsample, seed=0)
    if jitter_y:
        x_disp, y_disp = _jitter_display_coords(log_c0, log_fold, seed=0)
    else:
        rng = np.random.default_rng(0)
        x_disp = log_c0 + rng.normal(0.0, 0.02, size=len(log_c0))
        y_disp = log_fold

    for ne in tiers:
        mask = n_edges == ne
        col = _EDGE_COUNT_COLORS.get(ne, "#888888")
        show = mask.copy()
        if subsample is not None:
            show = mask & np.isin(np.arange(len(mask)), idx)
        ax.scatter(
            x_disp[show], y_disp[show], s=14 if subsample is None else 10,
            c=col, alpha=0.55 if subsample is None else 0.40,
            edgecolors="0.15" if subsample else "none",
            linewidths=0.25 if subsample else 0.0,
            zorder=3,
        )
        if int(mask.sum()) >= min_fit and float(np.std(log_c0[mask])) > 1e-9:
            b_e, _, r2_e = _ols(log_fold[mask], log_c0[mask])
            beta_tiers.append((ne, float(b_e[1]), r2_e, int(mask.sum())))
            x0, x1 = float(log_c0[mask].min()), float(log_c0[mask].max())
            x_line = np.linspace(x0, x1, 50)
            _outlined_regression_line(ax, x_line, b_e[0] + b_e[1] * x_line, col, lw=1.35)

    if len(log_c0) >= 3 and float(np.std(log_c0)) > 1e-12:
        x_line = np.linspace(float(log_c0.min()), float(log_c0.max()), 60)
        _outlined_regression_line(ax, x_line, beta[0] + beta[1] * x_line, "0.12", lw=1.9)

    ax.axhline(0.0, color="0.55", lw=0.7, ls="--", zorder=1)
    ax.set_xlabel("log start count", fontsize=7.0)
    ax.set_ylabel("log fold (end / start)", fontsize=7.0)
    ax.set_title(title, fontsize=7.4, pad=3.5)
    ax.tick_params(labelsize=6.0)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return float(beta[1]), float(r2), beta_tiers


def _draw_tier_beta_bars(
    ax,
    beta_tiers: list[tuple[int, float, float, int]],
    *,
    title: str,
) -> None:
    if not beta_tiers:
        ax.axis("off")
        return
    ne_vals = [t[0] for t in beta_tiers]
    betas = [t[1] for t in beta_tiers]
    cols = [_EDGE_COUNT_COLORS.get(ne, "#888888") for ne in ne_vals]
    ax.bar(np.arange(len(betas)), betas, color=cols, edgecolor="0.25", linewidth=0.4, zorder=2)
    ax.axhline(0.0, color="0.35", lw=0.8, zorder=1)
    ax.set_xticks(np.arange(len(betas)))
    ax.set_xticklabels([f"{ne}e" for ne in ne_vals], fontsize=7.0)
    for i, (ne, b, r2_e, _n) in enumerate(beta_tiers):
        ax.text(
            i, b + (0.03 if b >= 0 else -0.03), f"{b:+.2f}\n$R^2$={r2_e:.2f}",
            ha="center", va="bottom" if b >= 0 else "top", fontsize=6.2, color="0.2",
        )
    ax.set_ylabel(r"$\beta$ (log fold ~ log start)", fontsize=7.0)
    ax.set_title(title, fontsize=7.4, pad=3.5)
    ax.tick_params(labelsize=6.0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _motif_factor_rows_from_snaps(
    snaps: list[dict],
    *,
    motif_prefix: str,
    min_start: int,
) -> list[dict]:
    """Per-motif start/end counts with structural factor labels."""
    c0, c1 = snaps[0]["cnt"], snaps[-1]["cnt"]
    rows: list[dict] = []
    for key, v0_raw in c0.items():
        if not str(key).startswith(motif_prefix):
            continue
        v0 = float(v0_raw)
        if v0 < min_start:
            continue
        v1 = float(c1.get(key, 0.0))
        ne = _n_edges_from_key(key)
        ni = _n_inhibitory_from_key(key)
        pat = _edge_pattern_counts_from_key(key)
        rows.append({
            "key": str(key),
            "n_edges": ne,
            "n_inh": ni,
            "n_exc": ne - ni,
            "inh_frac": _inh_fraction_from_key(key),
            **pat,
            "start": v0,
            "end": v1,
            "fold": (v1 + EPS) / (v0 + EPS),
            "log_start": float(np.log(v0 + EPS)),
            "log_fold": float(np.log((v1 + EPS) / (v0 + EPS))),
        })
    return rows


def _factor_bin_stats(
    rows: list[dict],
    *,
    field: str,
    order: list[int | float] | None = None,
    min_n: int = 5,
) -> list[dict]:
    """Group motif rows by a discrete factor; return fold + dispersion summaries."""
    from collections import defaultdict

    groups: dict[int | float, list[dict]] = defaultdict(list)
    for r in rows:
        val = r[field]
        if val is None or not np.isfinite(val):
            continue
        groups[val].append(r)

    keys = order if order is not None else sorted(groups)
    out: list[dict] = []
    for k in keys:
        pts = groups.get(k, [])
        if len(pts) < min_n:
            continue
        starts = np.array([p["start"] for p in pts], dtype=float)
        ends = np.array([p["end"] for p in pts], dtype=float)
        folds = np.array([p["fold"] for p in pts], dtype=float)
        sd0, sd1 = float(np.std(starts)), float(np.std(ends))
        out.append({
            "label": k,
            "n": len(pts),
            "med_fold": float(np.median(folds)),
            "q25_fold": float(np.percentile(folds, 25)),
            "q75_fold": float(np.percentile(folds, 75)),
            "sd_start": sd0,
            "sd_end": sd1,
            "sd_ratio": sd1 / sd0 if sd0 > 1e-9 else float("nan"),
        })
    return out


def _draw_factor_median_fold_panel(
    ax,
    stats: list[dict],
    *,
    title: str,
    xlabel: str,
    color_fn,
    x_fmt,
) -> None:
    if not stats:
        ax.text(0.5, 0.5, "insufficient\ndata", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return
    xs = np.arange(len(stats))
    meds = [s["med_fold"] for s in stats]
    err_lo = [s["med_fold"] - s["q25_fold"] for s in stats]
    err_hi = [s["q75_fold"] - s["med_fold"] for s in stats]
    cols = [color_fn(s["label"]) for s in stats]
    ax.bar(xs, meds, color=cols, edgecolor="0.25", linewidth=0.4, zorder=2)
    ax.errorbar(xs, meds, yerr=[err_lo, err_hi], fmt="none", ecolor="0.25", capsize=2.5, lw=0.8, zorder=3)
    ax.axhline(1.0, color="0.45", lw=0.8, ls="--", zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([x_fmt(s["label"]) for s in stats], fontsize=6.5)
    ax.set_ylabel("median end / start", fontsize=7.0)
    ax.set_title(title, fontsize=7.4, pad=3.5)
    ax.tick_params(labelsize=6.0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, s in enumerate(stats):
        ax.text(i, s["med_fold"] + 0.02, f"n={s['n']}", ha="center", va="bottom", fontsize=5.5, color="0.35")


def _draw_factor_dispersion_panel(
    ax,
    stats: list[dict],
    *,
    title: str,
    xlabel: str,
    color_fn,
    x_fmt,
) -> None:
    if not stats:
        ax.text(0.5, 0.5, "insufficient\ndata", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return
    xs = np.arange(len(stats))
    ratios = [s["sd_ratio"] for s in stats]
    cols = [color_fn(s["label"]) for s in stats]
    ax.bar(xs, ratios, color=cols, edgecolor="0.25", linewidth=0.4, zorder=2)
    ax.axhline(1.0, color="0.45", lw=0.8, ls="--", zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([x_fmt(s["label"]) for s in stats], fontsize=6.5)
    ax.set_ylabel(r"$\sigma_{final} / \sigma_{start}$", fontsize=7.0)
    ax.set_title(title, fontsize=7.4, pad=3.5)
    ax.tick_params(labelsize=6.0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, s in enumerate(stats):
        ax.text(
            i, s["sd_ratio"] + 0.03, f"{s['sd_start']:.0f}→{s['sd_end']:.0f}",
            ha="center", va="bottom", fontsize=5.2, color="0.35",
        )


def _draw_ne_ninh_heatmap(
    ax,
    rows: list[dict],
    *,
    value: str,
    title: str,
    min_cell: int,
    cmap_name: str = "RdYlGn_r",
    vmin: float | None = None,
    vmax: float | None = None,
    fmt: str = ".2f",
) -> None:
    """Heatmap over (# directed edges × # inhibitory edges) cells."""
    from collections import defaultdict

    ne_vals = sorted({int(r["n_edges"]) for r in rows})
    ni_vals = sorted({int(r["n_inh"]) for r in rows})
    if not ne_vals or not ni_vals:
        ax.axis("off")
        return

    cells: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for r in rows:
        cells[(int(r["n_edges"]), int(r["n_inh"]))].append(r)

    grid = np.full((len(ni_vals), len(ne_vals)), np.nan, dtype=float)
    counts = np.zeros_like(grid)
    for j, ne in enumerate(ne_vals):
        for i, ni in enumerate(ni_vals):
            pts = cells.get((ne, ni), [])
            counts[i, j] = len(pts)
            if len(pts) < min_cell:
                continue
            if value == "med_fold":
                grid[i, j] = float(np.median([p["fold"] for p in pts]))
            elif value == "sd_ratio":
                s0 = float(np.std([p["start"] for p in pts]))
                s1 = float(np.std([p["end"] for p in pts]))
                grid[i, j] = s1 / s0 if s0 > 1e-9 else float("nan")
            elif value == "beta":
                if len(pts) < 8 or float(np.std([p["log_start"] for p in pts])) < 1e-9:
                    continue
                x = np.array([p["log_start"] for p in pts], dtype=float)
                y = np.array([p["log_fold"] for p in pts], dtype=float)
                b, _, _ = _ols(y, x)
                grid[i, j] = float(b[1])

    if vmin is None:
        finite = grid[np.isfinite(grid)]
        vmin = float(np.nanmin(finite)) if finite.size else 0.0
    if vmax is None:
        finite = grid[np.isfinite(grid)]
        vmax = float(np.nanmax(finite)) if finite.size else 1.0

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="0.92")
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(ne_vals)))
    ax.set_xticklabels([f"{ne}e" for ne in ne_vals], fontsize=6.5)
    ax.set_yticks(np.arange(len(ni_vals)))
    ax.set_yticklabels([f"{ni} inh" for ni in ni_vals], fontsize=6.5)
    ax.set_xlabel("# directed edges", fontsize=7.0)
    ax.set_ylabel("# inhibitory edges", fontsize=7.0)
    ax.set_title(title, fontsize=7.4, pad=3.5)
    for i in range(len(ni_vals)):
        for j in range(len(ne_vals)):
            n = int(counts[i, j])
            val = grid[i, j]
            if not np.isfinite(val):
                txt = f"n={n}" if n else ""
            else:
                txt = f"{val:{fmt}}\nn={n}"
            ax.text(
                j, i, txt, ha="center", va="center", fontsize=5.5,
                color="0.1" if np.isfinite(val) and val > 0.5 * (vmin + vmax) else "0.25",
            )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _draw_factor_regression_panel(
    ax,
    rows: list[dict],
    *,
    x_field: str,
    title: str,
    xlabel: str,
    color: str = "#4C78A8",
    jitter_x: bool = True,
    color_by_edges: bool = True,
) -> tuple[float, float]:
    """Scatter log fold vs a structural factor with OLS line."""
    x = np.array([float(r[x_field]) for r in rows], dtype=float)
    y = np.array([float(r["log_fold"]) for r in rows], dtype=float)
    n_edges = np.array([int(r["n_edges"]) for r in rows], dtype=int)

    if jitter_x:
        rng = np.random.default_rng(0)
        x_plot = x + rng.uniform(-0.10, 0.10, size=len(x))
    else:
        x_plot = x

    if color_by_edges:
        for ne in sorted(set(n_edges)):
            mask = n_edges == ne
            col = _EDGE_COUNT_COLORS.get(ne, "#888888")
            ax.scatter(
                x_plot[mask], y[mask], s=14, c=col, alpha=0.55,
                edgecolors="0.15", linewidths=0.2, zorder=3, label=f"{ne}e",
            )
    else:
        ax.scatter(x_plot, y, s=14, c=color, alpha=0.55, edgecolors="0.15", linewidths=0.2, zorder=3)

    beta, _, r2 = _ols(y, x)
    if len(x) >= 2 and float(np.std(x)) > 1e-12:
        x_line = np.linspace(float(x.min()), float(x.max()), 60)
        _outlined_regression_line(ax, x_line, beta[0] + beta[1] * x_line, "0.12", lw=1.8)

    ax.axhline(0.0, color="0.55", lw=0.7, ls="--", zorder=1)
    ax.set_xlabel(xlabel, fontsize=7.0)
    ax.set_ylabel("log fold (end / start)", fontsize=7.0)
    ax.set_title(
        rf"{title} ($\beta={beta[1]:+.3f}$, $R^2={r2:.3f}$, n={len(rows)})",
        fontsize=7.4, pad=3.5,
    )
    ax.tick_params(labelsize=6.0)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if color_by_edges:
        ax.legend(fontsize=5.5, frameon=True, fancybox=False, edgecolor="0.8", loc="best")
    return float(beta[1]), float(r2)


def plot_motif_factor_regressions(
    json_path: Path,
    out_path: Path,
    *,
    run_id: int,
    min_start: int,
    motif_prefix: str = "T|",
    n_dfa_states: int | None = None,
) -> Path:
    """OLS: log fold ~ #inh, ~ #exc, ~ inhibitory fraction."""
    snaps = json.loads(json_path.read_text(encoding="utf-8"))
    it0, it1 = int(snaps[0]["it"]), int(snaps[-1]["it"])
    rows = _motif_factor_rows_from_snaps(snaps, motif_prefix=motif_prefix, min_start=min_start)
    if not rows:
        raise ValueError(f"no motifs with start>={min_start} in {json_path}")

    panel_w, panel_h = 2.85, 2.65
    fig = plt.figure(figsize=(3 * panel_w + 0.5, panel_h + 0.75))
    gs = fig.add_gridspec(1, 3, wspace=0.32)

    b_inh, r2_inh = _draw_factor_regression_panel(
        fig.add_subplot(gs[0, 0]), rows,
        x_field="n_inh",
        title="log fold ~ # inhibitory",
        xlabel="# inhibitory edges",
    )
    b_exc, r2_exc = _draw_factor_regression_panel(
        fig.add_subplot(gs[0, 1]), rows,
        x_field="n_exc",
        title="log fold ~ # excitatory",
        xlabel="# excitatory edges",
    )
    b_frac, r2_frac = _draw_factor_regression_panel(
        fig.add_subplot(gs[0, 2]), rows,
        x_field="inh_frac",
        title="log fold ~ inh fraction",
        xlabel="inhibitory / (# edges)",
        jitter_x=True,
    )

    dfa_tag = f"{n_dfa_states} DFA states; " if n_dfa_states is not None else ""
    finalize_grid_figure(
        fig,
        suptitle=(
            f"r{run_id:02d} motif fold vs edge-sign structure  ({dfa_tag}iter {it0}→{it1}; "
            f"start>={min_start})"
        ),
        suptitle_fontsize=10,
        top=0.88,
        bottom=0.14,
        left=0.07,
        right=0.98,
        wspace=0.32,
    )
    save_figure(fig, out_path, dpi=150)
    print(f"wrote {out_path}")
    print(f"  log fold ~ n_inh:     beta={b_inh:+.4f}  R2={r2_inh:.3f}")
    print(f"  log fold ~ n_exc:     beta={b_exc:+.4f}  R2={r2_exc:.3f}")
    print(f"  log fold ~ inh_frac:  beta={b_frac:+.4f}  R2={r2_frac:.3f}")
    return out_path


def _ols_multi_with_se(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """OLS with intercept column already in X; returns beta, SE, R^2."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot else 0.0
    return beta, se, r2


def plot_motif_hypothesis_regressions(
    json_path: Path,
    out_path: Path,
    *,
    run_id: int,
    min_start: int,
    motif_prefix: str = "T|",
    n_dfa_states: int | None = None,
) -> Path:
    """Hypothesis battery: standardized OLS of log fold on ~14 structural features.

    Each feature is z-scored; shown univariate and controlled for # directed
    edges (density), since density confounds nearly every structural count.
    """
    snaps = json.loads(json_path.read_text(encoding="utf-8"))
    it0, it1 = int(snaps[0]["it"]), int(snaps[-1]["it"])
    c0, c1 = snaps[0]["cnt"], snaps[-1]["cnt"]

    feats: list[dict[str, float]] = []
    y_list: list[float] = []
    for key, v0_raw in c0.items():
        if not str(key).startswith(motif_prefix):
            continue
        v0 = float(v0_raw)
        if v0 < min_start:
            continue
        f = _hypothesis_features_from_key(key)
        if f is None:
            continue
        v1 = float(c1.get(key, 0.0))
        f["log_start"] = float(np.log(v0 + EPS))
        feats.append(f)
        y_list.append(float(np.log((v1 + EPS) / (v0 + EPS))))
    if len(feats) < 20:
        raise ValueError(f"only {len(feats)} motifs with start>={min_start} in {json_path}")

    y = np.array(y_list, dtype=float)
    n = len(y)
    ne_raw = np.array([f["n_edges"] for f in feats], dtype=float)
    z_ne = (ne_raw - ne_raw.mean()) / max(float(ne_raw.std()), 1e-12)
    ones = np.ones(n)

    # Baseline model with density only (for partial R^2 of other features).
    _, _, r2_ne_only = _ols_multi_with_se(y, np.column_stack([ones, z_ne]))

    results: list[dict] = []
    for field, label in _HYPOTHESIS_SPECS:
        x = np.array([f[field] for f in feats], dtype=float)
        sd = float(x.std())
        if sd < 1e-12:
            continue
        z = (x - x.mean()) / sd

        b_u, se_u, r2_u = _ols_multi_with_se(y, np.column_stack([ones, z]))
        if field == "n_edges":
            b_c, se_c, r2_c = b_u, se_u, r2_u
            partial_r2 = float("nan")
        else:
            b_c, se_c, r2_c = _ols_multi_with_se(y, np.column_stack([ones, z, z_ne]))
            partial_r2 = max(0.0, (r2_c - r2_ne_only) / max(1.0 - r2_ne_only, 1e-12))
        results.append({
            "field": field,
            "label": label,
            "beta_uni": float(b_u[1]),
            "se_uni": float(se_u[1]),
            "r2_uni": float(r2_u),
            "beta_ctl": float(b_c[1]),
            "se_ctl": float(se_c[1]),
            "partial_r2": partial_r2,
        })

    n_rows = len(results)
    fig = plt.figure(figsize=(9.4, 0.42 * n_rows + 1.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.06)
    ax_f = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1], sharey=ax_f)

    ys = np.arange(n_rows)[::-1]
    for yi, res in zip(ys, results):
        ax_f.errorbar(
            res["beta_uni"], yi + 0.16, xerr=1.96 * res["se_uni"],
            fmt="o", ms=4.5, color="#4C78A8", ecolor="#4C78A8",
            elinewidth=1.1, capsize=2.0, zorder=3,
        )
        ax_f.errorbar(
            res["beta_ctl"], yi - 0.16, xerr=1.96 * res["se_ctl"],
            fmt="s", ms=4.2, mfc="white", color="#E45756", ecolor="#E45756",
            elinewidth=1.1, capsize=2.0, zorder=3,
        )
    ax_f.axvline(0.0, color="0.4", lw=0.9, ls="--", zorder=1)
    ax_f.set_yticks(ys)
    ax_f.set_yticklabels([r["label"] for r in results], fontsize=7.2)
    ax_f.set_xlabel(r"standardized $\beta$ (log fold per 1 SD)", fontsize=8)
    ax_f.tick_params(labelsize=7)
    ax_f.grid(True, axis="x", alpha=0.25)
    ax_f.spines["top"].set_visible(False)
    ax_f.spines["right"].set_visible(False)
    from matplotlib.lines import Line2D
    ax_f.legend(
        handles=[
            Line2D([0], [0], marker="o", color="#4C78A8", lw=0, ms=5, label="univariate"),
            Line2D([0], [0], marker="s", mfc="white", color="#E45756", lw=0, ms=5,
                   label="controlling # edges"),
        ],
        fontsize=6.5, frameon=True, fancybox=False, edgecolor="0.8", loc="lower right",
    )
    ax_f.set_ylim(-0.7, n_rows - 0.3)

    bar_h = 0.30
    ax_r.barh(
        ys + 0.16, [r["r2_uni"] for r in results], height=bar_h,
        color="#4C78A8", alpha=0.75, zorder=2, label=r"univariate $R^2$",
    )
    partials = [0.0 if not np.isfinite(r["partial_r2"]) else r["partial_r2"] for r in results]
    ax_r.barh(
        ys - 0.16, partials, height=bar_h,
        color="#E45756", alpha=0.75, zorder=2, label=r"partial $R^2$ | # edges",
    )
    ax_r.tick_params(labelleft=False, labelsize=7)
    ax_r.set_xlabel(r"$R^2$", fontsize=8)
    ax_r.grid(True, axis="x", alpha=0.25)
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_visible(False)
    ax_r.legend(fontsize=6.5, frameon=True, fancybox=False, edgecolor="0.8", loc="lower right")

    dfa_tag = f"{n_dfa_states} DFA states; " if n_dfa_states is not None else ""
    finalize_grid_figure(
        fig,
        suptitle=(
            f"r{run_id:02d} hypothesis battery: log fold ~ structural features  "
            f"({dfa_tag}iter {it0}\u2192{it1}; start>={min_start}; n={n})"
        ),
        suptitle_fontsize=10,
        top=0.90,
        bottom=0.10,
        left=0.30,
        right=0.98,
        wspace=0.06,
    )
    save_figure(fig, out_path, dpi=150)
    print(f"wrote {out_path}")
    print(f"  n motifs = {n}; density-only R2 = {r2_ne_only:.3f}")
    ranked = sorted(
        results, key=lambda r: -(abs(r["beta_ctl"]) if r["field"] != "n_edges" else 0.0),
    )
    for r in ranked:
        pr = f"{r['partial_r2']:.3f}" if np.isfinite(r["partial_r2"]) else "  -  "
        print(
            f"  {r['label']:44s} uni b={r['beta_uni']:+.3f} (R2={r['r2_uni']:.3f})  "
            f"ctl b={r['beta_ctl']:+.3f} (partial R2={pr})"
        )
    return out_path


_STORY_FOREST_SPECS: list[tuple[str, str]] = [
    ("log_start", "initial abundance"),
    ("n_recip", "reciprocal pairs"),
    ("n_neg_loops", "negative feedback loops"),
    ("is_acyclic", "feedforward (no loops)"),
    ("max_out_deg", "max out-degree"),
    ("n_inh", "inhibitory edges"),
]


def _controlled_slope_between(
    c_base: dict,
    c_end: dict,
    *,
    motif_prefix: str,
    min_start: int,
    c_pred: dict | None = None,
) -> float:
    """Slope of log fold (base->end) on z(log pred count), controlling z(# edges)."""
    if c_pred is None:
        c_pred = c_base
    keys = [
        k for k in c_base
        if str(k).startswith(motif_prefix) and float(c_base[k]) >= min_start
    ]
    if len(keys) < 20:
        return float("nan")
    ls = np.array([np.log(float(c_pred.get(k, 0.0)) + EPS) for k in keys])
    ne = np.array([_n_edges_from_key(k) for k in keys], dtype=float)
    y = np.array([
        np.log((float(c_end.get(k, 0.0)) + EPS) / (float(c_base[k]) + EPS))
        for k in keys
    ])
    if float(ls.std()) < 1e-12 or float(ne.std()) < 1e-12:
        return float("nan")
    z_ls = (ls - ls.mean()) / ls.std()
    z_ne = (ne - ne.mean()) / ne.std()
    X = np.column_stack([np.ones(len(keys)), z_ls, z_ne])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[1])


def plot_motif_story_board(
    json_path: Path,
    out_path: Path,
    *,
    run_id: int,
    min_start: int,
    motif_prefix: str = "T|",
    n_dfa_states: int | None = None,
    focus_tier: int = 5,
) -> Path:
    """Four-panel story: abundant motifs compress, and it tracks learning.

    A: Simpson scatter (pooled slope positive, within-tier negative).
    B: raw count trajectories in one tier funneling together.
    C: compression slope vs training time, with word error and noise floor.
    D: condensed density-controlled hypothesis ranking.
    """
    snaps = json.loads(json_path.read_text(encoding="utf-8"))
    c0 = snaps[0]["cnt"]
    it0, it1 = int(snaps[0]["it"]), int(snaps[-1]["it"])
    iters = np.array([int(s["it"]) for s in snaps], dtype=float)

    fig = plt.figure(figsize=(10.2, 7.6))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.50)

    # --- Panel A: Simpson scatter -------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    sr_log_c0, sr_log_fold, sr_n_edges = _fold_arrays_from_snaps(
        snaps, motif_prefix=motif_prefix, min_start=min_start,
    )
    pooled_beta, _, _ = _draw_tier_fold_scatter(
        ax_a, sr_log_c0, sr_log_fold, sr_n_edges,
        min_fit=8, title="", jitter_y=True, subsample=None,
    )
    ax_a.set_title(
        f"A  pooled slope misleads (+{pooled_beta:.2f}, black):\n"
        "within every density tier, abundant motifs shrink",
        fontsize=8.2, pad=4,
    )

    # --- Panel B: trajectories funneling within focus tier ------------------
    ax_b = fig.add_subplot(gs[0, 1])
    tier_keys = [
        k for k in c0
        if str(k).startswith(motif_prefix)
        and _n_edges_from_key(k) == focus_tier
        and float(c0[k]) >= min_start
    ]
    if len(tier_keys) < 10:
        counts_by_tier = {}
        for k in c0:
            if str(k).startswith(motif_prefix) and float(c0[k]) >= min_start:
                counts_by_tier.setdefault(_n_edges_from_key(k), []).append(k)
        focus_tier, tier_keys = max(counts_by_tier.items(), key=lambda kv: len(kv[1]))
    starts = np.array([float(c0[k]) for k in tier_keys])
    ends = np.array([float(snaps[-1]["cnt"].get(k, 0.0)) for k in tier_keys])
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=float(starts.min()), vmax=float(starts.max()))
    series_mat = np.array([
        [float(s["cnt"].get(k, 0.0)) for s in snaps] for k in tier_keys
    ])
    for row, v0 in zip(series_mat, starts):
        ax_b.plot(iters, row, color=cmap(norm(v0)), lw=0.6, alpha=0.22, zorder=2)
    mean_t = series_mat.mean(axis=0)
    sd_t = series_mat.std(axis=0)
    tier_col = _EDGE_COUNT_COLORS.get(focus_tier, "#333333")
    ax_b.fill_between(
        iters, mean_t - sd_t, mean_t + sd_t,
        color="none", edgecolor=tier_col, lw=1.4, zorder=4,
    )
    ax_b.plot(iters, mean_t, color=tier_col, lw=1.6, zorder=5, label="tier mean")
    ax_b.plot(iters, mean_t - sd_t, color=tier_col, lw=1.2, ls="--", zorder=5)
    ax_b.plot(iters, mean_t + sd_t, color=tier_col, lw=1.2, ls="--", zorder=5,
              label=r"mean $\pm$ 1$\sigma$")
    ax_b.legend(fontsize=6.2, frameon=True, fancybox=False, edgecolor="0.8", loc="upper right")
    sd0, sd1 = float(starts.std()), float(ends.std())
    ax_b.set_title(
        f"B  {focus_tier}-edge classes converge: "
        rf"$\sigma$ {sd0:.0f}$\rightarrow${sd1:.0f}"
        f"\n(each line = one motif class, color = start count)",
        fontsize=8.2, pad=4,
    )
    ax_b.set_xlabel("training iteration", fontsize=7.5)
    ax_b.set_ylabel("count", fontsize=7.5)
    ax_b.tick_params(labelsize=6.5)
    ax_b.grid(True, alpha=0.25)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = plt.colorbar(sm, ax=ax_b, fraction=0.046, pad=0.03)
    cb.set_label("start count", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)

    # --- Panel C: slope deepens with learning -------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    slopes = [
        _controlled_slope_between(
            c0, s["cnt"], motif_prefix=motif_prefix, min_start=min_start,
        )
        for s in snaps[1:]
    ]
    noise_floor = _controlled_slope_between(
        snaps[-2]["cnt"], snaps[-1]["cnt"],
        motif_prefix=motif_prefix, min_start=min_start,
    )
    ax_c.plot(iters[1:], slopes, color="#E45756", lw=1.6, zorder=3,
              label=r"compression slope $\beta$")
    ax_c.axhline(0.0, color="0.55", lw=0.7, ls="--", zorder=1)
    if np.isfinite(noise_floor):
        ax_c.axhline(
            noise_floor, color="0.35", lw=1.0, ls=":", zorder=2,
            label=f"noise floor ({noise_floor:+.2f})",
        )
    we = np.array([float(s.get("we", float("nan"))) for s in snaps], dtype=float)
    if np.isfinite(we).sum() >= 2:
        ax_c2 = ax_c.twinx()
        ax_c2.plot(iters, we, color="0.55", lw=1.1, ls="-", alpha=0.75, zorder=2)
        ax_c2.set_ylabel("word error", fontsize=7.0, color="0.45", labelpad=1.0)
        ax_c2.tick_params(labelsize=6.0, colors="0.45", pad=1.0)
        ax_c2.spines["top"].set_visible(False)
        from matplotlib.lines import Line2D
        ax_c.legend(
            handles=[
                Line2D([0], [0], color="#E45756", lw=1.6,
                       label=r"compression slope $\beta$"),
                Line2D([0], [0], color="0.55", lw=1.1, label="word error (right)"),
                Line2D([0], [0], color="0.35", lw=1.0, ls=":",
                       label=f"noise floor ({noise_floor:+.2f})"),
            ],
            fontsize=6.2, frameon=True, fancybox=False, edgecolor="0.8", loc="center right",
        )
    else:
        ax_c.legend(fontsize=6.2, frameon=True, fancybox=False, edgecolor="0.8")
    ax_c.set_title(
        "C  compression deepens as the task is learned\n"
        rf"($\beta$ of log fold ~ log start | #edges, from iter {it0})",
        fontsize=8.2, pad=4,
    )
    ax_c.set_xlabel("training iteration", fontsize=7.5)
    ax_c.set_ylabel(r"standardized $\beta$", fontsize=7.5)
    ax_c.tick_params(labelsize=6.5)
    ax_c.grid(True, alpha=0.25)
    ax_c.spines["top"].set_visible(False)

    # --- Panel D: condensed hypothesis ranking ------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    feats: list[dict[str, float]] = []
    y_list: list[float] = []
    for key, v0_raw in c0.items():
        if not str(key).startswith(motif_prefix) or float(v0_raw) < min_start:
            continue
        f = _hypothesis_features_from_key(key)
        if f is None:
            continue
        f["log_start"] = float(np.log(float(v0_raw) + EPS))
        feats.append(f)
        y_list.append(float(np.log(
            (float(snaps[-1]["cnt"].get(key, 0.0)) + EPS) / (float(v0_raw) + EPS)
        )))
    y = np.array(y_list)
    ne_arr = np.array([f["n_edges"] for f in feats])
    z_ne = (ne_arr - ne_arr.mean()) / ne_arr.std()
    ones = np.ones(len(y))
    rows_d: list[tuple[str, float, float]] = []
    for field, label in _STORY_FOREST_SPECS:
        x = np.array([f[field] for f in feats])
        sd = float(x.std())
        if sd < 1e-12:
            continue
        z = (x - x.mean()) / sd
        b, se, _ = _ols_multi_with_se(y, np.column_stack([ones, z, z_ne]))
        rows_d.append((label, float(b[1]), float(se[1])))
    ys_d = np.arange(len(rows_d))[::-1]
    for yi, (label, b, se) in zip(ys_d, rows_d):
        col = "#E45756" if label == "initial abundance" else "0.35"
        ax_d.errorbar(
            b, yi, xerr=1.96 * se, fmt="o", ms=5.5 if col != "0.35" else 4.5,
            color=col, ecolor=col, elinewidth=1.2, capsize=2.5, zorder=3,
        )
    ax_d.axvline(0.0, color="0.4", lw=0.9, ls="--", zorder=1)
    ax_d.set_yticks(ys_d)
    ax_d.set_yticklabels([r[0] for r in rows_d], fontsize=7.2)
    ax_d.set_xlabel(r"standardized $\beta$ (density-controlled)", fontsize=7.5)
    ax_d.set_title(
        "D  initial abundance beats every structural feature\n"
        "(inhibition collapses once density is controlled)",
        fontsize=8.2, pad=4,
    )
    ax_d.tick_params(labelsize=6.5)
    ax_d.grid(True, axis="x", alpha=0.25)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)

    dfa_tag = f"{n_dfa_states} DFA states; " if n_dfa_states is not None else ""
    finalize_grid_figure(
        fig,
        suptitle=(
            f"r{run_id:02d}: training compresses abundant motif classes  "
            f"({dfa_tag}iter {it0}\u2192{it1}; start>={min_start})"
        ),
        suptitle_fontsize=10.5,
        top=0.89,
        bottom=0.08,
        left=0.08,
        right=0.97,
        hspace=0.42,
        wspace=0.46,
    )
    save_figure(fig, out_path, dpi=150)
    print(f"wrote {out_path}")
    print(f"  pooled beta={pooled_beta:+.3f}; focus tier={focus_tier}e sd {sd0:.1f}->{sd1:.1f}")
    print(f"  final controlled slope={slopes[-1]:+.3f}; noise floor={noise_floor:+.3f}")
    for label, b, se in rows_d:
        print(f"  D {label:28s} b={b:+.3f} +/- {1.96 * se:.3f}")
    return out_path


_PATTERN_REGRESSION_SPECS: list[tuple[str, str, str]] = [
    ("n_mut_exc", "mutual exc", "#4C78A8"),
    ("n_mut_inh", "mutual inh", "#E45756"),
    ("n_mut_mixed", "mutual mixed", "#B279A2"),
    ("n_uni_exc", "uni exc", "#72B7B2"),
    ("n_uni_inh", "uni inh", "#F58518"),
]


def _draw_tier_pattern_regression_panel(
    ax,
    rows: list[dict],
    *,
    tier_ne: int,
    x_field: str,
    color: str,
    min_n: int = 12,
) -> tuple[float, float]:
    """log fold ~ one edge-pattern count, within a single #edges tier."""
    tier_rows = [r for r in rows if int(r["n_edges"]) == int(tier_ne)]
    if len(tier_rows) < min_n:
        ax.text(
            0.5, 0.5, f"too few\n(n={len(tier_rows)})",
            ha="center", va="center", transform=ax.transAxes, fontsize=6.5,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        return float("nan"), float("nan")

    x = np.array([float(r[x_field]) for r in tier_rows], dtype=float)
    y = np.array([float(r["log_fold"]) for r in tier_rows], dtype=float)
    if float(np.std(x)) < 1e-12:
        ax.text(
            0.5, 0.5, "no variation\nin predictor",
            ha="center", va="center", transform=ax.transAxes, fontsize=6.5,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        return float("nan"), float("nan")

    rng = np.random.default_rng(int(tier_ne))
    x_plot = x + rng.uniform(-0.10, 0.10, size=len(x))
    ax.scatter(
        x_plot, y, s=16, c=color, alpha=0.60,
        edgecolors="0.15", linewidths=0.25, zorder=3,
    )

    beta, _, r2 = _ols(y, x)
    x_line = np.linspace(float(x.min()), float(x.max()), 60)
    _outlined_regression_line(ax, x_line, beta[0] + beta[1] * x_line, "0.12", lw=1.6)

    ax.axhline(0.0, color="0.55", lw=0.7, ls="--", zorder=1)
    ax.set_xlabel(f"# {x_field.replace('n_', '').replace('_', ' ')}", fontsize=6.5)
    ax.text(
        0.97, 0.97,
        rf"$\beta={beta[1]:+.3f}$  $R^2={r2:.2f}$  n={len(tier_rows)}",
        transform=ax.transAxes, ha="right", va="top", fontsize=6.0, color="0.25",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.5),
    )
    ax.tick_params(labelsize=6.0)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return float(beta[1]), float(r2)


def plot_motif_pattern_regressions(
    json_path: Path,
    out_path: Path,
    *,
    run_id: int,
    min_start: int,
    motif_prefix: str = "T|",
    n_dfa_states: int | None = None,
    min_tier_n: int = 20,
    min_cell_n: int = 12,
) -> Path:
    """log fold ~ edge-pattern counts, stratified by # directed edges."""
    snaps = json.loads(json_path.read_text(encoding="utf-8"))
    it0, it1 = int(snaps[0]["it"]), int(snaps[-1]["it"])
    rows = _motif_factor_rows_from_snaps(snaps, motif_prefix=motif_prefix, min_start=min_start)
    if not rows:
        raise ValueError(f"no motifs with start>={min_start} in {json_path}")

    tier_counts = {ne: sum(1 for r in rows if int(r["n_edges"]) == ne) for ne in (2, 3, 4, 5, 6)}
    tiers = [ne for ne in (2, 3, 4, 5, 6) if tier_counts.get(ne, 0) >= min_tier_n]
    if not tiers:
        raise ValueError(f"no tier with n>={min_tier_n}")

    ncols = len(_PATTERN_REGRESSION_SPECS)
    panel_w, panel_h = 2.05, 2.15
    fig = plt.figure(figsize=(ncols * panel_w + 0.55, len(tiers) * panel_h + 0.95))
    gs = fig.add_gridspec(len(tiers), ncols, hspace=0.52, wspace=0.35)

    col_titles = [spec[1] for spec in _PATTERN_REGRESSION_SPECS]
    for j, title in enumerate(col_titles):
        fig.text(
            0.07 + (j + 0.5) / ncols * 0.93, 0.965, title,
            ha="center", va="bottom", fontsize=8.0, fontweight="medium",
        )

    results: list[tuple[int, str, float, float]] = []
    for i, ne in enumerate(tiers):
        for j, (field, _label, color) in enumerate(_PATTERN_REGRESSION_SPECS):
            ax = fig.add_subplot(gs[i, j])
            beta, r2 = _draw_tier_pattern_regression_panel(
                ax, rows, tier_ne=ne, x_field=field, color=color, min_n=min_cell_n,
            )
            if j == 0:
                tier_col = _EDGE_COUNT_COLORS.get(ne, "#888888")
                ax.set_ylabel(
                    f"{ne}-edge\nlog fold",
                    fontsize=7.0, color=tier_col, fontweight="medium",
                )
            else:
                ax.tick_params(labelleft=False)
                ax.set_ylabel("")
            if np.isfinite(beta):
                results.append((ne, field, beta, r2))

    dfa_tag = f"{n_dfa_states} DFA states; " if n_dfa_states is not None else ""
    finalize_grid_figure(
        fig,
        suptitle=(
            f"r{run_id:02d} log fold ~ edge patterns (by #edges tier)  "
            f"({dfa_tag}iter {it0}→{it1}; start>={min_start})"
        ),
        suptitle_fontsize=10,
        top=0.92,
        bottom=0.07,
        left=0.09,
        right=0.98,
        hspace=0.52,
        wspace=0.35,
    )
    save_figure(fig, out_path, dpi=150)
    print(f"wrote {out_path}")
    for ne, field, beta, r2 in results:
        print(f"  {ne}e  {field}: beta={beta:+.4f}  R2={r2:.3f}")
    return out_path


def plot_motif_factor_panel_analysis(
    json_path: Path,
    out_path: Path,
    *,
    run_id: int,
    min_start: int,
    motif_prefix: str = "T|",
    n_dfa_states: int | None = None,
    min_cell: int = 8,
) -> Path:
    """Diagnostic panels: which structural factors drive fold change and dispersion."""
    snaps = json.loads(json_path.read_text(encoding="utf-8"))
    it0, it1 = int(snaps[0]["it"]), int(snaps[-1]["it"])
    rows = _motif_factor_rows_from_snaps(snaps, motif_prefix=motif_prefix, min_start=min_start)
    if not rows:
        raise ValueError(f"no motifs with start>={min_start} in {json_path}")

    ne_order = sorted({int(r["n_edges"]) for r in rows})
    ni_order = sorted({int(r["n_inh"]) for r in rows})

    by_ne = _factor_bin_stats(rows, field="n_edges", order=ne_order, min_n=min_cell // 2)
    by_ni = _factor_bin_stats(rows, field="n_inh", order=ni_order, min_n=min_cell // 2)

    panel_w, panel_h = 2.85, 2.55
    fig = plt.figure(figsize=(3 * panel_w + 0.5, 2 * panel_h + 0.9))
    gs = fig.add_gridspec(2, 3, hspace=0.48, wspace=0.38)

    _draw_factor_median_fold_panel(
        fig.add_subplot(gs[0, 0]), by_ne,
        title="median fold by # directed edges",
        xlabel="# edges",
        color_fn=lambda k: _EDGE_COUNT_COLORS.get(int(k), "#888888"),
        x_fmt=lambda k: f"{int(k)}e",
    )
    _draw_factor_median_fold_panel(
        fig.add_subplot(gs[0, 1]), by_ni,
        title="median fold by # inhibitory edges",
        xlabel="# inh edges",
        color_fn=lambda k: _INH_COUNT_COLORS.get(int(k), "#888888"),
        x_fmt=lambda k: f"{int(k)}",
    )
    _draw_ne_ninh_heatmap(
        fig.add_subplot(gs[0, 2]), rows,
        value="med_fold", title="median fold (#edges × #inh)",
        min_cell=min_cell, cmap_name="RdYlGn_r", vmin=0.5, vmax=1.2, fmt=".2f",
    )

    _draw_factor_dispersion_panel(
        fig.add_subplot(gs[1, 0]), by_ne,
        title=r"dispersion by # directed edges",
        xlabel="# edges",
        color_fn=lambda k: _EDGE_COUNT_COLORS.get(int(k), "#888888"),
        x_fmt=lambda k: f"{int(k)}e",
    )
    _draw_factor_dispersion_panel(
        fig.add_subplot(gs[1, 1]), by_ni,
        title=r"dispersion by # inhibitory edges",
        xlabel="# inh edges",
        color_fn=lambda k: _INH_COUNT_COLORS.get(int(k), "#888888"),
        x_fmt=lambda k: f"{int(k)}",
    )
    _draw_ne_ninh_heatmap(
        fig.add_subplot(gs[1, 2]), rows,
        value="sd_ratio", title=r"$\sigma$ ratio (#edges × #inh)",
        min_cell=min_cell, cmap_name="RdYlGn", vmin=0.5, vmax=1.5, fmt=".2f",
    )

    dfa_tag = f"{n_dfa_states} DFA states; " if n_dfa_states is not None else ""
    finalize_grid_figure(
        fig,
        suptitle=(
            f"r{run_id:02d} motif factor analysis  ({dfa_tag}iter {it0}→{it1}; "
            f"start>={min_start}; cells need n>={min_cell})"
        ),
        suptitle_fontsize=10,
        top=0.90,
        bottom=0.08,
        left=0.07,
        right=0.98,
        hspace=0.48,
        wspace=0.38,
    )
    save_figure(fig, out_path, dpi=150)
    print(f"wrote {out_path}")
    print(f"  n motifs={len(rows)}  tiers: {len(ne_order)} edge counts, {len(ni_order)} inh counts")
    for s in by_ni:
        print(
            f"  {int(s['label'])} inh: med fold={s['med_fold']:.3f}  "
            f"sd ratio={s['sd_ratio']:.2f}  n={s['n']}",
        )
    return out_path


def plot_single_run_motif_board(
    json_path: Path,
    cache_path: Path,
    out_path: Path,
    *,
    run_id: int,
    min_start: int,
    motif_prefix: str = "T|",
    n_demos: int = 4,
    n_cols: int = 4,
    n_dfa_states: int | None = None,
) -> Path:
    """Single-run motif board: uniform grid — demos, tier KDEs, summary metrics."""
    from viz.weight_structure import (
        draw_edge_signed_hl_motif,
        edge_signed_hl_schema_pseudo,
        motif_schema_box,
    )

    if not json_path.is_file():
        raise FileNotFoundError(json_path)
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"missing {cache_path}; run motif-folds or all-runs-over-learning first"
        )

    snaps = json.loads(json_path.read_text(encoding="utf-8"))
    c0, c1 = snaps[0]["cnt"], snaps[-1]["cnt"]
    it0, it1 = int(snaps[0]["it"]), int(snaps[-1]["it"])
    kind = "dyad" if motif_prefix.startswith("D") else "triad"
    min_fit = 3 if kind == "dyad" else 8

    demos = _pick_single_run_demos(
        snaps, motif_prefix=motif_prefix, min_start=min_start, n_demos=n_demos,
    )[:n_cols]

    kde_tiers = sorted({
        _n_edges_from_key(k)
        for k in c0
        if str(k).startswith(motif_prefix)
        and float(c0.get(k, 0.0)) >= min_start
    })
    # Prefer 2–5 edge tiers in the four histogram slots (6e is sparse).
    preferred = [ne for ne in (2, 3, 4, 5) if ne in kde_tiers]
    if len(preferred) < n_cols:
        preferred = sorted(kde_tiers)[:n_cols]
    kde_tiers = preferred[:n_cols]

    sr_log_c0, sr_log_fold, sr_n_edges = _fold_arrays_from_snaps(
        snaps, motif_prefix=motif_prefix, min_start=min_start,
    )
    mr_log_c0, mr_log_fold, mr_n_edges, per_run, payload = _load_all_runs_fold_context(
        cache_path, motif_prefix=motif_prefix,
    )
    if n_dfa_states is None:
        try:
            n_dfa_states = lookup_run_dfa(cache_path, run_id)
        except KeyError:
            n_dfa_states = None
    target_we = float(payload.get("target_we", TARGET_WE))
    n_solved = int(payload.get("n_solved", 0))
    n_runs = int(payload.get("n_runs", len(per_run)))

    panel_w, panel_h = 2.75, 2.55
    fig = plt.figure(figsize=(n_cols * panel_w + 0.45, 3 * panel_h + 0.85))
    gs = fig.add_gridspec(3, n_cols, hspace=0.40, wspace=0.30)

    schema_insets: list[tuple[Any, str]] = []

    # Row 0 — demo trajectories only (fig 19 inset style).
    for j, demo in enumerate(demos):
        key = demo["key"]
        ne = int(demo["n_edges"])
        col = _EDGE_COUNT_COLORS.get(ne, "#888888")
        counts = demo["counts"]
        iters = demo["iters"]
        fold = float(counts[-1] / max(counts[0], EPS))
        rising = bool(counts[-1] >= counts[0])

        ax = fig.add_subplot(gs[0, j])
        ax.plot(iters, counts, color=col, lw=1.25, zorder=2)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.34 * (hi - lo))
        ax.set_title(demo["label"], fontsize=7.4, pad=3.5, color=col)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=6.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("iteration", fontsize=7.0)
        if j == 0:
            ax.set_ylabel("count", fontsize=7.0)
        ax.text(
            0.97 if rising else 0.03, 0.035, rf"$\times${fold:.2f}",
            transform=ax.transAxes, ha="right" if rising else "left", va="bottom",
            fontsize=6.4, color="0.20",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, pad=0.8), zorder=6,
        )
        inset = ax.inset_axes([0.035, 0.58, 0.34, 0.36] if rising else [0.625, 0.58, 0.34, 0.36])
        inset.set_axis_off()
        inset.patch.set_visible(False)
        schema_insets.append((inset, key))
    for j in range(len(demos), n_cols):
        fig.add_subplot(gs[0, j]).axis("off")

    # Row 1 — before/after count histograms per tier (linear count axis).
    for j, ne in enumerate(kde_tiers):
        ax_k = fig.add_subplot(gs[1, j])
        _draw_tier_kde_before_after(
            ax_k,
            n_edges=ne,
            c0=c0,
            c1=c1,
            motif_prefix=motif_prefix,
            min_start=min_start,
            it0=it0,
            it1=it1,
        )
    for j in range(len(kde_tiers), n_cols):
        fig.add_subplot(gs[1, j]).axis("off")

    # Row 2 — summary metrics (equal panels).
    ax_sr = fig.add_subplot(gs[2, 0])
    sr_beta, sr_r2, sr_tiers = _draw_tier_fold_scatter(
        ax_sr, sr_log_c0, sr_log_fold, sr_n_edges, min_fit=min_fit,
        title=f"r{run_id:02d}: log fold vs log start",
        jitter_y=True, subsample=None,
    )
    ax_sr.set_title(
        rf"r{run_id:02d}: log fold vs log start ($\beta={sr_beta:+.2f}$, $R^2={sr_r2:.2f}$)",
        fontsize=7.4, pad=3.5,
    )

    ax_bb = fig.add_subplot(gs[2, 1])
    _draw_tier_beta_bars(ax_bb, sr_tiers, title=rf"r{run_id:02d}: $\beta$ by #edges")

    ax_mr = fig.add_subplot(gs[2, 2])
    mr_beta, mr_r2, _mr_tiers = _draw_tier_fold_scatter(
        ax_mr, mr_log_c0, mr_log_fold, mr_n_edges, min_fit=min_fit,
        title="all runs pooled",
        jitter_y=True, subsample=3500,
    )
    ax_mr.set_title(
        rf"all runs pooled ($\beta={mr_beta:+.2f}$, $R^2={mr_r2:.2f}$, n={len(mr_log_c0)})",
        fontsize=7.4, pad=3.5,
    )

    ax_dfa = fig.add_subplot(gs[2, 3])
    dfa_vals = [float(r["n_dfa_states"]) for r in per_run if np.isfinite(r.get("beta", float("nan")))]
    dfa_norm = plt.Normalize(vmin=min(dfa_vals), vmax=max(dfa_vals)) if dfa_vals else plt.Normalize(0, 1)
    _plot_beta_vs_dfa(
        per_run, ax=ax_dfa, dfa_cmap=plt.get_cmap("viridis"),
        dfa_norm=dfa_norm, target_we=target_we, highlight_run_id=run_id,
    )
    ax_dfa.set_title(
        f"per-run slope vs DFA (n={n_runs}, {n_solved} solved)",
        fontsize=7.4, pad=3.5,
    )

    dfa_tag = f"{n_dfa_states} DFA states; " if n_dfa_states is not None else ""
    finalize_grid_figure(
        fig,
        suptitle=(
            f"r{run_id:02d} rnn edge-sign {kind} motifs  "
            f"({dfa_tag}start>={min_start}; blue/red edges = excitatory/inhibitory)"
        ),
        suptitle_fontsize=10,
        top=0.92,
        bottom=0.07,
        left=0.07,
        right=0.98,
        hspace=0.40,
        wspace=0.30,
    )
    for inset, key in schema_insets:
        draw_edge_signed_hl_motif(
            inset, key,
            box=motif_schema_box(
                inset, edge_signed_hl_schema_pseudo(key), center_x=0.5, max_width=0.98,
            ),
        )

    save_figure(fig, out_path, dpi=150)
    print(f"wrote {out_path}")
    print(f"  r{run_id:02d} pooled beta={sr_beta:+.3f}  R2={sr_r2:.3f}  n={len(sr_log_c0)}")
    print(f"  all-runs pooled beta={mr_beta:+.3f}  R2={mr_r2:.3f}  n={len(mr_log_c0)}")
    for demo in demos:
        fold = float(demo["counts"][-1] / max(demo["counts"][0], EPS))
        print(f"  demo {demo['label']}  {demo['key']}  x{fold:.2f}  score={demo['score']:.3f}")
    return out_path


def _pick_demo_motifs(
    rows: list[dict],
    *,
    motif_prefix: str,
    n_demos: int = 4,
    min_runs: int = 8,
) -> list[dict]:
    """Pick representative motif keys: abundant compressors and rare growers."""
    from collections import defaultdict

    by_key: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        key = str(r["key"])
        if not key.startswith(motif_prefix):
            continue
        by_key[key].append(r)

    stats: list[dict] = []
    for key, pts in by_key.items():
        if len(pts) < min_runs:
            continue
        ne = _n_edges_from_key(key)
        log_c0 = np.array([p["log_c0"] for p in pts], dtype=float)
        log_fold = np.array([p["log_fold"] for p in pts], dtype=float)
        stats.append({
            "key": key,
            "n_edges": ne,
            "n_runs": len(pts),
            "med_log_c0": float(np.median(log_c0)),
            "med_log_fold": float(np.median(log_fold)),
        })
    if not stats:
        return []

    chosen: list[dict] = []
    used: set[str] = set()

    def _take(pool: list[dict], *, reverse_fold: bool = False) -> dict | None:
        pool = [s for s in pool if s["key"] not in used]
        if not pool:
            return None
        pick = sorted(pool, key=lambda s: s["med_log_fold"], reverse=reverse_fold)[0]
        used.add(pick["key"])
        return pick

    is_dyad = motif_prefix.startswith("D")
    if is_dyad:
        for ne_target, reverse in ((2, False), (2, True), (1, False), (1, True)):
            pick = _take([s for s in stats if s["n_edges"] == ne_target], reverse_fold=reverse)
            if pick is not None:
                chosen.append({**pick, "role": "grow" if reverse else "compress"})
            if len(chosen) >= n_demos:
                break
    else:
        # One homogenizer per edge tier (5→3), then a 2-edge grower counterexample.
        for ne_target in (5, 4, 3):
            pick = _take([s for s in stats if s["n_edges"] == ne_target])
            if pick is not None:
                chosen.append({**pick, "role": "compress"})
            if len(chosen) >= n_demos - 1:
                break
        grow = _take([s for s in stats if s["n_edges"] == 2], reverse_fold=True)
        if grow is not None:
            chosen.append({**grow, "role": "grow"})

    # Fill remaining slots with abundant compressors.
    while len(chosen) < n_demos:
        rest = [s for s in stats if s["key"] not in used]
        if not rest:
            break
        pick = sorted(rest, key=lambda s: (-s["med_log_c0"], s["med_log_fold"]))[0]
        used.add(pick["key"])
        chosen.append({**pick, "role": "compress"})
    return chosen[:n_demos]


def _best_run_for_key(rows: list[dict], key: str) -> int:
    pts = [r for r in rows if r["key"] == key]
    if not pts:
        raise KeyError(key)
    return int(max(pts, key=lambda p: float(p["log_c0"]))["run_id"])


def _load_motif_trajectory(
    run_id: int,
    key: str,
    *,
    model: str,
    seed: int,
    coloring: str,
    max_snaps: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    from experiment import checkpoint_path
    from rnn.learning_snaps import list_learning_snaps

    task = f"mixeddfa_r{run_id:02d}_ns"
    ckpt = checkpoint_path(task, model, seed=seed)
    snaps = [
        p for p in _dominant_session(list_learning_snaps(ckpt))
        if _snap_iteration(p) > 0
    ]
    snaps = _subsample_snaps(snaps, max_snaps)
    iters: list[float] = []
    counts: list[float] = []
    for snap in snaps:
        row = _snap_census(snap, coloring=coloring, dale_sign=None)
        it = float(row["it"])
        iters.append(it)
        counts.append(float(row["cnt"].get(key, 0.0)))
    return np.array(iters, dtype=float), np.array(counts, dtype=float)


def plot_all_runs_homogenization_board(
    cache_path: Path,
    out_path: Path,
    *,
    min_start: int,
    rebuild_cache: bool,
    model: str,
    seed: int,
    max_snaps_per_run: int,
    coloring: str = "edge_sign",
    motif_prefix: str = "T|",
    n_demos: int = 4,
) -> Path:
    """Multi-run homogenization board: demo trajectories + tier regressions."""
    from viz.weight_structure import (
        draw_edge_signed_hl_motif,
        edge_signed_hl_schema_pseudo,
        motif_schema_box,
    )

    if rebuild_cache or not cache_path.is_file():
        collect_all_runs_motif_cache(
            cache_path,
            min_start=min_start,
            max_snaps_per_run=max_snaps_per_run,
            model=model,
            seed=seed,
            coloring=coloring,
            motif_prefix=motif_prefix,
        )
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    rows = payload["fold_rows"]
    series = sorted(payload["run_series"], key=_run_sort_key)
    target_we = float(payload.get("target_we", TARGET_WE))
    n_solved = int(payload.get("n_solved", sum(1 for r in series if r.get("solved"))))
    color_tag = "edge-sign" if payload.get("coloring") == "edge_sign" else "Dale-node"
    motif_tag = "dyad" if str(payload.get("motif_prefix", motif_prefix)).startswith("D") else "triad"
    min_fit = 3 if motif_tag == "dyad" else 8
    min_demo_runs = 3 if motif_tag == "dyad" else 8

    by_run: dict[int, list[dict]] = {}
    for r in rows:
        by_run.setdefault(int(r["run_id"]), []).append(r)

    per_run: list[dict] = []
    for run in series:
        rid = int(run["run_id"])
        pts = by_run.get(rid, [])
        beta_val, r2 = float("nan"), float("nan")
        if len(pts) >= 3:
            x = np.array([p["log_c0"] for p in pts], dtype=float)
            y = np.array([p["log_fold"] for p in pts], dtype=float)
            if float(np.std(x)) > 1e-12:
                b, _, r2 = _ols(y, x)
                beta_val = float(b[1])
        per_run.append({
            "run_id": rid,
            "n_dfa_states": int(run["n_dfa_states"]),
            "solved": bool(run.get("solved", False)),
            "beta": beta_val,
            "r2": r2,
            "n": len(pts),
        })

    log_c0 = np.array([r["log_c0"] for r in rows], dtype=float)
    log_fold = np.array([r["log_fold"] for r in rows], dtype=float)
    n_edges = np.array([_n_edges_from_key(r["key"]) for r in rows], dtype=int)
    tiers = sorted(set(int(v) for v in n_edges))

    beta, _, r2_pool = _ols(log_fold, log_c0)
    beta_tiers: list[tuple[int, float, float, int]] = []
    for ne in tiers:
        mask = n_edges == ne
        if int(mask.sum()) < min_fit or float(np.std(log_c0[mask])) < 1e-9:
            continue
        b_e, _, r2_e = _ols(log_fold[mask], log_c0[mask])
        beta_tiers.append((ne, float(b_e[1]), r2_e, int(mask.sum())))

    demos = _pick_demo_trajectories(
        rows,
        motif_prefix=motif_prefix,
        model=model,
        seed=seed,
        coloring=coloring,
        n_demos=n_demos,
        min_runs=min_demo_runs,
    )
    n_demo = max(1, len(demos))

    fig = plt.figure(figsize=(13.4, 6.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[0.88, 1.22], hspace=0.44)
    demo_row = outer[0].subgridspec(1, n_demo, wspace=0.32)
    stat_row = outer[1].subgridspec(1, 2, width_ratios=[1.72, 1.0], wspace=0.28)

    schema_insets: list[tuple[Any, str, bool]] = []
    demo_axes: list[Any] = []

    # --- demo panels (fig 19-style: trajectory + corner inset glyph) ---
    for j, demo in enumerate(demos):
        key = demo["key"]
        ne = int(demo["n_edges"])
        col = _EDGE_COUNT_COLORS.get(ne, "#888888")
        iters = demo["iters"]
        counts = demo["counts"]
        run_id = int(demo["run_id"])
        fold = float(counts[-1] / max(counts[0], EPS))
        rising = bool(counts[-1] >= counts[0])

        ax = fig.add_subplot(demo_row[0, j])
        ax.plot(iters, counts, color=col, lw=1.25, zorder=2)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.34 * (hi - lo))
        ax.set_title(
            f"{demo['label']}  r{run_id:02d}",
            fontsize=7.4, pad=3.5, color=col,
        )
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=6.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if j == 0:
            ax.set_ylabel("count", fontsize=7.0)
        ax.set_xlabel("iteration", fontsize=7.0)
        ax.text(
            0.97 if rising else 0.03,
            0.035,
            rf"$\times${fold:.2f}",
            transform=ax.transAxes,
            ha="right" if rising else "left",
            va="bottom",
            fontsize=6.4,
            color="0.20",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, pad=0.8),
            zorder=6,
        )
        inset = ax.inset_axes(
            [0.035, 0.58, 0.34, 0.36] if rising else [0.625, 0.58, 0.34, 0.36],
        )
        inset.set_axis_off()
        inset.patch.set_visible(False)
        schema_insets.append((inset, key, rising))
        demo_axes.append(ax)

    # --- scatter with per-tier regression lines ---
    ax_sc = fig.add_subplot(stat_row[0, 0])
    disp_idx = _subsample_stratified_indices(n_edges, max_pts=3500, seed=0)
    x_disp, y_disp = _jitter_display_coords(log_c0, log_fold, seed=0)
    for ne in tiers:
        mask = n_edges == ne
        col = _EDGE_COUNT_COLORS.get(ne, "#888888")
        show = mask & np.isin(np.arange(len(mask)), disp_idx)
        ax_sc.scatter(
            x_disp[show], y_disp[show], s=12, c=col, alpha=0.45,
            edgecolors="0.15", linewidths=0.25, zorder=3,
        )
        if int(mask.sum()) >= min_fit and float(np.std(log_c0[mask])) > 1e-9:
            b_e, _, _r2_e = _ols(log_fold[mask], log_c0[mask])
            x0, x1 = float(log_c0[mask].min()), float(log_c0[mask].max())
            x_line = np.linspace(x0, x1, 50)
            _outlined_regression_line(
                ax_sc, x_line, b_e[0] + b_e[1] * x_line, col, lw=1.45,
            )

    for i, (ne, b1, _r2_e, _n) in enumerate(beta_tiers):
        col = _EDGE_COUNT_COLORS.get(ne, "#888888")
        ax_sc.text(
            0.02,
            0.97 - 0.08 * i,
            rf"{ne}e $\beta={b1:+.2f}$",
            transform=ax_sc.transAxes,
            ha="left", va="top", fontsize=6.2, color=col,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=col, alpha=0.92, lw=0.5),
            zorder=8,
        )

    if len(log_c0) >= 3 and float(np.std(log_c0)) > 1e-12:
        x_line = np.linspace(float(log_c0.min()), float(log_c0.max()), 60)
        _outlined_regression_line(ax_sc, x_line, beta[0] + beta[1] * x_line, "0.12", lw=2.0)

    ax_sc.axhline(0.0, color="0.55", lw=0.7, ls="--", zorder=1)
    ax_sc.set_xlabel("log start count", fontsize=7.0)
    ax_sc.set_ylabel("log fold (end / start)", fontsize=7.0)
    ax_sc.set_title(
        rf"log fold vs log start  (pooled $\beta={beta[1]:+.2f}$, $R^2={r2_pool:.2f}$)",
        fontsize=7.4, pad=3.5,
    )
    ax_sc.tick_params(labelsize=6.0)
    ax_sc.grid(True, alpha=0.25)
    ax_sc.spines["top"].set_visible(False)
    ax_sc.spines["right"].set_visible(False)

    # --- beta vs DFA ---
    ax_dfa = fig.add_subplot(stat_row[0, 1])
    dfa_vals = [float(r["n_dfa_states"]) for r in per_run if np.isfinite(r.get("beta", float("nan")))]
    dfa_norm = plt.Normalize(vmin=min(dfa_vals), vmax=max(dfa_vals)) if dfa_vals else plt.Normalize(0, 1)
    _plot_beta_vs_dfa(
        per_run, ax=ax_dfa, dfa_cmap=plt.get_cmap("viridis"),
        dfa_norm=dfa_norm, target_we=target_we,
    )
    ax_dfa.set_title("per-run slope vs DFA", fontsize=7.4, pad=3.5)
    ax_dfa.tick_params(labelsize=6.0)
    ax_dfa.grid(True, alpha=0.25)

    finalize_grid_figure(
        fig,
        suptitle=(
            f"Weight motifs vs learning ({color_tag}, {motif_tag}; "
            f"start>={min_start}, n={len(series)} runs, {n_solved} solved)"
        ),
        suptitle_fontsize=10,
        top=0.90,
        bottom=0.09,
        left=0.06,
        right=0.98,
        hspace=0.44,
        wspace=0.28,
    )
    for inset, key, _rising in schema_insets:
        pseudo = edge_signed_hl_schema_pseudo(key)
        draw_edge_signed_hl_motif(
            inset, key,
            box=motif_schema_box(inset, pseudo, center_x=0.5, max_width=0.98),
        )

    save_figure(fig, out_path, dpi=150)
    print(f"wrote {out_path}")
    print(f"  pooled beta={beta[1]:+.3f}  R2={r2_pool:.3f}  n={len(log_c0)}")
    for ne, b, r2_e, n in beta_tiers:
        print(f"  {ne}e  beta={b:+.3f}  R2={r2_e:.3f}  n={n}")
    for demo in demos:
        fold = float(demo["counts"][-1] / max(demo["counts"][0], EPS))
        print(
            f"  demo {demo['label']}  {demo['key']}  r{demo['run_id']:02d}  "
            f"x{fold:.2f}  score={demo['score']:.3f}",
        )
    return out_path


def plot_all_runs_homogenization_summary(
    cache_path: Path,
    out_path: Path,
    *,
    min_start: int,
    rebuild_cache: bool,
    model: str,
    seed: int,
    max_snaps_per_run: int,
    coloring: str = "edge_sign",
    motif_prefix: str = "T|",
) -> Path:
    return plot_all_runs_homogenization_board(
        cache_path,
        out_path,
        min_start=min_start,
        rebuild_cache=rebuild_cache,
        model=model,
        seed=seed,
        max_snaps_per_run=max_snaps_per_run,
        coloring=coloring,
        motif_prefix=motif_prefix,
    )


def _plot_beta_vs_dfa(
    per_run: list[dict],
    *,
    ax: plt.Axes,
    dfa_cmap,
    dfa_norm,
    target_we: float,
    highlight_run_id: int | None = None,
) -> None:
    from matplotlib.lines import Line2D

    xs, ys, cs, solved_flags, rids = [], [], [], [], []
    for row in per_run:
        beta = row.get("beta")
        if beta is None or not np.isfinite(beta):
            continue
        xs.append(float(row["n_dfa_states"]))
        ys.append(float(beta))
        cs.append(dfa_cmap(dfa_norm(float(row["n_dfa_states"]))))
        solved_flags.append(bool(row.get("solved", False)))
        rids.append(int(row["run_id"]))

    if not xs:
        ax.text(0.5, 0.5, "no per-run slopes", ha="center", va="center", transform=ax.transAxes)
        return

    x_arr = np.array(xs, dtype=float)
    y_arr = np.array(ys, dtype=float)
    rid_arr = np.array(rids, dtype=int)
    # x jitter so runs that share a DFA size are readable (fit uses true x).
    rng = np.random.default_rng(0)
    x_plot = x_arr + rng.uniform(-0.9, 0.9, size=len(x_arr))

    for x, y, c, solved in zip(x_plot, ys, cs, solved_flags):
        if solved:
            ax.scatter(x, y, s=36, c=[c], alpha=0.92, edgecolors="0.15", linewidths=0.45, zorder=3)
        else:
            ax.scatter(
                x, y, s=42, facecolors="none", edgecolors=c, linewidths=1.3, alpha=0.95, zorder=3,
            )

    if highlight_run_id is not None:
        hi = rid_arr == int(highlight_run_id)
        if np.any(hi):
            ax.scatter(
                x_plot[hi], y_arr[hi], s=120, facecolors="none", edgecolors="#c0392b",
                linewidths=2.0, zorder=5, marker="*",
            )
            ax.annotate(
                rf"r{int(highlight_run_id):02d}",
                (float(x_plot[hi][0]), float(y_arr[hi][0])),
                xytext=(6, 4), textcoords="offset points", fontsize=6.5, color="#c0392b",
                zorder=6,
            )

    if len(x_arr) >= 3 and np.std(x_arr) > 1e-12:
        X = np.column_stack([np.ones(len(x_arr)), x_arr])
        b, *_ = np.linalg.lstsq(X, y_arr, rcond=None)
        x_line = np.linspace(float(x_arr.min()), float(x_arr.max()), 100)
        y_line = b[0] + b[1] * x_line
        ax.plot(x_line, y_line, color="white", lw=3.4, zorder=1.9, solid_capstyle="round")
        ax.plot(x_line, y_line, color="#c0392b", lw=1.45, zorder=2, solid_capstyle="round")
        ss_res = float(np.sum((y_arr - (b[0] + b[1] * x_arr)) ** 2))
        ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
        meta_r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
        # Avoid "+ +" when the DFA slope is positive: use signed terms only.
        ax.set_title(
            rf"meta: $\beta$ = ${b[0]:+.2f}{b[1]:+.3f}\cdot\mathrm{{DFA}}$"
            rf"  ($R^2$={meta_r2:.2f})",
            fontsize=8, pad=4,
        )

    ax.axhline(0.0, color="0.55", lw=0.7, ls="--", zorder=1)
    ax.set_xlabel("DFA states", fontsize=8)
    ax.set_ylabel(r"$\beta$ (log fold ~ log start)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    x_pad = max(1.0, 0.04 * (float(x_arr.max()) - float(x_arr.min()) + 1.0))
    ax.set_xlim(float(x_arr.min()) - x_pad, float(x_arr.max()) + x_pad)
    # Keep legend clear of the cloud (low-DFA / near-zero β corner is crowded).
    handles = [
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor="0.35", markeredgecolor="0.15", markersize=7,
            label=rf"solved (best WE $\leq$ {100 * target_we:.0f}%)",
        ),
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor="none", markeredgecolor="0.35",
            markeredgewidth=1.4, markersize=7,
            label="unsolved",
        ),
    ]
    ax.legend(
        handles=handles, loc="upper left", fontsize=6.5, frameon=True,
        fancybox=False, edgecolor="0.8", framealpha=0.92,
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
    motif_prefix: str = "T|",
) -> Path:
    if rebuild_cache or not cache_path.is_file():
        collect_all_runs_motif_cache(
            cache_path,
            min_start=min_start,
            max_snaps_per_run=max_snaps_per_run,
            model=model,
            seed=seed,
            coloring=coloring,
            motif_prefix=motif_prefix,
        )
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    rows = payload["fold_rows"]
    series = sorted(payload["run_series"], key=_run_sort_key)
    target_we = float(payload.get("target_we", TARGET_WE))
    n_solved = int(payload.get("n_solved", sum(1 for r in series if r.get("solved"))))
    coloring_used = str(payload.get("coloring", coloring))
    color_tag = "edge-sign" if coloring_used == "edge_sign" else "Dale-node"
    motif_used = str(payload.get("motif_prefix", motif_prefix))
    motif_tag = "dyad" if motif_used.startswith("D") else "triad"
    # Dyads have few classes per run; allow smaller within-#edges fits.
    min_fit = 3 if motif_used.startswith("D") else 8

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
    # Room for per-panel β chips outside axes + centered meta β panel.
    fig_w = 1.65 * ncol + 1.8
    fig_h = 2.15 * nrow + 2.8
    fig = plt.figure(figsize=(fig_w, fig_h))
    height_ratios = [1.0] * nrow + [0.95]
    gs = fig.add_gridspec(nrow + 1, ncol, height_ratios=height_ratios, hspace=0.62, wspace=0.58)

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
        n_e = np.array([_n_edges_from_key(p["key"]) for p in pts], dtype=int)
        point_colors = [_EDGE_COUNT_COLORS.get(int(k), "#888888") for k in n_e]
        # Display-only jitter: integer start counts stack on log x. Fit uses true x.
        rng = np.random.default_rng(rid)
        x_disp = x + rng.normal(0.0, 0.045, size=len(x))
        ax.scatter(x_disp, y, s=9, c=point_colors, alpha=0.70, edgecolors="none", zorder=3)

        def _outlined_line(xs, ys, color, *, lw=1.35, z=5):
            ax.plot(xs, ys, color="white", lw=lw + 2.2, zorder=z - 0.1, solid_capstyle="round")
            ax.plot(xs, ys, color="0.05", lw=lw + 0.55, zorder=z - 0.05, solid_capstyle="round", alpha=0.35)
            ax.plot(xs, ys, color=color, lw=lw, zorder=z, solid_capstyle="round", alpha=0.98)

        beta_lines: list[tuple[str, float, str]] = []
        # Pooled regression (black) — confounded by #edges tiers.
        if len(x) >= 3 and np.std(x) > 1e-12:
            beta, _, r2 = ols_fn(y, x)
            beta_val = float(beta[1])
            x_line = np.linspace(float(x.min()), float(x.max()), 50)
            _outlined_line(x_line, beta[0] + beta[1] * x_line, "0.12", lw=1.25, z=4)
            beta_lines.append(("pool", beta_val, "0.12"))
        # Within-#edges regressions — span only that group's observed x-range.
        for ne in sorted(set(int(v) for v in n_e)):
            mask = n_e == ne
            if int(mask.sum()) < min_fit or float(np.std(x[mask])) < 1e-9:
                continue
            b_e, _, _ = ols_fn(y[mask], x[mask])
            x0, x1 = float(x[mask].min()), float(x[mask].max())
            x_e = np.linspace(x0, x1, 40)
            col = _EDGE_COUNT_COLORS.get(ne, "#888888")
            _outlined_line(x_e, b_e[0] + b_e[1] * x_e, col, lw=1.45, z=5)
            beta_lines.append((f"{ne}e", float(b_e[1]), col))

        per_run.append({
            "run_id": rid, "n_dfa_states": n_dfa, "solved": solved,
            "beta": beta_val, "r2": r2, "n": len(x),
        })
        ax.axhline(0.0, color="0.55", lw=0.6, ls="--", zorder=1)
        ax.set_xlim(*x_lim)
        # Per-panel y-scale (shared ylim hid within-tier structure).
        y_span = float(y.max() - y.min())
        y_pad = 0.14 * max(y_span, 0.25)
        ax.set_ylim(float(y.min()) - y_pad, float(y.max()) + y_pad)

        # β chips on the RIGHT (axes coords) — title is short/centered; left holds
        # the dense 5e/6e cloud. White chips keep labels readable and separated.
        # β chips OUTSIDE axes (right of panel) — never on data / fit lines.
        if beta_lines:
            n_chip = len(beta_lines)
            step = 0.105 if n_chip <= 5 else 0.092
            for i, (name, b, col) in enumerate(beta_lines):
                ax.text(
                    1.02, 0.98 - step * i, f"{name} {b:+.2f}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=4.6,
                    color=col, fontweight="bold" if name == "pool" else "normal",
                    zorder=11, clip_on=False,
                    bbox=dict(
                        boxstyle="round,pad=0.10", fc="white", ec=col,
                        alpha=0.96, lw=0.6,
                    ),
                )

        solve_tag = "" if solved else " · U"
        ax.set_title(f"r{rid:02d} DFA={n_dfa}{solve_tag}", fontsize=6.5, pad=2)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx // ncol == nrow - 1:
            ax.set_xlabel("log start", fontsize=6.5)
        else:
            ax.tick_params(labelbottom=False)
        # Per-panel ylims → keep y tick labels on every panel (not just col 0).
        if idx % ncol == 0:
            ax.set_ylabel("log fold", fontsize=6.5)
        ax.tick_params(labelleft=True, labelsize=5.5)

    for j in range(len(series), nrow * ncol):
        fig.add_subplot(gs[j // ncol, j % ncol]).axis("off")

    # Meta β panel: centered over a few columns, not stretched across the whole grid.
    meta_span = min(3, ncol)
    meta_c0 = (ncol - meta_span) // 2
    ax_beta = fig.add_subplot(gs[nrow, meta_c0 : meta_c0 + meta_span])
    _plot_beta_vs_dfa(
        per_run, ax=ax_beta, dfa_cmap=dfa_cmap, dfa_norm=dfa_norm, target_we=target_we,
    )
    ax_beta.text(
        0.5, -0.28,
        "Points by #edges. Black = pooled β; colored = within-#edges β (often flips).\n"
        "Stripes: integer start counts on log x. Points jittered for display only.  ·U = unsolved.",
        transform=ax_beta.transAxes, ha="center", va="top", fontsize=6.0, color="0.25",
    )

    from matplotlib.lines import Line2D
    edge_handles = [
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=col, markeredgecolor="none", markersize=6,
            label=f"{ne} edges",
        )
        for ne, col in sorted(_EDGE_COUNT_COLORS.items())
    ]
    edge_handles.append(Line2D([0], [0], color="0.15", lw=1.2, label="pooled fit"))
    fig.legend(
        handles=edge_handles, loc="center left",
        bbox_to_anchor=(0.97, 0.55), fontsize=6.5,
        frameon=True, fancybox=False, edgecolor="0.8",
        title="#edges", title_fontsize=7,
    )

    r2_vals = [r["r2"] for r in per_run if np.isfinite(r["r2"])]
    med_r2 = float(np.median(r2_vals)) if r2_vals else float("nan")
    finalize_grid_figure(
        fig,
        suptitle=(
            f"mixed DFA {model} ({color_tag}, {motif_tag}): start vs fold by #edges "
            f"(DFA order, post-init, start>={min_start}, "
            f"n={len(series)} runs, {n_solved} solved, median pooled R$^2$={med_r2:.2f})"
        ),
        top=0.945,
        bottom=0.08,
        left=0.05,
        right=0.90,
        hspace=0.62,
        wspace=0.58,
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
            f"{model} {motif_tag} compression slope vs DFA ({color_tag}; "
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
