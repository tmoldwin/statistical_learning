"""Canonical mixed-DFA motif census figures (mixed_vocab_dfa_ns)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from viz.plot_layout import finalize_grid_figure, save_figure
from viz.compare import mixed_dfa_motif_all_runs as mar

# Motif census boards live under motifs/, not trajectories/ (those are PCA/rollouts).
MOTIF_DIR = REPO / "experiments/comparisons/mixed_vocab_dfa_ns/motifs"

DEFAULT_JSON = MOTIF_DIR / "r43_motif_colored_all.json"
DEFAULT_EDGES_JSON = MOTIF_DIR / "r43_motif_counts_over_learning.json"
DEFAULT_OUT = MOTIF_DIR / "r43_motif_counts_raw_over_learning.png"
MIN_START = 20
EPS = 0.5
ALL_RUNS_CACHE = MOTIF_DIR / "mixed_dfa_rnn_motif_fold_all_runs.json"
ALL_RUNS_OUT = MOTIF_DIR / "mixed_dfa_rnn_motif_counts_raw_over_learning.png"
ALL_RUNS_BETA_OUT = MOTIF_DIR / "mixed_dfa_rnn_motif_fold_beta_vs_dfa.png"
R43_RNN_JSON = MOTIF_DIR / "r43_rnn_motif_edge_signed_all.json"
R43_RNN_EDGES_JSON = MOTIF_DIR / "r43_rnn_motif_counts_over_learning.json"
R43_RNN_OUT = MOTIF_DIR / "r43_rnn_motif_counts_raw_over_learning.png"
R43_RNN_UNSIGNED_JSON = MOTIF_DIR / "r43_rnn_motif_unsigned_all.json"
R43_RNN_UNSIGNED_OUT = MOTIF_DIR / "r43_rnn_motif_unsigned_over_learning.png"
R43_RNN_LOLLIPOP_OUT = MOTIF_DIR / "r43_rnn_motif_lollipop_before_after.png"
R43_RNN_SUMMARY_OUT = MOTIF_DIR / "r43_rnn_motif_homogenization_summary.png"
R43_RNN_BOARD_OUT = MOTIF_DIR / "r43_rnn_motif_board.png"
R43_RNN_DYAD_OUT = MOTIF_DIR / "r43_rnn_dyad_counts_raw_over_learning.png"
R43_RNN_DYAD_LOLLIPOP_OUT = MOTIF_DIR / "r43_rnn_dyad_lollipop_before_after.png"
R43_RNN_DYAD_SUMMARY_OUT = MOTIF_DIR / "r43_rnn_dyad_homogenization_summary.png"
ALL_RUNS_DYAD_CACHE = MOTIF_DIR / "mixed_dfa_rnn_dyad_fold_all_runs.json"
ALL_RUNS_DYAD_OUT = MOTIF_DIR / "mixed_dfa_rnn_dyad_counts_raw_over_learning.png"
ALL_RUNS_DYAD_BETA_OUT = MOTIF_DIR / "mixed_dfa_rnn_dyad_fold_beta_vs_dfa.png"
ALL_RUNS_SUMMARY_OUT = MOTIF_DIR / "mixed_dfa_rnn_motif_homogenization_summary.png"
ALL_RUNS_DYAD_SUMMARY_OUT = MOTIF_DIR / "mixed_dfa_rnn_dyad_homogenization_summary.png"
MODEL = "rnn"
COLORING = "edge_sign"
SEED = 1
MAX_SNAPS_PER_RUN = 8
SINGLE_RUN_ID = 43


def _single_run_paths(run_id: int) -> tuple[Path, Path, Path, Path, Path, Path]:
    """Census JSON, edges JSON, board, factor panels, factor & pattern regressions."""
    stem = f"r{run_id:02d}_rnn"
    return (
        MOTIF_DIR / f"{stem}_motif_edge_signed_all.json",
        MOTIF_DIR / f"{stem}_motif_counts_over_learning.json",
        MOTIF_DIR / f"{stem}_motif_board.png",
        MOTIF_DIR / f"{stem}_motif_factor_panels.png",
        MOTIF_DIR / f"{stem}_motif_factor_regressions.png",
        MOTIF_DIR / f"{stem}_motif_pattern_regressions.png",
    )


def _hypothesis_out_path(run_id: int) -> Path:
    return MOTIF_DIR / f"r{run_id:02d}_rnn_motif_hypothesis_regressions.png"


def _story_board_out_path(run_id: int) -> Path:
    return MOTIF_DIR / f"r{run_id:02d}_rnn_motif_story_board.png"


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    return beta, pred, r2


def collect_single_run_edge_signed_census(
    *,
    run_id: int = SINGLE_RUN_ID,
    model: str = "rnn",
    seed: int = SEED,
    max_snaps: int = 60,
    colored_json: Path = R43_RNN_JSON,
    edges_json: Path = R43_RNN_EDGES_JSON,
) -> tuple[Path, Path]:
    """Build edge-sign HL motif census over learning for one unconstrained run."""
    from experiment import checkpoint_path
    from rnn.learning_snaps import list_learning_snaps
    from viz.compare.mixed_dfa_motif_all_runs import (
        _dominant_session,
        _snap_iteration,
        _subsample_snaps,
    )
    from viz.weight_structure import compute_weight_edge_signed_hl_motif_counts

    task = f"mixeddfa_r{run_id:02d}_ns"
    ckpt = checkpoint_path(task, model, seed=seed)
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    snaps = [
        p for p in _dominant_session(list_learning_snaps(ckpt))
        if _snap_iteration(p) > 0
    ]
    if len(snaps) < 2:
        raise RuntimeError(f"r{run_id:02d}: need >=2 post-init learning snaps")
    snaps = _subsample_snaps(snaps, max_snaps)

    colored_rows: list[dict] = []
    edge_rows: list[dict] = []
    for snap in snaps:
        d = np.load(snap, allow_pickle=True)
        out = compute_weight_edge_signed_hl_motif_counts(
            d["weights_hidden_to_hidden"], mode="mean",
        )
        it = (
            int(d["learning_snap_iteration"])
            if "learning_snap_iteration" in d.files
            else _snap_iteration(snap)
        )
        we = (
            float(d["learning_snap_word_err"])
            if "learning_snap_word_err" in d.files
            else float("nan")
        )
        colored_rows.append({
            "it": it,
            "we": we,
            "dyads_conn": float(out["dyads_conn"]),
            "triples_conn": float(out["triples_conn"]),
            "cnt": out["cnt"],
        })
        edge_rows.append({"it": it, "edges": float(out["edges"])})
        n_t = sum(1 for k in out["cnt"] if k.startswith("T|"))
        print(f"  snap it={it}  edges={int(out['edges'])}  T-classes={n_t}", flush=True)

    colored_json.parent.mkdir(parents=True, exist_ok=True)
    colored_json.write_text(json.dumps(colored_rows, indent=2), encoding="utf-8")
    edges_json.write_text(json.dumps(edge_rows, indent=2), encoding="utf-8")
    print(f"wrote {colored_json}")
    print(f"wrote {edges_json}")
    return colored_json, edges_json


def plot_raw_counts_over_learning(
    json_path: Path = DEFAULT_JSON,
    out_path: Path = DEFAULT_OUT,
    *,
    edges_json_path: Path = DEFAULT_EDGES_JSON,
    min_start: int = MIN_START,
    title: str | None = None,
    motif_prefix: str = "T|",
) -> Path:
    snaps = json.loads(json_path.read_text(encoding="utf-8"))
    edge_snaps = json.loads(edges_json_path.read_text(encoding="utf-8"))
    if [s["it"] for s in snaps] != [s["it"] for s in edge_snaps]:
        raise ValueError("colored census and edge-count snapshots must share iteration grid")
    iters = np.array([s["it"] for s in snaps], dtype=float)
    n_edges = np.array([s["edges"] for s in edge_snaps], dtype=float)
    is_dyad = motif_prefix.startswith("D")
    conn_key = "dyads_conn" if is_dyad else "triples_conn"
    conn_total = np.array([s.get(conn_key, float("nan")) for s in snaps], dtype=float)
    motif_keys = sorted(k for k in snaps[0]["cnt"] if k.startswith(motif_prefix))
    if not motif_keys:
        raise ValueError(f"no keys with prefix {motif_prefix!r} in {json_path}")

    def mat(keys: list[str]) -> np.ndarray:
        return np.array([[s["cnt"].get(k, 0) for k in keys] for s in snaps], dtype=float)

    T = mat(motif_keys)
    t0 = np.maximum(T[0], EPS)
    t1 = np.maximum(T[-1], EPS)
    logt0 = np.log(t0)
    fold = t1 / t0
    log_fold = np.log(fold)

    n_e = np.array([mar._n_edges_from_key(k) for k in motif_keys], dtype=int)
    edge_colors = {
        int(ne): mar._EDGE_COUNT_COLORS.get(int(ne), "#888888")
        for ne in np.unique(n_e)
    }

    mask = t0 >= min_start
    x_sc = logt0[mask]
    y_sc = log_fold[mask]
    n_e_sc = n_e[mask]
    beta, _, r2 = _ols(y_sc, x_sc)
    min_fit = 3 if is_dyad else 8

    def _outlined_line(ax, xs, ys, color, *, lw=1.35, z=5, ls="-"):
        ax.plot(xs, ys, color="white", lw=lw + 2.2, zorder=z - 0.1,
                solid_capstyle="round", ls=ls)
        ax.plot(xs, ys, color="0.05", lw=lw + 0.55, zorder=z - 0.05,
                solid_capstyle="round", alpha=0.35, ls=ls)
        ax.plot(xs, ys, color=color, lw=lw, zorder=z,
                solid_capstyle="round", alpha=0.98, ls=ls)

    slog = np.log(np.maximum(T, EPS))
    sd = slog.std(axis=1)
    tiers = sorted(edge_colors)
    n_tier = len(tiers)
    mid = n_tier // 2
    rng = np.random.default_rng(0)
    beta_lines: list[tuple[str, float, str]] = [("pool", float(beta[1]), "0.12")]

    fig = plt.figure(figsize=(max(8.0, 3.2 * n_tier + 2.0), 10.8))
    outer = fig.add_gridspec(4, 1, height_ratios=[1.20, 1.15, 1.00, 0.90], hspace=0.55)
    gs_fold = outer[0].subgridspec(1, n_tier, wspace=0.38)
    gs_traj = outer[1].subgridspec(1, n_tier, wspace=0.38)
    gs_sd = outer[2].subgridspec(1, n_tier, wspace=0.38)
    gs_foot = outer[3].subgridspec(1, 3, wspace=0.36)

    def _style(ax):
        ax.tick_params(labelsize=6.2)
        ax.grid(True, alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j, ne in enumerate(tiers):
        ax = fig.add_subplot(gs_fold[0, j])
        m = n_e_sc == ne
        x, y = x_sc[m], y_sc[m]
        col = edge_colors[ne]
        # Label points for dyads (few classes).
        labels = [motif_keys[i] for i, ok in enumerate(mask) if ok and n_e[i] == ne]
        x_disp = x + rng.normal(0.0, 0.012 if is_dyad else 0.028, size=len(x))
        ax.scatter(x_disp, y, c=col, s=48 if is_dyad else 16, alpha=0.85,
                   edgecolors="0.25", linewidths=0.35, zorder=3)
        if is_dyad:
            for xi, yi, lab in zip(x_disp, y, labels):
                short = lab.replace("D|", "")
                ax.annotate(short, (xi, yi), textcoords="offset points",
                            xytext=(4, 3), fontsize=5.5, color="0.2")
        b1, r2_e = float("nan"), float("nan")
        if int(m.sum()) >= min_fit and float(np.std(x)) > 1e-9:
            b_e, _, r2_e = _ols(y, x)
            b1 = float(b_e[1])
            x_line = np.linspace(float(x.min()), float(x.max()), 40)
            _outlined_line(ax, x_line, b_e[0] + b_e[1] * x_line, col, lw=1.55, z=5)
            beta_lines.append((f"{ne}e", b1, col))
        ax.axhline(0.0, color="0.55", lw=0.65, ls="--", zorder=1)
        if len(y):
            y_span = float(y.max() - y.min())
            y_pad = 0.18 * max(y_span, 0.15)
            ax.set_ylim(float(y.min()) - y_pad, float(y.max()) + y_pad)
            x_span = float(x.max() - x.min())
            x_pad = 0.15 * max(x_span, 0.15)
            ax.set_xlim(float(x.min()) - x_pad, float(x.max()) + x_pad)
        ax.set_title(f"{ne}e  β={b1:+.2f}  $R^2$={r2_e:.2f}", fontsize=7.4,
                     color=col, pad=3)
        _style(ax)
        if j == 0:
            ax.set_ylabel("log fold", fontsize=7.5)
        if j == mid:
            ax.set_xlabel("log start", fontsize=7.5)

    for j, ne in enumerate(tiers):
        ax = fig.add_subplot(gs_traj[0, j])
        col = edge_colors[ne]
        tm = n_e == ne
        series = np.maximum(T[:, tm], EPS)
        for k in range(series.shape[1]):
            ax.plot(iters, series[:, k], color=col, lw=1.1 if is_dyad else 0.55,
                    alpha=0.55 if is_dyad else 0.28, zorder=2)
            if is_dyad:
                keys_ne = [motif_keys[i] for i, ok in enumerate(tm) if ok]
                ax.text(iters[-1], series[-1, k], " " + keys_ne[k].replace("D|", ""),
                        fontsize=5.5, color=col, va="center")
        med_e = np.median(series, axis=1)
        _outlined_line(ax, iters, med_e, col, lw=1.55, z=5)
        ax.set_yscale("log")
        ymin, ymax = float(series.min()), float(series.max())
        ax.set_ylim(max(ymin / 1.35, EPS), ymax * 1.35)
        fold_med = float(med_e[-1] / max(med_e[0], EPS))
        ax.set_title(f"{ne}e  med ×{fold_med:.2f}", fontsize=7.4, color=col, pad=3)
        _style(ax)
        if j == 0:
            ax.set_ylabel("count", fontsize=7.5)
        if j == mid:
            ax.set_xlabel("iteration", fontsize=7.5)

    for j, ne in enumerate(tiers):
        ax = fig.add_subplot(gs_sd[0, j])
        col = edge_colors[ne]
        sd_e = slog[:, n_e == ne].std(axis=1)
        _outlined_line(ax, iters, sd_e, col, lw=1.55, z=5)
        y0, y1 = float(sd_e.min()), float(sd_e.max())
        pad = 0.12 * max(y1 - y0, 0.05)
        ax.set_ylim(y0 - pad, y1 + pad)
        ax.set_title(f"{ne}e  sd {sd_e[0]:.2f}→{sd_e[-1]:.2f}", fontsize=7.4,
                     color=col, pad=3)
        _style(ax)
        if j == 0:
            ax.set_ylabel("sd log count", fontsize=7.5)
        if j == mid:
            ax.set_xlabel("iteration", fontsize=7.5)

    ax_edge = fig.add_subplot(gs_foot[0, 0])
    ax_spr = fig.add_subplot(gs_foot[0, 1])
    ax_conn = fig.add_subplot(gs_foot[0, 2])

    edge_fold = float(n_edges[-1] / n_edges[0])
    ax_edge.plot(iters, n_edges, color="#4c78a8", lw=1.6)
    ax_edge.set_xlabel("iteration", fontsize=7.5)
    ax_edge.set_ylabel("count", fontsize=7.5)
    ax_edge.set_title(
        f"strong |W_hh| edges  ({int(n_edges[0])}→{int(n_edges[-1])}, ×{edge_fold:.2f})",
        fontsize=8, pad=3,
    )
    _style(ax_edge)

    for ne in tiers:
        med_e = np.exp(np.median(slog[:, n_e == ne], axis=1))
        _outlined_line(ax_spr, iters, med_e, edge_colors[ne], lw=1.55, z=5)
    ax_spr.set_yscale("log")
    ax_spr.set_xlabel("iteration", fontsize=7.5)
    ax_spr.set_ylabel("count", fontsize=7.5)
    ax_spr.set_title("median count by #edges", fontsize=8, pad=3)
    _style(ax_spr)

    if np.all(np.isfinite(conn_total)):
        conn_fold = float(conn_total[-1] / conn_total[0])
        ax_conn.plot(iters, conn_total, color="#2ca02c", lw=1.6)
        kind = "dyad" if is_dyad else "triad"
        ax_conn.set_title(
            f"connected {kind} instances  "
            f"({int(conn_total[0])}→{int(conn_total[-1])}, ×{conn_fold:.2f})",
            fontsize=8, pad=3,
        )
    else:
        ax_conn.text(0.5, 0.5, "no connected-count series", ha="center", va="center",
                     transform=ax_conn.transAxes, fontsize=8)
        ax_conn.set_title("connected instances", fontsize=8, pad=3)
    ax_conn.set_xlabel("iteration", fontsize=7.5)
    ax_conn.set_ylabel("count", fontsize=7.5)
    _style(ax_conn)

    kind = "dyad" if is_dyad else "triad"
    finalize_grid_figure(
        fig,
        suptitle=(
            title
            or f"r43: {kind} class counts vs training (by #edges)"
        ) + f"   pooled β={beta[1]:+.2f}, $R^2$={r2:.2f}, n={int(mask.sum())}",
        top=0.935,
        bottom=0.055,
        left=0.055,
        right=0.985,
        hspace=0.50,
        wspace=0.32,
    )
    save_figure(fig, out_path, dpi=160)
    print(f"wrote {out_path}")
    print(f"log start vs log fold pooled R2={r2:.3f}  n={int(mask.sum())}")
    for name, b, _col in beta_lines:
        print(f"  {name:5s}  beta={b:+.3f}")
    print(
        f"edges {int(n_edges[0])} -> {int(n_edges[-1])}  "
        f"{conn_key} {conn_total[0]:.0f} -> {conn_total[-1]:.0f}"
    )
    print(f"sd log start/end {float(sd[0]):.2f} {float(sd[-1]):.2f}")
    return out_path


def plot_single_run_homogenization_summary(
    json_path: Path = R43_RNN_JSON,
    out_path: Path = R43_RNN_SUMMARY_OUT,
    *,
    min_start: int = MIN_START,
    title: str | None = None,
    motif_prefix: str = "T|",
) -> Path:
    """Compact single-run homogenization summary from edge-sign census JSON."""
    snaps = json.loads(json_path.read_text(encoding="utf-8"))
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
            int(mar._n_edges_from_key(key)),
        ))
    if not rows:
        raise ValueError(f"no motifs with start>={min_start} in {json_path}")
    log_c0 = np.array([r[0] for r in rows], dtype=float)
    log_fold = np.array([r[1] for r in rows], dtype=float)
    n_edges = np.array([r[2] for r in rows], dtype=int)
    kind = "dyad" if motif_prefix.startswith("D") else "triad"
    min_fit = 3 if kind == "dyad" else 8
    return mar.plot_homogenization_summary(
        log_c0, log_fold, n_edges, out_path,
        title=title or (
            f"r{SINGLE_RUN_ID:02d} rnn (edge-sign, {kind}): homogenization summary "
            f"(iter {int(snaps[0]['it'])}→{int(snaps[-1]['it'])}, start>={min_start})"
        ),
        min_fit=min_fit,
    )


def plot_r43_single_run_board(
    json_path: Path | None = None,
    cache_path: Path = ALL_RUNS_CACHE,
    out_path: Path | None = None,
    *,
    min_start: int = MIN_START,
    motif_prefix: str = "T|",
    run_id: int | None = None,
) -> Path:
    """Single-run motif board; defaults to highest-DFA exemplar from the fold cache."""
    if run_id is None:
        run_id, n_dfa = mar.pick_highest_dfa_exemplar(cache_path)
        print(f"board exemplar: r{run_id:02d} ({n_dfa} DFA states)", flush=True)
    else:
        n_dfa = mar.lookup_run_dfa(cache_path, run_id)

    default_json, default_edges, default_board, _, _ = _single_run_paths(run_id)
    json_path = json_path or default_json
    out_path = out_path or default_board

    if not json_path.is_file():
        colored, _edges = collect_single_run_edge_signed_census(
            run_id=run_id,
            model="rnn",
            seed=SEED,
            colored_json=json_path,
            edges_json=default_edges,
        )
        json_path = colored
    return mar.plot_single_run_motif_board(
        json_path,
        cache_path,
        out_path,
        run_id=run_id,
        min_start=min_start,
        motif_prefix=motif_prefix,
        n_dfa_states=n_dfa,
    )


def plot_single_run_factor_panels(
    json_path: Path | None = None,
    cache_path: Path = ALL_RUNS_CACHE,
    out_path: Path | None = None,
    *,
    min_start: int = MIN_START,
    motif_prefix: str = "T|",
    run_id: int | None = None,
    min_cell: int = 8,
) -> Path:
    """Structural factor panels; defaults to highest-DFA exemplar."""
    if run_id is None:
        run_id, n_dfa = mar.pick_highest_dfa_exemplar(cache_path)
        print(f"factor-panel exemplar: r{run_id:02d} ({n_dfa} DFA states)", flush=True)
    else:
        n_dfa = mar.lookup_run_dfa(cache_path, run_id)

    default_json, default_edges, _board, default_factors, _, _ = _single_run_paths(run_id)
    json_path = json_path or default_json
    out_path = out_path or default_factors

    if not json_path.is_file():
        colored, _edges = collect_single_run_edge_signed_census(
            run_id=run_id,
            model="rnn",
            seed=SEED,
            colored_json=json_path,
            edges_json=default_edges,
        )
        json_path = colored
    return mar.plot_motif_factor_panel_analysis(
        json_path,
        out_path,
        run_id=run_id,
        min_start=min_start,
        motif_prefix=motif_prefix,
        n_dfa_states=n_dfa,
        min_cell=min_cell,
    )


def plot_single_run_factor_regressions(
    json_path: Path | None = None,
    cache_path: Path = ALL_RUNS_CACHE,
    out_path: Path | None = None,
    *,
    min_start: int = MIN_START,
    motif_prefix: str = "T|",
    run_id: int | None = None,
) -> Path:
    """log fold ~ #inh / #exc / inh fraction; defaults to highest-DFA exemplar."""
    if run_id is None:
        run_id, n_dfa = mar.pick_highest_dfa_exemplar(cache_path)
        print(f"factor-regression exemplar: r{run_id:02d} ({n_dfa} DFA states)", flush=True)
    else:
        n_dfa = mar.lookup_run_dfa(cache_path, run_id)

    default_json, default_edges, _board, _factors, default_regressions, _ = _single_run_paths(run_id)
    json_path = json_path or default_json
    out_path = out_path or default_regressions

    if not json_path.is_file():
        colored, _edges = collect_single_run_edge_signed_census(
            run_id=run_id,
            model="rnn",
            seed=SEED,
            colored_json=json_path,
            edges_json=default_edges,
        )
        json_path = colored
    return mar.plot_motif_factor_regressions(
        json_path,
        out_path,
        run_id=run_id,
        min_start=min_start,
        motif_prefix=motif_prefix,
        n_dfa_states=n_dfa,
    )


def plot_single_run_pattern_regressions(
    json_path: Path | None = None,
    cache_path: Path = ALL_RUNS_CACHE,
    out_path: Path | None = None,
    *,
    min_start: int = MIN_START,
    motif_prefix: str = "T|",
    run_id: int | None = None,
) -> Path:
    """Edge-pattern regressions stratified by #edges; defaults to highest-DFA exemplar."""
    if run_id is None:
        run_id, n_dfa = mar.pick_highest_dfa_exemplar(cache_path)
        print(f"pattern-regression exemplar: r{run_id:02d} ({n_dfa} DFA states)", flush=True)
    else:
        n_dfa = mar.lookup_run_dfa(cache_path, run_id)

    default_json, default_edges, _b, _f, _r, default_patterns = _single_run_paths(run_id)
    json_path = json_path or default_json
    out_path = out_path or default_patterns

    if not json_path.is_file():
        colored, _edges = collect_single_run_edge_signed_census(
            run_id=run_id,
            model="rnn",
            seed=SEED,
            colored_json=json_path,
            edges_json=default_edges,
        )
        json_path = colored
    return mar.plot_motif_pattern_regressions(
        json_path,
        out_path,
        run_id=run_id,
        min_start=min_start,
        motif_prefix=motif_prefix,
        n_dfa_states=n_dfa,
    )


def plot_single_run_hypothesis_regressions(
    json_path: Path | None = None,
    cache_path: Path = ALL_RUNS_CACHE,
    out_path: Path | None = None,
    *,
    min_start: int = MIN_START,
    motif_prefix: str = "T|",
    run_id: int | None = None,
) -> Path:
    """~14-hypothesis regression battery; defaults to highest-DFA exemplar."""
    if run_id is None:
        run_id, n_dfa = mar.pick_highest_dfa_exemplar(cache_path)
        print(f"hypothesis-battery exemplar: r{run_id:02d} ({n_dfa} DFA states)", flush=True)
    else:
        n_dfa = mar.lookup_run_dfa(cache_path, run_id)

    default_json, default_edges, *_rest = _single_run_paths(run_id)
    json_path = json_path or default_json
    out_path = out_path or _hypothesis_out_path(run_id)

    if not json_path.is_file():
        colored, _edges = collect_single_run_edge_signed_census(
            run_id=run_id,
            model="rnn",
            seed=SEED,
            colored_json=json_path,
            edges_json=default_edges,
        )
        json_path = colored
    return mar.plot_motif_hypothesis_regressions(
        json_path,
        out_path,
        run_id=run_id,
        min_start=min_start,
        motif_prefix=motif_prefix,
        n_dfa_states=n_dfa,
    )


def plot_single_run_story_board(
    json_path: Path | None = None,
    cache_path: Path = ALL_RUNS_CACHE,
    out_path: Path | None = None,
    *,
    min_start: int = MIN_START,
    motif_prefix: str = "T|",
    run_id: int | None = None,
) -> Path:
    """Four-panel compression story board; defaults to highest-DFA exemplar."""
    if run_id is None:
        run_id, n_dfa = mar.pick_highest_dfa_exemplar(cache_path)
        print(f"story-board exemplar: r{run_id:02d} ({n_dfa} DFA states)", flush=True)
    else:
        n_dfa = mar.lookup_run_dfa(cache_path, run_id)

    default_json, default_edges, *_rest = _single_run_paths(run_id)
    json_path = json_path or default_json
    out_path = out_path or _story_board_out_path(run_id)

    if not json_path.is_file():
        colored, _edges = collect_single_run_edge_signed_census(
            run_id=run_id,
            model="rnn",
            seed=SEED,
            colored_json=json_path,
            edges_json=default_edges,
        )
        json_path = colored
    return mar.plot_motif_story_board(
        json_path,
        out_path,
        run_id=run_id,
        min_start=min_start,
        motif_prefix=motif_prefix,
        n_dfa_states=n_dfa,
    )


def plot_lollipop_before_after(
    json_path: Path = R43_RNN_JSON,
    out_path: Path = R43_RNN_LOLLIPOP_OUT,
    *,
    min_start: int = MIN_START,
    title: str | None = None,
    motif_prefix: str = "T|",
) -> Path:
    """Per-motif start / end / Δ lollipops, ordered by #edges then end−start."""
    snaps = json.loads(json_path.read_text(encoding="utf-8"))
    c0 = snaps[0]["cnt"]
    c1 = snaps[-1]["cnt"]
    it0, it1 = int(snaps[0]["it"]), int(snaps[-1]["it"])
    keys = [k for k in c0 if str(k).startswith(motif_prefix)]
    is_dyad = motif_prefix.startswith("D")

    rows: list[tuple[int, float, float, float, str]] = []
    for key in keys:
        start = float(c0.get(key, 0.0))
        if start < min_start:
            continue
        end = float(c1.get(key, 0.0))
        ne = int(mar._n_edges_from_key(key))
        rows.append((ne, start, end, end - start, key))
    rows.sort(key=lambda r: (r[0], -r[3], -r[1], r[4]))

    tiers = sorted({r[0] for r in rows})
    by_ne: dict[int, list[tuple[int, float, float, float, str]]] = {ne: [] for ne in tiers}
    for row in rows:
        by_ne[row[0]].append(row)

    n_tier = len(tiers)
    counts = [len(by_ne[ne]) for ne in tiers]
    row_h = 0.42 if is_dyad else 0.052
    fig_h = max(4.5, row_h * sum(counts) + 2.2)
    fig = plt.figure(figsize=(10.2 if is_dyad else 9.6, fig_h))
    gs = fig.add_gridspec(
        n_tier, 3,
        height_ratios=[max(n, 3 if is_dyad else 6) for n in counts],
        hspace=0.32, wspace=0.28,
    )

    for i, ne in enumerate(tiers):
        block = by_ne[ne]
        col = mar._EDGE_COUNT_COLORS.get(ne, "#888888")
        n = len(block)
        ys = np.arange(n, dtype=float)
        starts = np.array([r[1] for r in block], dtype=float)
        ends = np.array([r[2] for r in block], dtype=float)
        deltas = np.array([r[3] for r in block], dtype=float)
        labels = [r[4].replace("D|", "").replace("T|", "") for r in block]
        y_lim = (n - 0.5, -0.5)

        ax_s = fig.add_subplot(gs[i, 0])
        ax_e = fig.add_subplot(gs[i, 1])
        ax_d = fig.add_subplot(gs[i, 2])

        for ax, vals in ((ax_s, starts), (ax_e, ends)):
            ax.hlines(ys, 0.0, vals, colors=col, lw=1.2 if is_dyad else 0.75, zorder=2)
            ax.scatter(vals, ys, s=36 if is_dyad else 9, c=col, zorder=3,
                       edgecolors="0.2", linewidths=0.35)
            ax.axvline(0.0, color="0.75", lw=0.5, zorder=1)
            xmax = float(max(starts.max(), ends.max()))
            ax.set_xlim(-0.02 * xmax, xmax * 1.06)
            ax.set_ylim(*y_lim)
            ax.tick_params(labelsize=6.5)
            ax.grid(True, axis="x", alpha=0.22)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)

        ax_d.axvline(0.0, color="0.35", lw=0.8, zorder=1)
        ax_d.hlines(ys, 0.0, deltas, colors=col, lw=1.2 if is_dyad else 0.75, zorder=2)
        ax_d.scatter(deltas, ys, s=36 if is_dyad else 9, c=col, zorder=3,
                     edgecolors="0.2", linewidths=0.35)
        dmax = float(np.max(np.abs(deltas))) if len(deltas) else 1.0
        ax_d.set_xlim(-1.08 * dmax, 1.08 * dmax)
        ax_d.set_ylim(*y_lim)
        ax_d.tick_params(labelleft=False, labelsize=6.5)
        ax_d.grid(True, axis="x", alpha=0.22)
        ax_d.spines["top"].set_visible(False)
        ax_d.spines["right"].set_visible(False)
        ax_d.spines["left"].set_visible(False)

        if is_dyad:
            ax_s.set_yticks(ys)
            ax_s.set_yticklabels(labels, fontsize=7)
            ax_e.tick_params(labelleft=False)
        else:
            ax_s.tick_params(labelleft=False)
            ax_e.tick_params(labelleft=False)
            ax_s.set_ylabel(f"{ne}e  n={n}", fontsize=8, color=col, fontweight="bold")

        if is_dyad:
            ax_s.set_ylabel(f"{ne}e  n={n}", fontsize=8, color=col, fontweight="bold")

        if i == 0:
            ax_s.set_title("start", fontsize=9, pad=4)
            ax_e.set_title("end", fontsize=9, pad=4)
            ax_d.set_title("end − start", fontsize=9, pad=4)
        if i == n_tier - 1:
            ax_s.set_xlabel("count", fontsize=8)
            ax_e.set_xlabel("count", fontsize=8)
            ax_d.set_xlabel("Δ count", fontsize=8)
        else:
            ax_s.tick_params(labelbottom=False)
            ax_e.tick_params(labelbottom=False)
            ax_d.tick_params(labelbottom=False)

    kind = "dyad" if is_dyad else "triad"
    finalize_grid_figure(
        fig,
        suptitle=title or (
            f"r{SINGLE_RUN_ID:02d} rnn (edge-sign): {kind} lollipops  "
            f"iter {it0}→{it1}, start>={min_start}, n={len(rows)}  "
            f"(order: #edges, end−start)"
        ),
        top=0.94 if is_dyad else 0.965,
        bottom=0.08 if is_dyad else 0.035,
        left=0.16 if is_dyad else 0.07,
        right=0.985,
        hspace=0.32,
        wspace=0.28,
    )
    save_figure(fig, out_path, dpi=140)
    print(f"wrote {out_path}")
    for ne in tiers:
        block = by_ne[ne]
        d = np.array([r[3] for r in block], dtype=float)
        print(
            f"  {ne}e  n={len(block):3d}  "
            f"median d={float(np.median(d)):+.1f}  "
            f"mean d={float(np.mean(d)):+.1f}"
        )
        if is_dyad:
            for r in block:
                print(f"    {r[4]:18s}  start={r[1]:6.0f}  end={r[2]:6.0f}  d={r[3]:+7.0f}")
    return out_path


def collect_r43_rnn_unsigned_census(
    *,
    run_id: int = SINGLE_RUN_ID,
    model: str = "rnn",
    seed: int = SEED,
    max_snaps: int = 60,
    out_json: Path = R43_RNN_UNSIGNED_JSON,
) -> Path:
    """Unsigned HL topology census over learning for one unconstrained run."""
    from experiment import checkpoint_path
    from rnn.learning_snaps import list_learning_snaps
    from viz.compare.mixed_dfa_motif_all_runs import (
        _dominant_session,
        _snap_iteration,
        _subsample_snaps,
    )
    from viz.weight_structure import compute_weight_unsigned_hl_motif_counts

    task = f"mixeddfa_r{run_id:02d}_ns"
    ckpt = checkpoint_path(task, model, seed=seed)
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    snaps = [
        p for p in _dominant_session(list_learning_snaps(ckpt))
        if _snap_iteration(p) > 0
    ]
    if len(snaps) < 2:
        raise RuntimeError(f"r{run_id:02d}: need >=2 post-init learning snaps")
    snaps = _subsample_snaps(snaps, max_snaps)

    rows: list[dict] = []
    for snap in snaps:
        d = np.load(snap, allow_pickle=True)
        out = compute_weight_unsigned_hl_motif_counts(
            d["weights_hidden_to_hidden"], mode="mean",
        )
        it = (
            int(d["learning_snap_iteration"])
            if "learning_snap_iteration" in d.files
            else _snap_iteration(snap)
        )
        we = (
            float(d["learning_snap_word_err"])
            if "learning_snap_word_err" in d.files
            else float("nan")
        )
        rows.append({
            "it": it,
            "we": we,
            "edges": float(out["edges"]),
            "dyads_conn": float(out["dyads_conn"]),
            "triples_conn": float(out["triples_conn"]),
            "cnt": out["cnt"],
        })
        print(
            f"  snap it={it}  edges={int(out['edges'])}  "
            f"classes={len(out['cnt'])}",
            flush=True,
        )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {out_json}")
    return out_json


def plot_unsigned_motifs_over_learning(
    json_path: Path = R43_RNN_UNSIGNED_JSON,
    out_path: Path = R43_RNN_UNSIGNED_OUT,
) -> Path:
    """Grid: HL motif diagram + raw count trajectory for each unsigned class."""
    from matplotlib.gridspec import GridSpec
    from viz.weight_structure import draw_unsigned_hl_motif

    snaps = json.loads(json_path.read_text(encoding="utf-8"))
    iters = np.array([s["it"] for s in snaps], dtype=float)
    keys = sorted(
        {k for s in snaps for k in s["cnt"]},
        key=lambda k: (-float(snaps[0]["cnt"].get(k, 0)), k),
    )
    # Prefer triads first (diagrams), then dyads.
    triad_keys = [k for k in keys if k.startswith("T|")]
    dyad_keys = [k for k in keys if k.startswith("D|")]
    keys = triad_keys + dyad_keys

    n = len(keys)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(2.55 * ncol + 0.8, 3.55 * nrow + 1.1))
    # Extra outer hspace so bottom-row x labels never hit the title band below.
    outer = GridSpec(nrow, ncol, figure=fig, hspace=0.78, wspace=0.40)

    cmap = plt.get_cmap("viridis")
    starts = np.array([float(snaps[0]["cnt"].get(k, 0)) for k in keys], dtype=float)
    norm = plt.Normalize(vmin=float(starts.min()), vmax=float(max(starts.max(), 1.0)))

    panels: list[tuple[str, object, object, object]] = []
    for idx, key in enumerate(keys):
        r, c = divmod(idx, ncol)
        # Reserved bands: title | schematic | data (never share title+glyph axes).
        cell = outer[r, c].subgridspec(
            3, 1, height_ratios=[0.20, 1.20, 1.15], hspace=0.08,
        )
        ax_lab = fig.add_subplot(cell[0, 0])
        ax_d = fig.add_subplot(cell[1, 0])
        ax_t = fig.add_subplot(cell[2, 0])
        label = key.replace("T|", "").replace("D|", "")
        ax_lab.axis("off")
        ax_lab.text(
            0.5, 0.5, label,
            transform=ax_lab.transAxes,
            ha="center", va="center", fontsize=8, fontweight="bold",
            clip_on=False,
        )
        panels.append((key, ax_d, ax_t, (r, c)))

    for j in range(n, nrow * ncol):
        r, c = divmod(j, ncol)
        fig.add_subplot(outer[r, c]).axis("off")

    # Lock margins BEFORE drawing glyphs — motif_schema_box uses ax.get_position()
    # for physical aspect; drawing earlier used pre-adjust sizes and squashed /
    # clipped dense mutuals on later panels.
    finalize_grid_figure(
        fig,
        suptitle=(
            f"r{SINGLE_RUN_ID:02d} rnn unsigned HL topologies over learning "
            f"(diagram + raw count; color = start abundance)"
        ),
        top=0.90,
        bottom=0.05,
        left=0.06,
        right=0.98,
        hspace=0.72,
        wspace=0.38,
    )

    for key, ax_d, ax_t, (r, c) in panels:
        draw_unsigned_hl_motif(ax_d, key, color="#1a1a1a")
        series = np.array([float(s["cnt"].get(key, 0)) for s in snaps], dtype=float)
        col = cmap(norm(float(series[0])))
        ax_t.plot(iters, series, color=col, lw=1.5)
        ax_t.tick_params(labelsize=6)
        ax_t.grid(True, alpha=0.22)
        ax_t.spines["top"].set_visible(False)
        ax_t.spines["right"].set_visible(False)
        if r == nrow - 1:
            ax_t.set_xlabel("iter", fontsize=7)
        else:
            ax_t.tick_params(labelbottom=False)
        if c == 0:
            ax_t.set_ylabel("count", fontsize=7)
        c0 = max(float(series[0]), EPS)
        fold = float(series[-1]) / c0
        ax_t.text(
            0.98, 0.95,
            f"×{fold:.2f}",
            transform=ax_t.transAxes, fontsize=6.5, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.85", alpha=0.9),
        )

    save_figure(fig, out_path, dpi=160)
    print(f"wrote {out_path}")

    # companion start-vs-fold with diagrams is overkill; print summary
    x = np.log(np.maximum(starts, EPS))
    y = np.array([
        np.log((float(snaps[-1]["cnt"].get(k, 0)) + EPS) / (float(snaps[0]["cnt"].get(k, 0)) + EPS))
        for k in keys
    ], dtype=float)
    if len(x) >= 3 and np.std(x) > 1e-12:
        _, _, r2 = _ols(y, x)
        print(f"unsigned start vs fold R2={r2:.3f}  n={len(keys)}")
    for k in keys:
        c0 = float(snaps[0]["cnt"].get(k, 0))
        c1 = float(snaps[-1]["cnt"].get(k, 0))
        print(f"  {k:10s}  start={c0:7.0f}  end={c1:7.0f}  fold={c1/max(c0,EPS):.3f}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--only",
        choices=(
            "raw-over-learning",
            "all-runs-over-learning",
            "all-runs-dyad-over-learning",
            "all-runs-homogenization-summary",
            "all-runs-dyad-homogenization-summary",
            "r43-rnn-over-learning",
            "r43-rnn-dyad-over-learning",
            "r43-rnn-homogenization-summary",
            "r43-rnn-board",
            "r43-rnn-factor-panels",
            "r43-rnn-factor-regressions",
            "r43-rnn-pattern-regressions",
            "r43-rnn-hypothesis-regressions",
            "r43-rnn-story-board",
            "r43-rnn-dyad-homogenization-summary",
            "r43-rnn-unsigned",
            "r43-rnn-lollipop",
            "r43-rnn-dyad-lollipop",
        ),
        default="raw-over-learning",
        help="figure to generate",
    )
    p.add_argument("--json", type=Path, default=DEFAULT_JSON)
    p.add_argument("--edges-json", type=Path, default=DEFAULT_EDGES_JSON)
    p.add_argument("--cache", type=Path, default=ALL_RUNS_CACHE)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--min-start", type=int, default=MIN_START)
    p.add_argument(
        "--run-id",
        type=int,
        default=None,
        help="single-run board exemplar (default: highest DFA states in fold cache)",
    )
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--model", default=MODEL, choices=("rnn", "rnn_dale"))
    p.add_argument(
        "--coloring",
        default=COLORING,
        choices=("edge_sign", "dale_node"),
        help="edge_sign: HL by W polarity; dale_node: Dale E/I (or column-sign fallback)",
    )
    args = p.parse_args()
    if args.only == "raw-over-learning":
        plot_raw_counts_over_learning(
            args.json,
            args.out or DEFAULT_OUT,
            edges_json_path=args.edges_json,
            min_start=args.min_start,
        )
    elif args.only == "r43-rnn-over-learning":
        colored, edges = collect_single_run_edge_signed_census(
            run_id=SINGLE_RUN_ID,
            model="rnn",
            seed=SEED,
        )
        plot_raw_counts_over_learning(
            colored,
            args.out or R43_RNN_OUT,
            edges_json_path=edges,
            min_start=args.min_start,
            title="r43 rnn (edge-sign): start vs fold by #edges",
        )
    elif args.only == "r43-rnn-dyad-over-learning":
        plot_raw_counts_over_learning(
            args.json if args.json != DEFAULT_JSON else R43_RNN_JSON,
            args.out or R43_RNN_DYAD_OUT,
            edges_json_path=(
                args.edges_json if args.edges_json != DEFAULT_EDGES_JSON
                else R43_RNN_EDGES_JSON
            ),
            min_start=args.min_start,
            motif_prefix="D|",
            title="r43 rnn (edge-sign): dyad start vs fold by #edges",
        )
    elif args.only == "r43-rnn-dyad-lollipop":
        plot_lollipop_before_after(
            args.json if args.json != DEFAULT_JSON else R43_RNN_JSON,
            args.out or R43_RNN_DYAD_LOLLIPOP_OUT,
            min_start=args.min_start,
            motif_prefix="D|",
            title="r43 rnn (edge-sign): dyad lollipops by #edges",
        )
    elif args.only == "r43-rnn-lollipop":
        plot_lollipop_before_after(
            args.json if args.json != DEFAULT_JSON else R43_RNN_JSON,
            args.out or R43_RNN_LOLLIPOP_OUT,
            min_start=args.min_start,
            motif_prefix="T|",
        )
    elif args.only == "r43-rnn-homogenization-summary":
        plot_single_run_homogenization_summary(
            args.json if args.json != DEFAULT_JSON else R43_RNN_JSON,
            args.out or R43_RNN_SUMMARY_OUT,
            min_start=args.min_start,
            motif_prefix="T|",
        )
    elif args.only == "r43-rnn-board":
        plot_r43_single_run_board(
            None if args.json == DEFAULT_JSON else args.json,
            args.cache,
            args.out,
            min_start=args.min_start,
            motif_prefix="T|",
            run_id=args.run_id,
        )
    elif args.only == "r43-rnn-factor-panels":
        plot_single_run_factor_panels(
            None if args.json == DEFAULT_JSON else args.json,
            args.cache,
            args.out,
            min_start=args.min_start,
            motif_prefix="T|",
            run_id=args.run_id,
        )
    elif args.only == "r43-rnn-factor-regressions":
        plot_single_run_factor_regressions(
            None if args.json == DEFAULT_JSON else args.json,
            args.cache,
            args.out,
            min_start=args.min_start,
            motif_prefix="T|",
            run_id=args.run_id,
        )
    elif args.only == "r43-rnn-pattern-regressions":
        plot_single_run_pattern_regressions(
            None if args.json == DEFAULT_JSON else args.json,
            args.cache,
            args.out,
            min_start=args.min_start,
            motif_prefix="T|",
            run_id=args.run_id,
        )
    elif args.only == "r43-rnn-hypothesis-regressions":
        plot_single_run_hypothesis_regressions(
            None if args.json == DEFAULT_JSON else args.json,
            args.cache,
            args.out,
            min_start=args.min_start,
            motif_prefix="T|",
            run_id=args.run_id,
        )
    elif args.only == "r43-rnn-story-board":
        plot_single_run_story_board(
            None if args.json == DEFAULT_JSON else args.json,
            args.cache,
            args.out,
            min_start=args.min_start,
            motif_prefix="T|",
            run_id=args.run_id,
        )
    elif args.only == "r43-rnn-dyad-homogenization-summary":
        plot_single_run_homogenization_summary(
            args.json if args.json != DEFAULT_JSON else R43_RNN_JSON,
            args.out or R43_RNN_DYAD_SUMMARY_OUT,
            min_start=args.min_start,
            motif_prefix="D|",
            title="r43 rnn (edge-sign, dyad): homogenization summary",
        )
    elif args.only == "r43-rnn-unsigned":
        collect_r43_rnn_unsigned_census(run_id=SINGLE_RUN_ID, model="rnn", seed=SEED)
        plot_unsigned_motifs_over_learning(
            R43_RNN_UNSIGNED_JSON,
            args.out or R43_RNN_UNSIGNED_OUT,
        )
    elif args.only == "all-runs-dyad-over-learning":
        mar.plot_all_runs_over_learning(
            ALL_RUNS_DYAD_CACHE,
            args.out or ALL_RUNS_DYAD_OUT,
            min_start=args.min_start,
            rebuild_cache=args.rebuild_cache or not ALL_RUNS_DYAD_CACHE.is_file(),
            model=args.model,
            seed=SEED,
            max_snaps_per_run=MAX_SNAPS_PER_RUN,
            ols_fn=_ols,
            beta_out_path=ALL_RUNS_DYAD_BETA_OUT,
            coloring=args.coloring,
            motif_prefix="D|",
        )
    elif args.only == "all-runs-over-learning":
        model = args.model
        coloring = args.coloring
        if model == "rnn" and coloring == "dale_node":
            print(
                "note: unconstrained rnn + dale_node uses column-sign pseudo-E/I "
                "(Dale signs empty in ckpts)",
                flush=True,
            )
        mar.plot_all_runs_over_learning(
            args.cache,
            args.out or ALL_RUNS_OUT,
            min_start=args.min_start,
            rebuild_cache=args.rebuild_cache,
            model=model,
            seed=SEED,
            max_snaps_per_run=MAX_SNAPS_PER_RUN,
            ols_fn=_ols,
            beta_out_path=ALL_RUNS_BETA_OUT,
            coloring=coloring,
            motif_prefix="T|",
        )
    elif args.only == "all-runs-homogenization-summary":
        mar.plot_all_runs_homogenization_summary(
            args.cache,
            args.out or ALL_RUNS_SUMMARY_OUT,
            min_start=args.min_start,
            rebuild_cache=args.rebuild_cache,
            model=args.model,
            seed=SEED,
            max_snaps_per_run=MAX_SNAPS_PER_RUN,
            coloring=args.coloring,
            motif_prefix="T|",
        )
    elif args.only == "all-runs-dyad-homogenization-summary":
        mar.plot_all_runs_homogenization_summary(
            ALL_RUNS_DYAD_CACHE,
            args.out or ALL_RUNS_DYAD_SUMMARY_OUT,
            min_start=args.min_start,
            rebuild_cache=args.rebuild_cache or not ALL_RUNS_DYAD_CACHE.is_file(),
            model=args.model,
            seed=SEED,
            max_snaps_per_run=MAX_SNAPS_PER_RUN,
            coloring=args.coloring,
            motif_prefix="D|",
        )


if __name__ == "__main__":
    main()
