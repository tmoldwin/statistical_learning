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
MODEL = "rnn"
COLORING = "edge_sign"
SEED = 1
MAX_SNAPS_PER_RUN = 8
SINGLE_RUN_ID = 43


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
) -> Path:
    snaps = json.loads(json_path.read_text(encoding="utf-8"))
    edge_snaps = json.loads(edges_json_path.read_text(encoding="utf-8"))
    if [s["it"] for s in snaps] != [s["it"] for s in edge_snaps]:
        raise ValueError("colored census and edge-count snapshots must share iteration grid")
    iters = np.array([s["it"] for s in snaps], dtype=float)
    n_edges = np.array([s["edges"] for s in edge_snaps], dtype=float)
    triples_total = np.array([s["triples_conn"] for s in snaps], dtype=float)
    triad_keys = sorted(k for k in snaps[0]["cnt"] if k.startswith("T|"))

    def mat(keys: list[str]) -> np.ndarray:
        return np.array([[s["cnt"].get(k, 0) for k in keys] for s in snaps], dtype=float)

    T = mat(triad_keys)
    t0 = np.maximum(T[0], EPS)
    t1 = np.maximum(T[-1], EPS)
    logt0 = np.log(t0)
    fold = t1 / t0
    log_fold = np.log(fold)

    n_e = np.array([mar._n_edges_from_key(k) for k in triad_keys], dtype=int)
    edge_colors = {
        int(ne): mar._EDGE_COUNT_COLORS.get(int(ne), "#888888")
        for ne in np.unique(n_e)
    }

    mask = t0 >= min_start
    x_sc = logt0[mask]
    y_sc = log_fold[mask]
    n_e_sc = n_e[mask]
    beta, _, r2 = _ols(y_sc, x_sc)

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

    fig = plt.figure(figsize=(13.2, 10.8))
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

    fold_axes: list = []
    for j, ne in enumerate(tiers):
        ax = fig.add_subplot(gs_fold[0, j])
        fold_axes.append(ax)
        m = n_e_sc == ne
        x, y = x_sc[m], y_sc[m]
        col = edge_colors[ne]
        x_disp = x + rng.normal(0.0, 0.028, size=len(x))
        ax.scatter(x_disp, y, c=col, s=16, alpha=0.80, edgecolors="0.25",
                   linewidths=0.25, zorder=3)
        b1, r2_e = float("nan"), float("nan")
        if int(m.sum()) >= 8 and float(np.std(x)) > 1e-9:
            b_e, _, r2_e = _ols(y, x)
            b1 = float(b_e[1])
            x_line = np.linspace(float(x.min()), float(x.max()), 40)
            _outlined_line(ax, x_line, b_e[0] + b_e[1] * x_line, col, lw=1.55, z=5)
            beta_lines.append((f"{ne}e", b1, col))
        ax.axhline(0.0, color="0.55", lw=0.65, ls="--", zorder=1)
        if len(y):
            y_span = float(y.max() - y.min())
            y_pad = 0.14 * max(y_span, 0.15)
            ax.set_ylim(float(y.min()) - y_pad, float(y.max()) + y_pad)
            x_span = float(x.max() - x.min())
            x_pad = 0.12 * max(x_span, 0.15)
            ax.set_xlim(float(x.min()) - x_pad, float(x.max()) + x_pad)
        ax.set_title(f"{ne}e  β={b1:+.2f}  $R^2$={r2_e:.2f}", fontsize=7.4,
                     color=col, pad=3)
        _style(ax)
        if j == 0:
            ax.set_ylabel("log fold", fontsize=7.5)
        if j == mid:
            ax.set_xlabel("log start", fontsize=7.5)

    traj_axes: list = []
    for j, ne in enumerate(tiers):
        ax = fig.add_subplot(gs_traj[0, j])
        traj_axes.append(ax)
        col = edge_colors[ne]
        tm = n_e == ne
        series = np.maximum(T[:, tm], EPS)
        for k in range(series.shape[1]):
            ax.plot(iters, series[:, k], color=col, lw=0.55, alpha=0.28, zorder=2)
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

    sd_axes: list = []
    for j, ne in enumerate(tiers):
        ax = fig.add_subplot(gs_sd[0, j])
        sd_axes.append(ax)
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
    ax_tri = fig.add_subplot(gs_foot[0, 2])

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

    tri_fold = float(triples_total[-1] / triples_total[0])
    ax_tri.plot(iters, triples_total, color="#2ca02c", lw=1.6)
    ax_tri.set_xlabel("iteration", fontsize=7.5)
    ax_tri.set_ylabel("count", fontsize=7.5)
    ax_tri.set_title(
        f"connected triad instances  ({int(triples_total[0])}→{int(triples_total[-1])}, ×{tri_fold:.2f})",
        fontsize=8, pad=3,
    )
    _style(ax_tri)

    finalize_grid_figure(
        fig,
        suptitle=(
            title
            or "r43: motif class counts vs training (by #edges)"
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
        f"triples {int(triples_total[0])} -> {int(triples_total[-1])}"
    )
    print(f"sd log start/end {float(sd[0]):.2f} {float(sd[-1]):.2f}")
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
            "r43-rnn-over-learning",
            "r43-rnn-unsigned",
        ),
        default="raw-over-learning",
        help="figure to generate",
    )
    p.add_argument("--json", type=Path, default=DEFAULT_JSON)
    p.add_argument("--edges-json", type=Path, default=DEFAULT_EDGES_JSON)
    p.add_argument("--cache", type=Path, default=ALL_RUNS_CACHE)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--min-start", type=int, default=MIN_START)
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
    elif args.only == "r43-rnn-unsigned":
        collect_r43_rnn_unsigned_census(run_id=SINGLE_RUN_ID, model="rnn", seed=SEED)
        plot_unsigned_motifs_over_learning(
            R43_RNN_UNSIGNED_JSON,
            args.out or R43_RNN_UNSIGNED_OUT,
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
        )


if __name__ == "__main__":
    main()
