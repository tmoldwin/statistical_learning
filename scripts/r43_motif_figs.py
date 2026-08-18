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

DEFAULT_JSON = (
    REPO / "experiments/comparisons/mixed_vocab_dfa_ns/trajectories/r43_motif_colored_all.json"
)
DEFAULT_EDGES_JSON = (
    REPO / "experiments/comparisons/mixed_vocab_dfa_ns/trajectories/r43_motif_counts_over_learning.json"
)
DEFAULT_OUT = (
    REPO / "experiments/comparisons/mixed_vocab_dfa_ns/trajectories/r43_motif_counts_raw_over_learning.png"
)
MIN_START = 20
EPS = 0.5
ALL_RUNS_CACHE = REPO / "experiments/comparisons/mixed_vocab_dfa_ns/trajectories/mixed_dfa_motif_fold_all_runs.json"
ALL_RUNS_OUT = REPO / "experiments/comparisons/mixed_vocab_dfa_ns/trajectories/mixed_dfa_motif_counts_raw_over_learning.png"
MODEL = "rnn_dale"
SEED = 1
MAX_SNAPS_PER_RUN = 8


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


def plot_raw_counts_over_learning(
    json_path: Path = DEFAULT_JSON,
    out_path: Path = DEFAULT_OUT,
    *,
    edges_json_path: Path = DEFAULT_EDGES_JSON,
    min_start: int = MIN_START,
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

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=float(logt0.min()), vmax=float(logt0.max()))
    colors = [cmap(norm(v)) for v in logt0]

    mask = t0 >= min_start
    x_sc = logt0[mask]
    y_sc = log_fold[mask]
    c_sc = [colors[i] for i, ok in enumerate(mask) if ok]
    beta, pred, r2 = _ols(y_sc, x_sc)
    x_line = np.linspace(float(x_sc.min()), float(x_sc.max()), 100)
    y_line = beta[0] + beta[1] * x_line

    fig = plt.figure(figsize=(12.0, 6.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 0.85])
    ax_sc = fig.add_subplot(gs[0, 0])
    ax_log = fig.add_subplot(gs[0, 1])
    ax_spr = fig.add_subplot(gs[0, 2])
    ax_edge = fig.add_subplot(gs[1, 0])
    ax_sd = fig.add_subplot(gs[1, 1])
    ax_tri = fig.add_subplot(gs[1, 2])

    ax_sc.scatter(x_sc, y_sc, c=c_sc, s=22, alpha=0.85, edgecolors="0.35", linewidths=0.25)
    ax_sc.plot(x_line, y_line, color="#c0392b", lw=1.4, label=f"OLS  R2={r2:.2f}")
    ax_sc.axhline(0.0, color="0.55", lw=0.7, ls="--")
    ax_sc.set_xlabel("log start count", fontsize=8)
    ax_sc.set_ylabel("log fold (end / start)", fontsize=8)
    ax_sc.set_title(f"start vs fold  (start>={min_start}, n={int(mask.sum())})", fontsize=8, pad=4)
    ax_sc.legend(fontsize=7, loc="upper right", frameon=True)
    ax_sc.tick_params(labelsize=7)
    ax_sc.grid(True, alpha=0.25)

    for i, col in enumerate(colors):
        ax_log.plot(iters, np.maximum(T[:, i], EPS), color=col, lw=0.9, alpha=0.75)
    med = np.median(T, axis=1)
    geo = np.exp(np.mean(np.log(np.maximum(T, EPS)), axis=1))
    ax_log.plot(iters, med, color="0.1", lw=1.6, label="median class")
    ax_log.plot(iters, geo, color="#c0392b", lw=1.4, ls="--", label="geo mean")
    ax_log.set_yscale("log")
    ax_log.set_xlabel("iteration", fontsize=8)
    ax_log.set_ylabel("count", fontsize=8)
    ax_log.set_title("triads, raw (log scale)", fontsize=9, pad=4)
    ax_log.legend(fontsize=6, loc="lower right", frameon=True)
    ax_log.tick_params(labelsize=7)
    ax_log.grid(True, alpha=0.25)

    slog = np.log(np.maximum(T, EPS))
    sd = slog.std(axis=1)
    q25 = np.percentile(slog, 25, axis=1)
    q75 = np.percentile(slog, 75, axis=1)
    ax_spr.fill_between(iters, np.exp(q25), np.exp(q75), color="0.75", label="middle 50% of classes")
    ax_spr.plot(iters, np.exp(np.median(slog, axis=1)), color="0.1", lw=1.5, label="median class")
    ax_spr.set_yscale("log")
    ax_spr.set_xlabel("iteration", fontsize=8)
    ax_spr.set_ylabel("count", fontsize=8)
    ax_spr.set_title(
        f"spread  (sd log-count {sd[0]:.2f} -> {sd[-1]:.2f})",
        fontsize=8,
        pad=4,
    )
    ax_spr.legend(fontsize=6, loc="upper right", frameon=True)
    ax_spr.tick_params(labelsize=7)
    ax_spr.grid(True, alpha=0.25)

    edge_fold = n_edges[-1] / n_edges[0]
    ax_edge.plot(iters, n_edges, color="#4c78a8", lw=1.6)
    ax_edge.set_xlabel("iteration", fontsize=8)
    ax_edge.set_ylabel("count", fontsize=8)
    ax_edge.set_title(
        f"strong |W_hh| edges  ({int(n_edges[0])} -> {int(n_edges[-1])}, x{edge_fold:.2f})",
        fontsize=8,
        pad=4,
    )
    ax_edge.tick_params(labelsize=7)
    ax_edge.grid(True, alpha=0.25)

    ax_sd.plot(iters, sd, color="#c0392b", lw=1.6)
    ax_sd.set_xlabel("iteration", fontsize=8)
    ax_sd.set_ylabel("sd log count", fontsize=8)
    ax_sd.set_title(f"triad spread  (sd {sd[0]:.2f} -> {sd[-1]:.2f})", fontsize=8, pad=4)
    ax_sd.tick_params(labelsize=7)
    ax_sd.grid(True, alpha=0.25)

    tri_fold = triples_total[-1] / triples_total[0]
    ax_tri.plot(iters, triples_total, color="#2ca02c", lw=1.6)
    ax_tri.set_xlabel("iteration", fontsize=8)
    ax_tri.set_ylabel("count", fontsize=8)
    ax_tri.set_title(
        f"connected triad instances  ({int(triples_total[0])} -> {int(triples_total[-1])}, x{tri_fold:.2f})",
        fontsize=8,
        pad=4,
    )
    ax_tri.tick_params(labelsize=7)
    ax_tri.grid(True, alpha=0.25)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cax = fig.add_axes([0.935, 0.34, 0.012, 0.52])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("start count", fontsize=7)
    ticks = np.array([3.0, 5.0, 7.0, 9.0, 11.0])
    ticks = ticks[(ticks >= logt0.min() - 0.2) & (ticks <= logt0.max() + 0.2)]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{int(round(np.exp(t)))}" for t in ticks])
    cbar.ax.tick_params(labelsize=6)

    finalize_grid_figure(
        fig,
        suptitle="r43: motif class counts vs training (start size drives fold)",
        top=0.92,
        bottom=0.08,
        left=0.06,
        right=0.92,
        hspace=0.42,
        wspace=0.32,
    )
    save_figure(fig, out_path, dpi=160)
    plt.close(fig)
    print(f"wrote {out_path}")
    print(f"log start vs log fold R2={r2:.3f}  (n={int(mask.sum())})")
    print(f"edges {int(n_edges[0])} -> {int(n_edges[-1])}  triples {int(triples_total[0])} -> {int(triples_total[-1])}")
    print(f"sd log start/end {float(sd[0]):.2f} {float(sd[-1]):.2f}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--only",
        choices=("raw-over-learning", "all-runs-over-learning"),
        default="raw-over-learning",
        help="figure to generate",
    )
    p.add_argument("--json", type=Path, default=DEFAULT_JSON)
    p.add_argument("--edges-json", type=Path, default=DEFAULT_EDGES_JSON)
    p.add_argument("--cache", type=Path, default=ALL_RUNS_CACHE)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--min-start", type=int, default=MIN_START)
    p.add_argument("--rebuild-cache", action="store_true")
    args = p.parse_args()
    if args.only == "raw-over-learning":
        plot_raw_counts_over_learning(
            args.json,
            args.out or DEFAULT_OUT,
            edges_json_path=args.edges_json,
            min_start=args.min_start,
        )
    elif args.only == "all-runs-over-learning":
        mar.plot_all_runs_over_learning(
            args.cache,
            args.out or ALL_RUNS_OUT,
            min_start=args.min_start,
            rebuild_cache=args.rebuild_cache,
            model=MODEL,
            seed=SEED,
            max_snaps_per_run=MAX_SNAPS_PER_RUN,
            ols_fn=_ols,
        )


if __name__ == "__main__":
    main()
