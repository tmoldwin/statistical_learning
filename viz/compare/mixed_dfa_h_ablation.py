"""Hidden-size ablation for mixed-vocab DFA geometry (exploratory; not paper).

Compares H ∈ {50, 100, 150} on the same 50-run vocab plan: between/within
cosine, Euclidean, and learning curves (word-error vs iteration).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiment import checkpoint_path, comparison_dir
from vocab_mixed_dfa import (
    HIDDEN_SIZE,
    HIDDEN_SIZE_ABLATION,
    comparison_name_for_h,
    iter_tasks_for_h,
)
from viz.compare.mixed_dfa_viz import (
    _WITHIN_CORR_FEATURES,
    _dfa_states,
    collect_mixed_dfa_within_corr,
)
from viz.compare.pow2_sweep_metric_board import _fit_trend
from viz.compare.sweep_output import sweep_data_dir
from viz.plot_layout import finalize_grid_figure, save_figure

ABLATION_HS: tuple[int, ...] = (50, HIDDEN_SIZE, 150)
H_COLORS: dict[int, str] = {
    50: "#E69F00",
    100: "#0072B2",
    150: "#009E73",
}
# Focus feature for geometry packing story (most DFA-aligned).
FOCUS_FEATURE = "dfa"


def _all_hs() -> tuple[int, ...]:
    return tuple(sorted(set(ABLATION_HS) | set(HIDDEN_SIZE_ABLATION) | {HIDDEN_SIZE}))


def collect_training_panels_for_h(
    *,
    hidden_size: int,
    seed: int = 1,
    model_type: str = "rnn",
    recompute: bool = False,
) -> Path:
    """Per-run training summary + word-error learning curve from checkpoints."""
    comp = comparison_name_for_h(hidden_size)
    out = sweep_data_dir(comp) / "training_vs_dfa_h.json"
    if out.is_file() and not recompute:
        try:
            old = json.loads(out.read_text(encoding="utf-8"))
            if int(old.get("hidden_size", -1)) == int(hidden_size) and old.get("panels"):
                return out
        except Exception:  # noqa: BLE001
            pass

    panels: list[dict[str, Any]] = []
    for entry in iter_tasks_for_h(hidden_size):
        task = entry["task"]
        ckpt = checkpoint_path(task, model_type, seed=seed)
        if not ckpt.is_file():
            continue
        try:
            data = np.load(ckpt, allow_pickle=True)
            h_ckpt = int(np.asarray(data["hidden_size"]).reshape(-1)[0])
            if h_ckpt != int(hidden_size):
                print(f"  skip {task}: ckpt H={h_ckpt} != {hidden_size}", flush=True)
                continue
            iters = np.asarray(data["metric_iterations"], dtype=float).ravel()
            we = np.asarray(data["metric_word_error_frac"], dtype=float).ravel()
            best_iter = float(np.asarray(data["best_metric_iter"]).reshape(-1)[0])
            best_we = float(np.asarray(data["best_metric_word_error_frac"]).reshape(-1)[0])
            demo_we = float(np.asarray(data["demo_word_error_frac"]).reshape(-1)[0])
            # Iters to first reaching 3% word error (if ever).
            hit = np.where(np.isfinite(we) & (we <= 0.03))[0]
            iter_to_3 = float(iters[hit[0]]) if len(hit) else float("nan")
            n_dfa = _dfa_states(list(entry["words"]))
            panels.append({
                "task": task,
                "run_id": int(entry["run_id"]),
                "n_words": int(entry["n_words"]),
                "n_dfa_states": n_dfa,
                "hidden_size": int(hidden_size),
                "best_metric_iter": best_iter,
                "best_metric_word_error_frac": best_we,
                "demo_word_error_frac": demo_we,
                "iter_to_3pct": iter_to_3,
                "metric_iterations": iters.tolist(),
                "metric_word_error_frac": we.tolist(),
            })
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {task}: {exc}", flush=True)
            continue

    payload = {
        "hidden_size": int(hidden_size),
        "seed": seed,
        "comparison": comp,
        "panels": panels,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out} ({len(panels)} panels)", flush=True)
    return out


def ensure_h_ablation_data(
    *,
    seed: int = 1,
    recompute: bool = False,
    hs: tuple[int, ...] = ABLATION_HS,
) -> dict[int, dict[str, Path]]:
    """Collect within-corr + training panels for each H (skip missing ckpts)."""
    paths: dict[int, dict[str, Path]] = {}
    for h in hs:
        geo = collect_mixed_dfa_within_corr(
            seed=seed, recompute=recompute, hidden_size=h,
        )
        train = collect_training_panels_for_h(
            hidden_size=h, seed=seed, recompute=recompute,
        )
        paths[h] = {"geometry": geo, "training": train}
    return paths


def _load_panels(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")).get("panels", [])


def _scatter_by_h(
    ax,
    by_h: dict[int, list[dict[str, Any]]],
    *,
    y_key: str,
    feature: str | None = None,
    ylabel: str,
    title: str,
    ylim: tuple[float, float] | None = None,
    y_floor: float | None = None,
) -> None:
    y_all: list[float] = []
    for h, panels in sorted(by_h.items()):
        color = H_COLORS.get(h, "#666")
        xs, ys = [], []
        for p in panels:
            x = float(p["n_dfa_states"])
            if feature is None:
                y = float(p.get(y_key, float("nan")))
            else:
                y = float(p.get(y_key, {}).get(feature, float("nan")))
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            xs.append(x)
            ys.append(y)
        if not xs:
            continue
        xa = np.asarray(xs, dtype=float)
        ya = np.asarray(ys, dtype=float)
        y_all.extend(ya.tolist())
        ax.scatter(
            xa, ya, s=22, color=color, alpha=0.75, edgecolors="white",
            linewidths=0.3, zorder=3, label=f"H={h}",
        )
        x_fit, y_fit, _r2, _ = _fit_trend(xa, ya)
        if x_fit is not None:
            ax.plot(x_fit, y_fit, color=color, lw=1.6, zorder=4)
            y_all.extend(np.asarray(y_fit, dtype=float).tolist())
    ax.set_xlabel("DFA states", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        finite = [y for y in y_all if np.isfinite(y)]
        if finite:
            lo, hi = float(np.min(finite)), float(np.max(finite))
            pad = max(0.02, 0.08 * (hi - lo if hi > lo else 0.1))
            y0 = lo - pad
            if y_floor is not None:
                y0 = max(y_floor, y0)
            ax.set_ylim(y0, hi + pad)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=7)


def _plot_learning_curves(
    ax,
    by_h: dict[int, list[dict[str, Any]]],
    *,
    dfa_lo: float,
    dfa_hi: float,
    title: str,
) -> None:
    """Mean ± std word-error curves over runs in a DFA band."""
    for h, panels in sorted(by_h.items()):
        color = H_COLORS.get(h, "#666")
        curves = [
            p for p in panels
            if dfa_lo <= float(p["n_dfa_states"]) <= dfa_hi
            and p.get("metric_iterations") and p.get("metric_word_error_frac")
        ]
        if not curves:
            continue
        # Common grid in log-ish linear space up to max shared length.
        max_t = max(float(np.max(p["metric_iterations"])) for p in curves)
        grid = np.linspace(0.0, max_t, 80)
        mats = []
        for p in curves:
            t = np.asarray(p["metric_iterations"], dtype=float)
            y = np.asarray(p["metric_word_error_frac"], dtype=float)
            order = np.argsort(t)
            t, y = t[order], y[order]
            if len(t) < 2:
                continue
            mats.append(np.interp(grid, t, y, left=y[0], right=y[-1]))
        if not mats:
            continue
        arr = np.vstack(mats)
        mu = arr.mean(axis=0)
        sd = arr.std(axis=0)
        ax.plot(grid, mu, color=color, lw=1.6, label=f"H={h} (n={len(mats)})")
        ax.fill_between(grid, mu - sd, mu + sd, color=color, alpha=0.18, linewidth=0)
    ax.axhline(0.03, color="0.45", ls="--", lw=0.9, zorder=1)
    ax.set_xlabel("training iteration", fontsize=8)
    ax.set_ylabel("word error frac", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=7)


def plot_mixed_dfa_h_ablation(
    *,
    seed: int = 1,
    recompute: bool = False,
    hs: tuple[int, ...] = ABLATION_HS,
    outfile: str | None = None,
) -> Path:
    """Exploratory figure: geometry packing + learning across hidden sizes."""
    paths = ensure_h_ablation_data(seed=seed, recompute=recompute, hs=hs)
    geo_by_h = {h: _load_panels(paths[h]["geometry"]) for h in hs}
    train_by_h = {h: _load_panels(paths[h]["training"]) for h in hs}

    out = Path(
        outfile
        or comparison_dir("mixed_vocab_dfa_ns", "trajectories")
        / "h_ablation_geometry_learning.png"
    )

    fig, axes = plt.subplots(2, 3, figsize=(12.8, 6.8), squeeze=False)

    _scatter_by_h(
        axes[0, 0], geo_by_h,
        y_key="between_cosine", feature=FOCUS_FEATURE,
        ylabel="between-label cosine (DFA)",
        title="Between-label cosine (DFA)",
    )
    _scatter_by_h(
        axes[0, 1], geo_by_h,
        y_key="within_cosine", feature=FOCUS_FEATURE,
        ylabel="within-label cosine (DFA)",
        title="Within-label cosine (DFA)",
        ylim=(-0.05, 1.05),
    )
    _scatter_by_h(
        axes[0, 2], geo_by_h,
        y_key="pairwise_between_median", feature=FOCUS_FEATURE,
        ylabel="between-label Euclidean (DFA)",
        title="Between-label Euclidean (DFA)",
        y_floor=0.0,
    )
    _scatter_by_h(
        axes[1, 0], train_by_h,
        y_key="iter_to_3pct", feature=None,
        ylabel="iters to 3% word error",
        title="Learning speed vs DFA",
        y_floor=0.0,
    )

    # Learning curves: easy and hard DFA bands.
    dfa_all = [
        float(p["n_dfa_states"])
        for panels in train_by_h.values() for p in panels
    ]
    if dfa_all:
        q33, q66 = np.quantile(dfa_all, [0.33, 0.66])
        _plot_learning_curves(
            axes[1, 1], train_by_h,
            dfa_lo=0.0, dfa_hi=float(q33),
            title=f"Word-error curves (DFA ≤ {q33:.0f})",
        )
        _plot_learning_curves(
            axes[1, 2], train_by_h,
            dfa_lo=float(q66), dfa_hi=1e9,
            title=f"Word-error curves (DFA ≥ {q66:.0f})",
        )
    else:
        axes[1, 1].set_axis_off()
        axes[1, 2].set_axis_off()

    # Shared legend from first axis that has labels.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[1, 1].get_legend_handles_labels()
    if handles:
        # Deduplicate by label.
        seen: set[str] = set()
        uniq_h, uniq_l = [], []
        for hnd, lab in zip(handles, labels):
            if lab in seen:
                continue
            seen.add(lab)
            uniq_h.append(hnd)
            uniq_l.append(lab)
        fig.legend(
            uniq_h, uniq_l,
            loc="lower center", bbox_to_anchor=(0.5, 0.01),
            ncol=len(uniq_l), fontsize=8, frameon=False,
        )

    finalize_grid_figure(
        fig,
        suptitle=(
            f"Hidden-size ablation on mixed-vocab DFA runs "
            f"(H={','.join(str(h) for h in hs)}; seed {seed}; exploratory)"
        ),
        top=0.90,
        bottom=0.10,
        left=0.07,
        right=0.99,
        wspace=0.34,
        hspace=0.40,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out
