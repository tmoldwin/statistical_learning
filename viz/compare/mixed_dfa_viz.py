"""Analyses for mixed-vocab runs, organized by minimized DFA state count."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiment import checkpoint_path, comparison_dir
from vocab_diagrams import build_minimized_vocabulary_automaton
from vocab_mixed_dfa import (
    COMPARISON_NAME,
    DEFAULT_SEEDS,
    iter_runs,
    write_run_manifest,
)
from viz.compare._data import load_task_decoding_context
from viz.compare.decoding import (
    DECODING_FEATURES,
    DECODE_FEATURE_COLORS,
    _DEFAULT_MAX_PCS,
    _DEFAULT_NEURON_RANDOM_TRIALS,
    chance_corrected,
    compute_panel_decoding,
    feature_display_name,
)
from viz.compare.geometry import _GEOMETRY_TRIALS, _ROLLOUT_SEED, _mean_closed_loop_hidden
from viz.compare.pow2_sweep_heatmap import _training_panel_from_checkpoint
from viz.compare.state_space_metrics import pc_variance_percent
from viz.compare.sweep_output import sweep_data_dir, sweep_decoding_dir, sweep_figures_dir
from viz.plot_layout import finalize_grid_figure, hide_x_tick_labels, save_figure
from visualize import (
    _closed_loop_summary_seed,
    _one_vocab_cycle_steps,
    _trajectory_seed_letters,
)

_FEATURE_ORDER = DECODING_FEATURES  # char, dfa, position, position_from_end


def _training_metrics(task: str, *, model_type: str, seed: int, n_words: int) -> dict[str, Any]:
    return _training_panel_from_checkpoint(
        task, n_words=n_words, length="mixed", seed=seed, model_type=model_type,
    )


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _dfa_states(words: list[str]) -> int:
    return int(build_minimized_vocabulary_automaton(words).dfa._n)


def _loop_pc_spectrum(ctx) -> np.ndarray:
    vocab_words = list(ctx.words)
    seed_letters = _trajectory_seed_letters(ctx.model, vocab_words)
    summary_seed = _closed_loop_summary_seed(vocab_words, seed_letters, spaced=ctx.spaced)
    summary_steps = _one_vocab_cycle_steps(vocab_words, spaced=ctx.spaced)
    if vocab_words:
        summary_steps += max(len(w) for w in vocab_words)
    mean_loop = _mean_closed_loop_hidden(
        ctx.model,
        summary_seed=summary_seed,
        steps=summary_steps,
        rollout_seed=_ROLLOUT_SEED,
        n_trials=_GEOMETRY_TRIALS,
        vocab_words=vocab_words,
        spaced=ctx.spaced,
    )
    if mean_loop is None or len(mean_loop) < 3:
        return np.empty(0)
    return pc_variance_percent(mean_loop)


def _pad_spectrum(spectrum: np.ndarray | list[float], max_pcs: int) -> np.ndarray:
    arr = np.asarray(spectrum, dtype=float)
    out = np.zeros(max_pcs, dtype=float)
    n = min(len(arr), max_pcs)
    if n:
        out[:n] = arr[:n]
    return out


def collect_mixed_dfa_panels(
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    model_type: str = "rnn",
    max_k: int = _DEFAULT_MAX_PCS,
    recompute: bool = True,
) -> Path:
    """Decode + spectrum + training metrics for every mixed-dfa run; axis = DFA size."""
    data_dir = sweep_data_dir(COMPARISON_NAME)
    out_path = data_dir / "mixed_dfa_panels.json"
    manifest_path = write_run_manifest(data_dir / "run_manifest.json")
    print(f"wrote {manifest_path}", flush=True)

    if out_path.is_file() and not recompute:
        return out_path

    panels: list[dict[str, Any]] = []
    for entry in iter_runs():
        task = entry["task"]
        words = list(entry["words"])
        n_dfa = _dfa_states(words)
        for seed in seeds:
            if not checkpoint_path(task, model_type, seed=seed).is_file():
                print(f"  skip {task} seed {seed} (no checkpoint)", flush=True)
                continue
            print(f"  {task} seed {seed}  n_words={entry['n_words']}  dfa={n_dfa}", flush=True)
            try:
                ctx = load_task_decoding_context(task, model_type=model_type, seed=seed)
                decoding = compute_panel_decoding(
                    ctx,
                    max_k=max_k,
                    neuron_sampling="random",
                    n_random_trials=_DEFAULT_NEURON_RANDOM_TRIALS,
                )
            except Exception as exc:  # noqa: BLE001 — keep sweep going
                panels.append({
                    "task": task,
                    "seed": seed,
                    "run_id": entry["run_id"],
                    "n_words": entry["n_words"],
                    "n_dfa_states": n_dfa,
                    "words": words,
                    "error": str(exc),
                })
                continue

            spectrum = _loop_pc_spectrum(ctx)
            try:
                train = _training_metrics(
                    task, model_type=model_type, seed=seed, n_words=int(entry["n_words"]),
                )
            except Exception as exc:  # noqa: BLE001
                train = {"error": str(exc)}

            feat_summary: dict[str, Any] = {}
            for feat in _FEATURE_ORDER:
                blob = decoding.get("features", {}).get(feat, {})
                if blob.get("error"):
                    feat_summary[feat] = {"error": blob["error"]}
                    continue
                chance = float(blob.get("chance", float("nan")))
                full = float(blob.get("full_hidden", float("nan")))
                feat_summary[feat] = {
                    "chance": chance,
                    "full_hidden": full,
                    "full_hidden_cc": (
                        chance_corrected(full, chance)
                        if np.isfinite(full) and np.isfinite(chance)
                        else float("nan")
                    ),
                    "by_k": blob.get("by_k"),
                    "by_k_neurons": blob.get("by_k_neurons"),
                    "by_k_neurons_std": blob.get("by_k_neurons_std"),
                    "neuron_sampling": blob.get("neuron_sampling", "random"),
                }

            panels.append({
                "task": task,
                "seed": seed,
                "run_id": entry["run_id"],
                "n_words": entry["n_words"],
                "n_dfa_states": n_dfa,
                "words": words,
                "length_counts": {
                    str(L): sum(1 for w in words if len(w) == L) for L in (3, 4, 5, 6)
                },
                "features": feat_summary,
                "spectrum_pct": (
                    _pad_spectrum(spectrum, max_k).tolist() if len(spectrum) else []
                ),
                "training": train,
            })

    payload = {
        "comparison": COMPARISON_NAME,
        "model_type": model_type,
        "seeds": list(seeds),
        "features": list(_FEATURE_ORDER),
        "max_k": max_k,
        "panels": panels,
    }
    out_path.write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")
    print(f"wrote {out_path}", flush=True)
    return out_path


def _load_panels(path: Path | None = None) -> dict[str, Any]:
    p = path or (sweep_data_dir(COMPARISON_NAME) / "mixed_dfa_panels.json")
    return json.loads(p.read_text(encoding="utf-8"))


def plot_decoding_vs_dfa(
    payload: dict[str, Any] | None = None,
    *,
    outfile: str = "decoding_vs_dfa.png",
) -> Path:
    """Scatter: chance-corrected full-hidden readout accuracy vs DFA state count."""
    payload = payload or _load_panels()
    panels = [p for p in payload["panels"] if "error" not in p]

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, feat in zip(axes, _FEATURE_ORDER):
        xs: list[float] = []
        ys: list[float] = []
        for panel in panels:
            blob = panel.get("features", {}).get(feat, {})
            y = blob.get("full_hidden_cc")
            if y is None or not np.isfinite(y):
                continue
            xs.append(float(panel["n_dfa_states"]))
            ys.append(float(y))
        color = DECODE_FEATURE_COLORS.get(feat, "#333333")
        if xs:
            # Jitter tiny x so overlapping runs remain visible.
            rng = np.random.default_rng(0)
            xj = np.asarray(xs, dtype=float) + rng.uniform(-0.25, 0.25, size=len(xs))
            ax.scatter(xj, ys, s=28, alpha=0.75, color=color, edgecolors="white", linewidths=0.4)
            # Running mean by DFA size.
            by_x: dict[int, list[float]] = {}
            for x, y in zip(xs, ys):
                by_x.setdefault(int(round(x)), []).append(y)
            mx = sorted(by_x)
            my = [float(np.mean(by_x[k])) for k in mx]
            ax.plot(mx, my, color=color, lw=1.6, alpha=0.9)
        ax.set_title(feature_display_name(feat), fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.0, color="0.7", lw=0.6, ls=":")
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=8)

    for ax in axes[2:]:
        ax.set_xlabel("minimized DFA states", fontsize=9)
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("chance-corrected\nfull-hidden accuracy", fontsize=8)

    finalize_grid_figure(
        fig,
        suptitle="Linear readouts vs DFA size (mixed English vocabs)",
        bottom=0.12,
        left=0.12,
        right=0.98,
        top=0.86,
        wspace=0.22,
        hspace=0.40,
    )
    out = sweep_decoding_dir(COMPARISON_NAME) / outfile
    save_figure(fig, out)
    plt.close(fig)
    return out


def _mean_chance_corrected_curve(
    subset: list[dict[str, Any]],
    *,
    feat: str,
    field: str,
    max_k: int,
) -> np.ndarray | None:
    rows: list[list[float]] = []
    chance_vals: list[float] = []
    for p in subset:
        blob = p.get("features", {}).get(feat, {})
        row = blob.get(field) or []
        if not row:
            continue
        rows.append(list(row[:max_k]) + [float("nan")] * max(0, max_k - len(row)))
        if np.isfinite(blob.get("chance", float("nan"))):
            chance_vals.append(float(blob["chance"]))
    if not rows:
        return None
    arr = np.asarray(rows, dtype=float)
    if not np.any(np.isfinite(arr)):
        return None
    with np.errstate(all="ignore"):
        mean = np.nanmean(arr, axis=0)
    chance = float(np.mean(chance_vals)) if chance_vals else 0.0
    return np.asarray([chance_corrected(v, chance) for v in mean], dtype=float)


def _pca_readout_cc(blob: dict[str, Any], *, k: int | None) -> float | None:
    """Chance-corrected accuracy from top-``k`` PCs, or full hidden if ``k`` is None."""
    chance = blob.get("chance", float("nan"))
    if not np.isfinite(chance):
        return None
    if k is None:
        y = blob.get("full_hidden_cc")
        if y is not None and np.isfinite(y):
            return float(y)
        full = blob.get("full_hidden")
        if full is None or not np.isfinite(full):
            return None
        return float(chance_corrected(float(full), float(chance)))
    row = blob.get("by_k") or []
    if k < 1 or k > len(row):
        return None
    v = row[k - 1]
    if v is None or not np.isfinite(v):
        return None
    return float(chance_corrected(float(v), float(chance)))


def plot_decoding_curves_by_dfa_bins(
    payload: dict[str, Any] | None = None,
    *,
    outfile: str = "decoding_curves_by_dfa.png",
    n_bins: int = 4,
    pc_ks: tuple[int | None, ...] = (1, 5, None),
) -> Path:
    """PCA/neuron curves by DFA bin, plus readout-vs-DFA scatters at fixed PC counts."""
    from viz.compare.pow2_sweep_metric_board import _fit_trend

    payload = payload or _load_panels()
    panels = [p for p in payload["panels"] if "error" not in p and p.get("features")]
    if not panels:
        raise FileNotFoundError("no mixed-dfa panels with decoding features")

    dfa_vals = np.asarray([p["n_dfa_states"] for p in panels], dtype=float)
    edges = np.unique(np.quantile(dfa_vals, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        edges = np.asarray([dfa_vals.min() - 0.5, dfa_vals.max() + 0.5])
    n_bin_cols = max(1, len(edges) - 1)
    n_feat = len(_FEATURE_ORDER)
    n_cols = max(n_bin_cols, n_feat)
    n_k_rows = len(pc_ks)
    n_rows = 2 + n_k_rows

    fig = plt.figure(figsize=(3.2 * n_cols + 0.8, 2.15 * n_rows + 1.2))
    gs = fig.add_gridspec(n_rows, n_cols, height_ratios=[1.05, 1.05] + [1.0] * n_k_rows)
    axes = np.empty((n_rows, n_cols), dtype=object)
    for r in range(n_rows):
        for c in range(n_cols):
            axes[r, c] = fig.add_subplot(gs[r, c])

    max_k = int(payload.get("max_k", _DEFAULT_MAX_PCS))
    ks = np.arange(1, max_k + 1, dtype=float)
    row_specs = (
        ("by_k", "PCA dims (k)", "PCA"),
        ("by_k_neurons", "# neurons (k)", "random neurons"),
    )
    words_cmap = plt.get_cmap("YlOrRd")
    words_norm = plt.Normalize(vmin=1.0, vmax=25.0)

    for bi in range(n_bin_cols):
        lo, hi = float(edges[bi]), float(edges[bi + 1])
        if bi == n_bin_cols - 1:
            subset = [p for p in panels if lo <= p["n_dfa_states"] <= hi]
            title = f"DFA {int(round(lo))}–{int(round(hi))}"
        else:
            subset = [p for p in panels if lo <= p["n_dfa_states"] < hi]
            title = f"DFA {int(round(lo))}–{int(round(hi - 1e-9))}"

        for ri, (field, xlabel, basis_label) in enumerate(row_specs):
            ax = axes[ri, bi]
            for feat in _FEATURE_ORDER:
                y = _mean_chance_corrected_curve(
                    subset, feat=feat, field=field, max_k=max_k,
                )
                if y is None:
                    continue
                ax.plot(
                    ks,
                    y,
                    color=DECODE_FEATURE_COLORS[feat],
                    lw=1.6,
                    label=feature_display_name(feat) if (ri == 0 and bi == 0) else None,
                )
            if ri == 0:
                ax.set_title(f"{title}  (n={len(subset)})", fontsize=8)
            ax.set_xlim(1, max_k)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=6)
            if bi == 0:
                ax.set_ylabel(f"{basis_label}\nchance-corr. acc.", fontsize=7, labelpad=6)
            if ri == 1:
                ax.set_xlabel(xlabel, fontsize=7)
            else:
                hide_x_tick_labels(ax)
    for bi in range(n_bin_cols, n_cols):
        axes[0, bi].set_axis_off()
        axes[1, bi].set_axis_off()

    k_ylabels = {
        1: "1 PC",
        5: "5 PCs",
        None: "full hidden\n(all H)",
    }
    for ki, k in enumerate(pc_ks):
        ri = 2 + ki
        for bi, feat in enumerate(_FEATURE_ORDER):
            ax = axes[ri, bi]
            xs: list[float] = []
            ys: list[float] = []
            ns: list[float] = []
            for p in panels:
                y = _pca_readout_cc(p.get("features", {}).get(feat, {}), k=k)
                if y is None:
                    continue
                xs.append(float(p["n_dfa_states"]))
                ys.append(y)
                ns.append(float(p["n_words"]))
            color = DECODE_FEATURE_COLORS[feat]
            if xs:
                x = np.asarray(xs, dtype=float)
                y = np.asarray(ys, dtype=float)
                n_words = np.asarray(ns, dtype=float)
                ax.scatter(
                    x, y,
                    c=n_words, cmap=words_cmap, norm=words_norm,
                    s=14, alpha=0.8, linewidths=0.25, edgecolors="white", zorder=2,
                )
                x_fit, y_fit, r2, _model = _fit_trend(x, y)
                if x_fit is not None and y_fit is not None and np.isfinite(r2):
                    ax.plot(x_fit, y_fit, color=color, lw=1.3, zorder=3)
                    if ki == 0:
                        ax.set_title(
                            f"{feature_display_name(feat)}\n$R^2$={r2:.2f}",
                            fontsize=7.5, pad=3, color=color,
                        )
                    else:
                        ax.set_title(f"$R^2$={r2:.2f}", fontsize=7, pad=2, color=color)
                elif ki == 0:
                    ax.set_title(feature_display_name(feat), fontsize=7.5, pad=3, color=color)
            elif ki == 0:
                ax.set_title(feature_display_name(feat), fontsize=7.5, pad=3, color=color)
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(0.0, color="0.7", lw=0.6, ls=":")
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=6)
            if ki == n_k_rows - 1:
                ax.set_xlabel("DFA states", fontsize=7)
            else:
                hide_x_tick_labels(ax)
            if bi == 0:
                ax.set_ylabel(
                    f"{k_ylabels.get(k, f'{k} PCs')}\nchance-corr. acc.",
                    fontsize=6.5, labelpad=6,
                )
        for bi in range(n_feat, n_cols):
            axes[ri, bi].set_axis_off()

    if axes[0, 0].get_legend_handles_labels()[0]:
        axes[0, 0].legend(fontsize=6, frameon=False, loc="lower right")

    cax = fig.add_axes([0.92, 0.08, 0.014, 0.42])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=words_cmap, norm=words_norm),
        cax=cax,
    )
    cbar.set_label("# words", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    finalize_grid_figure(
        fig,
        suptitle="Readouts by DFA bin; PC-count probes vs DFA size",
        bottom=0.06,
        left=0.11,
        right=0.90,
        top=0.90,
        wspace=0.28,
        hspace=0.55,
    )
    out = sweep_decoding_dir(COMPARISON_NAME) / outfile
    save_figure(fig, out)
    plt.close(fig)
    return out


def plot_spectra_vs_dfa(
    payload: dict[str, Any] | None = None,
    *,
    outfile: str = "pc_spectra_vs_dfa.png",
) -> Path:
    """Overlay closed-loop PC scree curves; color = DFA state count."""
    payload = payload or _load_panels()
    panels = [p for p in payload["panels"] if p.get("spectrum_pct")]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    max_pcs = int(payload.get("max_k", _DEFAULT_MAX_PCS))
    ks = np.arange(1, max_pcs + 1, dtype=float)

    dfa_vals = [float(p["n_dfa_states"]) for p in panels]
    vmin = min(dfa_vals) if dfa_vals else 0.0
    vmax = max(dfa_vals) if dfa_vals else 1.0
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-6))

    for panel in panels:
        y = np.asarray(panel["spectrum_pct"], dtype=float)
        n = min(len(y), max_pcs)
        ax.plot(
            ks[:n],
            y[:n],
            color=cmap(norm(float(panel["n_dfa_states"]))),
            lw=1.1,
            alpha=0.75,
        )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("DFA states", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.set_xlabel("PC index", fontsize=9)
    ax.set_ylabel("% variance", fontsize=9)
    ax.set_xlim(1, max_pcs)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=8)

    finalize_grid_figure(
        fig,
        suptitle="Closed-loop PC spectra (colored by DFA size)",
        bottom=0.16,
        left=0.14,
        right=0.88,
        top=0.86,
    )
    out = sweep_figures_dir(COMPARISON_NAME) / outfile
    save_figure(fig, out)
    plt.close(fig)
    return out


def plot_training_vs_dfa(
    payload: dict[str, Any] | None = None,
    *,
    outfile: str = "training_vs_dfa.png",
) -> Path:
    """Word-error / iters-to-target vs DFA size."""
    payload = payload or _load_panels()
    panels = [p for p in payload["panels"] if isinstance(p.get("training"), dict) and "error" not in p.get("training", {})]

    metrics = [
        ("demo_word_error_pct", "demo word error (%)"),
        ("best_metric_word_error_pct", "best metric word error (%)"),
        ("iter_to_threshold", "iters to word-error target"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    for ax, (key, ylabel) in zip(axes, metrics):
        xs, ys = [], []
        for p in panels:
            tr = p["training"]
            if key not in tr or tr[key] is None:
                continue
            val = float(tr[key])
            if not np.isfinite(val):
                continue
            xs.append(float(p["n_dfa_states"]))
            ys.append(val)
        if xs:
            ax.scatter(xs, ys, s=28, alpha=0.75, color="#4C78A8", edgecolors="white", linewidths=0.4)
        ax.set_xlabel("DFA states", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)

    finalize_grid_figure(
        fig,
        suptitle="Training outcomes vs DFA size",
        bottom=0.20,
        left=0.08,
        right=0.98,
        top=0.84,
        wspace=0.35,
    )
    out = comparison_dir(COMPARISON_NAME, "learning_curves")
    out.mkdir(parents=True, exist_ok=True)
    path = out / outfile
    save_figure(fig, path)
    plt.close(fig)
    return path


def _nwords_dfa_xy(
    payload: dict[str, Any] | None = None,
) -> tuple[list[float], list[float]]:
    man_path = sweep_data_dir(COMPARISON_NAME) / "run_manifest.json"
    if man_path.is_file():
        runs = json.loads(man_path.read_text(encoding="utf-8"))["runs"]
        return (
            [float(r["n_words"]) for r in runs],
            [float(r["n_dfa_states"]) for r in runs],
        )
    payload = payload or _load_panels()
    panels = payload["panels"]
    return (
        [float(p["n_words"]) for p in panels],
        [float(p["n_dfa_states"]) for p in panels],
    )


def plot_nwords_vs_dfa_sanity(
    payload: dict[str, Any] | None = None,
    *,
    outfile: str = "nwords_vs_dfa.png",
) -> Path:
    """Show that n_words and DFA size are related but not identical (sanity / caption)."""
    xs, ys = _nwords_dfa_xy(payload)

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.scatter(xs, ys, s=36, alpha=0.8, color="#E45756", edgecolors="white", linewidths=0.5)
    ax.set_xlabel("# words sampled", fontsize=9)
    ax.set_ylabel("minimized DFA states", fontsize=9)
    ax.grid(True, alpha=0.25)
    finalize_grid_figure(
        fig,
        suptitle="Vocabulary size vs DFA complexity",
        bottom=0.18,
        left=0.18,
        right=0.96,
        top=0.86,
    )
    out = sweep_figures_dir(COMPARISON_NAME) / outfile
    save_figure(fig, out)
    plt.close(fig)
    return out


_OVERVIEW_METRICS: tuple[tuple[str, str, bool], ...] = (
    ("geometry.state_space.loop_top2_variance_frac", "loop top-2 var frac", False),
    ("geometry.state_space.loop_effective_dim", "loop effective dim", False),
    ("training.iter_to_threshold", "iters to 3% word err", True),
)


def plot_mixed_dfa_scaling_overview(
    payload: dict[str, Any] | None = None,
    *,
    outfile: str = "scaling_overview.png",
    recompute: bool = False,
) -> Path:
    """Paper overview: vocab–DFA sanity + PC spectra + key metric-vs-DFA scatters."""
    from viz.compare.pow2_sweep_metric_board import _fit_trend

    decode_payload = payload or _load_panels()
    decode_panels = [
        p for p in decode_payload["panels"]
        if "error" not in p and p.get("spectrum_pct")
    ]
    metric_path = collect_mixed_dfa_metric_board(recompute=recompute)
    metric_panels = [
        p for p in json.loads(metric_path.read_text(encoding="utf-8"))["panels"]
        if "error" not in p
    ]

    n_met = len(_OVERVIEW_METRICS)
    fig = plt.figure(figsize=(10.8, 7.0))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.2, 1.0])
    ax_nw = fig.add_subplot(gs[0, 0:2])
    ax_sp = fig.add_subplot(gs[0, 2:6])
    metric_axes = [
        fig.add_subplot(gs[1, i * 2:(i + 1) * 2]) for i in range(n_met)
    ]

    xs, ys = _nwords_dfa_xy(decode_payload)
    ax_nw.scatter(xs, ys, s=28, alpha=0.8, color="#E45756", edgecolors="white", linewidths=0.4)
    ax_nw.set_xlabel("# words", fontsize=8)
    ax_nw.set_ylabel("DFA states", fontsize=8)
    ax_nw.set_title("vocab size vs DFA", fontsize=9, pad=4)
    ax_nw.grid(True, alpha=0.25)
    ax_nw.tick_params(labelsize=7)

    max_pcs = int(decode_payload.get("max_k", _DEFAULT_MAX_PCS))
    ks = np.arange(1, max_pcs + 1, dtype=float)
    dfa_vals = [float(p["n_dfa_states"]) for p in decode_panels]
    vmin = min(dfa_vals) if dfa_vals else 0.0
    vmax = max(dfa_vals) if dfa_vals else 1.0
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-6))
    for panel in decode_panels:
        y = np.asarray(panel["spectrum_pct"], dtype=float)
        n = min(len(y), max_pcs)
        ax_sp.plot(
            ks[:n], y[:n],
            color=cmap(norm(float(panel["n_dfa_states"]))),
            lw=1.0, alpha=0.75,
        )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_sp = fig.colorbar(sm, ax=ax_sp, pad=0.015, fraction=0.035)
    cbar_sp.set_label("DFA states", fontsize=7)
    cbar_sp.ax.tick_params(labelsize=6)
    ax_sp.set_xlabel("PC index", fontsize=8)
    ax_sp.set_ylabel("% variance", fontsize=8)
    ax_sp.set_title("closed-loop PC spectra", fontsize=9, pad=4)
    ax_sp.set_xlim(1, max_pcs)
    ax_sp.grid(True, alpha=0.25)
    ax_sp.tick_params(labelsize=7)

    words_cmap = plt.get_cmap("YlOrRd")
    words_norm = plt.Normalize(vmin=1.0, vmax=25.0)
    for ax, (path_key, title, log_y) in zip(metric_axes, _OVERVIEW_METRICS):
        mx: list[float] = []
        my: list[float] = []
        mn: list[float] = []
        for p in metric_panels:
            y = _dig(p, path_key)
            if y is None:
                continue
            mx.append(float(p["n_dfa_states"]))
            my.append(y)
            mn.append(float(p["n_words"]))
        if len(mx) < 3:
            ax.set_axis_off()
            continue
        x = np.asarray(mx, dtype=float)
        y = np.asarray(my, dtype=float)
        n_words = np.asarray(mn, dtype=float)
        use_log = bool(log_y and np.all(y > 0))
        y_plot = np.log10(np.clip(y, 1e-12, None)) if use_log else y
        ax.scatter(
            x, y_plot,
            c=n_words, cmap=words_cmap, norm=words_norm,
            s=16, alpha=0.75, linewidths=0.25, edgecolors="white", zorder=2,
        )
        x_fit, y_fit, r2, _model = _fit_trend(x, y_plot)
        panel_title = title
        if x_fit is not None and y_fit is not None and np.isfinite(r2):
            ax.plot(x_fit, y_fit, color="#111111", lw=1.15, zorder=3)
            panel_title = f"{title}\n$R^2$={r2:.2f}"
        ax.set_title(panel_title, fontsize=8, pad=4)
        ax.set_xlabel("DFA states", fontsize=7)
        if use_log:
            ax.set_ylabel("log10", fontsize=7)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=6)

    cax = fig.add_axes([0.92, 0.12, 0.015, 0.30])
    cbar_w = fig.colorbar(
        plt.cm.ScalarMappable(cmap=words_cmap, norm=words_norm),
        cax=cax,
    )
    cbar_w.set_label("# words", fontsize=7)
    cbar_w.ax.tick_params(labelsize=6)

    finalize_grid_figure(
        fig,
        suptitle="Mixed-vocab scaling with DFA size",
        bottom=0.10,
        left=0.07,
        right=0.90,
        top=0.88,
        wspace=0.35,
        hspace=0.45,
    )
    out = sweep_figures_dir(COMPARISON_NAME) / outfile
    save_figure(fig, out)
    plt.close(fig)
    return out


_WEIGHT_DFA_N_FINAL = 5


def _pick_mixed_dfa_span_examples(
    *,
    n_final: int = _WEIGHT_DFA_N_FINAL,
    seed: int = 1,
    model_type: str = "rnn",
) -> list[tuple[int, int, str]]:
    """Pick the smallest available DFA sizes in ascending order."""
    by_dfa: dict[int, list[tuple[int, str]]] = {}
    for entry in iter_runs():
        task = entry["task"]
        if not checkpoint_path(task, model_type, seed=seed).is_file():
            continue
        n_dfa = _dfa_states(list(entry["words"]))
        by_dfa.setdefault(n_dfa, []).append((int(entry["n_words"]), task))

    levels = sorted(by_dfa)
    if not levels:
        raise FileNotFoundError(f"no mixed-dfa weight checkpoints for seed {seed}")

    chosen = levels[: max(1, min(n_final, len(levels)))]
    out: list[tuple[int, int, str]] = []
    for lv in chosen:
        cands = sorted(by_dfa[lv], key=lambda t: (t[0], t[1]))
        n_words, task = cands[0]
        out.append((lv, n_words, task))
    return out


def plot_mixed_dfa_weight_matrices_by_dfa(
    *,
    outfile: str = "weight_matrices_by_dfa.png",
    seed: int = 1,
    n_final: int = _WEIGHT_DFA_N_FINAL,
    model_type: str = "rnn",
) -> Path:
    """Init Wxh/Whh plus clustered finals at the smallest successive DFA sizes.

    Each matrix panel is color-scaled independently. A third row shows, per column,
    overlaid histograms of W_xh and W_hh (signed).
    """
    from visualize import load_model_for_viz, weights_for_plot
    from viz.weight_structure import _cluster_unit_order, init_weights_for_model

    examples = _pick_mixed_dfa_span_examples(n_final=n_final, seed=seed, model_type=model_type)
    _dfa_hi, _nw_hi, task_init = examples[-1]
    model0 = load_model_for_viz(str(checkpoint_path(task_init, model_type, seed=seed)), model_type)
    w_in_i, w_rec_i, w_out_i = init_weights_for_model(model0, seed)
    dale0 = model0.get("dale_signs")
    if dale0 is not None and len(dale0) == w_in_i.shape[0]:
        from rnn.rnn_dyn import dale_signs_ordered, permute_hidden_by_dale

        if not dale_signs_ordered(np.asarray(dale0)):
            w_in_i, w_rec_i, w_out_i, _, _ = permute_hidden_by_dale(
                w_in_i, w_rec_i, w_out_i, np.zeros(w_in_i.shape[0]), np.asarray(dale0),
            )
    order_i = _cluster_unit_order(w_in_i, w_rec_i)
    init_xh = w_in_i[order_i].T
    init_hh = w_rec_i[np.ix_(order_i, order_i)]

    finals: list[tuple[str, np.ndarray, np.ndarray]] = []
    for n_dfa, n_words, task in examples:
        model = load_model_for_viz(str(checkpoint_path(task, model_type, seed=seed)), model_type)
        w_in_f, w_rec_f, _w_out_f, _dale = weights_for_plot(model)
        order_f = _cluster_unit_order(w_in_f, w_rec_f)
        label = f"DFA={n_dfa}\n{n_words}w"
        finals.append((label, w_in_f[order_f].T, w_rec_f[np.ix_(order_f, order_f)]))

    n_cols = 1 + len(finals)
    fig = plt.figure(figsize=(1.55 * n_cols + 0.6, 4.55))
    gs = fig.add_gridspec(
        3, n_cols,
        height_ratios=[0.55, 1.0, 0.62],
        wspace=0.22,
        hspace=0.42,
        left=0.09,
        right=0.99,
        top=0.86,
        bottom=0.09,
    )
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(3)])
    cmap = plt.cm.RdBu_r

    xh_panels = [init_xh] + [xh for _lab, xh, _hh in finals]
    hh_panels = [init_hh] + [hh for _lab, _xh, hh in finals]
    col_titles = ["Init"] + [lab for lab, *_ in finals]
    row_panels = (xh_panels, hh_panels)
    row_ylabels = (r"$W_{xh}$", r"$W_{hh}$")
    # Avoid red/blue (reserved for signed weight heatmaps / E–I).
    xh_color = "#2ca02c"
    hh_color = "#ff1493"

    for row in range(2):
        for col in range(n_cols):
            ax = axes[row, col]
            data = row_panels[row][col]
            # Independent color scale per matrix panel (not shared across row/column).
            vmax = max(float(np.max(np.abs(data))), 1e-9)
            ax.imshow(
                data, aspect="auto", cmap=cmap,
                vmin=-vmax, vmax=vmax,
                interpolation="nearest", origin="lower",
            )
            if row == 0:
                ax.set_title(col_titles[col], fontsize=7, pad=6)
            ax.text(
                0.97, 0.97, f"±{vmax:.2g}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=5.5, color="0.2",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75),
            )
            ny, nx = data.shape
            if row == 1:
                ax.set_xticks([0, max(nx - 1, 0)])
                ax.tick_params(axis="x", labelsize=5)
            else:
                ax.set_xticks([])
            if col == 0:
                ax.set_ylabel(row_ylabels[row], fontsize=9, labelpad=4)
            else:
                ax.set_ylabel("")
            ax.set_yticks([0, max(ny - 1, 0)])
            ax.tick_params(axis="y", labelsize=5, labelleft=(col == 0))

    axes[1, 0].set_xlabel("hidden unit", fontsize=6)
    if n_cols > 1:
        axes[1, -1].set_xlabel("source h", fontsize=6)

    xh_flat = [xh.ravel() for xh in xh_panels]
    hh_flat = [hh.ravel() for hh in hh_panels]
    w_min = min(
        min(float(np.min(a)) for a in xh_flat),
        min(float(np.min(a)) for a in hh_flat),
    )
    w_max = max(
        max(float(np.max(a)) for a in xh_flat),
        max(float(np.max(a)) for a in hh_flat),
    )
    if not np.isfinite(w_min) or not np.isfinite(w_max) or w_min == w_max:
        w_min, w_max = -1e-3, 1e-3
    bins = np.linspace(w_min, w_max, 41)
    ymax = 0.0
    for arr_xh, arr_hh in zip(xh_flat, hh_flat):
        for arr in (arr_xh, arr_hh):
            counts, _ = np.histogram(arr, bins=bins, density=True)
            if counts.size:
                ymax = max(ymax, float(np.max(counts)))
    ymax = max(ymax * 1.05, 1e-9)

    for col in range(n_cols):
        ax = axes[2, col]
        ax.hist(
            xh_flat[col], bins=bins, density=True, histtype="stepfilled",
            alpha=0.45, color=xh_color, edgecolor=xh_color, linewidth=0.8,
            label=r"$W_{xh}$",
        )
        ax.hist(
            hh_flat[col], bins=bins, density=True, histtype="stepfilled",
            alpha=0.45, color=hh_color, edgecolor=hh_color, linewidth=0.8,
            label=r"$W_{hh}$",
        )
        ax.axvline(0.0, color="0.5", linewidth=0.5, linestyle=":")
        ax.set_xlim(w_min, w_max)
        ax.set_ylim(0.0, ymax)
        ax.tick_params(axis="both", labelsize=5)
        ax.set_xlabel(r"$W$", fontsize=6)
        if col == 0:
            ax.set_ylabel("density", fontsize=7)
            ax.legend(fontsize=5.5, frameon=False, loc="upper right", handlelength=1.0)
        else:
            ax.tick_params(axis="y", labelleft=False)

    fig.suptitle(
        rf"Weight matrices by DFA size (seed {seed})",
        fontsize=10,
        y=0.97,
    )
    out = sweep_figures_dir(COMPARISON_NAME) / outfile
    save_figure(fig, out, dpi=150)
    print(f"wrote {out}", flush=True)
    return out


_METRIC_SPECS: tuple[tuple[str, str, bool], ...] = (
    # (path.in.panel, display title, log_y)
    ("geometry.state_space.loop_top2_variance_frac", "loop top-2 var frac", False),
    ("geometry.state_space.corpus_top2_variance_frac", "corpus top-2 var frac", False),
    ("geometry.state_space.loop_effective_dim", "loop effective dim", False),
    ("geometry.state_space.corpus_effective_dim", "corpus effective dim", False),
    ("geometry.state_space.loop_dims_90pct", "loop dims to 90%", False),
    ("geometry.full_space.planarity_top2", "planarity top-2", False),
    ("geometry.full_space.turn_regularity", "turn regularity", False),
    ("training.iter_to_threshold", "iters to 3% word err", True),
    ("training.demo_word_error_pct", "word error (demo %)", False),
    ("training.best_metric_word_error_pct", "best word error %", False),
    ("weights.final.input_over_recurrent_norm", "input / recurrent Frobenius", False),
    ("weights.final.mean_input_drive_fraction", "mean input-drive fraction", False),
    ("weights.motif_final.hh_adjacent_corr", r"$W_{hh}$ adjacent |corr|", False),
    ("weights.motif_final.xh_top1_mass", r"$W_{xh}$ top-1 mass", False),
    ("weights.motif_final.cluster_cohesion_xh", r"$W_{xh}$ cluster cohesion", False),
    ("weights.motif_final.hh_within_between_ratio", r"$W_{hh}$ within/between |w|", False),
    ("decoding.position_from_end.full_hidden_cc", "pos-from-end readout", False),
    ("decoding.dfa.full_hidden_cc", "DFA-state readout", False),
)


def _dig(panel: dict[str, Any], path: str) -> float | None:
    cur: Any = panel
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    try:
        val = float(cur)
    except (TypeError, ValueError):
        return None
    return val if np.isfinite(val) else None


def collect_mixed_dfa_metric_board(
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    model_type: str = "rnn",
    recompute: bool = True,
) -> Path:
    """Geometry + weight metrics for every run (for DFA correlation scatters)."""
    from viz.compare._data import load_task_viz_context
    from viz.compare.geometry import compute_panel_geometry
    from viz.compare.pow2_sweep_weights import _metrics_for_checkpoint

    data_dir = sweep_data_dir(COMPARISON_NAME)
    out_path = data_dir / "mixed_dfa_metric_board.json"
    decode_path = data_dir / "mixed_dfa_panels.json"
    decode_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    if decode_path.is_file():
        for p in json.loads(decode_path.read_text(encoding="utf-8")).get("panels", []):
            if "error" in p:
                continue
            decode_by_key[(p["task"], int(p["seed"]))] = p

    if out_path.is_file() and not recompute:
        return out_path

    panels: list[dict[str, Any]] = []
    for entry in iter_runs():
        task = entry["task"]
        words = list(entry["words"])
        n_dfa = _dfa_states(words)
        for seed in seeds:
            if not checkpoint_path(task, model_type, seed=seed).is_file():
                print(f"  skip metrics {task} seed {seed}", flush=True)
                continue
            print(f"  metrics {task} seed {seed}  dfa={n_dfa}", flush=True)
            row: dict[str, Any] = {
                "task": task,
                "seed": seed,
                "run_id": entry["run_id"],
                "n_words": entry["n_words"],
                "n_dfa_states": n_dfa,
                "words": words,
            }
            try:
                ctx = load_task_viz_context(task, model_type=model_type, seed=seed)
                row["geometry"] = compute_panel_geometry(ctx)
            except Exception as exc:  # noqa: BLE001
                row["geometry"] = {"error": str(exc)}
            try:
                w = _metrics_for_checkpoint(task, seed=seed, model_type=model_type)
                row["weights"] = w if w is not None else {"error": "missing"}
            except Exception as exc:  # noqa: BLE001
                row["weights"] = {"error": str(exc)}
            try:
                row["training"] = _training_metrics(
                    task, model_type=model_type, seed=seed, n_words=int(entry["n_words"]),
                )
            except Exception as exc:  # noqa: BLE001
                row["training"] = {"error": str(exc)}
            dec = decode_by_key.get((task, seed))
            if dec is not None:
                row["decoding"] = dec.get("features", {})
                row["spectrum_pct"] = dec.get("spectrum_pct", [])
            panels.append(row)

    payload = {
        "comparison": COMPARISON_NAME,
        "model_type": model_type,
        "seeds": list(seeds),
        "panels": panels,
    }
    out_path.write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")
    print(f"wrote {out_path}", flush=True)
    return out_path


def plot_metrics_vs_dfa(
    *,
    outfile: str = "metrics_vs_dfa.png",
    min_r2: float = 0.1,
    wrap: int = 4,
    recompute: bool = False,
) -> Path:
    """Paper-style grid: geometry / training / weight metrics vs DFA size."""
    from viz.compare.pow2_sweep_metric_board import _fit_line, _fit_trend

    path = collect_mixed_dfa_metric_board(recompute=recompute)
    payload = json.loads(path.read_text(encoding="utf-8"))
    panels = [p for p in payload["panels"] if "error" not in p]

    prepared: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, bool, np.ndarray | None, np.ndarray | None]] = []
    for path_key, title, log_y in _METRIC_SPECS:
        xs: list[float] = []
        ys: list[float] = []
        ns: list[float] = []
        for p in panels:
            y = _dig(p, path_key)
            if y is None:
                continue
            xs.append(float(p["n_dfa_states"]))
            ys.append(y)
            ns.append(float(p["n_words"]))
        if len(xs) < 4:
            continue
        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
        n_words = np.asarray(ns, dtype=float)
        use_log = bool(log_y and np.all(y > 0))
        y_plot = np.log10(np.clip(y, 1e-12, None)) if use_log else y
        _coef, r2_lin = _fit_line(x, y_plot)
        if not np.isfinite(r2_lin) or r2_lin < min_r2:
            continue
        x_fit, y_fit, r2, _model = _fit_trend(x, y_plot)
        if x_fit is None or not np.isfinite(r2):
            continue
        prepared.append((
            f"{title}\n$R^2$={r2:.2f}",
            x, y_plot, n_words, use_log, x_fit, y_fit,
        ))

    if not prepared:
        raise ValueError(f"no metrics with R^2 >= {min_r2}")

    # Prefer paper priority; skip overview + full-hidden decoding (shown in fig 15/16).
    skip_titles = {t for _k, t, _log in _OVERVIEW_METRICS} | {
        "pos-from-end readout",
        "DFA-state readout",
    }
    priority = (
        "corpus top-2 var frac",
        "corpus effective dim",
        "input / recurrent Frobenius",
        r"$W_{hh}$ adjacent |corr|",
        r"$W_{xh}$ top-1 mass",
        "planarity top-2",
        "mean input-drive fraction",
        r"$W_{xh}$ cluster cohesion",
    )
    ranked: list[tuple[int, tuple]] = []
    for panel in prepared:
        short = panel[0].split("\n")[0]
        if short in skip_titles:
            continue
        try:
            rank = priority.index(short)
        except ValueError:
            rank = 100 + len(ranked)
        ranked.append((rank, panel))
    ranked.sort(key=lambda t: t[0])
    core = [p for r, p in ranked if r < 100][:8]
    prepared = core if core else [p for _, p in ranked[:8]]

    n_metrics = len(prepared)
    n_wrap = max(1, min(wrap, n_metrics))
    n_rows = int(np.ceil(n_metrics / n_wrap))
    fig, axes = plt.subplots(
        n_rows, n_wrap,
        figsize=(2.35 * n_wrap + 0.8, 2.35 * n_rows + 1.1),
        squeeze=False,
        sharex=True,
    )
    words_cmap = plt.get_cmap("YlOrRd")
    words_norm = plt.Normalize(vmin=1.0, vmax=25.0)

    for mi, (title, x, y_plot, n_words, use_log, x_fit, y_fit) in enumerate(prepared):
        row, col = divmod(mi, n_wrap)
        ax = axes[row][col]
        ax.scatter(
            x, y_plot,
            c=n_words, cmap=words_cmap, norm=words_norm,
            s=18, alpha=0.75, linewidths=0.25, edgecolors="white", zorder=2,
        )
        if x_fit is not None and y_fit is not None:
            ax.plot(x_fit, y_fit, color="#111111", lw=1.2, zorder=3)
        ax.set_title(title, fontsize=7, pad=6)
        ax.tick_params(labelsize=6)
        if row == n_rows - 1:
            ax.set_xlabel("DFA states", fontsize=7)
        else:
            hide_x_tick_labels(ax)
        if use_log:
            ax.set_ylabel("log10", fontsize=6)
        ax.grid(True, alpha=0.2)

    for mi in range(n_metrics, n_rows * n_wrap):
        row, col = divmod(mi, n_wrap)
        axes[row][col].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=words_cmap, norm=words_norm)
    sm.set_array([])
    cax = fig.add_axes([0.92, 0.18, 0.015, 0.55])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("# words", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    finalize_grid_figure(
        fig,
        suptitle="Metrics vs DFA size (mixed English vocabs)",
        bottom=0.14,
        left=0.08,
        right=0.90,
        top=0.78,
        wspace=0.40,
        hspace=0.55,
    )
    out = sweep_figures_dir(COMPARISON_NAME) / outfile
    save_figure(fig, out)
    plt.close(fig)
    return out


def _thin_learning_snaps(
    snaps: list[Path],
    *,
    max_snaps: int = 30,
    early_frac: float = 0.12,
) -> list[Path]:
    """Keep all early snaps (for zoom) plus evenly spaced later ones."""
    if max_snaps < 2 or len(snaps) <= max_snaps:
        return snaps
    iters = np.asarray([int(s.stem.split("_")[1]) for s in snaps], dtype=float)
    stop = float(max(iters.max(), 1.0))
    early_mask = (iters / stop) <= float(early_frac)
    early = [s for s, m in zip(snaps, early_mask) if m]
    late = [s for s, m in zip(snaps, early_mask) if not m]
    keep_late = max(2, int(max_snaps) - len(early))
    if len(late) <= keep_late:
        return early + late
    idxs = np.unique(np.round(np.linspace(0, len(late) - 1, keep_late)).astype(int))
    return early + [late[i] for i in idxs]


def _collect_learning_decode_seed(
    task: str,
    *,
    seed: int,
    model_type: str = "rnn",
    pc_ks: tuple[int | None, ...] = (1, 5, None),
    max_snaps: int = 30,
) -> tuple[list[dict[str, Any]], int]:
    """Decode each sparse learning snap for one seed; return rows and stop_iter."""
    from rnn.learning_snaps import list_learning_snaps
    from viz.compare._data import load_task_viz_context

    ckpt = checkpoint_path(task, model_type, seed=seed)
    snaps = _thin_learning_snaps(list_learning_snaps(ckpt), max_snaps=max_snaps)
    if not snaps:
        raise FileNotFoundError(
            f"no learning snaps for {task} seed {seed} under {ckpt.stem}_learning/"
        )

    from experiment import TASKS
    cfg = TASKS[task]
    # Learning curves need many snaps; cap rollout so 15×50 snaps stay tractable.
    metric_len = int(cfg.get("metric_rollout_len", cfg.get("viz_length", 50)))
    text_chars = min(metric_len, 500)

    rows: list[dict[str, Any]] = []
    for snap in snaps:
        print(f"  seed {seed} decode {snap.name}", flush=True)
        meta = np.load(snap, allow_pickle=True)
        iteration = int(meta["learning_snap_iteration"]) if "learning_snap_iteration" in meta.files else int(
            snap.stem.split("_")[1]
        )
        word_err = (
            float(meta["learning_snap_word_err"])
            if "learning_snap_word_err" in meta.files
            else float("nan")
        )
        ctx = load_task_viz_context(
            task,
            model_type=model_type,
            seed=seed,
            text_chars=text_chars,
            checkpoint=snap,
        )
        panel = compute_panel_decoding(
            ctx,
            max_k=max((k for k in pc_ks if k is not None), default=5),
        )
        feat_out: dict[str, Any] = {}
        for feat in _FEATURE_ORDER:
            blob = panel.get("features", {}).get(feat, {})
            if blob.get("error"):
                feat_out[feat] = {"error": blob["error"]}
                continue
            chance = float(blob.get("chance", float("nan")))
            by_k = blob.get("by_k") or []
            basis_vals: dict[str, float] = {}
            for k in pc_ks:
                if k is None:
                    full = blob.get("full_hidden")
                    y = blob.get("full_hidden_cc")
                    if y is None and full is not None and np.isfinite(full) and np.isfinite(chance):
                        y = chance_corrected(float(full), chance)
                    basis_vals["full"] = float(y) if y is not None and np.isfinite(y) else float("nan")
                else:
                    raw = by_k[k - 1] if k <= len(by_k) else None
                    if raw is None or not np.isfinite(raw) or not np.isfinite(chance):
                        basis_vals[f"pc{k}"] = float("nan")
                    else:
                        basis_vals[f"pc{k}"] = float(chance_corrected(float(raw), chance))
            feat_out[feat] = {"chance": chance, **basis_vals}
        rows.append({
            "iteration": iteration,
            "word_err": word_err,
            "snap": snap.name,
            "features": feat_out,
        })

    rows.sort(key=lambda r: int(r["iteration"]))
    stop_iter = max(int(r["iteration"]) for r in rows) if rows else 1
    for r in rows:
        r["progress"] = float(r["iteration"]) / float(max(stop_iter, 1))
    return rows, stop_iter


def _interp_learning_curve(
    progress: np.ndarray,
    values: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    order = np.argsort(progress)
    xp = np.asarray(progress, dtype=float)[order]
    yp = np.asarray(values, dtype=float)[order]
    mask = np.isfinite(xp) & np.isfinite(yp)
    if int(mask.sum()) < 2:
        return np.full(grid.shape, np.nan, dtype=float)
    return np.interp(grid, xp[mask], yp[mask], left=np.nan, right=np.nan)


_LEARNING_PC_KS: tuple[int | None, ...] = (1, 2, 3, 4, 5, None)


def _learning_basis_specs(pc_ks: tuple[int | None, ...] | None = None) -> tuple[tuple[str, str], ...]:
    ks = pc_ks if pc_ks is not None else _LEARNING_PC_KS
    out: list[tuple[str, str]] = []
    for k in ks:
        if k is None:
            out.append(("full", "full H"))
        elif int(k) == 1:
            out.append((f"pc{int(k)}", "1 PC"))
        else:
            out.append((f"pc{int(k)}", f"{int(k)} PCs"))
    return tuple(out)


def _aggregate_learning_decode_rows(
    seed_rows: list[list[dict[str, Any]]],
    *,
    n_grid: int = 101,
    early_xlim: float = 0.12,
    n_early: int = 25,
    basis_keys: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    # Denser samples in the early window so the zoom row has real resolution.
    late_n = max(int(n_grid) - int(n_early) + 1, 20)
    grid = np.unique(
        np.concatenate([
            np.linspace(0.0, float(early_xlim), int(n_early)),
            np.linspace(float(early_xlim), 1.0, late_n),
        ])
    )
    if basis_keys is None:
        basis_keys = tuple(b for b, _ in _learning_basis_specs())
    feat_out: dict[str, Any] = {}
    for feat in _FEATURE_ORDER:
        feat_out[feat] = {}
        for bkey in basis_keys:
            mats = []
            for rows in seed_rows:
                prog = np.asarray([r["progress"] for r in rows], dtype=float)
                vals = np.asarray(
                    [float(r.get("features", {}).get(feat, {}).get(bkey, float("nan"))) for r in rows],
                    dtype=float,
                )
                mats.append(_interp_learning_curve(prog, vals, grid))
            mat = np.vstack(mats)
            feat_out[feat][f"{bkey}_mean"] = np.nanmean(mat, axis=0).tolist()
            feat_out[feat][f"{bkey}_std"] = np.nanstd(mat, axis=0).tolist()

    we_mats = []
    for rows in seed_rows:
        prog = np.asarray([r["progress"] for r in rows], dtype=float)
        vals = np.asarray([r["word_err"] for r in rows], dtype=float)
        we_mats.append(_interp_learning_curve(prog, vals, grid))
    we_mat = np.vstack(we_mats)
    return {
        "progress_grid": grid.tolist(),
        "word_err_mean": np.nanmean(we_mat, axis=0).tolist(),
        "word_err_std": np.nanstd(we_mat, axis=0).tolist(),
        "features": feat_out,
        "n_seeds": len(seed_rows),
        "basis_keys": list(basis_keys),
    }


def collect_learning_decode(
    task: str = "mixeddfa_r26_ns",
    *,
    seeds: tuple[int, ...] = (1,),
    model_type: str = "rnn",
    pc_ks: tuple[int | None, ...] = _LEARNING_PC_KS,
) -> Path:
    """Decode each sparse learning snap; write JSON next to the learning directory."""
    per_seed: list[dict[str, Any]] = []
    all_rows: list[list[dict[str, Any]]] = []
    for seed in seeds:
        rows, stop_iter = _collect_learning_decode_seed(
            task, seed=seed, model_type=model_type, pc_ks=pc_ks,
        )
        per_seed.append({"seed": int(seed), "stop_iter": int(stop_iter), "snaps": rows})
        all_rows.append(rows)

    basis_keys = tuple(b for b, _ in _learning_basis_specs(pc_ks))
    if len(seeds) == 1:
        out = sweep_decoding_dir(COMPARISON_NAME) / f"learning_decode_{task}.json"
        payload: dict[str, Any] = {
            "task": task,
            "seed": int(seeds[0]),
            "model_type": model_type,
            "pc_ks": [k if k is not None else "full" for k in pc_ks],
            "stop_iter": int(per_seed[0]["stop_iter"]),
            "snaps": per_seed[0]["snaps"],
        }
    else:
        out = sweep_decoding_dir(COMPARISON_NAME) / f"learning_decode_{task}_seed_mean.json"
        payload = {
            "task": task,
            "seeds": [int(s) for s in seeds],
            "model_type": model_type,
            "pc_ks": [k if k is not None else "full" for k in pc_ks],
            "aggregated": _aggregate_learning_decode_rows(all_rows, basis_keys=basis_keys),
            "per_seed": per_seed,
        }
    out.write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return out


def plot_learning_decode(
    task: str = "mixeddfa_r26_ns",
    *,
    json_path: Path | None = None,
    outfile: str | None = None,
    early_xlim: float = 0.15,
) -> Path:
    """Two rows x PC bases (1..5 + full H): full progress, then early zoom."""
    path = json_path or (sweep_decoding_dir(COMPARISON_NAME) / f"learning_decode_{task}.json")
    if not path.is_file():
        alt = sweep_decoding_dir(COMPARISON_NAME) / f"learning_decode_{task}_seed_mean.json"
        path = alt if alt.is_file() else path
    if not path.is_file():
        path = collect_learning_decode(task)

    payload = json.loads(path.read_text(encoding="utf-8"))
    aggregated = payload.get("aggregated")
    rows = payload.get("snaps") or []
    n_seeds = len(payload.get("seeds") or [])
    if aggregated is None and not rows:
        raise FileNotFoundError(f"no snaps in {path}")

    raw_ks = payload.get("pc_ks")
    if raw_ks:
        pc_ks_parsed: list[int | None] = []
        for k in raw_ks:
            if k is None or k == "full":
                pc_ks_parsed.append(None)
            else:
                pc_ks_parsed.append(int(k))
        basis_keys = _learning_basis_specs(tuple(pc_ks_parsed))
    else:
        basis_keys = _learning_basis_specs()

    row_specs = (
        (0.0, 1.02, "full training"),
        (0.0, float(early_xlim), f"early (0-{early_xlim:g})"),
    )
    n_cols = len(basis_keys)
    fig, axes = plt.subplots(
        len(row_specs),
        n_cols,
        figsize=(2.15 * n_cols + 0.6, 2.55 * len(row_specs) + 0.55),
        sharey=True,
        squeeze=False,
    )
    word_err_line = None

    if aggregated is not None:
        progress_all = np.asarray(aggregated["progress_grid"], dtype=float)
        word_err_mean_all = np.asarray(aggregated["word_err_mean"], dtype=float)
        word_err_std_all = np.asarray(aggregated["word_err_std"], dtype=float)
        n_s = int(aggregated.get("n_seeds", n_seeds))
        early_counts = [
            sum(1 for s in (ps.get("snaps") or []) if float(s.get("progress", 1.0)) <= early_xlim)
            for ps in (payload.get("per_seed") or [])
        ]
        mean_early = int(round(float(np.mean(early_counts)))) if early_counts else 0
        title = f"Readout over learning ({task}, mean ± std over {n_s} seeds; ~{mean_early} early snaps/seed)"
    else:
        progress_all = np.asarray([r["progress"] for r in rows], dtype=float)
        word_err_mean_all = np.asarray([r["word_err"] for r in rows], dtype=float)
        word_err_std_all = None
        early_count = int(np.sum(progress_all <= early_xlim))
        title = f"Readout over learning ({task}; {early_count} snaps in early window)"

    for ri, (x0, x1, row_label) in enumerate(row_specs):
        zoom_mask = (progress_all >= x0) & (progress_all <= x1 + 1e-9)
        progress = progress_all[zoom_mask]
        word_err_mean = word_err_mean_all[zoom_mask]
        word_err_std = word_err_std_all[zoom_mask] if word_err_std_all is not None else None
        for ci, (bkey, blabel) in enumerate(basis_keys):
            ax = axes[ri, ci]
            for feat in _FEATURE_ORDER:
                color = DECODE_FEATURE_COLORS[feat]
                if aggregated is not None:
                    blob = aggregated.get("features", {}).get(feat, {})
                    y_mean = np.asarray(blob.get(f"{bkey}_mean", []), dtype=float)
                    y_std = np.asarray(blob.get(f"{bkey}_std", []), dtype=float)
                    if y_mean.size == 0:
                        continue
                    y_mean = y_mean[zoom_mask]
                    if y_std.size == zoom_mask.size:
                        y_std = y_std[zoom_mask]
                    ax.plot(
                        progress,
                        y_mean,
                        color=color,
                        lw=1.6,
                        marker="o" if ri == 1 else None,
                        ms=2.4 if ri == 1 else None,
                        label=feature_display_name(feat) if (ri == 0 and ci == 0) else None,
                    )
                    if y_std.size == y_mean.size and np.any(np.isfinite(y_std)):
                        ax.fill_between(
                            progress,
                            y_mean - y_std,
                            y_mean + y_std,
                            color=color,
                            alpha=0.16,
                            linewidth=0,
                        )
                else:
                    ys = [
                        float(r.get("features", {}).get(feat, {}).get(bkey, float("nan")))
                        for r in rows
                    ]
                    y = np.asarray(ys, dtype=float)[zoom_mask]
                    ax.plot(
                        progress,
                        y,
                        color=color,
                        lw=1.6,
                        marker="o",
                        ms=2.8,
                        label=feature_display_name(feat) if (ri == 0 and ci == 0) else None,
                    )
            if ri == 0:
                ax.set_title(blabel, fontsize=8, pad=3)
            if ri == len(row_specs) - 1:
                ax.set_xlabel("progress", fontsize=7)
            else:
                hide_x_tick_labels(ax)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(x0, x1)
            ax.axhline(0.0, color="0.7", lw=0.6, ls=":")
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=5.5)
            if ci == 0:
                ax.set_ylabel(f"{row_label}\nchance-corr. acc.", fontsize=7, labelpad=5)
            if np.any(np.isfinite(word_err_mean)):
                ax2 = ax.twinx()
                (line,) = ax2.plot(
                    progress, word_err_mean, color="0.45", lw=0.95, ls="--", alpha=0.8,
                )
                if word_err_line is None:
                    word_err_line = line
                if word_err_std is not None and np.any(np.isfinite(word_err_std)):
                    ax2.fill_between(
                        progress,
                        word_err_mean - word_err_std,
                        word_err_mean + word_err_std,
                        color="0.45",
                        alpha=0.10,
                        linewidth=0,
                    )
                ax2.set_ylim(-0.02, 1.05)
                ax2.tick_params(labelsize=5, colors="0.45")
                if ci == n_cols - 1:
                    ax2.set_ylabel("word err", fontsize=6, color="0.45")
                else:
                    ax2.set_yticklabels([])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if word_err_line is not None:
        handles = [*handles, word_err_line]
        labels = [*labels, "word err"]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=len(labels),
        fontsize=6.5,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.4,
    )

    finalize_grid_figure(
        fig,
        suptitle=title,
        bottom=0.11,
        left=0.07,
        right=0.94,
        top=0.90,
        wspace=0.22,
        hspace=0.30,
    )
    if outfile:
        out_name = outfile
    elif aggregated is not None and task.startswith("mixeddfa_r") and task.endswith("_ns"):
        rid = task[len("mixeddfa_"):-len("_ns")]
        out_name = f"learning_decode_{rid}_seed_mean.png"
    elif task.startswith("mixeddfa_r") and task.endswith("_ns"):
        rid = task[len("mixeddfa_"):-len("_ns")]
        out_name = f"learning_decode_{rid}.png"
    else:
        out_name = f"learning_decode_{task.replace('mixeddfa_', '').replace('_ns', '')}.png"
    out = sweep_decoding_dir(COMPARISON_NAME) / out_name
    save_figure(fig, out)
    plt.close(fig)
    return out



def _dfa_quantile_edges(dfa_vals: np.ndarray, n_bins: int = 4) -> np.ndarray:
    edges = np.unique(np.quantile(np.asarray(dfa_vals, dtype=float), np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        lo = float(np.min(dfa_vals)) - 0.5
        hi = float(np.max(dfa_vals)) + 0.5
        edges = np.asarray([lo, hi], dtype=float)
    return edges


def _subset_for_dfa_bin(
    panels: list[dict[str, Any]],
    *,
    edges: np.ndarray,
    bin_index: int,
) -> list[dict[str, Any]]:
    lo = float(edges[bin_index])
    hi = float(edges[bin_index + 1])
    if bin_index == len(edges) - 2:
        return [p for p in panels if lo <= float(p["n_dfa_states"]) <= hi]
    return [p for p in panels if lo <= float(p["n_dfa_states"]) < hi]


def _dfa_bin_title(edges: np.ndarray, bin_index: int) -> str:
    lo = float(edges[bin_index])
    hi = float(edges[bin_index + 1])
    if bin_index == len(edges) - 2:
        return f"DFA {int(round(lo))}-{int(round(hi))}"
    return f"DFA {int(round(lo))}-{int(round(hi - 1e-9))}"


def collect_learning_decode_by_dfa(
    *,
    seed: int = 1,
    model_type: str = "rnn",
    pc_ks: tuple[int | None, ...] = _LEARNING_PC_KS,
    recompute: bool = False,
) -> Path:
    """Learning-decode for every mixed run (default seed 1), aggregated in Fig-12 DFA bins."""
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
        n_words = int(entry["n_words"])
        print(f"learning-decode collect {task} seed {seed}  dfa={n_dfa}", flush=True)
        rows, stop_iter = _collect_learning_decode_seed(
            task, seed=seed, model_type=model_type, pc_ks=pc_ks,
        )
        run_payloads.append({
            "run_id": rid,
            "task": task,
            "seed": int(seed),
            "n_dfa_states": n_dfa,
            "n_words": n_words,
            "stop_iter": int(stop_iter),
            "snaps": rows,
        })

    dfa_vals = np.asarray([r["n_dfa_states"] for r in run_payloads], dtype=float)
    edges = _dfa_quantile_edges(dfa_vals, n_bins=4)
    basis_keys = tuple(b for b, _ in _learning_basis_specs(pc_ks))

    bins: list[dict[str, Any]] = []
    for bi in range(len(edges) - 1):
        subset = _subset_for_dfa_bin(run_payloads, edges=edges, bin_index=bi)
        seed_rows = [r["snaps"] for r in subset]
        agg = (
            _aggregate_learning_decode_rows(seed_rows, basis_keys=basis_keys)
            if seed_rows else None
        )
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
        "model_type": model_type,
        "pc_ks": [k if k is not None else "full" for k in pc_ks],
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
    pc_row_ks: tuple[int | None, ...] = (1, 5, None),
) -> Path:
    """Columns = Fig-12 DFA bins; rows = 1 PC / 5 PCs / full H over early progress."""
    path = json_path or (sweep_decoding_dir(COMPARISON_NAME) / "learning_decode_by_dfa.json")
    if not path.is_file():
        path = collect_learning_decode_by_dfa()
    payload = json.loads(path.read_text(encoding="utf-8"))
    bins = [b for b in payload.get("bins", []) if b.get("aggregated")]
    if not bins:
        raise FileNotFoundError(f"no binned learning curves in {path}")

    raw_ks = payload.get("pc_ks") or [1, 2, 3, 4, 5, "full"]
    parsed: list[int | None] = []
    for k in raw_ks:
        if k is None or k == "full":
            parsed.append(None)
        else:
            parsed.append(int(k))
    all_basis = _learning_basis_specs(tuple(parsed))
    want = {"full" if k is None else f"pc{int(k)}" for k in pc_row_ks}
    basis_keys = tuple((b, lab) for b, lab in all_basis if b in want)
    if not basis_keys:
        basis_keys = (("pc1", "1 PC"), ("pc5", "5 PCs"), ("full", "full H"))

    row_blocks = (
        ("early", 0.0, float(early_xlim)),
    )
    n_basis = len(basis_keys)
    n_bins = len(bins)
    n_rows = len(row_blocks) * n_basis
    fig, axes = plt.subplots(
        n_rows,
        n_bins,
        figsize=(2.6 * n_bins + 0.7, 2.15 * n_rows + 0.7),
        sharey=True,
        squeeze=False,
    )
    word_err_line = None
    seed = int(payload.get("seed", 1))

    for bi, blob in enumerate(bins):
        agg = blob["aggregated"]
        progress_all = np.asarray(agg["progress_grid"], dtype=float)
        we_mean_all = np.asarray(agg["word_err_mean"], dtype=float)
        we_std_all = np.asarray(agg["word_err_std"], dtype=float)
        for block_i, (block_name, x0, x1) in enumerate(row_blocks):
            zoom = (progress_all >= x0) & (progress_all <= x1 + 1e-9)
            progress = progress_all[zoom]
            we_mean = we_mean_all[zoom]
            we_std = we_std_all[zoom]
            for ki, (bkey, blabel) in enumerate(basis_keys):
                ri = block_i * n_basis + ki
                ax = axes[ri, bi]
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
                    ax.plot(
                        progress,
                        y_mean,
                        color=color,
                        lw=1.5,
                        marker="o" if block_name == "early" else None,
                        ms=2.2 if block_name == "early" else None,
                        label=feature_display_name(feat) if (ri == 0 and bi == 0) else None,
                    )
                    if y_std.size == y_mean.size and np.any(np.isfinite(y_std)):
                        ax.fill_between(
                            progress,
                            y_mean - y_std,
                            y_mean + y_std,
                            color=color,
                            alpha=0.14,
                            linewidth=0,
                        )
                if ri == 0:
                    ax.set_title(f"{blob['title']}  (n={blob['n_runs']})", fontsize=8, pad=3)
                if ri == n_rows - 1:
                    ax.set_xlabel("progress", fontsize=7)
                else:
                    hide_x_tick_labels(ax)
                ax.set_xlim(x0, x1)
                ax.set_ylim(-0.05, 1.05)
                ax.axhline(0.0, color="0.7", lw=0.6, ls=":")
                ax.grid(True, alpha=0.25)
                ax.tick_params(labelsize=5.5)
                if bi == 0:
                    ax.set_ylabel(f"{blabel}\nchance-corr. acc.", fontsize=7, labelpad=4)
                if np.any(np.isfinite(we_mean)):
                    ax2 = ax.twinx()
                    (line,) = ax2.plot(progress, we_mean, color="0.45", lw=0.9, ls="--", alpha=0.8)
                    if word_err_line is None:
                        word_err_line = line
                    if np.any(np.isfinite(we_std)):
                        ax2.fill_between(
                            progress,
                            we_mean - we_std,
                            we_mean + we_std,
                            color="0.45",
                            alpha=0.10,
                            linewidth=0,
                        )
                    ax2.set_ylim(-0.02, 1.05)
                    ax2.tick_params(labelsize=5, colors="0.45")
                    if bi == n_bins - 1:
                        ax2.set_ylabel("word err", fontsize=6, color="0.45")
                    else:
                        ax2.set_yticklabels([])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if word_err_line is not None:
        handles = [*handles, word_err_line]
        labels = [*labels, "word err"]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=len(labels),
        fontsize=6.5,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    n_runs = sum(int(b.get("n_runs", 0)) for b in bins)
    title = "Readout over learning by DFA bin (seed {}, {} mixed runs)".format(seed, n_runs)
    finalize_grid_figure(
        fig,
        suptitle=title,
        top=0.90,
        bottom=0.08,
        left=0.08,
        right=0.94,
        wspace=0.22,
        hspace=0.28,
    )
    out = sweep_decoding_dir(COMPARISON_NAME) / outfile
    save_figure(fig, out)
    plt.close(fig)
    return out


def run_all_mixed_dfa_plots(
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    recompute: bool = True,
) -> list[Path]:
    path = collect_mixed_dfa_panels(seeds=seeds, recompute=recompute)
    payload = json.loads(path.read_text(encoding="utf-8"))
    outs = [
        plot_nwords_vs_dfa_sanity(payload),
        plot_decoding_vs_dfa(payload),
        plot_decoding_curves_by_dfa_bins(payload),
        plot_spectra_vs_dfa(payload),
        plot_training_vs_dfa(payload),
        plot_metrics_vs_dfa(recompute=recompute),
        plot_mixed_dfa_scaling_overview(payload, recompute=False),
        plot_mixed_dfa_weight_matrices_by_dfa(),
    ]
    for p in outs:
        print(f"wrote {p}", flush=True)
    return outs
