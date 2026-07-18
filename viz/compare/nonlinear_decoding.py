"""Linear vs nonlinear readout comparison (sibling to existing linear decoding).

Does not modify ``compute_panel_decoding`` or any ``decoding/`` artifacts.
Writes under ``decoding_linear_vs_nonlinear/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from viz.compare._data import load_task_decoding_context
from viz.compare.decoding import (
    DECODING_FEATURES,
    DECODE_FEATURE_COLORS,
    _DEFAULT_RANDOM_STATE,
    _DEFAULT_TEST_SIZE,
    _fit_probe_accuracy,
    chance_corrected,
    dfa_oracle_cc_from_feat,
    dfa_oracles_from_timestep_labels,
    empirical_null_chance,
    feature_display_name,
    fit_pca_k,
    prepare_decoding_data,
    project_pca_k,
)
from viz.plot_layout import finalize_grid_figure, save_figure

DEFAULT_PC_KS: tuple[int, ...] = (1, 5, 15)
MLP_HIDDEN = (64,)


def _stratified_split(
    x: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = _DEFAULT_TEST_SIZE,
    random_state: int = _DEFAULT_RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int] | None:
    """Same filtering + stratified split as ``decode_feature_curve``."""
    from sklearn.model_selection import train_test_split

    y_arr = np.asarray(y)
    if len(y_arr) < 4:
        return None
    classes, counts = np.unique(y_arr, return_counts=True)
    supported = set(classes[counts >= 2])
    if len(supported) < 2:
        return None
    keep = np.array([yi in supported for yi in y_arr])
    x = np.asarray(x, dtype=float)[keep]
    y_arr = y_arr[keep]
    if len(y_arr) < 4:
        return None
    n_classes = len(supported)
    chance = float(1.0 / n_classes)
    n = len(y_arr)
    abs_test = max(int(np.ceil(n * test_size)), 1)
    if abs_test < n_classes:
        abs_test = n_classes
    if n - abs_test < n_classes:
        return None
    try:
        train_idx, test_idx = train_test_split(
            np.arange(n),
            test_size=abs_test / n,
            random_state=random_state,
            stratify=y_arr,
        )
    except ValueError:
        return None
    return (
        x[train_idx], y_arr[train_idx],
        x[test_idx], y_arr[test_idx],
        chance, n_classes,
    )


def _fit_mlp_accuracy(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    n_classes: int,
) -> float:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    del n_classes
    enc = LabelEncoder()
    try:
        ytr = enc.fit_transform(y_train)
        yte = enc.transform(y_test)
    except ValueError:
        return float("nan")
    if len(np.unique(ytr)) < 2:
        return float("nan")

    scaler = StandardScaler()
    try:
        xtr = scaler.fit_transform(x_train)
        xte = scaler.transform(x_test)
    except ValueError:
        return float("nan")

    # early_stopping needs enough val samples; disable on tiny splits.
    n_tr = len(ytr)
    use_early = n_tr >= 40
    clf = MLPClassifier(
        hidden_layer_sizes=MLP_HIDDEN,
        activation="relu",
        alpha=1e-3,
        max_iter=500 if use_early else 800,
        early_stopping=use_early,
        validation_fraction=0.15 if use_early else 0.1,
        n_iter_no_change=20,
        random_state=_DEFAULT_RANDOM_STATE,
    )
    try:
        clf.fit(xtr, ytr)
        return float(clf.score(xte, yte))
    except ValueError:
        return float("nan")


def probe_linear_and_nonlinear(
    x: np.ndarray,
    y: np.ndarray,
    *,
    pc_ks: tuple[int, ...] = DEFAULT_PC_KS,
) -> dict[str, Any] | None:
    """Same train/test split: linear logistic vs small MLP on full H and selected PCs."""
    split = _stratified_split(x, y)
    if split is None:
        return None
    x_train, y_train, x_test, y_test, chance, n_classes = split

    acc_lin = _fit_probe_accuracy(
        x_train, y_train, x_test, y_test, n_classes=n_classes, k=None,
    )
    acc_mlp = _fit_mlp_accuracy(
        x_train, y_train, x_test, y_test, n_classes=n_classes,
    )

    by_k: dict[str, Any] = {}
    for k in pc_ks:
        if k < 1 or k >= x_train.shape[1]:
            continue
        _, mean, components = fit_pca_k(x_train, k)
        xtr = project_pca_k(x_train, mean, components)
        xte = project_pca_k(x_test, mean, components)
        by_k[str(k)] = {
            "linear": _fit_probe_accuracy(
                xtr, y_train, xte, y_test, n_classes=n_classes, k=k,
            ),
            "mlp": _fit_mlp_accuracy(
                xtr, y_train, xte, y_test, n_classes=n_classes,
            ),
        }

    return {
        "chance": chance,
        "n_classes": int(n_classes),
        "n_samples": int(len(y_train) + len(y_test)),
        "full_hidden": {"linear": float(acc_lin), "mlp": float(acc_mlp)},
        "by_pc_k": by_k,
    }


def compute_panel_linear_vs_nonlinear(
    ctx,
    *,
    features: tuple[str, ...] | None = None,
    pc_ks: tuple[int, ...] = DEFAULT_PC_KS,
) -> dict[str, Any]:
    """Per-feature linear vs MLP probe panel for one (task, seed)."""
    from unit_selectivity import build_timestep_labels
    from vocab_diagrams import build_minimized_vocabulary_automaton

    feats = features if features is not None else DECODING_FEATURES
    feature_data = prepare_decoding_data(ctx, features=feats)
    automaton = build_minimized_vocabulary_automaton(ctx.words)
    ts_labels = build_timestep_labels(
        ctx.text, automaton,
        spaced=ctx.spaced, words=ctx.words,
        model=ctx.model,
        activations=np.asarray(ctx.hidden_states, dtype=float),
    )
    dfa_oracles = dfa_oracles_from_timestep_labels(ts_labels, feats)

    features_out: dict[str, Any] = {}
    for feat in feats:
        x_feat, y_feat = feature_data[feat]
        probe = probe_linear_and_nonlinear(x_feat, y_feat, pc_ks=pc_ks)
        if probe is None:
            features_out[feat] = {"error": "insufficient samples or classes"}
            continue
        chance = float(probe["chance"])
        entry: dict[str, Any] = {
            "chance": chance,
            "null_chance": empirical_null_chance(feat, y_feat),
            "n_classes": probe["n_classes"],
            "n_samples": probe["n_samples"],
            "full_hidden_linear": probe["full_hidden"]["linear"],
            "full_hidden_mlp": probe["full_hidden"]["mlp"],
            "full_hidden_linear_cc": chance_corrected(probe["full_hidden"]["linear"], chance),
            "full_hidden_mlp_cc": chance_corrected(probe["full_hidden"]["mlp"], chance),
            "by_pc_k": {},
        }
        oracle = dfa_oracles.get(feat)
        if oracle is not None:
            entry["dfa_oracle"] = float(oracle)
            entry["dfa_oracle_cc"] = chance_corrected(float(oracle), chance)
        for k_str, blob in probe["by_pc_k"].items():
            entry["by_pc_k"][k_str] = {
                "linear": blob["linear"],
                "mlp": blob["mlp"],
                "linear_cc": chance_corrected(blob["linear"], chance),
                "mlp_cc": chance_corrected(blob["mlp"], chance),
            }
        features_out[feat] = entry

    return {
        "task": ctx.task,
        "seed": int(ctx.seed),
        "hidden_size": int(ctx.hidden_states.shape[1]),
        "n_timesteps": int(ctx.hidden_states.shape[0]),
        "features": features_out,
        "feature_order": list(feats),
        "pc_ks": list(pc_ks),
        "mlp_hidden": list(MLP_HIDDEN),
        "probe": "logistic_vs_mlp",
    }


def plot_linear_vs_nonlinear_panel(
    panel: dict[str, Any],
    outfile: Path,
    *,
    title: str | None = None,
) -> Path:
    """Grouped bars: linear vs MLP (chance-corrected) for full H and selected PCs."""
    feats = [f for f in panel.get("feature_order", DECODING_FEATURES)
             if f in panel.get("features", {}) and "error" not in panel["features"][f]]
    pc_ks = [str(k) for k in panel.get("pc_ks", DEFAULT_PC_KS)]
    bases = ["full"] + [f"pc{k}" for k in pc_ks]
    n_feat = len(feats)
    n_bases = len(bases)
    if n_feat == 0:
        raise FileNotFoundError("no features to plot in linear-vs-nonlinear panel")

    fig, axes = plt.subplots(
        1, n_bases,
        figsize=(2.4 * n_bases + 1.2, 3.6),
        sharey=True,
        squeeze=False,
    )
    x = np.arange(n_feat)
    width = 0.36

    for bi, basis in enumerate(bases):
        ax = axes[0, bi]
        lin_vals: list[float] = []
        mlp_vals: list[float] = []
        oracle_vals: list[float | None] = []
        for feat in feats:
            blob = panel["features"][feat]
            if basis == "full":
                lin_vals.append(float(blob.get("full_hidden_linear_cc", np.nan)))
                mlp_vals.append(float(blob.get("full_hidden_mlp_cc", np.nan)))
            else:
                k = basis[2:]
                sub = (blob.get("by_pc_k") or {}).get(k, {})
                lin_vals.append(float(sub.get("linear_cc", np.nan)))
                mlp_vals.append(float(sub.get("mlp_cc", np.nan)))
            oracle_vals.append(dfa_oracle_cc_from_feat(blob) if feat != "dfa" else None)

        ax.bar(
            x - width / 2, lin_vals, width,
            color="#4C78A8", edgecolor="0.2", linewidth=0.5, label="linear",
        )
        ax.bar(
            x + width / 2, mlp_vals, width,
            color="#F58518", edgecolor="0.2", linewidth=0.5, label="MLP",
        )
        for i, o in enumerate(oracle_vals):
            if o is None or not np.isfinite(o):
                continue
            ax.hlines(
                o, i - 0.45, i + 0.45,
                colors="0.35", linestyles="--", linewidths=1.0, zorder=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [feature_display_name(f) for f in feats],
            rotation=35, ha="right", fontsize=6.5,
        )
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.0, color="0.7", lw=0.6, ls=":")
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(labelsize=6.5)
        label = "full H" if basis == "full" else f"top {basis[2:]} PCs"
        ax.set_title(label, fontsize=9)
        if bi == 0:
            ax.set_ylabel("chance-corr. acc.", fontsize=8)
            ax.legend(fontsize=7, frameon=False, loc="lower right")

    from matplotlib.lines import Line2D

    fig.legend(
        [Line2D([0], [0], color="0.35", ls="--", lw=1.0)],
        ["DFA-state oracle"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        fontsize=7,
        frameon=False,
    )
    task = panel.get("task", "")
    seed = panel.get("seed", "")
    finalize_grid_figure(
        fig,
        suptitle=title or (
            f"Linear vs nonlinear readout ({task}, seed {seed}; "
            f"MLP{list(MLP_HIDDEN)})"
        ),
        top=0.86,
        bottom=0.22,
        left=0.08,
        right=0.99,
        wspace=0.18,
    )
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, outfile, dpi=150)
    plt.close(fig)
    print(f"wrote {outfile}", flush=True)
    return outfile


def run_task_linear_vs_nonlinear(
    task: str,
    out_dir: Path | str,
    *,
    model_type: str = "rnn",
    seed: int = 1,
    features: tuple[str, ...] | None = None,
    pc_ks: tuple[int, ...] = DEFAULT_PC_KS,
) -> dict[str, Any]:
    """Compute + plot linear vs MLP for one task/seed."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ctx = load_task_decoding_context(task, model_type=model_type, seed=seed)
    panel = compute_panel_linear_vs_nonlinear(ctx, features=features, pc_ks=pc_ks)
    json_path = out / "linear_vs_nonlinear.json"
    json_path.write_text(json.dumps(panel, indent=2), encoding="utf-8")
    print(f"wrote {json_path}", flush=True)
    plot_linear_vs_nonlinear_panel(panel, out / "linear_vs_nonlinear.png")
    return panel


def collect_mixed_linear_vs_nonlinear(
    *,
    model_type: str = "rnn",
    seeds: tuple[int, ...] | None = None,
    features: tuple[str, ...] | None = None,
    pc_ks: tuple[int, ...] = (5, 15),
    max_tasks: int | None = None,
) -> dict[str, Any]:
    """Full-H (+ sparse PC-k) linear vs MLP across mixed-vocab runs."""
    from vocab_mixed_dfa import COMPARISON_NAME, iter_runs
    from experiment import seeds_for_task

    feats = features if features is not None else DECODING_FEATURES
    panels: list[dict[str, Any]] = []
    tasks_seen = 0
    for entry in iter_runs():
        if max_tasks is not None and tasks_seen >= max_tasks:
            break
        task = entry["task"]
        seed_list = list(seeds) if seeds is not None else sorted(seeds_for_task(task, model_type))
        if not seed_list:
            continue
        # Default: seed 1 only (full multi-seed is expensive with MLP).
        if seeds is None:
            seed_list = [1] if 1 in seed_list else seed_list[:1]
        tasks_seen += 1
        for seed in seed_list:
            print(f"linear-vs-nonlinear {task} seed {seed} ...", flush=True)
            try:
                ctx = load_task_decoding_context(task, model_type=model_type, seed=int(seed))
                panel = compute_panel_linear_vs_nonlinear(ctx, features=feats, pc_ks=pc_ks)
                panel["n_dfa_states"] = int(
                    __import__("vocab_diagrams", fromlist=["build_minimized_vocabulary_automaton"])
                    .build_minimized_vocabulary_automaton(list(entry["words"])).dfa._n
                )
                panel["n_words"] = int(entry["n_words"])
                panels.append(panel)
            except Exception as exc:  # noqa: BLE001 — keep sweep going
                panels.append({
                    "task": task, "seed": int(seed), "error": str(exc),
                })
    return {
        "comparison": COMPARISON_NAME,
        "model_type": model_type,
        "features": list(feats),
        "pc_ks": list(pc_ks),
        "mlp_hidden": list(MLP_HIDDEN),
        "panels": panels,
    }


def plot_mixed_linear_vs_nonlinear_vs_dfa(
    payload: dict[str, Any],
    outfile: Path,
) -> Path:
    """Scatter: chance-corr linear vs MLP full-H accuracy against DFA size."""
    feats = tuple(payload.get("features") or DECODING_FEATURES)
    panels = [p for p in payload.get("panels", []) if "error" not in p and p.get("features")]
    n_feat = len(feats)
    fig, axes = plt.subplots(
        2, n_feat,
        figsize=(2.35 * n_feat + 0.6, 5.0),
        squeeze=False,
    )
    for fi, feat in enumerate(feats):
        xs, lin_y, mlp_y, delta = [], [], [], []
        for p in panels:
            blob = p.get("features", {}).get(feat, {})
            if "full_hidden_linear_cc" not in blob:
                continue
            xs.append(float(p["n_dfa_states"]))
            lin_y.append(float(blob["full_hidden_linear_cc"]))
            mlp_y.append(float(blob["full_hidden_mlp_cc"]))
            delta.append(float(blob["full_hidden_mlp_cc"]) - float(blob["full_hidden_linear_cc"]))
        color = DECODE_FEATURE_COLORS.get(feat, "#888")
        ax0 = axes[0, fi]
        ax1 = axes[1, fi]
        if xs:
            ax0.scatter(xs, lin_y, s=12, c=color, alpha=0.55, label="linear", marker="o")
            ax0.scatter(xs, mlp_y, s=12, c="0.25", alpha=0.55, label="MLP", marker="^")
            ax1.scatter(xs, delta, s=12, c=color, alpha=0.65)
            ax1.axhline(0.0, color="0.5", lw=0.8, ls="--")
        ax0.set_ylim(-0.05, 1.05)
        ax0.set_title(feature_display_name(feat), fontsize=8, color=color)
        ax0.grid(True, alpha=0.25)
        ax0.tick_params(labelsize=6)
        ax1.set_xlabel("DFA states", fontsize=7)
        ax1.grid(True, alpha=0.25)
        ax1.tick_params(labelsize=6)
        if fi == 0:
            ax0.set_ylabel("chance-corr. acc.\n(full H)", fontsize=7)
            ax1.set_ylabel("MLP − linear", fontsize=7)
            ax0.legend(fontsize=6, frameon=False, loc="lower left")
        else:
            ax0.set_yticklabels([])
            ax1.set_yticklabels([])

    finalize_grid_figure(
        fig,
        suptitle="Linear vs nonlinear full-H readout across mixed DFA sizes",
        top=0.90,
        bottom=0.10,
        left=0.08,
        right=0.99,
        wspace=0.22,
        hspace=0.32,
    )
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, outfile, dpi=150)
    plt.close(fig)
    print(f"wrote {outfile}", flush=True)
    return outfile


def run_mixed_linear_vs_nonlinear(
    *,
    recompute: bool = True,
    max_tasks: int | None = None,
    seeds: tuple[int, ...] | None = None,
) -> tuple[Path, Path]:
    from vocab_mixed_dfa import COMPARISON_NAME
    from viz.compare.sweep_output import sweep_decoding_dir

    out_dir = sweep_decoding_dir(COMPARISON_NAME) / "linear_vs_nonlinear"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "mixed_dfa_linear_vs_nonlinear.json"
    fig_path = out_dir / "linear_vs_nonlinear_vs_dfa.png"
    if json_path.is_file() and not recompute:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    else:
        payload = collect_mixed_linear_vs_nonlinear(max_tasks=max_tasks, seeds=seeds)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {json_path}", flush=True)
    plot_mixed_linear_vs_nonlinear_vs_dfa(payload, fig_path)
    return json_path, fig_path
