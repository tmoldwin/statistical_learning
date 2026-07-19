"""Fixed small-alphabet synthetic vocab sweep (DFA axis, |Σ| held fixed)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment import EXPERIMENTS_ROOT, checkpoint_path
from vocab_fixed_letters_dfa import (
    ALPHABET,
    COMPARISON_NAME,
    DEFAULT_SEEDS,
    N_LETTERS,
    N_RUNS,
    iter_runs,
    write_run_manifest,
)
from viz.compare._data import load_task_decoding_context
from viz.compare.mixed_dfa_viz import _loop_pc_spectrum, _pad_spectrum
from viz.compare.decoding import _DEFAULT_MAX_PCS
from viz.compare.sweep_output import sweep_data_dir, sweep_figures_dir
from viz.plot_layout import finalize_grid_figure, save_figure


def _train_one(task: str, seeds: tuple[int, ...], *, smoke: bool, device: str) -> None:
    need = [s for s in seeds if not checkpoint_path(task, "rnn", seed=s).is_file()]
    if not need:
        return
    cmd = [
        sys.executable, "scripts/run_task.py", task,
        "--models", "rnn",
        "--seeds", *[str(s) for s in need],
        "--skip-viz",
        "--device", device,
    ]
    if smoke:
        cmd.append("--smoke")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def cmd_plan(args: argparse.Namespace) -> None:
    path = write_run_manifest(sweep_data_dir(COMPARISON_NAME) / "run_manifest.json")
    seeds = tuple(args.seeds) if args.seeds else DEFAULT_SEEDS
    missing = sum(
        1 for e in iter_runs() for s in seeds
        if not checkpoint_path(e["task"], "rnn", seed=s).is_file()
    )
    dfas = [int(e["n_dfa_states"]) for e in iter_runs()]
    print(f"fixed-letters DFA sweep: {N_RUNS} runs  |Alphabet|={N_LETTERS} alphabet={ALPHABET!r}")
    print(f"comparison: {COMPARISON_NAME}")
    print(f"DFA states: {min(dfas)}-{max(dfas)}  unique={len(set(dfas))}")
    print(f"manifest: {path}")
    print(f"missing checkpoints (seeds={list(seeds)}): {missing}")


def cmd_train(args: argparse.Namespace) -> None:
    seeds = tuple(args.seeds) if args.seeds else DEFAULT_SEEDS
    run_ids = set(args.runs) if args.runs else None
    tasks = [
        e["task"] for e in iter_runs()
        if run_ids is None or int(e["run_id"]) in run_ids
    ]
    jobs = max(1, int(args.jobs))
    print(f"training {len(tasks)} tasks x seeds={list(seeds)} jobs={jobs}", flush=True)
    if jobs == 1:
        for t in tasks:
            print(f"  train {t}", flush=True)
            _train_one(t, seeds, smoke=args.smoke, device=args.device)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futs = {
                pool.submit(_train_one, t, seeds, smoke=args.smoke, device=args.device): t
                for t in tasks
            }
            for fut in as_completed(futs):
                t = futs[fut]
                fut.result()
                print(f"  done {t}", flush=True)


def collect_panels(*, seed: int = 1, max_k: int = _DEFAULT_MAX_PCS) -> Path:
    panels = []
    for entry in iter_runs():
        task = entry["task"]
        ckpt = checkpoint_path(task, "rnn", seed=seed)
        if not ckpt.is_file():
            panels.append({**entry, "seed": seed, "error": f"missing {ckpt}"})
            continue
        print(f"spectrum {task} seed {seed} dfa={entry['n_dfa_states']}", flush=True)
        try:
            ctx = load_task_decoding_context(task, model_type="rnn", seed=seed)
            spectrum = _loop_pc_spectrum(ctx)
            panels.append({
                "task": task,
                "seed": seed,
                "run_id": int(entry["run_id"]),
                "n_words": int(entry["n_words"]),
                "n_letters": int(entry["n_letters"]),
                "n_dfa_states": int(entry["n_dfa_states"]),
                "words": list(entry["words"]),
                "alphabet": ALPHABET,
                "spectrum_pct": _pad_spectrum(spectrum, max_k).tolist() if len(spectrum) else [],
            })
        except Exception as exc:  # noqa: BLE001
            panels.append({**entry, "seed": seed, "error": str(exc)})
    out = sweep_data_dir(COMPARISON_NAME) / "fixed_letters_dfa_panels.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "comparison": COMPARISON_NAME,
        "alphabet": ALPHABET,
        "n_letters": N_LETTERS,
        "seed": seed,
        "max_k": max_k,
        "panels": panels,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return out


def plot_spectra(*, seed: int = 1) -> Path:
    path = sweep_data_dir(COMPARISON_NAME) / "fixed_letters_dfa_panels.json"
    if not path.is_file():
        path = collect_panels(seed=seed)
    payload = json.loads(path.read_text(encoding="utf-8"))
    panels = [p for p in payload["panels"] if p.get("spectrum_pct") and "error" not in p]
    if not panels:
        raise FileNotFoundError("no spectrum panels")

    max_pcs = int(payload.get("max_k", _DEFAULT_MAX_PCS))
    ks = np.arange(1, max_pcs + 1, dtype=float)
    dfa_vals = [float(p["n_dfa_states"]) for p in panels]
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=min(dfa_vals), vmax=max(dfa_vals))

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for p in sorted(panels, key=lambda r: int(r["n_dfa_states"])):
        y = np.asarray(p["spectrum_pct"], dtype=float)
        n = min(len(y), max_pcs)
        ax.plot(
            ks[:n], y[:n],
            color=cmap(norm(float(p["n_dfa_states"]))),
            lw=1.2, alpha=0.85,
        )
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, pad=0.02)
    cbar.set_label("DFA states", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.set_xlabel("PC index", fontsize=9)
    ax.set_ylabel("% variance", fontsize=9)
    ax.set_xlim(1, max_pcs)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=8)

    n_let = int(payload.get("n_letters", N_LETTERS))
    finalize_grid_figure(
        fig,
        suptitle=(
            f"PC spectra vs DFA with fixed alphabet |Σ|={n_let} ({ALPHABET})  "
            f"seed {seed}; synthetic words"
        ),
        top=0.86, bottom=0.14, left=0.12, right=0.88,
    )
    out = sweep_figures_dir(COMPARISON_NAME) / "pc_spectra_vs_dfa_fixed_letters.png"
    save_figure(fig, out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}", flush=True)
    return out


def cmd_weight_spikiness(*, seed: int = 1) -> Path:
    from viz.compare.weight_spikiness import (
        collect_weight_spikiness_panels,
        plot_weight_spikiness_vs_dfa,
    )

    collect_weight_spikiness_panels(
        comparison=COMPARISON_NAME,
        runs=list(iter_runs()),
        seed=seed,
    )
    return plot_weight_spikiness_vs_dfa(
        comparison=COMPARISON_NAME,
        seed=seed,
        title=(
            f"Weight spikiness / ergodicity vs DFA  "
            f"|Σ|={N_LETTERS} ({ALPHABET}), seed {seed}"
        ),
    )


ABLATION_RUN_IDS: tuple[int, ...] = (0, 9, 19)  # DFA ~6, 24, 44
ABLATION_CONDS: tuple[tuple[str, dict], ...] = (
    ("baseline", {"dropout": 0.25, "l2_lambda": 1e-4, "l2_on": "all"}),
    ("dropout0", {"dropout": 0.0, "l2_lambda": 1e-4, "l2_on": "all"}),
    ("l2_0", {"dropout": 0.25, "l2_lambda": 0.0, "l2_on": "none"}),
    ("l2_hh", {"dropout": 0.25, "l2_lambda": 1e-4, "l2_on": "hh"}),
)


def _ablation_root() -> Path:
    return EXPERIMENTS_ROOT / "comparisons" / COMPARISON_NAME / "ablations"


def _ablation_model_path(cond: str, run_id: int, *, seed: int = 1) -> Path:
    return _ablation_root() / cond / f"r{run_id:02d}" / "rnn" / f"model_seed{seed}.npz"


def cmd_ablate_weights(args: argparse.Namespace) -> None:
    """Train low/mid/high DFA under dropout/L2 ablations; plot norms + dynamics."""
    from visualize import load_model_for_viz, weights_for_plot
    from viz.weight_structure import (
        compute_weight_spikiness_pair,
        compute_weight_structure_metrics,
    )

    seed = args.seeds[0] if args.seeds else 1
    run_ids = tuple(args.runs) if args.runs else ABLATION_RUN_IDS
    by_id = {int(e["run_id"]): e for e in iter_runs()}
    smoke = bool(args.smoke)

    # --- train ---
    jobs = []
    for cond, kw in ABLATION_CONDS:
        for rid in run_ids:
            if rid not in by_id:
                raise KeyError(f"run_id {rid} not in fixed-letters plan")
            task = by_id[rid]["task"]
            out = _ablation_model_path(cond, rid, seed=seed)
            log_every = 100 if cond == "baseline" and rid in (run_ids[0], run_ids[-1]) else 0
            jobs.append((cond, rid, task, out, kw, log_every))

    print(f"ablate-weights: {len(jobs)} jobs seed={seed}", flush=True)
    for cond, rid, task, out, kw, log_every in jobs:
        if out.is_file() and not args.force:
            print(f"  skip existing {cond} r{rid:02d}", flush=True)
            continue
        print(f"  train {cond} r{rid:02d} dfa={by_id[rid]['n_dfa_states']}", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "scripts/run_task.py", task,
            "--models", "rnn",
            "--seeds", str(seed),
            "--skip-viz",
            "--device", args.device,
            "--dropout", str(kw["dropout"]),
            "--l2-lambda", str(kw["l2_lambda"]),
            "--l2-on", str(kw["l2_on"]),
            "--model-out", str(out),
        ]
        if log_every:
            cmd.extend(["--log-weight-norms-every", str(log_every)])
        if smoke:
            cmd.append("--smoke")
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    # --- collect final metrics ---
    panels = []
    for cond, kw in ABLATION_CONDS:
        for rid in run_ids:
            entry = by_id[rid]
            out = _ablation_model_path(cond, rid, seed=seed)
            row = {
                "cond": cond,
                "run_id": rid,
                "task": entry["task"],
                "n_dfa_states": int(entry["n_dfa_states"]),
                "n_words": int(entry["n_words"]),
                "dropout": kw["dropout"],
                "l2_lambda": kw["l2_lambda"],
                "l2_on": kw["l2_on"],
                "seed": seed,
            }
            if not out.is_file():
                row["error"] = f"missing {out}"
                panels.append(row)
                continue
            model = load_model_for_viz(str(out), "rnn")
            w_in, w_rec, w_out, _ = weights_for_plot(model)
            row["structure"] = compute_weight_structure_metrics(w_in, w_rec, w_out)
            row["spikiness"] = compute_weight_spikiness_pair(w_in, w_rec)
            panels.append(row)

    data_dir = sweep_data_dir(COMPARISON_NAME)
    data_dir.mkdir(parents=True, exist_ok=True)
    json_path = data_dir / "weight_ablation_fixed_letters.json"
    json_path.write_text(json.dumps({"seed": seed, "panels": panels}, indent=2), encoding="utf-8")
    print(f"wrote {json_path}", flush=True)

    # --- plot final norms by condition ---
    ok = [p for p in panels if "structure" in p]
    conds = [c for c, _ in ABLATION_CONDS]
    colors = {
        "baseline": "#1f77b4",
        "dropout0": "#ff7f0e",
        "l2_0": "#2ca02c",
        "l2_hh": "#d62728",
    }
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
    for cond in conds:
        rows = sorted([p for p in ok if p["cond"] == cond], key=lambda r: r["n_dfa_states"])
        if not rows:
            continue
        xs = [r["n_dfa_states"] for r in rows]
        axes[0].plot(xs, [r["structure"]["input_frobenius"] for r in rows],
                     "o-", color=colors[cond], label=cond, lw=1.3, ms=5)
        axes[1].plot(xs, [r["structure"]["recurrent_frobenius"] for r in rows],
                     "o-", color=colors[cond], label=cond, lw=1.3, ms=5)
        axes[2].plot(xs, [r["structure"]["input_over_recurrent_norm"] for r in rows],
                     "o-", color=colors[cond], label=cond, lw=1.3, ms=5)
    axes[0].set_ylabel(r"$||W_{xh}||_F$", fontsize=8)
    axes[1].set_ylabel(r"$||W_{hh}||_F$", fontsize=8)
    axes[2].set_ylabel(r"$||W_{xh}||_F / ||W_{hh}||_F$", fontsize=8)
    for ax, title in zip(axes, ["input norm", "recurrent norm", "input / recurrent"]):
        ax.set_xlabel("DFA states", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, loc="best", framealpha=0.9)
    finalize_grid_figure(
        fig,
        suptitle=f"Fixed-|Σ| weight ablations (seed {seed})",
        top=0.82, bottom=0.16, left=0.08, right=0.98, wspace=0.35,
    )
    fig_path = sweep_figures_dir(COMPARISON_NAME) / "weight_ablation_fixed_letters.png"
    save_figure(fig, fig_path, dpi=150)
    plt.close(fig)
    print(f"wrote {fig_path}", flush=True)

    # --- dynamics plot (baseline low vs high) ---
    dyn_paths = []
    for rid in (run_ids[0], run_ids[-1]):
        p = Path(str(_ablation_model_path("baseline", rid, seed=seed)) + ".norms.jsonl")
        if p.is_file() and p.stat().st_size > 0:
            dyn_paths.append((rid, int(by_id[rid]["n_dfa_states"]), p))
    if dyn_paths:
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), sharey=False)
        for ax, (rid, dfa, path) in zip(axes, dyn_paths):
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            it = [r["iter"] for r in rows]
            ax.plot(it, [r["input_frobenius"] for r in rows], color="#1f77b4", lw=1.2, label=r"$W_{xh}$")
            ax.plot(it, [r["recurrent_frobenius"] for r in rows], color="#d62728", lw=1.2, label=r"$W_{hh}$")
            ax.set_title(f"baseline training  DFA={dfa} (r{rid:02d})", fontsize=8)
            ax.set_xlabel("iteration", fontsize=8)
            ax.set_ylabel("Frobenius norm", fontsize=8)
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7, loc="best")
        finalize_grid_figure(
            fig,
            suptitle="When do norms reallocate? (baseline, fixed alphabet)",
            top=0.82, bottom=0.16, left=0.10, right=0.98, wspace=0.28,
        )
        dyn_out = sweep_figures_dir(COMPARISON_NAME) / "weight_norm_dynamics_fixed_letters.png"
        save_figure(fig, dyn_out, dpi=150)
        plt.close(fig)
        print(f"wrote {dyn_out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "plan", "train", "collect", "plot", "weight-spikiness",
            "ablate-weights", "all",
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto", "gpu"))
    parser.add_argument("--runs", type=int, nargs="+", default=None)
    parser.add_argument("--force", action="store_true", help="retrain ablations even if present")
    args = parser.parse_args()

    (EXPERIMENTS_ROOT / "comparisons" / COMPARISON_NAME).mkdir(parents=True, exist_ok=True)

    if args.command == "plan":
        cmd_plan(args)
    elif args.command == "train":
        cmd_plan(args)
        cmd_train(args)
    elif args.command == "collect":
        seed = args.seeds[0] if args.seeds else 1
        collect_panels(seed=seed)
    elif args.command == "plot":
        seed = args.seeds[0] if args.seeds else 1
        collect_panels(seed=seed)
        plot_spectra(seed=seed)
    elif args.command == "weight-spikiness":
        seed = args.seeds[0] if args.seeds else 1
        cmd_weight_spikiness(seed=seed)
    elif args.command == "ablate-weights":
        cmd_ablate_weights(args)
    elif args.command == "all":
        cmd_plan(args)
        cmd_train(args)
        seed = args.seeds[0] if args.seeds else 1
        collect_panels(seed=seed)
        plot_spectra(seed=seed)
        cmd_weight_spikiness(seed=seed)

if __name__ == "__main__":
    main()

