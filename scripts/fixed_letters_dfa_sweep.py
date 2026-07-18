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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "train", "collect", "plot", "all"))
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto", "gpu"))
    parser.add_argument("--runs", type=int, nargs="+", default=None)
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
    elif args.command == "all":
        cmd_plan(args)
        cmd_train(args)
        seed = args.seeds[0] if args.seeds else 1
        collect_panels(seed=seed)
        plot_spectra(seed=seed)


if __name__ == "__main__":
    main()
