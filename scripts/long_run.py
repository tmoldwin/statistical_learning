"""Long unconstrained RNN run on the full mixed English bank, then motif board."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment import COMPARISONS_ROOT, checkpoint_path
from rnn.learning_snaps import list_learning_snaps
from vocab_long_run import (
    COMPARISON_NAME,
    DEFAULT_SEEDS,
    STEPS,
    TASK,
    n_dfa_states,
    words,
    write_run_manifest,
)
from viz.compare.sweep_output import sweep_data_dir

MOTIF_DIR = COMPARISONS_ROOT / COMPARISON_NAME / "motifs"
CURVE_DIR = COMPARISONS_ROOT / COMPARISON_NAME / "learning_curves"
CENSUS_JSON = MOTIF_DIR / "long_run_rnn_motif_edge_signed_all.json"
EDGES_JSON = MOTIF_DIR / "long_run_rnn_motif_counts_over_learning.json"
BOARD_OUT = MOTIF_DIR / "long_run_rnn_motif_board.png"
CURVE_OUT = CURVE_DIR / "live_curve.png"
METRICS_JSON = sweep_data_dir(COMPARISON_NAME) / "live_metrics.json"

_METRIC_RE = re.compile(
    r"metric iter (?P<it>\d+), word_err: (?P<we>[-0-9.]+)%, val CE: (?P<ce>[-0-9.]+)/char"
)


def cmd_plan(_args: argparse.Namespace) -> None:
    manifest = write_run_manifest(sweep_data_dir(COMPARISON_NAME) / "run_manifest.json")
    vocab = words()
    n_dfa = n_dfa_states()
    ckpt = checkpoint_path(TASK, "rnn", seed=DEFAULT_SEEDS[0])
    snaps = list_learning_snaps(ckpt) if ckpt.is_file() else []
    print(f"long_run: task={TASK}  words={len(vocab)}  DFA={n_dfa}")
    print(f"comparison: {COMPARISON_NAME}")
    print(f"manifest: {manifest}")
    print(f"checkpoint: {ckpt}  exists={ckpt.is_file()}  snaps={len(snaps)}")


def cmd_train(args: argparse.Namespace) -> None:
    seeds = tuple(args.seeds) if args.seeds else DEFAULT_SEEDS
    cmd_plan(args)
    cmd = [
        sys.executable, "scripts/run_task.py", TASK,
        "--models", "rnn",
        "--seeds", *[str(s) for s in seeds],
        "--skip-viz",
        "--device", args.device,
        "--save-learning-snaps",
    ]
    if args.smoke:
        cmd.append("--smoke")
    print(f">> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _metrics_from_log(log_path: Path) -> list[dict]:
    rows: list[dict] = []
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for m in _METRIC_RE.finditer(text):
        rows.append({
            "it": int(m.group("it")),
            "we": float(m.group("we")) / 100.0,
            "ce": float(m.group("ce")),
        })
    return rows


def _metrics_from_snaps(seed: int) -> list[dict]:
    ckpt = checkpoint_path(TASK, "rnn", seed=seed)
    rows: list[dict] = []
    for snap in list_learning_snaps(ckpt):
        d = np.load(snap, allow_pickle=True)
        it = int(d["learning_snap_iteration"]) if "learning_snap_iteration" in d.files else int(snap.stem.split("_")[1])
        we = float(d["learning_snap_word_err"]) if "learning_snap_word_err" in d.files else float("nan")
        rows.append({"it": it, "we": we, "ce": float("nan")})
    return rows


def _load_live_metrics(*, seed: int, log_path: Path | None) -> list[dict]:
    rows = _metrics_from_log(log_path) if log_path is not None and log_path.is_file() else []
    if len(rows) < 2:
        rows = _metrics_from_snaps(seed)
    by_it = {int(r["it"]): r for r in rows}
    return [by_it[k] for k in sorted(by_it)]


def _curve_health(rows: list[dict]) -> dict:
    we = np.array([r["we"] for r in rows], dtype=float)
    it = np.array([r["it"] for r in rows], dtype=int)
    ce = np.array([r["ce"] for r in rows], dtype=float)
    finite = we[np.isfinite(we)]
    last = float(finite[-1]) if len(finite) else float("nan")
    best = float(np.min(finite)) if len(finite) else float("nan")
    if len(finite) >= 8:
        mid = max(1, len(finite) // 2)
        early = float(np.median(finite[:max(2, mid // 2)]))
        late = float(np.median(finite[-min(20, len(finite)):]))
        improving = late < early - 0.01
    else:
        early = late = float("nan")
        improving = False
    n_nan = int(np.sum(~np.isfinite(we)))
    exploding = bool(len(finite) and (finite[-1] > 1.5 or np.nanmax(ce) > 20))
    return {
        "n": len(rows),
        "last_it": int(it[-1]) if len(it) else -1,
        "last_we": last,
        "best_we": best,
        "early_med_we": early,
        "late_med_we": late,
        "improving": improving,
        "n_nan": n_nan,
        "exploding": exploding,
    }


def cmd_curve(args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from viz.plot_layout import finalize_grid_figure, save_figure

    seed = (args.seeds[0] if args.seeds else DEFAULT_SEEDS[0])
    rows = _load_live_metrics(seed=seed, log_path=args.log)
    if len(rows) < 2:
        raise RuntimeError("need >=2 metric points (pass --log or wait for learning snaps)")
    METRICS_JSON.write_text(json.dumps(rows), encoding="utf-8")
    health = _curve_health(rows)
    it = np.array([r["it"] for r in rows], dtype=int)
    we = 100.0 * np.array([r["we"] for r in rows], dtype=float)
    ce = np.array([r["ce"] for r in rows], dtype=float)

    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(it, we, color="#4C78A8", lw=1.35, label="word error %")
    ax.axhline(3.0, color="0.45", ls="--", lw=0.9, label="3% target")
    ax.set_xlabel("iteration", fontsize=9)
    ax.set_ylabel("word error %", fontsize=9, color="#4C78A8")
    ax.tick_params(axis="y", labelcolor="#4C78A8", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_ylim(0, max(100.0, float(np.nanmax(we)) + 2.0))
    ax.set_xlim(0, max(int(STEPS), int(it[-1]) * 1.02))
    ax.grid(True, alpha=0.28)
    ax.spines["top"].set_visible(False)

    if np.isfinite(ce).any():
        ax2 = ax.twinx()
        ax2.plot(it, ce, color="#54A24B", lw=1.15, alpha=0.9, label="val CE / char")
        ax2.set_ylabel("val CE / char", fontsize=9, color="#54A24B")
        ax2.tick_params(axis="y", labelcolor="#54A24B", labelsize=8)
        ax2.spines["top"].set_visible(False)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7.5, frameon=False)
    else:
        ax.legend(loc="upper right", fontsize=7.5, frameon=False)

    n_dfa = n_dfa_states()
    status = "improving" if health["improving"] else "flat / noisy"
    if health["exploding"] or health["n_nan"]:
        status = "UNHEALTHY"
    finalize_grid_figure(
        fig,
        suptitle=(
            f"long_run live curve  (DFA={n_dfa}; iter {health['last_it']}/{STEPS}; "
            f"WE {100.0 * health['last_we']:.1f}%; best {100.0 * health['best_we']:.1f}%; {status})"
        ),
        suptitle_fontsize=10,
        top=0.86,
        bottom=0.16,
        left=0.10,
        right=0.90,
        hspace=0.30,
        wspace=0.25,
    )
    out = args.out or CURVE_OUT
    save_figure(fig, out, dpi=140)
    print(f"wrote {out}")
    print(
        f"curve: n={health['n']}  iter={health['last_it']}  "
        f"last WE={100.0 * health['last_we']:.2f}%  best={100.0 * health['best_we']:.2f}%  "
        f"late med={100.0 * health['late_med_we']:.2f}%  "
        f"improving={health['improving']}  nan={health['n_nan']}  exploding={health['exploding']}",
        flush=True,
    )


def cmd_motif_board(args: argparse.Namespace) -> None:
    from viz.compare import mixed_dfa_motif_all_runs as mar

    seed = (args.seeds[0] if args.seeds else DEFAULT_SEEDS[0])
    n_dfa = n_dfa_states()
    MOTIF_DIR.mkdir(parents=True, exist_ok=True)
    colored, _edges = mar.collect_task_edge_signed_census(
        task=TASK,
        model="rnn",
        seed=seed,
        max_snaps=int(args.max_snaps),
        colored_json=CENSUS_JSON,
        edges_json=EDGES_JSON,
    )
    out = mar.plot_single_run_motif_board(
        colored,
        None,
        args.out or BOARD_OUT,
        run_label="long_run",
        min_start=int(args.min_start),
        motif_prefix="T|",
        n_dfa_states=n_dfa,
    )
    print(f"motif board: {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "train", "curve", "motif-board"))
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto", "gpu"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--min-start", type=int, default=20)
    parser.add_argument("--max-snaps", type=int, default=60)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None, help="trainer stdout to parse for live metrics")
    args = parser.parse_args()

    (REPO_ROOT / "experiments" / "comparisons" / COMPARISON_NAME).mkdir(
        parents=True, exist_ok=True,
    )
    if args.command == "plan":
        cmd_plan(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "curve":
        cmd_curve(args)
    elif args.command == "motif-board":
        cmd_motif_board(args)


if __name__ == "__main__":
    main()