#!/usr/bin/env python3
"""Cluster sweep progress for fixgrid Dale (single recipe at a time)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vocab_fixed_letters_grid import N_RUNS

TARGET = 0.03


def _squeue_count(job_name: str) -> int:
    p = subprocess.run(
        ["squeue", "-u", "toviah.moldwin", "-n", job_name, "-h"],
        capture_output=True,
        text=True,
        check=False,
    )
    return len([ln for ln in p.stdout.splitlines() if ln.strip()])


def _parse_progress(progress_path: Path) -> tuple[int, float, float] | None:
    if not progress_path.is_file():
        return None
    try:
        parts = progress_path.read_text(encoding="utf-8").strip().split()
        if len(parts) < 3:
            return None
        return int(parts[0]), float(parts[1]), float(parts[2])
    except (OSError, ValueError):
        return None


def _parse_log(log_path: Path) -> tuple[int, float] | None:
    if not log_path.is_file():
        return None
    last = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.search(r"metric iter (\d+), word_err: ([0-9.]+)%", line)
        if m:
            last = (int(m.group(1)), float(m.group(2)) / 100.0)
    return last


def tally(root: Path, *, model_dir: str, log_dir: Path, log_glob: str) -> dict:
    best_wes: list[float] = []
    hits = 0
    failed = 0
    done = 0
    live: list[tuple[int, float, int]] = []
    for i in range(N_RUNS):
        ckpt = root / f"r{i:02d}" / model_dir / "model_seed1.npz"
        prog = root / f"r{i:02d}" / model_dir / "model_seed1.progress"
        log = log_dir / log_glob.format(i=i)
        if ckpt.is_file() and ckpt.stat().st_size > 1000:
            done += 1
            try:
                d = np.load(ckpt, allow_pickle=True)
                we = float(np.asarray(d["best_metric_word_error_frac"]).reshape(-1)[0])
            except Exception:
                we = float("nan")
                failed += 1
            best_wes.append(we)
            if np.isfinite(we) and we <= TARGET:
                hits += 1
            continue
        pr = _parse_progress(prog)
        if pr is None:
            lg = _parse_log(log)
            if lg is not None:
                live.append((i, lg[1], lg[0]))
            continue
        live.append((i, pr[1], pr[0]))
    worst_done = max(best_wes) if best_wes else float("nan")
    median_done = float(np.median(best_wes)) if best_wes else float("nan")
    worst_live = max(live, key=lambda t: t[1]) if live else None
    return {
        "still": len(live),
        "done": done,
        "failed": failed,
        "hits": hits,
        "median": median_done,
        "worst_done": worst_done,
        "worst_live": worst_live,
    }


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=str(REPO_ROOT))
    p.add_argument("--job-name", default="dale_fixgrid_v3")
    p.add_argument("--model-dir", default="rnn_dale")
    p.add_argument("--log-glob", default="fixgrid_v3_{i}.out")
    args = p.parse_args()
    root = Path(args.root)
    ck_root = root / "experiments" / "comparisons" / "fixed_letters_grid_ns" / "checkpoints"
    log_dir = root / "logs"

    queued = _squeue_count(args.job_name)
    stats = tally(ck_root, model_dir=args.model_dir, log_dir=log_dir, log_glob=args.log_glob)
    still = max(queued, stats["still"])
    total = N_RUNS
    print(f"{args.job_name}: still-going/total {still}/{total}")
    print(
        f"completed {stats['done']}/{total}  failed {stats['failed']}  "
        f"at-3% {stats['hits']}/{max(stats['done'], 1)}"
    )
    if stats["done"]:
        print(
            f"of completed: median best-WE {stats['median']:.3f}  "
            f"worst best-WE {stats['worst_done']:.3f}"
        )
    if stats["worst_live"] is not None:
        rid, we, it = stats["worst_live"]
        print(f"worst live: r{rid:02d} WE={we:.3f} iter={it}")


if __name__ == "__main__":
    main()
