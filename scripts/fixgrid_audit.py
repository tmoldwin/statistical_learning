#!/usr/bin/env python3
"""Classify fixed_letters_grid_ns Dale runs by best word error / CE."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vocab_fixed_letters_grid import COMPARISON_NAME, N_RUNS, run_plan

LN4 = float(np.log(4.0))


def _classify(best_we: float, final_ce: float) -> str:
    if not np.isfinite(best_we):
        return "missing"
    if best_we <= 0.03:
        return "hit"
    if best_we >= 0.90 and (not np.isfinite(final_ce) or final_ce > 1.2):
        return "dead"
    if best_we < 0.10:
        return "near"
    return "partial"


def audit_one(ckpt: Path) -> dict:
    if not ckpt.is_file():
        return {"exists": False}
    data = np.load(ckpt, allow_pickle=True)
    best_we = float(np.asarray(data["best_metric_word_error_frac"]).reshape(-1)[0])
    we = np.asarray(data["metric_word_error_frac"], dtype=np.float64)
    ce = np.asarray(data["metric_val_ce"], dtype=np.float64) if "metric_val_ce" in data.files else np.array([])
    iters = np.asarray(data["metric_iterations"], dtype=np.int32) if "metric_iterations" in data.files else np.array([])
    final_ce = float(ce[-1]) if ce.size else float("nan")
    final_we = float(we[-1]) if we.size else float("nan")
    last_iter = int(iters[-1]) if iters.size else -1
    dropout = float(np.asarray(data["dropout_rate"]).reshape(-1)[0]) if "dropout_rate" in data.files else float("nan")
    lr_key = "learning_rate" if "learning_rate" in data.files else None
    return {
        "exists": True,
        "best_we": best_we,
        "final_we": final_we,
        "final_ce": final_ce,
        "last_iter": last_iter,
        "dropout": dropout,
        "ln4": LN4,
        "class": _classify(best_we, final_ce),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ckpt-root",
        default=str(REPO_ROOT / "experiments" / "comparisons" / COMPARISON_NAME / "checkpoints"),
    )
    p.add_argument("--model-dir", default="rnn_dale")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="")
    args = p.parse_args()
    ckpt_root = Path(args.ckpt_root)
    plan = run_plan()
    rows = []
    counts: dict[str, int] = {}
    for entry in plan:
        rid = int(entry["run_id"])
        ckpt = ckpt_root / f"r{rid:02d}" / args.model_dir / f"model_seed{args.seed}.npz"
        info = audit_one(ckpt)
        row = {
            "run_id": rid,
            "n_words": int(entry["n_words"]),
            "word_length": int(entry["word_length"]),
            "rep": int(entry["rep"]),
            "ckpt": str(ckpt),
            **info,
        }
        rows.append(row)
        cls = str(row.get("class", "missing"))
        counts[cls] = counts.get(cls, 0) + 1

    hits = counts.get("hit", 0)
    print(f"fixgrid audit  n={N_RUNS}  model={args.model_dir}  seed={args.seed}")
    print(f"counts: {counts}   hits={hits}/{N_RUNS}")
    print(f"{'run':>4} {'L':>2} {'nw':>3} {'rep':>3} {'class':<8} {'bestWE':>8} {'finalCE':>8} {'iter':>7}")
    for row in rows:
        if not row.get("exists"):
            print(f"{row['run_id']:4d} {row['word_length']:2d} {row['n_words']:3d} {row['rep']:3d} {'missing':<8}")
            continue
        print(
            f"{row['run_id']:4d} {row['word_length']:2d} {row['n_words']:3d} {row['rep']:3d} "
            f"{row['class']:<8} {row['best_we']:8.3f} {row['final_ce']:8.3f} {row['last_iter']:7d}"
        )
    payload = {"comparison": COMPARISON_NAME, "model_dir": args.model_dir, "seed": args.seed, "counts": counts, "runs": rows}
    out = Path(args.out) if args.out else (
        REPO_ROOT / "experiments" / "comparisons" / COMPARISON_NAME / "data" / "fixgrid_audit.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
