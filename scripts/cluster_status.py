#!/usr/bin/env python3
"""Per-seed sweep status with word accuracy from NPZ, progress files, or logs."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment import REPO_ROOT, checkpoint_path
from viz.compare.spec import COMPARISON_PRESETS

OK_THRESHOLD = 0.15


def latest_log_dir(glob_pat: str) -> Path | None:
    matches = sorted((REPO_ROOT / "cluster_logs").glob(glob_pat), reverse=True)
    return matches[0] if matches else None


def slurm_counts(user: str = "toviah.moldwin") -> tuple[int, int]:
    def count(state: str | None = None) -> int:
        cmd = ["squeue", "-u", user, "-h"]
        if state:
            cmd.extend(["-t", state])
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return 0
        return len([ln for ln in out.splitlines() if ln.strip()])
    return count(), count("PENDING")


def tail_text(path: Path, n: int = 512_000) -> str:
    if not path.is_file():
        return ""
    data = path.read_bytes()
    if len(data) > n:
        data = data[-n:]
    return data.decode("utf-8", errors="replace")


def word_err_to_acc(word_err_pct: float) -> float:
    return 100.0 - word_err_pct


def parse_log(log_path: Path) -> dict:
    text = tail_text(log_path)
    info: dict = {"state": "RUN", "iter": None, "word_err": None, "best_we": None}
    if "saved trained model" in text:
        info["state"] = "DONE"
    elif "Traceback" in text:
        info["state"] = "FAIL"
    elif ">> /usr/bin/python3 rnn/" not in text:
        info["state"] = "WAIT"

    m = re.findall(r"^iter (\d+), loss:", text, re.M)
    if m:
        info["iter"] = int(m[-1])
    mm = re.findall(r"metric iter (\d+), word_err: ([0-9.]+)%", text, re.M)
    if mm:
        info["iter"] = int(mm[-1][0])
        info["word_err"] = float(mm[-1][1])
    elif info["state"] == "DONE":
        fm = re.search(r"final word error rate[^:]*: ([0-9.]+)%", text)
        if fm:
            info["word_err"] = float(fm.group(1))
    bm = re.search(r"checkpoint from iter \d+ \(([0-9.]+)% word error", text)
    if bm:
        info["best_we"] = float(bm.group(1))
    return info


def read_progress(progress_path: Path) -> dict | None:
    if not progress_path.is_file():
        return None
    line = progress_path.read_text(encoding="utf-8").strip().splitlines()[-1]
    parts = line.split("\t")
    if len(parts) < 2:
        return None
    return {"iter": int(parts[0]), "word_err": float(parts[1]) * 100.0}


def read_npz(npz_path: Path) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    final_we = float(d["metric_word_error_frac"][-1]) * 100.0
    best_we = float(d["best_metric_word_error_frac"]) * 100.0
    last_iter = int(d["metric_iterations"][-1]) if len(d["metric_iterations"]) else -1
    return {"state": "DONE", "iter": last_iter, "word_err": final_we, "best_we": best_we}


def job_status(task: str, seed: int, log_dir: Path | None) -> dict:
    ckpt = checkpoint_path(task, "rnn", seed=seed)
    progress = ckpt.parent / f"{ckpt.stem}.progress"
    log_path = log_dir / f"{task}_s{seed}.out" if log_dir else None

    if ckpt.is_file():
        try:
            info = read_npz(ckpt)
        except Exception:
            info = {"state": "DONE", "iter": None, "word_err": None, "best_we": None}
    else:
        info = parse_log(log_path) if log_path else {"state": "WAIT", "iter": None, "word_err": None, "best_we": None}
        prog = read_progress(progress)
        if prog:
            info.update(prog)
            if info["state"] not in ("FAIL",):
                info["state"] = "RUN"

    info["seed"] = seed
    we = info.get("word_err")
    info["acc"] = word_err_to_acc(we) if we is not None else None
    best_we = info.get("best_we")
    info["best_acc"] = word_err_to_acc(best_we) if best_we is not None else None
    info["ok"] = we is not None and we / 100.0 < OK_THRESHOLD
    return info


def fmt_acc(acc: float | None) -> str:
    return f"{acc:.1f}%" if acc is not None else "  --"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="sixteen_word_lengths_ns_h500")
    p.add_argument("--log-glob", default="sixteen_word_lengths_ns_h500_*")
    args = p.parse_args()

    spec = COMPARISON_PRESETS[args.preset]
    log_dir = latest_log_dir(args.log_glob)
    running, pending = slurm_counts()
    batch = log_dir.name if log_dir else "?"

    print(f"=== {datetime.now().isoformat(timespec='seconds')} === {batch}")
    print(f"slurm: running={running} pending={pending}")
    print(f"acc = 100% - invalid_word_rate  |  ok threshold: word_err < {OK_THRESHOLD*100:.0f}%")
    print()

    total_done = total_fail = total_ok = 0
    total = len(spec.tasks) * len(spec.seeds)

    for task in spec.tasks:
        label = spec.labels.get(task, task)
        rows = [job_status(task, seed, log_dir) for seed in spec.seeds]
        done = sum(r["state"] == "DONE" for r in rows)
        fail = sum(r["state"] == "FAIL" for r in rows)
        ok = sum(r.get("ok") for r in rows if r["state"] == "DONE")
        total_done += done
        total_fail += fail
        total_ok += ok

        print(f"{label}  ({done}/{len(spec.seeds)} done, {ok} ok, {fail} fail)")
        print(f"  {'seed':>5} {'state':>6} {'iter':>7} {'acc':>8} {'best_acc':>9}")
        for r in rows:
            st = r["state"][:6]
            it = str(r["iter"]) if r["iter"] is not None else "--"
            print(f"  {r['seed']:5d} {st:>6} {it:>7} {fmt_acc(r.get('acc')):>8} {fmt_acc(r.get('best_acc')):>9}")
        print()

    print(f"sweep: {total_done}/{total} done  |  {total_ok} ok  |  {total_fail} fail")


if __name__ == "__main__":
    main()