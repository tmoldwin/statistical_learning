#!/usr/bin/env python3
"""Sweep status: per-task rollup + running job iter/word-error from logs, NPZ, or .progress files."""
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

OK_THRESHOLD = 0.15  # word error fraction


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


def tail_text(path: Path, n: int = 4000) -> str:
    if not path.is_file():
        return ""
    data = path.read_bytes()
    if len(data) > n:
        data = data[-n:]
    return data.decode("utf-8", errors="replace")


def parse_log(log_path: Path) -> dict:
    text = tail_text(log_path)
    info = {"state": "RUN", "iter": None, "loss": None, "word_err": None, "best": None}
    if "saved trained model" in text:
        info["state"] = "DONE"
    elif "Traceback" in text:
        info["state"] = "FAIL"
    elif ">> /usr/bin/python3 rnn/" not in text:
        info["state"] = "WAIT"

    m = re.findall(r"^iter (\d+), loss: ([0-9.]+)", text, re.M)
    if m:
        info["iter"] = int(m[-1][0])
        info["loss"] = float(m[-1][1])
    mm = re.findall(r"metric iter (\d+), word_err: ([0-9.]+)%", text, re.M)
    if mm:
        info["word_err"] = float(mm[-1][1])
    elif info["state"] == "DONE":
        fm = re.search(r"final word error rate[^:]*: ([0-9.]+)%", text)
        if fm:
            info["word_err"] = float(fm.group(1))
    bm = re.search(r"checkpoint from iter \d+ \(([0-9.]+)% word error", text)
    if bm:
        info["best"] = float(bm.group(1))
    return info


def read_progress(progress_path: Path) -> dict | None:
    if not progress_path.is_file():
        return None
    line = progress_path.read_text(encoding="utf-8").strip().splitlines()[-1]
    parts = line.split("\t")
    if len(parts) < 2:
        return None
    out = {"iter": int(parts[0]), "word_err": float(parts[1]) * 100.0}
    if len(parts) > 2:
        out["loss"] = float(parts[2])
    return out


def read_npz(npz_path: Path) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    final_we = float(d["metric_word_error_frac"][-1]) * 100.0
    best_we = float(d["best_metric_word_error_frac"]) * 100.0
    last_iter = int(d["metric_iterations"][-1]) if len(d["metric_iterations"]) else -1
    return {"state": "DONE", "iter": last_iter, "word_err": final_we, "best": best_we}


def fmt_pct(x: float | None) -> str:
    return f"{x:.1f}" if x is not None else "-"


def fmt_iter(x: int | None) -> str:
    return str(x) if x is not None else "-"


def job_status(task: str, seed: int, log_dir: Path | None) -> dict:
    ckpt = checkpoint_path(task, "rnn", seed=seed)
    progress = ckpt.parent / f"{ckpt.stem}.progress"
    log_path = log_dir / f"{task}_s{seed}.out" if log_dir else Path("/dev/null")

    info = parse_log(log_path) if log_dir else {"state": "?", "iter": None, "loss": None, "word_err": None, "best": None}

    prog = read_progress(progress)
    if prog and info["state"] not in ("DONE", "FAIL"):
        info.update(prog)
        info["state"] = "RUN"

    if info["state"] == "DONE" and ckpt.is_file():
        try:
            info = read_npz(ckpt)
        except Exception:
            pass
    elif info["state"] == "DONE" and not ckpt.is_file():
        info["state"] = "DONE?"

    info["seed"] = seed
    info["ok"] = info.get("word_err") is not None and info["word_err"] / 100.0 < OK_THRESHOLD
    return info


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

    total_done = total_fail = total_ok = 0
    total = len(spec.tasks) * len(spec.seeds)

    for task in spec.tasks:
        label = spec.labels.get(task, task)
        rows = [job_status(task, seed, log_dir) for seed in spec.seeds]
        done = sum(r["state"] == "DONE" for r in rows)
        fail = sum(r["state"] == "FAIL" for r in rows)
        ok = sum(r.get("ok") for r in rows if r["state"] == "DONE")
        run_rows = [r for r in rows if r["state"] not in ("DONE", "FAIL")]
        total_done += done
        total_fail += fail
        total_ok += ok

        running_bits = []
        for r in run_rows:
            if r["iter"] is not None and r.get("word_err") is not None:
                running_bits.append(f"s{r['seed']}@{r['iter']}/{fmt_pct(r['word_err'])}%")
            elif r["iter"] is not None:
                running_bits.append(f"s{r['seed']}@{r['iter']}")
            else:
                running_bits.append(f"s{r['seed']}?")

        run_str = ", ".join(running_bits) if running_bits else "(none)"
        print(f"{label:12s}  done {done:2d}/{len(spec.seeds)}  ok {ok:2d}  fail {fail}  |  running: {run_str}")

        # Detail for running jobs with any signal
        detail = [r for r in run_rows if r["iter"] is not None or r.get("word_err") is not None]
        if detail:
            print(f"  {'seed':>5} {'iter':>7} {'word_err%':>10} {'best%':>8}")
            for r in detail:
                print(f"  {r['seed']:5d} {fmt_iter(r['iter']):>7} {fmt_pct(r.get('word_err')):>10} {fmt_pct(r.get('best')):>8}")

    print("---")
    print(f"sweep: done {total_done}/{total}  ok(<15% err) {total_ok}  fail {total_fail}")


if __name__ == "__main__":
    main()