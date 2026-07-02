"""Submit, monitor, and plot experiment sweeps on the ELSC SLURM cluster.

Single entry point for arbitrary task × seed grids. Run on the login node.

Examples:
  # Preview jobs (no submission)
  python scripts/cluster.py plan --preset sixteen_word_lengths_ns_h500

  # Submit one SLURM job per (task, seed)
  python scripts/cluster.py submit --preset sixteen_word_lengths_ns_h500 \\
      --partition ss.cpu --time 04:00:00

  # Custom task list and seeds
  python scripts/cluster.py submit --tasks sixteen_word_ns_h500 --seeds 1 2 3 5 7

  # Wait for queue to drain, then generate comparison figures
  python scripts/cluster.py wait
  python scripts/cluster.py plot --preset sixteen_word_lengths_ns_h500 \\
      --kinds trajectory_geometry closed_loop_trajectories learning_curves

  # submit → wait → plot
  python scripts/cluster.py all --preset sixteen_word_lengths_ns_h500
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment import DEFAULT_SEED, TASKS, checkpoint_path
from viz.compare.spec import COMPARISON_PRESETS, ComparisonSpec

COMPARISON_KINDS = (
    "learning_curves",
    "trajectory_geometry",
    "closed_loop_trajectories",
    "closed_loop_trajectories_2d",
    "closed_loop_trajectories_3d",
)


@dataclass(frozen=True)
class ClusterPlan:
  tasks: tuple[str, ...]
  seeds: tuple[int, ...]
  model_type: str
  comparison_name: str | None = None

  @property
  def job_count(self) -> int:
    return len(self.tasks) * len(self.seeds)


def resolve_plan(args: argparse.Namespace) -> ClusterPlan:
  if args.preset:
    if args.tasks:
      raise SystemExit("--tasks cannot be used with --preset")
    spec = COMPARISON_PRESETS[args.preset]
    if args.name and args.name != spec.name:
      raise SystemExit("--name conflicts with --preset")
    seeds = tuple(args.seeds) if args.seeds else spec.seeds
    model_type = args.model_type or spec.model_type
    return ClusterPlan(spec.tasks, seeds, model_type, spec.name)

  if not args.tasks:
    raise SystemExit("provide --preset or --tasks")

  seeds = tuple(args.seeds) if args.seeds else (DEFAULT_SEED,)
  model_type = args.model_type or "rnn"
  return ClusterPlan(tuple(args.tasks), seeds, model_type, args.name)


def iter_jobs(
  plan: ClusterPlan,
  *,
  skip_existing: bool,
) -> list[tuple[str, int]]:
  jobs: list[tuple[str, int]] = []
  for task in plan.tasks:
    if task not in TASKS:
      raise SystemExit(f"unknown task: {task!r}")
    for seed in plan.seeds:
      if skip_existing and checkpoint_path(task, plan.model_type, seed=seed).is_file():
        continue
      jobs.append((task, seed))
  return jobs


def train_command(
  task: str,
  seed: int,
  *,
  model_type: str,
  smoke: bool,
  repo_dir: Path,
) -> str:
  py = shlex.quote(sys.executable)
  cmd = [
    py, "scripts/run_task.py", task,
    "--models", model_type,
    "--seeds", str(seed),
    "--skip-viz",
  ]
  if smoke:
    cmd.append("--smoke")
  inner = " ".join(shlex.quote(part) for part in cmd)
  return f"cd {shlex.quote(str(repo_dir))} && {inner}"


def submit_jobs(
  plan: ClusterPlan,
  jobs: list[tuple[str, int]],
  *,
  repo_dir: Path,
  partition: str,
  walltime: str,
  mem: str,
  smoke: bool,
  log_dir: Path,
  dry_run: bool,
) -> int:
  log_dir.mkdir(parents=True, exist_ok=True)
  submitted = 0
  for task, seed in jobs:
    run_name = f"{task}_s{seed}"
    job_name = f"sl_{run_name}"[:64]
    output_log = log_dir / f"{run_name}.out"
    wrap = train_command(task, seed, model_type=plan.model_type, smoke=smoke, repo_dir=repo_dir)
    sbatch = [
      "sbatch",
      f"--job-name={job_name}",
      f"--partition={partition}",
      f"--time={walltime}",
      f"--mem={mem}",
      f"--output={output_log}",
      f"--wrap={wrap}",
    ]
    line = " ".join(shlex.quote(part) for part in sbatch)
    print(line)
    if not dry_run:
      subprocess.run(sbatch, check=True, cwd=repo_dir)
      submitted += 1
      time.sleep(0.2)
  return submitted


def wait_for_queue(*, poll_seconds: int = 60) -> None:
  user = os.environ.get("USER") or os.environ.get("LOGNAME")
  if not user:
    raise SystemExit("cannot determine username for squeue")
  print(f"waiting for SLURM jobs (user={user})...")
  while True:
    proc = subprocess.run(
      ["squeue", "-u", user, "-h"],
      capture_output=True,
      text=True,
      check=False,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
      print("queue empty.")
      return
    print(f"{datetime.now().isoformat(timespec='seconds')}  {len(lines)} jobs remaining")
    time.sleep(poll_seconds)


def run_plot(plan: ClusterPlan, args: argparse.Namespace) -> None:
  if not plan.comparison_name:
    raise SystemExit("plot requires --preset or --name (with --tasks for a custom comparison)")

  cmd = [sys.executable, "scripts/compare.py"]
  if args.preset:
    cmd.extend(["--preset", args.preset])
  else:
    cmd.extend(["--name", plan.comparison_name, "--tasks", *plan.tasks])
    if args.title:
      cmd.extend(["--title", args.title])
  cmd.extend(["--model-type", plan.model_type])
  cmd.extend(["--seeds", *[str(s) for s in plan.seeds]])
  if args.kinds:
    cmd.extend(["--kinds", *args.kinds])
  if args.truncate_to_plateau:
    cmd.append("--truncate-to-plateau")
  subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def cmd_plan(plan: ClusterPlan, jobs: list[tuple[str, int]]) -> None:
  print(f"comparison: {plan.comparison_name or '(train only)'}")
  print(f"model_type: {plan.model_type}")
  print(f"tasks:      {', '.join(plan.tasks)}")
  print(f"seeds:      {', '.join(str(s) for s in plan.seeds)}")
  print(f"grid:       {plan.job_count} total, {len(jobs)} to submit, {plan.job_count - len(jobs)} skipped")
  for task, seed in jobs:
    print(f"  {task}  seed {seed}")


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument(
    "action",
    choices=["plan", "submit", "wait", "plot", "all"],
    help="plan=preview; submit=sbatch; wait=drain queue; plot=compare.py; all=submit+wait+plot",
  )
  parser.add_argument("--preset", choices=sorted(COMPARISON_PRESETS))
  parser.add_argument("--name", help="comparison folder name (custom --tasks sweeps)")
  parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS.keys()))
  parser.add_argument("--seeds", nargs="+", type=int)
  parser.add_argument("--model-type", default="rnn", choices=["rnn", "rnn_dale", "transformer"])
  parser.add_argument("--title", default="")
  parser.add_argument(
    "--kinds",
    nargs="+",
    default=["trajectory_geometry", "closed_loop_trajectories", "learning_curves"],
    choices=sorted(COMPARISON_KINDS),
  )
  parser.add_argument("--truncate-to-plateau", action="store_true")

  parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
  parser.add_argument("--partition", default=os.environ.get("PARTITION", "ss.q"))
  parser.add_argument("--time", dest="walltime", default=os.environ.get("TIME", "04:00:00"))
  parser.add_argument("--mem", default=os.environ.get("MEM", "8G"))
  parser.add_argument("--log-dir", type=Path, help="defaults to cluster_logs/<comparison>_<timestamp>/")
  parser.add_argument("--smoke", action="store_true")
  parser.add_argument("--force", action="store_true", help="submit even if checkpoint exists")
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--poll-seconds", type=int, default=60)
  return parser


def main() -> None:
  parser = build_parser()
  args = parser.parse_args()
  plan = resolve_plan(args)
  jobs = iter_jobs(plan, skip_existing=not args.force)

  if args.action == "plan":
    cmd_plan(plan, jobs)
    return

  if args.action == "plot":
    run_plot(plan, args)
    return

  if args.action == "wait":
    wait_for_queue(poll_seconds=args.poll_seconds)
    return

  if args.action in ("submit", "all"):
    if not jobs:
      print("nothing to submit (all checkpoints present; use --force to rerun)")
    else:
      stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      label = plan.comparison_name or "tasks"
      log_dir = args.log_dir or (args.repo_dir / "cluster_logs" / f"{label}_{stamp}")
      submitted = submit_jobs(
        plan,
        jobs,
        repo_dir=args.repo_dir,
        partition=args.partition,
        walltime=args.walltime,
        mem=args.mem,
        smoke=args.smoke,
        log_dir=log_dir,
        dry_run=args.dry_run,
      )
      print(f"submitted {submitted} jobs → {log_dir}")

  if args.action == "all":
    if args.dry_run:
      print("dry-run: skipping wait/plot")
      return
    wait_for_queue(poll_seconds=args.poll_seconds)
    if plan.comparison_name:
      run_plot(plan, args)
    else:
      print("no comparison name; skipping plot")


if __name__ == "__main__":
  main()
