#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
LOG_GLOB="${1:-sixteen_word_lengths_ns_h500_*}"
LOGDIR="$(ls -dt "$REPO/cluster_logs"/$LOG_GLOB 2>/dev/null | head -1 || true)"
if [[ -z "$LOGDIR" ]]; then echo "no log dir under $REPO/cluster_logs/$LOG_GLOB"; exit 1; fi
USER_NAME="${USER:-toviah.moldwin}"
running="$(squeue -u "$USER_NAME" -h 2>/dev/null | wc -l)"
pending="$(squeue -u "$USER_NAME" -t PENDING -h 2>/dev/null | wc -l)"
echo "=== $(date -Iseconds) === ${LOGDIR##*/}"
echo "queue: running=$running pending=$pending"
done_n=0; fail_n=0; run_n=0; start_n=0
printf '%-32s %5s %6s %8s %8s %10s\n' JOB SEED STATE ITER LOSS WORD_ERR%
for f in "$LOGDIR"/*.out; do
  [[ -f "$f" ]] || continue
  base="${f##*/}"; base="${base%.out}"; seed="${base##*_s}"; job="${base%_s*}"
  state=RUN; iter=-; loss=-; we=-
  if grep -q 'saved trained model' "$f" 2>/dev/null; then state=DONE; done_n=$((done_n+1))
  elif grep -q Traceback "$f" 2>/dev/null; then state=FAIL; fail_n=$((fail_n+1))
  elif ! grep -q '^iter ' "$f" 2>/dev/null; then state=START; start_n=$((start_n+1))
  else run_n=$((run_n+1)); fi
  last_iter="$(grep '^iter ' "$f" 2>/dev/null | tail -1 || true)"
  if [[ -n "$last_iter" ]]; then
    iter="$(echo "$last_iter" | sed -n 's/^iter \([0-9]*\).*/\1/p')"
    loss="$(echo "$last_iter" | sed -n 's/.*loss: \([^ ]*\).*/\1/p')"
  fi
  if grep -q 'metric iter ' "$f" 2>/dev/null; then
    we="$(grep 'metric iter ' "$f" 2>/dev/null | tail -1 | sed -n 's/.*word_err: \([^%]*\)%.*/\1/p')"
  elif grep -q 'final word error' "$f" 2>/dev/null; then
    we="$(grep 'final word error' "$f" 2>/dev/null | tail -1 | sed -n 's/.*: \([0-9.]*\)%.*/\1/p')"
  elif grep -q 'word error' "$f" 2>/dev/null; then
    we="$(grep 'word error' "$f" 2>/dev/null | tail -1 | sed -n 's/.*(\([0-9.]*\)% word error.*/\1/p')"
  fi
  if [[ "$state" == RUN || "$state" == START ]]; then
    printf '%-32s %5s %6s %8s %8s %10s\n' "$job" "$seed" "$state" "$iter" "$loss" "$we"
  fi
done
echo "---"
echo "active: run=$run_n starting=$start_n | finished: done=$done_n fail=$fail_n"