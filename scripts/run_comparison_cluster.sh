#!/bin/bash
# Submit statistical_learning comparison training jobs to the ELSC SLURM cluster.
#
# Modeled on SynClass/run_synclass.sh (sbatch --wrap, one job per run).
#
# Usage (on loginserver, from repo root):
#   ./scripts/run_comparison_cluster.sh sixteen_word_lengths_ns_h500
#   ./scripts/run_comparison_cluster.sh sixteen_word_lengths_ns_h500 --plot
#
# Environment overrides:
#   REPO_DIR=~/code/statistical_learning   path to repo on cluster
#   PARTITION=ss.cpu                       SLURM partition (CPU for numpy RNN)
#   TIME=02:00:00                          wall time per (task, seed) job
#   MEM=8G                                 memory per job
#
# After training, sync models back to your laptop (from Windows / MobaXterm):
#   scp -r toviah.moldwin@loginserver.elsc.huji.ac.il:~/code/statistical_learning/experiments/*_h500 ./experiments/

set -euo pipefail

PRESET="${1:-}"
PLOT_ONLY=false
if [[ "${2:-}" == "--plot" ]]; then
    PLOT_ONLY=true
fi
if [[ -z "$PRESET" ]]; then
    echo "usage: $0 <preset> [--plot]" >&2
    echo "  presets: sixteen_word_lengths_ns_h500, sixteen_word_lengths_ns, ..." >&2
    exit 1
fi

REPO_DIR="${REPO_DIR:-$HOME/code/statistical_learning}"
PARTITION="${PARTITION:-ss.cpu}"
TIME="${TIME:-04:00:00}"
MEM="${MEM:-8G}"

cd "$REPO_DIR"

if [[ "$PLOT_ONLY" == true ]]; then
    echo "Generating comparison figures for preset: $PRESET"
    python scripts/compare.py --preset "$PRESET" \
        --kinds trajectory_geometry closed_loop_trajectories learning_curves
    exit 0
fi

echo "--- statistical_learning cluster sweep: $PRESET ---"
echo "repo:      $REPO_DIR"
echo "partition: $PARTITION"
echo "time:      $TIME"
echo "mem:       $MEM"
echo "pulling latest..."
git pull
echo "---"

# Preset -> tasks and seeds (keep in sync with viz/compare/spec.py)
declare -a TASKS=()
declare -a SEEDS=()

case "$PRESET" in
    sixteen_word_lengths_ns_h500)
        TASKS=(
            sixteen_word_ns_h500
            sixteen_word_four_letter_ns_h500
            sixteen_word_five_letter_ns_h500
            sixteen_word_mixed_ns_h500
        )
        SEEDS=(1 2 3 5 7 8 11 13 17 19 23 29 31 37 53)
        ;;
    sixteen_word_lengths_ns)
        TASKS=(
            sixteen_word_ns
            sixteen_word_four_letter_ns
            sixteen_word_five_letter_ns
            sixteen_word_mixed_ns
        )
        SEEDS=(1 2 3 5 7 8 11 13 17 19 23 29 31 37 53)
        ;;
    *)
        echo "unknown preset: $PRESET (add a case block in $0)" >&2
        exit 1
        ;;
esac

SWEEP_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="cluster_logs/${PRESET}_${SWEEP_TIMESTAMP}"
mkdir -p "$LOG_DIR"
echo "logs: $LOG_DIR"

SUBMITTED=0
SKIPPED=0

for TASK in "${TASKS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        MODEL="experiments/${TASK}/rnn/model_seed${SEED}.npz"
        if [[ -f "$MODEL" ]]; then
            echo "skip ${TASK} seed ${SEED} (checkpoint exists)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        RUN_NAME="${TASK}_s${SEED}"
        JOB_NAME="sl_${RUN_NAME}"
        OUTPUT_LOG="${LOG_DIR}/${RUN_NAME}.out"

        FULL_CMD="cd ${REPO_DIR} && python scripts/run_task.py ${TASK} --models rnn --seed ${SEED} --skip-viz"

        SBATCH_CMD="sbatch \
          --job-name=${JOB_NAME} \
          --partition=${PARTITION} \
          --time=${TIME} \
          --mem=${MEM} \
          --output=${OUTPUT_LOG} \
          --wrap=\"${FULL_CMD}\""

        echo "submit ${JOB_NAME}"
        eval "$SBATCH_CMD"
        SUBMITTED=$((SUBMITTED + 1))
        sleep 0.2
    done
done

echo "--- submitted ${SUBMITTED} jobs, skipped ${SKIPPED} existing checkpoints ---"
echo "monitor: squeue -u \$USER"
echo "after all jobs finish, run:"
echo "  ./scripts/run_comparison_cluster.sh ${PRESET} --plot"
