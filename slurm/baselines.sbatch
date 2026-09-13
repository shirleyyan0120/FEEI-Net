#!/usr/bin/env bash
#SBATCH --job-name=feei-baselines
#SBATCH --partition=GPUA40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --array=0-104
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT before calling sbatch}"
REPO_ROOT=${REPO_ROOT:-$SLURM_SUBMIT_DIR}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$REPO_ROOT/runs"}
PYTHON_BIN=${PYTHON_BIN:-python}
SUBJECTS=${SUBJECTS:-all}

MODELS=(stanet xanet darnet dbpnet listennet mhanet hcan)
VIEWS=(frontal_only ear_only frontal_ear)
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / 15))
VIEW_INDEX=$(((SLURM_ARRAY_TASK_ID / 5) % 3))
FOLD=$((SLURM_ARRAY_TASK_ID % 5 + 1))
MODEL=${MODELS[$MODEL_INDEX]}
VIEW=${VIEWS[$VIEW_INDEX]}
RUN_ID=baseline_${VIEW}_${MODEL}_fold${FOLD}_seed20261010

cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  exec "$PYTHON_BIN" -m feei_aad.train \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --run-id "$RUN_ID" \
  --models "$MODEL" \
  --views "$VIEW" \
  --subjects "$SUBJECTS" \
  --folds "$FOLD" \
  --base-seed 20261010 \
  --split-seed 20260807 \
  --max-epochs 100 \
  --patience 10 \
  --learning-rate 3e-4 \
  --weight-decay 3e-4 \
  --batch-size 32
