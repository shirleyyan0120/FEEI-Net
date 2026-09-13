#!/usr/bin/env bash
#SBATCH --job-name=feei-ablation
#SBATCH --partition=GPUA40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --array=0-24
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT before calling sbatch}"
REPO_ROOT=${REPO_ROOT:-$SLURM_SUBMIT_DIR}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$REPO_ROOT/runs"}
PYTHON_BIN=${PYTHON_BIN:-python}
SUBJECTS=${SUBJECTS:-all}

MODELS=(
  unified_carrier
  dual_branch
  dual_branch_cd
  dual_branch_cd_interaction
  feei_full
)
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
FOLD=$((SLURM_ARRAY_TASK_ID % 5 + 1))
MODEL=${MODELS[$MODEL_INDEX]}
RUN_ID=ablation_${MODEL}_fold${FOLD}_seed20261010

cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  exec "$PYTHON_BIN" -m feei_aad.train \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --run-id "$RUN_ID" \
  --models "$MODEL" \
  --views frontal_ear \
  --subjects "$SUBJECTS" \
  --folds "$FOLD" \
  --base-seed 20261010 \
  --split-seed 20260807 \
  --max-epochs 100 \
  --patience 10
