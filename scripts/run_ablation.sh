#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 DATA_ROOT [OUTPUT_ROOT]"
  exit 2
fi

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT=$1
OUTPUT_ROOT=${2:-"$REPO_ROOT/runs"}
PYTHON_BIN=${PYTHON_BIN:-python}

PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" -m feei_aad.train \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --run-id ablation_seed20261010 \
  --models unified_carrier,dual_branch,dual_branch_cd,dual_branch_cd_interaction,feei_full \
  --views frontal_ear \
  --subjects all \
  --folds 1,2,3,4,5 \
  --base-seed 20261010 \
  --split-seed 20260807
