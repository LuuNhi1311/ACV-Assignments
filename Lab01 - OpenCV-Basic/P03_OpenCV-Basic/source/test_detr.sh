#!/usr/bin/env bash
# Evaluate DETR with the shared COCO evaluator.  DATA=data/megafauna.yaml SPLIT=test bash test_detr.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA="${DATA:-$ROOT/data/megafauna.yaml}"
SPLIT="${SPLIT:-test}"
GPU="${GPU:-0}"
TAG="$(basename "${DATA%.*}")"
WEIGHTS="${WEIGHTS:-$ROOT/runs/detr_${TAG}/best}"

python "$ROOT/detr/evaluate.py" -weights "$WEIGHTS" -data "$DATA" -split "$SPLIT" \
  -out-json "$ROOT/runs/detr_${TAG}_pred_${SPLIT}.json" -g "$GPU"
