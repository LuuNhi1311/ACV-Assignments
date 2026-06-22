#!/usr/bin/env bash
# Eval DETR (shared COCO). env: DATA SPLIT GPU WEIGHTS
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${DATA:-$ROOT/data/megafauna.yaml}"; SPLIT="${SPLIT:-test}"
case "$DATA" in /*) ;; *) DATA="$ROOT/$DATA" ;; esac; TAG="$(basename "${DATA%.*}")"
python "$ROOT/detr/evaluate.py" -weights "${WEIGHTS:-$ROOT/runs/detr_${TAG}/best}" \
  -data "$DATA" -split "$SPLIT" -out-json "$ROOT/runs/detr_${TAG}_pred_${SPLIT}.json" -g "${GPU:-0}"
