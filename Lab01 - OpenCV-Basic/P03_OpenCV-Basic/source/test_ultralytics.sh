#!/usr/bin/env bash
# Evaluate YOLOv8 / RT-DETR with the shared COCO evaluator.
#   ARCH=yolov8 DATA=data/megafauna.yaml SPLIT=test bash test_ultralytics.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARCH="${ARCH:-yolov8}"
DATA="${DATA:-$ROOT/data/megafauna.yaml}"
SPLIT="${SPLIT:-test}"
GPU="${GPU:-0}"
TAG="$(basename "${DATA%.*}")"
WEIGHTS="${WEIGHTS:-$ROOT/runs/${ARCH}_${TAG}/weights/best.pt}"

python "$ROOT/ultralytics_det/evaluate.py" -arch "$ARCH" -weights "$WEIGHTS" \
  -data "$DATA" -split "$SPLIT" -device "$GPU" \
  -out-json "$ROOT/runs/${ARCH}_${TAG}_pred_${SPLIT}.json"
