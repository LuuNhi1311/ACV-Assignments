#!/usr/bin/env bash
# Eval YOLOv8/RT-DETR (shared COCO). env: ARCH DATA SPLIT GPU WEIGHTS
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH="${ARCH:-yolov8}"; DATA="${DATA:-$ROOT/data/megafauna.yaml}"; SPLIT="${SPLIT:-test}"; TAG="$(basename "${DATA%.*}")"
python "$ROOT/ultralytics_det/evaluate.py" -arch "$ARCH" \
  -weights "${WEIGHTS:-$ROOT/runs/${ARCH}_${TAG}/weights/best.pt}" \
  -data "$DATA" -split "$SPLIT" -device "${GPU:-0}" -out-json "$ROOT/runs/${ARCH}_${TAG}_pred_${SPLIT}.json"
