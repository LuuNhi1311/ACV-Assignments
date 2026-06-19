#!/usr/bin/env bash
# Train YOLOv8 or RT-DETR.  Override via env: ARCH, DATA, GPU, EPOCHS, BATCH, IMGSZ
#   ARCH=yolov8  DATA=data/megafauna.yaml  bash train_ultralytics.sh
#   ARCH=rtdetr  DATA=data/fishinv.yaml    bash train_ultralytics.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARCH="${ARCH:-yolov8}"
DATA="${DATA:-$ROOT/data/megafauna.yaml}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-16}"
IMGSZ="${IMGSZ:-640}"

python "$ROOT/ultralytics_det/train.py" -arch "$ARCH" -data "$DATA" \
  -epochs "$EPOCHS" -batch "$BATCH" -imgsz "$IMGSZ" -device "$GPU" -project "$ROOT/runs"
