#!/usr/bin/env bash
# Draw YOLOv8 / RT-DETR predictions. Set SOURCE to an image or folder.
#   ARCH=yolov8 DATA=data/megafauna.yaml SOURCE=data/megafauna/images/test bash demo_ultralytics.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARCH="${ARCH:-yolov8}"
DATA="${DATA:-$ROOT/data/megafauna.yaml}"
GPU="${GPU:-0}"
CONF="${CONF:-0.25}"
TAG="$(basename "${DATA%.*}")"
WEIGHTS="${WEIGHTS:-$ROOT/runs/${ARCH}_${TAG}/weights/best.pt}"
SOURCE="${SOURCE:?set SOURCE to an image file or directory}"

python "$ROOT/ultralytics_det/demo.py" -arch "$ARCH" -weights "$WEIGHTS" \
  -source "$SOURCE" -conf "$CONF" -device "$GPU" -project "$ROOT/runs" -name "demo_${ARCH}_${TAG}"
