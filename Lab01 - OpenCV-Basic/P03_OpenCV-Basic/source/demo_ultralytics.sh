#!/usr/bin/env bash
# Demo YOLOv8/RT-DETR. env: ARCH DATA GPU CONF WEIGHTS  SOURCE(required)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH="${ARCH:-yolov8}"; DATA="${DATA:-$ROOT/data/megafauna.yaml}"
case "$DATA" in /*) ;; *) DATA="$ROOT/$DATA" ;; esac; TAG="$(basename "${DATA%.*}")"
python "$ROOT/ultralytics_det/demo.py" -arch "$ARCH" \
  -weights "${WEIGHTS:-$ROOT/runs/${ARCH}_${TAG}/weights/best.pt}" \
  -source "${SOURCE:?set SOURCE to an image/dir}" -conf "${CONF:-0.25}" -device "${GPU:-0}" \
  -project "$ROOT/runs" -name "demo_${ARCH}_${TAG}"
