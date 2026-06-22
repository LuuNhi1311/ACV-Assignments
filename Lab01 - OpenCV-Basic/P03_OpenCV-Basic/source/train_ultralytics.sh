#!/usr/bin/env bash
# Train YOLOv8/RT-DETR. env: ARCH DATA GPU EPOCHS BATCH IMGSZ WORKERS
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${DATA:-$ROOT/data/megafauna.yaml}"; case "$DATA" in /*) ;; *) DATA="$ROOT/$DATA" ;; esac
python "$ROOT/ultralytics_det/train.py" -arch "${ARCH:-yolov8}" -data "$DATA" \
  -epochs "${EPOCHS:-200}" -batch "${BATCH:-16}" -imgsz "${IMGSZ:-640}" -workers "${WORKERS:-8}" \
  -device "${GPU:-0}" -project "$ROOT/runs"
