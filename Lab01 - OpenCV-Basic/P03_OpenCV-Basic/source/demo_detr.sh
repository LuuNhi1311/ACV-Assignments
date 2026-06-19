#!/usr/bin/env bash
# Draw DETR predictions.  DATA=data/megafauna.yaml SOURCE=data/megafauna/images/test bash demo_detr.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA="${DATA:-$ROOT/data/megafauna.yaml}"
GPU="${GPU:-0}"
CONF="${CONF:-0.5}"
TAG="$(basename "${DATA%.*}")"
WEIGHTS="${WEIGHTS:-$ROOT/runs/detr_${TAG}/best}"
SOURCE="${SOURCE:?set SOURCE to an image file or directory}"

python "$ROOT/detr/demo.py" -weights "$WEIGHTS" -source "$SOURCE" \
  -outdir "$ROOT/runs/demo_detr_${TAG}" -conf "$CONF" -g "$GPU"
