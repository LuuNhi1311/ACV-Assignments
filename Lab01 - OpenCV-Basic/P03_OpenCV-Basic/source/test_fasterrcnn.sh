#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$ROOT/data"

SPLIT="${SPLIT:-test}"
GPU="${GPU:-0}"
WEIGHTS="${WEIGHTS:-$ROOT/faster-rcnn/checkpoints/best.pt}"

python "$ROOT/faster-rcnn/evaluate.py" -weights "$WEIGHTS" \
  -split-dir "$DATA/$SPLIT" -out-json "$ROOT/faster-rcnn/predictions_$SPLIT.json" -g "$GPU"
