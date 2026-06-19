#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$ROOT/data"

SPLIT="${SPLIT:-test}"
GPU="${GPU:-0}"
WEIGHTS="${WEIGHTS:-$ROOT/detr/checkpoints/best}"

python "$ROOT/detr/evaluate.py" -weights "$WEIGHTS" \
  -split-dir "$DATA/$SPLIT" -out-json "$ROOT/detr/predictions_$SPLIT.json" -g "$GPU"
