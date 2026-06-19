#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$ROOT/data"

SPLIT="${SPLIT:-test}"
GPU="${GPU:-0}"
CONF="${CONF:-0.5}"
WEIGHTS="${WEIGHTS:-$ROOT/detr/checkpoints/best}"

python "$ROOT/detr/demo.py" -weights "$WEIGHTS" -imgfile "$DATA/$SPLIT" \
  -names "$DATA/$SPLIT/_classes.txt" -outdir "$ROOT/detr/visualizations" \
  -conf "$CONF" -g "$GPU"
