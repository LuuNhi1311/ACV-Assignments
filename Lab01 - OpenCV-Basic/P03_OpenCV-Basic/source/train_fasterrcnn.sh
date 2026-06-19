#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$ROOT/data"

GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-4}"
LR="${LR:-5e-3}"
OUT="$ROOT/faster-rcnn/checkpoints"

python "$ROOT/faster-rcnn/train.py" -data "$DATA" \
  -epochs "$EPOCHS" -batch "$BATCH" -lr "$LR" -out "$OUT" -g "$GPU"
