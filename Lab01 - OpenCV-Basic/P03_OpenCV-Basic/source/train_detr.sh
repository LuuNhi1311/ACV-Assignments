#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$ROOT/data"

GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-4}"
LR="${LR:-1e-4}"
MODEL="${MODEL:-facebook/detr-resnet-50}"
OUT="$ROOT/detr/checkpoints"

python "$ROOT/detr/train.py" -data "$DATA" -model "$MODEL" \
  -epochs "$EPOCHS" -batch "$BATCH" -lr "$LR" -out "$OUT" -g "$GPU"
