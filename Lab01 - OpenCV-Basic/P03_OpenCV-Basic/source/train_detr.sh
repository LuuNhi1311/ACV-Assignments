#!/usr/bin/env bash
# Fine-tune DETR (facebook/detr-resnet-50).  DATA=data/megafauna.yaml bash train_detr.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA="${DATA:-$ROOT/data/megafauna.yaml}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-4}"
LR="${LR:-1e-4}"
TAG="$(basename "${DATA%.*}")"
OUT="$ROOT/runs/detr_${TAG}"

python "$ROOT/detr/train.py" -data "$DATA" -epochs "$EPOCHS" -batch "$BATCH" \
  -lr "$LR" -out "$OUT" -g "$GPU"
