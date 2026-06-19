#!/usr/bin/env bash
# Fine-tune DETR. env: DATA GPU EPOCHS BATCH LR WORKERS
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${DATA:-$ROOT/data/megafauna.yaml}"; TAG="$(basename "${DATA%.*}")"
python "$ROOT/detr/train.py" -data "$DATA" -epochs "${EPOCHS:-100}" -batch "${BATCH:-4}" \
  -lr "${LR:-1e-4}" -workers "${WORKERS:-4}" -out "$ROOT/runs/detr_${TAG}" -g "${GPU:-0}"
