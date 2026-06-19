#!/usr/bin/env bash
# Demo DETR. env: DATA GPU CONF WEIGHTS  SOURCE(required)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${DATA:-$ROOT/data/megafauna.yaml}"; TAG="$(basename "${DATA%.*}")"
python "$ROOT/detr/demo.py" -weights "${WEIGHTS:-$ROOT/runs/detr_${TAG}/best}" \
  -source "${SOURCE:?set SOURCE to an image/dir}" -outdir "$ROOT/runs/demo_detr_${TAG}" \
  -conf "${CONF:-0.5}" -g "${GPU:-0}"
