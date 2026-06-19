#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YDIR="$ROOT/pytorch-YOLOv4"; DATA="$ROOT/data"

GPUS="${GPUS:-7}"
EPOCHS=500
LR=0.04 
BATCH=64 
SUBDIV=16 
BURN_IN=200
CFG="$YDIR/cfg/yolov4-csp-x-mish.cfg"
WEIGHTS="$YDIR/yolov4-pacsp-x-mish.weights"
WEIGHTS_URL="https://github.com/WongKinYiu/PyTorch_YOLOv4/releases/download/weights/yolov4-pacsp-x-mish.weights"
NC="$(grep -c . "$DATA/train/_classes.txt")"

EPOCHS=$EPOCHS BATCH=$BATCH SUBDIV=$SUBDIV BURN_IN=$BURN_IN python "$YDIR/tool/patch_repo.py"

OBJ="$YDIR/data/obj"
rm -rf "$OBJ"; mkdir -p "$OBJ"
ln -sf "$DATA"/train/*.jpg "$DATA"/valid/*.jpg "$OBJ"/
cp "$DATA/train/_annotations.txt" "$YDIR/data/train.txt"
cp "$DATA/valid/_annotations.txt" "$YDIR/data/val.txt"

[ -f "$WEIGHTS" ] || curl -fSL "$WEIGHTS_URL" -o "$WEIGHTS"

cd "$YDIR"
python train.py -g "$GPUS" -l "$LR" -classes "$NC" -dir "$OBJ" \
  -train_label_path "$YDIR/data/train.txt" -optimizer adam -iou-type ciou \
  -cfg "$CFG" -data "$YDIR/data/chess.yaml" -weights "$WEIGHTS"
