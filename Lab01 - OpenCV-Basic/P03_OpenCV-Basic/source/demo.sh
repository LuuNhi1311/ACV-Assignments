#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YDIR="$ROOT/pytorch-YOLOv4"; DATA="$ROOT/data"

SPLIT=test
GPU=7
CONF=0.5
NMS=0.6
WEIGHTS="$YDIR/checkpoints/best.pt"
SRC_CFG="$YDIR/cfg/yolov4-csp-x-mish.cfg"
NC="$(grep -c . "$DATA/$SPLIT/_classes.txt")"
CFG="$YDIR/cfg/yolov4-csp-x-mish_${NC}cls.cfg"

python "$YDIR/tool/patch_repo.py"

[ -f "$CFG" ] || sed -e "s/classes=80/classes=$NC/g" \
  -e "s/filters=255/filters=$(( (NC + 5) * 3 ))/g" "$SRC_CFG" > "$CFG"

cd "$YDIR"
CUDA_VISIBLE_DEVICES="$GPU" python demo.py -cfgfile "$CFG" -weightfile "$WEIGHTS" \
  -imgfile "$DATA/$SPLIT" -names "$DATA/$SPLIT/_classes.txt" \
  -outdir visualizations -conf "$CONF" -nms "$NMS"
