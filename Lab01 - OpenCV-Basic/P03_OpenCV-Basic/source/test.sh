#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YDIR="$ROOT/pytorch-YOLOv4"; DATA="$ROOT/data"

SPLIT="${SPLIT:-test}"
GPU="${GPU:-7}"
WEIGHTS="$YDIR/checkpoints/best.pt"
CFG="$YDIR/cfg/yolov4-csp-x-mish.cfg"
NC="$(grep -c . "$DATA/$SPLIT/_classes.txt")"

python "$YDIR/tool/patch_repo.py"

OBJ="$YDIR/data/obj_$SPLIT"
rm -rf "$OBJ"; mkdir -p "$OBJ"
ln -sf "$DATA/$SPLIT"/*.jpg "$OBJ"/
cp "$DATA/$SPLIT/_annotations.txt" "$YDIR/data/$SPLIT.txt"

cd "$YDIR"
python evaluate_metrics.py -weights "$WEIGHTS" -dir "$OBJ" -gt "data/$SPLIT.txt" \
  -classes "$NC" -cfg "$CFG" -data "data/chess.yaml" -g "$GPU"
python visualize.py --mode all -weights "$WEIGHTS" -dir "$OBJ" -gt "data/$SPLIT.txt" \
  -classes "$NC" -cfg "$CFG" -data "data/chess.yaml" -names "$DATA/$SPLIT/_classes.txt" \
  -g "$GPU" -out visualizations
