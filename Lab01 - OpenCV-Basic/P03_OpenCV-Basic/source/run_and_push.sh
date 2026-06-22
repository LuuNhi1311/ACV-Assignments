#!/usr/bin/env bash
# Train all models, collect light results into ../results/, git-push after each.
#   nohup bash source/run_and_push.sh > run_and_push.out 2>&1 &
# env: GROUP(all|ultra|detr) DATASETS GPU EPOCHS_ULTRA EPOCHS_DETR BATCH_ULTRA
#      BATCH_RTDETR BATCH_DETR IMGSZ WORKERS PIN_MEMORY ULTRA_ENV DETR_ENV PUSH(0/1)
# Note: RT-DETR needs a smaller batch than YOLOv8 (transformer = VRAM-heavy).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$ROOT/.." && pwd)"
REPO="$(git -C "$ROOT" rev-parse --show-toplevel)"
RESULTS="$PROJ/results"; LOGS="$RESULTS/logs"; FIGS="$RESULTS/figures"; PREDS="$RESULTS/preds"
mkdir -p "$LOGS" "$FIGS" "$PREDS"
GROUP="${GROUP:-all}"; DATASETS="${DATASETS:-megafauna fishinv}"; GPU="${GPU:-0}"
EPOCHS_ULTRA="${EPOCHS_ULTRA:-200}"; EPOCHS_DETR="${EPOCHS_DETR:-200}"
BATCH_ULTRA="${BATCH_ULTRA:-32}"; BATCH_RTDETR="${BATCH_RTDETR:-8}"; BATCH_DETR="${BATCH_DETR:-8}"
IMGSZ="${IMGSZ:-640}"; PUSH="${PUSH:-1}"

log() { echo "[$(date '+%F %T')] $*"; }

activate_env() {
  [ -z "${1:-}" ] && return 0
  source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
  conda activate "$1" 2>/dev/null && log "env $1" || log "WARN: env '$1' not activated"
}

snapshot() {  # $1 = label
  python "$ROOT/common/collect_results.py" -datasets $DATASETS > "$RESULTS/metrics.md" 2>>"$LOGS/collect.log" || log "WARN collect"
  python "$ROOT/common/plots.py" -datasets $DATASETS >>"$LOGS/plots.log" 2>&1 || log "WARN plots"
  git -C "$REPO" add -f "$RESULTS" 2>/dev/null || true
  git -C "$REPO" diff --cached --quiet 2>/dev/null && { log "nothing to commit"; return; }
  git -C "$REPO" commit -m "results: $1 @ $(date '+%F %H:%M')" >/dev/null 2>&1 && log "commit $1" || log "WARN commit"
  [ "$PUSH" = 1 ] || return 0
  git -C "$REPO" pull --rebase --autostash >/dev/null 2>&1 || true
  git -C "$REPO" push origin HEAD >/dev/null 2>&1 && log "pushed $1" || log "WARN push failed (committed locally)"
}

save_extra() {  # $1 tag, $2 ultra-run-dir (optional)
  [ -f "$ROOT/runs/${1}_pred_test.json" ] && cp -f "$ROOT/runs/${1}_pred_test.json" "$PREDS/" || true
  [ -n "${2:-}" ] || return 0
  for f in results.png results.csv confusion_matrix.png val_batch0_pred.jpg; do
    [ -f "$2/$f" ] && cp -f "$2/$f" "$FIGS/${1}__$f"
  done
}

ultra() {
  for DS in $DATASETS; do
    DATA="$ROOT/data/${DS}.yaml"; [ -f "$DATA" ] || { log "SKIP $DS (no yaml)"; continue; }
    for ARCH in yolov8 rtdetr; do
      TAG="${ARCH}_${DS}"
      B=$BATCH_ULTRA; [ "$ARCH" = rtdetr ] && B=$BATCH_RTDETR
      ARCH=$ARCH DATA=$DATA GPU=$GPU EPOCHS=$EPOCHS_ULTRA BATCH=$B IMGSZ=$IMGSZ \
        bash "$ROOT/train_ultralytics.sh" 2>&1 | tee "$LOGS/train_$TAG.log" || log "WARN train $TAG"
      ARCH=$ARCH DATA=$DATA GPU=$GPU SPLIT=test \
        bash "$ROOT/test_ultralytics.sh" 2>&1 | tee "$LOGS/test_$TAG.log" || log "WARN eval $TAG"
      save_extra "$TAG" "$ROOT/runs/$TAG"; snapshot "$TAG"
    done
  done
}

detr() {
  for DS in $DATASETS; do
    DATA="$ROOT/data/${DS}.yaml"; [ -f "$DATA" ] || { log "SKIP $DS (no yaml)"; continue; }
    TAG="detr_${DS}"
    DATA=$DATA GPU=$GPU EPOCHS=$EPOCHS_DETR BATCH=$BATCH_DETR \
      bash "$ROOT/train_detr.sh" 2>&1 | tee "$LOGS/train_$TAG.log" || log "WARN train $TAG"
    DATA=$DATA GPU=$GPU SPLIT=test \
      bash "$ROOT/test_detr.sh" 2>&1 | tee "$LOGS/test_$TAG.log" || log "WARN eval $TAG"
    save_extra "$TAG"; snapshot "$TAG"
  done
}

log "START group=$GROUP datasets='$DATASETS' gpu=$GPU"
case "$GROUP" in all|ultra) activate_env "${ULTRA_ENV:-}"; ultra ;; esac
case "$GROUP" in all|detr)  activate_env "${DETR_ENV:-}";  detr  ;; esac
snapshot final
log "DONE -> $RESULTS"
