#!/usr/bin/env bash
# Download + prepare marine datasets (MegaFauna, FishInv) -> data/<name>.yaml
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; DATA="$ROOT/data"; mkdir -p "$DATA"
SAS="sv=2022-11-02&ss=bf&srt=co&sp=rltf&se=2099-12-31T18:55:46Z&st=2025-02-03T10:55:46Z&spr=https,http&sig=w%2FTQzrECsYsjtkBXNnnuFtn%2BC06PkjgLxDgRw%2FaUUKI%3D"
BASE="https://stpubtenakanclyw.blob.core.windows.net/marine-detect"

dl() {  # $1 zip, $2 subdir  (resumable; survives 'connection reset')
  [ -d "$DATA/$2" ] && [ -n "$(ls -A "$DATA/$2" 2>/dev/null)" ] && { echo ">> $2 ready, skip"; return; }
  echo ">> downloading $1"
  curl -fL -C - --retry 10 --retry-delay 5 --retry-all-errors -o "$DATA/$1" "$BASE/$1?$SAS"
  rm -rf "$DATA/$2"; mkdir -p "$DATA/$2"; unzip -q -o "$DATA/$1" -d "$DATA/$2"; rm -f "$DATA/$1"
}

dl "MegaFauna-dataset.zip" "megafauna"
dl "FishInv-dataset.zip"   "fishinv"
python "$ROOT/common/prepare_data.py" "$DATA/megafauna" -o "$DATA/megafauna.yaml"
python "$ROOT/common/prepare_data.py" "$DATA/fishinv"   -o "$DATA/fishinv.yaml"
echo "Done. DATA=$DATA/megafauna.yaml | $DATA/fishinv.yaml"
