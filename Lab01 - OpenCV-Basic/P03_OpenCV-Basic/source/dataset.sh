#!/usr/bin/env bash
# Download + prepare the marine-detect datasets (MegaFauna, FishInv).
# Both are Ultralytics-YOLO format. We download, unzip, then normalise each into
# a clean data.yaml (data/megafauna.yaml, data/fishinv.yaml) with absolute paths.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$ROOT/data"
mkdir -p "$DATA"

SAS="sv=2022-11-02&ss=bf&srt=co&sp=rltf&se=2099-12-31T18:55:46Z&st=2025-02-03T10:55:46Z&spr=https,http&sig=w%2FTQzrECsYsjtkBXNnnuFtn%2BC06PkjgLxDgRw%2FaUUKI%3D"
BASE="https://stpubtenakanclyw.blob.core.windows.net/marine-detect"

download() {  # $1 = remote zip name, $2 = local subfolder
  local url="$BASE/$1?$SAS"
  echo ">> downloading $1"
  curl -fSL "$url" -o "$DATA/$1"
  echo ">> extracting $1 -> $DATA/$2"
  rm -rf "$DATA/$2"; mkdir -p "$DATA/$2"
  unzip -q -o "$DATA/$1" -d "$DATA/$2"
  rm -f "$DATA/$1"
}

download "MegaFauna-dataset.zip" "megafauna"
download "FishInv-dataset.zip"   "fishinv"

python "$ROOT/common/prepare_data.py" "$DATA/megafauna" -o "$DATA/megafauna.yaml"
python "$ROOT/common/prepare_data.py" "$DATA/fishinv"   -o "$DATA/fishinv.yaml"

echo "Done. Use DATA=$DATA/megafauna.yaml or $DATA/fishinv.yaml in the train/test scripts."
