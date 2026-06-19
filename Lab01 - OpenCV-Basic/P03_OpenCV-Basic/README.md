# Marine Species Detection — YOLOv8, DETR & RT-DETR

*Tiếng Việt: [README-vi.md](README-vi.md)*

**Advanced Computer Vision — Lab 03**

| Member | Student ID |
|---|---|
| Lưu Thị Yến Nhi | 25C11014 |
| Hoàng Trọng Vũ | 25C15028 |

## Introduction
This homework studies **object detection** on the **marine-detect** datasets
([Orange-OpenSource/marine-detect](https://github.com/Orange-OpenSource/marine-detect))
across **three detector families**, all scored with the **same COCO metric protocol**:

| Family | Model | Framework | Paradigm |
|---|---|---|---|
| One-stage, anchor-free | **YOLOv8** | Ultralytics | dense prediction + NMS |
| Transformer, set-based | **DETR (ResNet-50)** | HuggingFace | bipartite matching, NMS-free |
| Real-time Transformer | **RT-DETR** | Ultralytics | hybrid encoder, NMS-free |

YOLOv8 and RT-DETR share the exact same Ultralytics API/data format; DETR is
HuggingFace [`facebook/detr-resnet-50`](https://huggingface.co/facebook/detr-resnet-50),
re-coded with a transparent training loop.

## Datasets
Two ready-to-use YOLO-format datasets from marine-detect (each split into train/val/test):

| Dataset | Classes | Examples |
|---|---|---|
| **MegaFauna** | 3 | Sharks, Sea Turtles, Rays |
| **FishInv** | 14 | Grouper, Parrotfish, Snapper, Moray Eel, Giant Clam, Urchin, Lobster, … (fish + invertebrates) |

The authoritative class list/order is in each dataset's `data.yaml` after download.

**Links (per assignment requirement — original + a re-up):**
- **MegaFauna — original:** `https://stpubtenakanclyw.blob.core.windows.net/marine-detect/MegaFauna-dataset.zip?sv=2022-11-02&ss=bf&srt=co&sp=rltf&se=2099-12-31T18:55:46Z&st=2025-02-03T10:55:46Z&spr=https,http&sig=w%2FTQzrECsYsjtkBXNnnuFtn%2BC06PkjgLxDgRw%2FaUUKI%3D`
- **FishInv — original:** `https://stpubtenakanclyw.blob.core.windows.net/marine-detect/FishInv-dataset.zip?sv=2022-11-02&ss=bf&srt=co&sp=rltf&se=2099-12-31T18:55:46Z&st=2025-02-03T10:55:46Z&spr=https,http&sig=w%2FTQzrECsYsjtkBXNnnuFtn%2BC06PkjgLxDgRw%2FaUUKI%3D`
- **Re-up (Google Drive):** `<TODO: upload both zips to Drive and paste the share link here>`

> The original links are Azure blobs with a SAS token that may eventually expire — that is
> exactly why the assignment asks for a Drive/OneDrive re-up. **Upload both zips to Drive and fill the link above.**

Download + prepare both with [`source/dataset.sh`](source/dataset.sh) (writes
`data/megafauna.yaml` and `data/fishinv.yaml` with absolute paths).

## Evaluation metrics
We use **COCO detection metrics** (via `pycocotools`) as the primary, model-agnostic
comparison, computed by **one shared evaluator** ([`common/coco_metrics.py`](source/common/coco_metrics.py))
for all three models — so the numbers are directly comparable.

| Metric | What it measures |
|---|---|
| **mAP@[0.50:0.95]** | headline — averaged over 10 IoU thresholds |
| **mAP@0.50 / @0.75** | lenient / strict localization |
| **AP small/medium/large** | accuracy by object size |
| **AR@[1/10/100]** | recall ceiling |
| **Per-class AP** | which species are hard |
| **Params (M), FPS** | efficiency trade-off |

## Setup
YOLOv8/RT-DETR (Ultralytics) and DETR (HuggingFace) work best in **separate envs**:
```bash
# --- Ultralytics env (YOLOv8 + RT-DETR) ---
conda create -n ultra python=3.10 -y && conda activate ultra
pip install -r source/ultralytics_det/requirements.txt

# --- DETR env (HuggingFace) ---
conda create -n detr python=3.10 -y && conda activate detr
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r source/detr/requirements.txt
```

## Usage
```bash
cd source
bash dataset.sh                                   # download + prepare both datasets

# ---- YOLOv8 / RT-DETR (Ultralytics) ----
ARCH=yolov8 DATA=data/megafauna.yaml GPU=0 bash train_ultralytics.sh
ARCH=yolov8 DATA=data/megafauna.yaml      bash test_ultralytics.sh
ARCH=rtdetr DATA=data/fishinv.yaml        bash train_ultralytics.sh
ARCH=rtdetr DATA=data/fishinv.yaml        bash test_ultralytics.sh
ARCH=yolov8 DATA=data/megafauna.yaml SOURCE=data/megafauna/images/test bash demo_ultralytics.sh

# ---- DETR (HuggingFace) ----
DATA=data/megafauna.yaml GPU=0 bash train_detr.sh
DATA=data/megafauna.yaml      bash test_detr.sh
DATA=data/megafauna.yaml SOURCE=data/megafauna/images/test bash demo_detr.sh
```
Override via env: `ARCH`, `DATA`, `SPLIT`, `GPU`, `EPOCHS`, `BATCH`, `IMGSZ`, `CONF`, `WEIGHTS`, `SOURCE`.

## Code layout
```
source/
├── common/
│   ├── yolo_to_coco.py     # Ultralytics YOLO split -> COCO json
│   ├── prepare_data.py     # normalise downloaded dataset -> clean data.yaml
│   └── coco_metrics.py     # single shared pycocotools evaluator (fair comparison)
├── ultralytics_det/        # YOLOv8 + RT-DETR: train.py / evaluate.py / demo.py
├── detr/                   # HuggingFace DETR: train.py / evaluate.py / demo.py / dataset.py
├── dataset.sh              # download + prepare MegaFauna & FishInv
└── *_ultralytics.sh, *_detr.sh
```

## Results
> Fill in after running `test_*.sh`. Report one table per dataset.

### MegaFauna (3 classes)
| Model | mAP@[.5:.95] | mAP@0.50 | mAP@0.75 | AR@100 | Params (M) | FPS |
|---|---|---|---|---|---|---|
| YOLOv8 | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ |
| DETR (R50) | _TODO_ | _TODO_ | _TODO_ | _TODO_ | ~41 | _TODO_ |
| RT-DETR | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ |

### FishInv (14 classes)
| Model | mAP@[.5:.95] | mAP@0.50 | mAP@0.75 | AR@100 | Params (M) | FPS |
|---|---|---|---|---|---|---|
| YOLOv8 | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ |
| DETR (R50) | _TODO_ | _TODO_ | _TODO_ | _TODO_ | ~41 | _TODO_ |
| RT-DETR | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ |

**Analysis (to write up):** YOLOv8 vs the two transformer detectors; RT-DETR's real-time
NMS-free design vs vanilla DETR's slower convergence on a modest dataset; per-class AP to
spot the hardest species; accuracy vs speed vs params.

## Contribution
| Work item | Lưu Thị Yến Nhi (25C11014) | Hoàng Trọng Vũ (25C15028) |
|---|---|---|
| Dataset download & YOLO→COCO conversion |  |  |
| YOLOv8 (Ultralytics) |  |  |
| RT-DETR (Ultralytics) |  |  |
| DETR (HuggingFace) |  |  |
| Shared COCO evaluation & comparison |  |  |
| Report writing & figures |  |  |
| **Overall effort** |  |  |
