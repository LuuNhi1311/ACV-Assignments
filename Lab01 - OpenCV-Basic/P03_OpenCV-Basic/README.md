# Object Detection on Chess Pieces — YOLOv4, DETR & Faster R-CNN

*Tiếng Việt: [README-vi.md](README-vi.md)*

**Advanced Computer Vision — Lab 03**

| Member | Student ID |
|---|---|
| Lưu Thị Yến Nhi | 25C11014 |
| Hoàng Trọng Vũ | 25C15028 |

## Introduction
This homework studies **object detection** on the Roboflow **Chess Pieces** dataset
(12 classes) across **three architecturally different detector families**, all scored
with the **same COCO metric protocol** so the numbers are directly comparable:

| Family | Model | Paradigm |
|---|---|---|
| One-stage, anchor-based | **YOLOv4 (CSP-X / Mish)** | dense grid prediction + NMS |
| Transformer, set-based | **DETR (ResNet-50)** | bipartite matching, NMS-free |
| Two-stage, anchor-based | **Faster R-CNN (ResNet-50 FPN)** | region proposals → ROI head |

YOLOv4 is the [Tianxiaomo/pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4)
implementation; DETR is HuggingFace [`facebook/detr-resnet-50`](https://huggingface.co/facebook/detr-resnet-50);
Faster R-CNN is `torchvision` `fasterrcnn_resnet50_fpn`. On top of plain detection,
YOLOv4 also adds two interpretability views (**Grad-CAM**, **t-SNE**).

The 12 classes are: `black-bishop, black-king, black-knight, black-pawn, black-queen,
black-rook, white-bishop, white-king, white-knight, white-pawn, white-queen, white-rook`.

## Dataset
**Chess Pieces** (Roboflow Public, *yolov4pytorch* export, auto-orient, 416×416).
Each split (`train` / `valid` / `test`) is a folder of `.jpg` images plus
`_annotations.txt` (one line per image: `image.jpg x1,y1,x2,y2,class_id ...`, pixel xyxy,
0-indexed class) and `_classes.txt` (class names). License: **Public Domain** (exported by
Roboflow, 2020-05-31). All photos were captured from a constant angle (tripod left of the
board), so occlusion of densely-packed pieces is the main challenge.

**Statistics** — 12 classes, **289 images / 2870 boxes** total (~10 objects/image), uniform
`416×416`:

| Split | Images | Boxes | Boxes/img |
|---|---|---|---|
| train | 202 | 2108 | 10.44 |
| valid | 58 | 386 | 6.66 |
| test | 29 | 376 | 12.97 |
| **Total** | **289** | **2870** | **9.93** |

Class distribution is **imbalanced**: pawns dominate (black-pawn 659, white-pawn 639) while
queens are rarest (black-queen 87, white-queen 111).

- **Original:** https://public.roboflow.com/object-detection/chess-full
- **Re-up (Google Drive):** https://drive.google.com/uc?id=18-AUodrP2NDTvPWC2NsDbhrZt9XO21OR

Download with [`source/dataset.sh`](source/dataset.sh) (uses the Drive re-up via `gdown`).

## Evaluation metrics
We chose **COCO detection metrics** as the primary, model-agnostic comparison because
they integrate precision over the full recall range and over **10 IoU thresholds**
(0.50→0.95), which is the standard, fair way to compare detectors with very different
output styles (dense YOLO grids vs. set-based DETR vs. region proposals). The same
`pycocotools` evaluator ([`common/coco_metrics.py`](source/common/coco_metrics.py)) is
used for **all three** models.

| Metric | What it measures | Why it matters here |
|---|---|---|
| **mAP @[0.50:0.95]** | mean AP averaged over IoU 0.50→0.95 | primary headline number — strict localization |
| **mAP @0.50** | AP at IoU=0.50 (PASCAL-VOC style) | lenient; "did we find the piece at all" |
| **mAP @0.75** | AP at IoU=0.75 | strict localization quality |
| **AP small / medium / large** | AP by object area | chess pieces are mostly *small/medium* |
| **AR @[1/10/100]** | average recall vs. #detections | recall ceiling, missed objects |
| **Per-class AP** | AP per piece type | which pieces are hard (e.g. knight vs. bishop) |

Beyond accuracy we also report **efficiency** so the trade-off is visible:
**#parameters (M)** and **inference speed (FPS)** (printed by each `evaluate.py`).

## Setup
DETR and Faster R-CNN share one environment; YOLOv4 keeps its own (see its section).
```bash
conda create -n detlab python=3.10 -y
conda activate detlab
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r source/detr/requirements.txt          # transformers, timm, pycocotools, ...
pip install gdown
```

## Usage
```bash
cd source
bash dataset.sh                 # download dataset into data/

# ---- DETR ----------------------------------------------------------------
GPU=0 bash train_detr.sh        # fine-tune facebook/detr-resnet-50
GPU=0 bash test_detr.sh         # COCO metrics + FPS + per-class AP on test
GPU=0 CONF=0.5 bash demo_detr.sh

# ---- Faster R-CNN --------------------------------------------------------
GPU=0 bash train_fasterrcnn.sh  # fine-tune torchvision fasterrcnn_resnet50_fpn
GPU=0 bash test_fasterrcnn.sh
GPU=0 CONF=0.5 bash demo_fasterrcnn.sh

# ---- YOLOv4 (see "YOLOv4 details" below) ---------------------------------
GPUS=0 bash train.sh
GPU=0 bash test.sh
GPU=0 bash demo.sh
```
Overridable env vars: `GPU`, `EPOCHS`, `BATCH`, `LR`, `SPLIT`, `CONF`, `WEIGHTS`.

## Code layout
```
source/
├── common/            # shared across all 3 models
│   ├── roboflow_to_coco.py   # Roboflow _annotations.txt -> COCO json
│   └── coco_metrics.py       # single pycocotools evaluator (fair comparison)
├── detr/              # train.py / evaluate.py / demo.py / dataset.py
├── faster-rcnn/       # train.py / evaluate.py / demo.py / dataset.py
├── pytorch-YOLOv4/    # YOLOv4 implementation (+ Grad-CAM, t-SNE)
└── *.sh               # train/test/demo wrappers per model
```

## Results

### Model comparison (test split)
> Fill in after running `test_*.sh`. YOLOv4 row is from the existing run.

| Model | mAP@[.5:.95] | mAP@0.50 | mAP@0.75 | AP (small) | AR@100 | Params (M) | FPS |
|---|---|---|---|---|---|---|---|
| **YOLOv4 (CSP-X)** | **0.7621** | **0.9803** | 0.9610 | 0.7660 | 0.8032 | ~99 | _TODO_ |
| **DETR (R50)** | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | ~41 | _TODO_ |
| **Faster R-CNN (R50-FPN)** | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | ~41 | _TODO_ |

**Analysis (to write up):** discuss accuracy vs. speed vs. params; DETR's NMS-free
set prediction and slower convergence on a small dataset; Faster R-CNN's strong small-object
recall from FPN; YOLOv4's speed/accuracy balance. Use the per-class AP printout to point
out the hardest pieces.

### YOLOv4 details
We **rewrote the darknet config to 12 classes**
([`cfg/yolov4-csp-x-mish_12cls.cfg`](source/pytorch-YOLOv4/cfg/yolov4-csp-x-mish_12cls.cfg))
and **fine-tuned from the pretrained `yolov4-pacsp-x-mish.weights`** (COCO, 80 classes).

***Checkpoint: [`yolov4-csp-x-mish_12cls`](https://drive.google.com/file/d/1vSblt8nzjVz2d8pSmQIJbhj4xForsPqE/view?usp=sharing)***

| Metric | Value |
|---|---|
| **mAP @[IoU=0.50:0.95]** | **0.7621** |
| **mAP @[IoU=0.50]** | **0.9803** |
| AP @[IoU=0.75] | 0.9610 |
| AP (small) | 0.7660 |
| AP (medium) | 0.7694 |
| AR @[maxDets=100] | 0.8032 |

![detections](source/pytorch-YOLOv4/visualizations/detections_sample.png)
![gradcam](source/pytorch-YOLOv4/visualizations/gradcam.png)
![tsne](source/pytorch-YOLOv4/visualizations/tsne_features.png)
![AP1](source/pytorch-YOLOv4/visualizations/AP1.png)
![AP2](source/pytorch-YOLOv4/visualizations/AP2.png)
![avg_loss](source/pytorch-YOLOv4/visualizations/avg_loss.png)
![losses](source/pytorch-YOLOv4/visualizations/losses.png)

### DETR / Faster R-CNN qualitative
> Add `source/detr/visualizations/*` and `source/faster-rcnn/visualizations/*` after running the demos.

## Contribution
| Work item | Lưu Thị Yến Nhi (25C11014) | Hoàng Trọng Vũ (25C15028) |
|---|---|---|
| Dataset prep & COCO conversion |  |  |
| YOLOv4 training + Grad-CAM/t-SNE |  |  |
| DETR pipeline |  |  |
| Faster R-CNN pipeline |  |  |
| Shared COCO evaluation & comparison |  |  |
| Report writing & figures |  |  |
| **Overall effort** |  |  |
