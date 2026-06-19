# Phát hiện quân cờ vua — YOLOv4, DETR & Faster R-CNN

*English: [README.md](README.md)*

**Thị giác máy tính nâng cao — Lab 03**

| Thành viên | MSSV |
|---|---|
| Lưu Thị Yến Nhi | 25C11014 |
| Hoàng Trọng Vũ | 25C15028 |

## Giới thiệu
Bài tập nghiên cứu **phát hiện đối tượng (Object Detection)** trên tập dữ liệu Roboflow
**Chess Pieces** (12 lớp) với **ba họ kiến trúc khác nhau**, tất cả được chấm bằng **cùng
một bộ độ đo COCO** để kết quả so sánh được trực tiếp:

| Họ | Mô hình | Cách tiếp cận |
|---|---|---|
| One-stage, anchor-based | **YOLOv4 (CSP-X / Mish)** | dự đoán dày trên lưới + NMS |
| Transformer, set-based | **DETR (ResNet-50)** | so khớp song ánh, không cần NMS |
| Two-stage, anchor-based | **Faster R-CNN (ResNet-50 FPN)** | đề xuất vùng → đầu ROI |

YOLOv4 dùng cài đặt [Tianxiaomo/pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4);
DETR là HuggingFace [`facebook/detr-resnet-50`](https://huggingface.co/facebook/detr-resnet-50);
Faster R-CNN là `torchvision` `fasterrcnn_resnet50_fpn`. Ngoài phát hiện thông thường,
YOLOv4 còn bổ sung hai góc nhìn diễn giải (**Grad-CAM**, **t-SNE**).

12 lớp: `black-bishop, black-king, black-knight, black-pawn, black-queen, black-rook,
white-bishop, white-king, white-knight, white-pawn, white-queen, white-rook`.

## Tập dữ liệu
**Chess Pieces** (Roboflow Public, bản *v1 416x416 auto-orient*, định dạng *yolov4pytorch*).
Giấy phép **Public Domain** (Roboflow xuất ngày 31/05/2020). Mỗi split (`train` / `valid` /
`test`) là thư mục ảnh `.jpg` kèm `_annotations.txt` (mỗi dòng một ảnh:
`image.jpg x1,y1,x2,y2,class_id ...`, toạ độ pixel xyxy, class 0-index) và `_classes.txt`.
Ảnh được chụp từ một góc cố định (chân máy bên trái bàn cờ) nên thách thức chính là các quân
cờ nhỏ, dày đặc và **che khuất lẫn nhau**.

**Thống kê** — 12 lớp, **289 ảnh / 2870 hộp** (~10 đối tượng/ảnh), kích thước đồng nhất `416×416`:

| Tập | Số ảnh | Số hộp | Hộp/ảnh |
|---|---|---|---|
| train | 202 | 2108 | 10.44 |
| valid | 58 | 386 | 6.66 |
| test | 29 | 376 | 12.97 |
| **Tổng** | **289** | **2870** | **9.93** |

Dữ liệu **mất cân bằng lớp**: quân tốt nhiều nhất (black-pawn 659, white-pawn 639), quân hậu
ít nhất (black-queen 87, white-queen 111).

- **Link gốc:** https://public.roboflow.com/object-detection/chess-full
- **Link reup (Google Drive):** https://drive.google.com/uc?id=18-AUodrP2NDTvPWC2NsDbhrZt9XO21OR

Tải bằng [`source/dataset.sh`](source/dataset.sh) (dùng bản reup trên Drive qua `gdown`).

## Độ đo đánh giá
Chúng tôi chọn **độ đo COCO** làm thước đo chính, độc lập mô hình, vì nó lấy trung bình
precision trên toàn dải recall và trên **10 ngưỡng IoU** (0.50→0.95) — cách chuẩn và công
bằng để so các bộ phát hiện có kiểu đầu ra rất khác nhau (lưới dày YOLO vs. tập hợp DETR vs.
đề xuất vùng). Cùng một evaluator `pycocotools`
([`common/coco_metrics.py`](source/common/coco_metrics.py)) được dùng cho **cả ba** mô hình.

| Độ đo | Đo cái gì | Vì sao quan trọng |
|---|---|---|
| **mAP @[0.50:0.95]** | AP trung bình qua IoU 0.50→0.95 | con số chính — định vị nghiêm ngặt |
| **mAP @0.50** | AP tại IoU=0.50 (kiểu PASCAL-VOC) | dễ; "có tìm thấy quân cờ hay không" |
| **mAP @0.75** | AP tại IoU=0.75 | chất lượng định vị chặt |
| **AP small / medium / large** | AP theo diện tích | quân cờ chủ yếu *nhỏ/vừa* |
| **AR @[1/10/100]** | recall theo số lượng phát hiện | trần recall, đối tượng bị bỏ sót |
| **AP từng lớp** | AP theo loại quân | quân nào khó (vd. mã vs. tượng) |

Ngoài độ chính xác, còn báo cáo **hiệu năng** để thấy đánh đổi:
**số tham số (M)** và **tốc độ suy luận (FPS)** (mỗi `evaluate.py` đều in ra).

## Cài đặt
DETR và Faster R-CNN dùng chung một môi trường; YOLOv4 giữ môi trường riêng (xem phần YOLOv4).
```bash
conda create -n detlab python=3.10 -y
conda activate detlab
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r source/detr/requirements.txt          # transformers, timm, pycocotools, ...
pip install gdown
```

## Cách chạy
```bash
cd source
bash dataset.sh                 # tải dữ liệu vào data/

# ---- DETR ----------------------------------------------------------------
GPU=0 bash train_detr.sh        # fine-tune facebook/detr-resnet-50
GPU=0 bash test_detr.sh         # COCO metrics + FPS + AP từng lớp trên test
GPU=0 CONF=0.5 bash demo_detr.sh

# ---- Faster R-CNN --------------------------------------------------------
GPU=0 bash train_fasterrcnn.sh  # fine-tune torchvision fasterrcnn_resnet50_fpn
GPU=0 bash test_fasterrcnn.sh
GPU=0 CONF=0.5 bash demo_fasterrcnn.sh

# ---- YOLOv4 (xem "Chi tiết YOLOv4" bên dưới) -----------------------------
GPUS=0 bash train.sh
GPU=0 bash test.sh
GPU=0 bash demo.sh
```
Biến môi trường có thể chỉnh: `GPU`, `EPOCHS`, `BATCH`, `LR`, `SPLIT`, `CONF`, `WEIGHTS`.

## Cấu trúc mã nguồn
```
source/
├── common/            # dùng chung cho cả 3 mô hình
│   ├── roboflow_to_coco.py   # _annotations.txt -> COCO json
│   └── coco_metrics.py       # 1 evaluator pycocotools duy nhất (so sánh công bằng)
├── detr/              # train.py / evaluate.py / demo.py / dataset.py
├── faster-rcnn/       # train.py / evaluate.py / demo.py / dataset.py
├── pytorch-YOLOv4/    # cài đặt YOLOv4 (+ Grad-CAM, t-SNE)
└── *.sh               # script train/test/demo cho từng mô hình
```

## Kết quả

### So sánh các mô hình (tập test)
> Điền sau khi chạy `test_*.sh`. Dòng YOLOv4 lấy từ lần chạy hiện có.

| Mô hình | mAP@[.5:.95] | mAP@0.50 | mAP@0.75 | AP (small) | AR@100 | Params (M) | FPS |
|---|---|---|---|---|---|---|---|
| **YOLOv4 (CSP-X)** | **0.7621** | **0.9803** | 0.9610 | 0.7660 | 0.8032 | ~99 | _TODO_ |
| **DETR (R50)** | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | ~41 | _TODO_ |
| **Faster R-CNN (R50-FPN)** | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | ~41 | _TODO_ |

**Phân tích (cần viết):** bàn về đánh đổi độ chính xác – tốc độ – tham số; DETR không cần
NMS nhưng hội tụ chậm trên tập nhỏ; Faster R-CNN recall vật nhỏ tốt nhờ FPN; YOLOv4 cân
bằng tốc độ/độ chính xác. Dùng bảng AP từng lớp để chỉ ra quân cờ khó nhất.

### Chi tiết YOLOv4
Chúng tôi **viết lại file cấu hình darknet cho 12 lớp**
([`cfg/yolov4-csp-x-mish_12cls.cfg`](source/pytorch-YOLOv4/cfg/yolov4-csp-x-mish_12cls.cfg))
và **fine-tune từ trọng số tiền huấn luyện `yolov4-pacsp-x-mish.weights`** (COCO, 80 lớp).

***Checkpoint: [`yolov4-csp-x-mish_12cls`](https://drive.google.com/file/d/1vSblt8nzjVz2d8pSmQIJbhj4xForsPqE/view?usp=sharing)***

| Độ đo | Giá trị |
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

### Định tính DETR / Faster R-CNN
> Bổ sung `source/detr/visualizations/*` và `source/faster-rcnn/visualizations/*` sau khi chạy demo.

## Đóng góp
| Hạng mục công việc | Lưu Thị Yến Nhi (25C11014) | Hoàng Trọng Vũ (25C15028) |
|---|---|---|
| Chuẩn bị dữ liệu & chuyển đổi COCO |  |  |
| Huấn luyện YOLOv4 + Grad-CAM/t-SNE |  |  |
| Pipeline DETR |  |  |
| Pipeline Faster R-CNN |  |  |
| Đánh giá COCO dùng chung & so sánh |  |  |
| Viết báo cáo & hình ảnh |  |  |
| **Tổng đóng góp** |  |  |
