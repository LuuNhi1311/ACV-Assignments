# Phát hiện sinh vật biển — YOLOv8, DETR & RT-DETR

*English: [README.md](README.md)*

**Thị giác máy tính nâng cao — Lab 03**

| Thành viên | MSSV |
|---|---|
| Lưu Thị Yến Nhi | 25C11014 |
| Hoàng Trọng Vũ | 25C15028 |

## Giới thiệu
Bài tập nghiên cứu **phát hiện đối tượng** trên bộ dữ liệu **marine-detect**
([Orange-OpenSource/marine-detect](https://github.com/Orange-OpenSource/marine-detect))
với **ba họ kiến trúc**, tất cả chấm bằng **cùng một bộ độ đo COCO**:

| Họ | Mô hình | Framework | Cách tiếp cận |
|---|---|---|---|
| One-stage, anchor-free | **YOLOv8** | Ultralytics | dự đoán dày + NMS |
| Transformer, set-based | **DETR (ResNet-50)** | HuggingFace | so khớp song ánh, không NMS |
| Transformer thời gian thực | **RT-DETR** | Ultralytics | hybrid encoder, không NMS |

YOLOv8 và RT-DETR dùng **chung API/định dạng dữ liệu** của Ultralytics; DETR là
HuggingFace [`facebook/detr-resnet-50`](https://huggingface.co/facebook/detr-resnet-50),
được viết lại với vòng lặp huấn luyện minh bạch.

## Tập dữ liệu
Hai bộ dữ liệu định dạng YOLO từ marine-detect (đều chia sẵn train/val/test):

| Bộ | Số lớp | Ví dụ |
|---|---|---|
| **MegaFauna** | 3 | Sharks, Sea Turtles, Rays |
| **FishInv** | 14 | Grouper, Parrotfish, Snapper, Moray Eel, Giant Clam, Urchin, Lobster, … (cá + động vật không xương) |

Danh sách/thứ tự lớp chuẩn nằm trong `data.yaml` của từng bộ sau khi tải.

**Liên kết (theo yêu cầu — link gốc + link reup):**
- **MegaFauna — gốc:** `https://stpubtenakanclyw.blob.core.windows.net/marine-detect/MegaFauna-dataset.zip?sv=2022-11-02&ss=bf&srt=co&sp=rltf&se=2099-12-31T18:55:46Z&st=2025-02-03T10:55:46Z&spr=https,http&sig=w%2FTQzrECsYsjtkBXNnnuFtn%2BC06PkjgLxDgRw%2FaUUKI%3D`
- **FishInv — gốc:** `https://stpubtenakanclyw.blob.core.windows.net/marine-detect/FishInv-dataset.zip?sv=2022-11-02&ss=bf&srt=co&sp=rltf&se=2099-12-31T18:55:46Z&st=2025-02-03T10:55:46Z&spr=https,http&sig=w%2FTQzrECsYsjtkBXNnnuFtn%2BC06PkjgLxDgRw%2FaUUKI%3D`
- **Reup (Google Drive):** `<TODO: upload 2 file zip lên Drive và dán link share vào đây>`

> Link gốc là Azure blob kèm SAS token có thể hết hạn — đây đúng là lý do yêu cầu cần thêm
> link reup Drive/OneDrive. **Hãy upload 2 file zip lên Drive và điền link ở trên.**

Tải + chuẩn hóa cả hai bằng [`source/dataset.sh`](source/dataset.sh) (sinh
`data/megafauna.yaml` và `data/fishinv.yaml` với đường dẫn tuyệt đối).

## Độ đo đánh giá
Dùng **độ đo COCO** (qua `pycocotools`) làm thước đo chính, độc lập mô hình, tính bằng
**một evaluator dùng chung** ([`common/coco_metrics.py`](source/common/coco_metrics.py))
cho cả ba mô hình — nên số liệu so sánh được trực tiếp.

| Độ đo | Đo cái gì |
|---|---|
| **mAP@[0.50:0.95]** | con số chính — trung bình 10 ngưỡng IoU |
| **mAP@0.50 / @0.75** | định vị dễ / chặt |
| **AP small/medium/large** | độ chính xác theo kích thước |
| **AR@[1/10/100]** | trần recall |
| **AP từng lớp** | loài nào khó |
| **Params (M), FPS** | đánh đổi hiệu năng |

## Cài đặt
YOLOv8/RT-DETR (Ultralytics) và DETR (HuggingFace) nên dùng **hai môi trường riêng**:
```bash
# --- Môi trường Ultralytics (YOLOv8 + RT-DETR) ---
conda create -n ultra python=3.10 -y && conda activate ultra
pip install -r source/ultralytics_det/requirements.txt

# --- Môi trường DETR (HuggingFace) ---
conda create -n detr python=3.10 -y && conda activate detr
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r source/detr/requirements.txt
```

## Cách chạy
```bash
cd source
bash dataset.sh                                   # tải + chuẩn hóa cả hai bộ

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
Biến môi trường: `ARCH`, `DATA`, `SPLIT`, `GPU`, `EPOCHS`, `BATCH`, `IMGSZ`, `CONF`, `WEIGHTS`, `SOURCE`.

## Cấu trúc mã nguồn
```
source/
├── common/
│   ├── yolo_to_coco.py     # YOLO split -> COCO json
│   ├── prepare_data.py     # chuẩn hóa dataset tải về -> data.yaml sạch
│   └── coco_metrics.py     # evaluator pycocotools dùng chung (so sánh công bằng)
├── ultralytics_det/        # YOLOv8 + RT-DETR: train.py / evaluate.py / demo.py
├── detr/                   # DETR HuggingFace: train.py / evaluate.py / demo.py / dataset.py
├── dataset.sh              # tải + chuẩn hóa MegaFauna & FishInv
└── *_ultralytics.sh, *_detr.sh
```

## Kết quả
> Điền sau khi chạy `test_*.sh`. Báo cáo một bảng cho mỗi bộ dữ liệu.

### MegaFauna (3 lớp)
| Mô hình | mAP@[.5:.95] | mAP@0.50 | mAP@0.75 | AR@100 | Params (M) | FPS |
|---|---|---|---|---|---|---|
| YOLOv8 | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ |
| DETR (R50) | _TODO_ | _TODO_ | _TODO_ | _TODO_ | ~41 | _TODO_ |
| RT-DETR | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ |

### FishInv (14 lớp)
| Mô hình | mAP@[.5:.95] | mAP@0.50 | mAP@0.75 | AR@100 | Params (M) | FPS |
|---|---|---|---|---|---|---|
| YOLOv8 | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ |
| DETR (R50) | _TODO_ | _TODO_ | _TODO_ | _TODO_ | ~41 | _TODO_ |
| RT-DETR | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ | _TODO_ |

**Phân tích (cần viết):** YOLOv8 so với hai mô hình transformer; RT-DETR thời gian thực
không cần NMS so với DETR gốc hội tụ chậm trên tập vừa; AP từng lớp để tìm loài khó nhất;
đánh đổi độ chính xác – tốc độ – tham số.

## Đóng góp
| Hạng mục | Lưu Thị Yến Nhi (25C11014) | Hoàng Trọng Vũ (25C15028) |
|---|---|---|
| Tải dữ liệu & chuyển YOLO→COCO |  |  |
| YOLOv8 (Ultralytics) |  |  |
| RT-DETR (Ultralytics) |  |  |
| DETR (HuggingFace) |  |  |
| Đánh giá COCO dùng chung & so sánh |  |  |
| Viết báo cáo & hình ảnh |  |  |
| **Tổng đóng góp** |  |  |
