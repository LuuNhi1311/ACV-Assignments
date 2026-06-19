# Làm trơn ảnh & Phát hiện biên cạnh: Tự cài đặt vs Thư viện

**Advanced Computer Vision — Bài thực hành 02**

| Thành viên | MSHV |
|---|---|
| Lưu Thị Yến Nhi | 25C11014 |
| Hoàng Trọng Vũ | 25C15028 |

## Giới thiệu
Bài tập cài đặt hai nhóm thao tác xử lý ảnh **tự viết bằng NumPy** và so sánh từng
nhóm với OpenCV theo từng pixel:

1. **Làm trơn ảnh (smoothing)** — bộ lọc Mean, Gaussian, Median, Bilateral và Box so
   với `cv2.blur` / `cv2.GaussianBlur` / `cv2.medianBlur` / `cv2.bilateralFilter` / `cv2.boxFilter`.
2. **Phát hiện biên cạnh (edge detection)** — Roberts, Prewitt, Sobel, Scharr,
   Laplacian, LoG/DoG và toàn bộ quy trình Canny so với `cv2.filter2D` / `cv2.Sobel`
   / `cv2.Scharr` / `cv2.Laplacian` / `cv2.Canny`.

Mỗi kết quả tự cài đặt được chấm so với kết quả thư viện bằng **MAE** và **PSNR**.

Cấu trúc thư mục:
```
source/
  smoothing.py        # Chức năng 1: Mean/Gaussian/Median/Bilateral/Box + bộ tiện ích dùng chung
  edge_detection.py   # Chức năng 2: Roberts/Prewitt/Sobel/Scharr/Laplacian/LoG/DoG/Canny
  images/             # ảnh mẫu
  results/            # các hình so sánh được sinh ra
  requirements.txt
doc/                  # báo cáo LaTeX và hình ảnh
README.md             # bản tiếng Anh
README-vi.md          # file này
```
Toàn bộ mã nguồn nằm trong đúng hai file. `smoothing.py` chứa các thành phần dùng
chung (hàm `correlate2d` tự viết, nhân Gaussian, MAE/PSNR, đọc ảnh và hàm vẽ lưới
so sánh); `edge_detection.py` import lại các thành phần đó.

## Cài đặt môi trường
```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r source/requirements.txt
```
Thư viện phụ thuộc: `numpy`, `opencv-python`, `pillow`, `matplotlib`.

## Cách chạy
```bash
cd source

# Chức năng 1 — Làm trơn ảnh  (ghi ra results/01_smoothing.png)
python smoothing.py
python smoothing.py path/to/img.jpg

# Chức năng 2 — Phát hiện biên cạnh  (ghi ra results/02_edge_detection.png)
python edge_detection.py
python edge_detection.py path/to/img.jpg
```
Các chỉ số so sánh được in ra màn hình; mỗi script ghi một hình vào `source/results/`
(và sao chép sang `doc/figures/` để dùng cho báo cáo).

## Kết quả
Mỗi hình là một lưới: **cột là phương pháp (Tự cài đặt vs Thư viện)** và **hàng là
các phép xử lý**, để hai cách cài đặt nằm cạnh nhau dễ đối chiếu.

**Chức năng 1 — Làm trơn ảnh** (Mean, Gaussian, Median, Bilateral, Box):
![smoothing](source/results/01_smoothing.png)

**Chức năng 2 — Phát hiện biên cạnh** (Roberts, Prewitt, Sobel, Scharr, Laplacian, LoG, DoG, Canny):
![edges](source/results/02_edge_detection.png)

### Mức độ trùng khớp định lượng (ảnh mẫu `lena.jpg`)
| Phép xử lý | MAE | PSNR (dB) | Ghi chú |
|---|---|---|---|
| Mean 5×5 | 0.000 | ∞ | cùng nhân box |
| Gaussian 5×5, σ=1 | 0.006 | 70.61 | khớp với `cv2.getGaussianKernel` |
| Median 5×5 | 0.000 | ∞ | chính xác (median lấy mẫu có thật) |
| Bilateral d=5 | 0.932 | 43.05 | OpenCV xấp xỉ bằng bảng tra nội bộ |
| Box 5×5 | 0.000 | ∞ | giống hệt mean |
| Roberts | 0.000 | ∞ | cùng nhân 2×2 |
| Prewitt | 0.000 | ∞ | cùng nhân 3×3 |
| Sobel | 0.000 | ∞ | giống hệt (so với `cv2.Sobel`) |
| Scharr | 0.000 | ∞ | giống hệt (so với `cv2.Scharr`) |
| Laplacian | 0.000 | ∞ | giống hệt (nhân `ksize=1`) |
| LoG | 0.000 | ∞ | Gaussian + Laplacian, đều tuyến tính |
| DoG | 0.000 | ∞ | hiệu của hai lần làm mờ Gaussian |
| Canny | 1.222 | 23.19 | 99.52% pixel cùng nhãn biên/không-biên |

Sau khi ghép mỗi phép tự cài đặt với đúng hàm thư viện tương ứng — cùng nhân và đệm
viền `BORDER_REFLECT_101` giống `cv2.filter2D` — mọi toán tử **tuyến tính** đều trùng
khớp **tuyệt đối** (MAE = 0): mean, median, box, Roberts, Prewitt, Sobel, Scharr,
Laplacian, LoG và DoG. Gaussian chỉ còn ≈0.006 MAE do làm tròn số thực. Hai trường
hợp khác biệt về cấu trúc là **bộ lọc Bilateral** (OpenCV dùng bảng tra lượng tử hoá
nội bộ, ≈0.93 MAE) và **Canny** (NMS + liên kết biên phi tuyến, nhưng vẫn trùng 99.5%
nhãn biên). Xem `doc/report.tex` để có phân tích đầy đủ.

## Phân công công sức
| Hạng mục công việc | Lưu Thị Yến Nhi (25C11014) | Hoàng Trọng Vũ (25C15028) |
|---|---|---|
| Chức năng 1 — Làm trơn ảnh (Mean/Gaussian/Median/Bilateral/Box) | 60% | 40% |
| Chức năng 2 — Toán tử gradient (Roberts/Prewitt/Sobel/Scharr) | 40% | 60% |
| Chức năng 2 — Laplacian, LoG/DoG, quy trình Canny | 50% | 50% |
| Kiểm thử & so sánh tự-cài-đặt-vs-thư-viện | 40% | 60% |
| Viết báo cáo & dựng hình | 60% | 40% |
| **Tổng công sức** | **50%** | **50%** |
