"""Run Lab 02 features, save report figures, and write benchmark CSV files"""

import argparse
import csv
import os

from edge_detection import (
    canny_manual, canny_opencv, dog_response_map_manual, dog_response_map_opencv,
    edge_agreement, laplacian_response_map_manual, laplacian_response_map_opencv,
    log_response_map_manual, log_response_map_opencv, prewitt_manual, prewitt_opencv,
    roberts_manual, roberts_opencv, scharr_manual, scharr_opencv, sobel_manual, sobel_opencv,
)
from smoothing import (
    bilateral_manual, bilateral_opencv, gaussian_blur_manual, gaussian_blur_opencv,
    mean_blur_manual, mean_blur_opencv, median_blur_manual, median_blur_opencv,
)
from utils import (
    DEFAULT_IMAGE, DOCFIG, RESULTS, benchmark_report, compare_report, ensure_dir,
    imread_pillow, measure_run, show_compare_grid, show_image_grid, to_gray,
)

CSV_COLUMNS = (
    "feature",
    "operation",
    "implementation",
    "time_ms",
    "peak_kib",
    "mae_vs_opencv",
    "psnr_vs_opencv",
    "edge_agreement_percent",
)

SMOOTHING_KSIZE = 11
GAUSSIAN_SIGMA = 2.0
BILATERAL_SIGMA_COLOR = 50.0
BILATERAL_SIGMA_SPACE = 5.0
LOG_KSIZE = 11
LOG_SIGMA = 2.0
DOG_KSIZE = 5
DOG_SIGMA1 = 1.0
DOG_SIGMA2 = 2.0
SECOND_ORDER_DISPLAY_MAX = 40.0
CANNY_PARAM_SETS = ((30, 90), (80, 160))


def output_path(name: str) -> str:
    """Return a path under source/results"""
    return os.path.join(RESULTS, name)


def csv_path(name: str) -> str:
    """Return a benchmark CSV path under source/results"""
    return output_path(name)


def setup_output_dirs() -> None:
    """Create output folders for generated figures and benchmark CSV files"""
    ensure_dir(RESULTS)
    ensure_dir(DOCFIG)


def parse_args():
    """Parse command-line options for the assignment runner"""
    parser = argparse.ArgumentParser(
        description="Run Lab 02 NumPy-vs-OpenCV features and write figures/benchmark CSV files"
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=DEFAULT_IMAGE,
        help="Input image path. Defaults to source/images/lena.jpg",
    )
    parser.add_argument(
        "-f",
        "--feature",
        choices=("all", "smoothing", "edge"),
        default="all",
        help="Feature group to run. Default: all",
    )
    return parser.parse_args()


def load_image(path: str):
    """Read an image once and keep RGB uint8 data"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    print(f"Loading: {path}")
    img = imread_pillow(path)
    print("Image shape:", img.shape, "| dtype:", img.dtype)
    return img


def _format_metric(value):
    if value is None:
        return ""
    if isinstance(value, float) and value == float("inf"):
        return "inf"
    return f"{value:.6f}" if isinstance(value, float) else value


def _append_row(rows, feature, operation, implementation, stats, quality=None):
    quality = quality or {}
    rows.append({
        "feature": feature,
        "operation": operation,
        "implementation": implementation,
        "time_ms": f"{stats['time_ms']:.6f}",
        "peak_kib": f"{stats['peak_kib']:.6f}",
        "mae_vs_opencv": _format_metric(quality.get("mae")),
        "psnr_vs_opencv": _format_metric(quality.get("psnr")),
        "edge_agreement_percent": _format_metric(quality.get("edge_agreement_percent")),
    })


def write_benchmark_csv(path: str, rows) -> None:
    """Write benchmark rows to CSV"""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SAVED] {path}")


def save_fig(name: str, rows, pair_cols=None) -> None:
    """Render a Manual-vs-OpenCV comparison grid into results/ and doc/figures/"""
    for path in (os.path.join(RESULTS, name), os.path.join(DOCFIG, name)):
        show_compare_grid(rows, save_path=path, pair_cols=pair_cols)


def save_grid_fig(name: str, rows, col_titles) -> None:
    """Render a generic image grid into results/ and doc/figures/"""
    for path in (os.path.join(RESULTS, name), os.path.join(DOCFIG, name)):
        show_image_grid(rows, col_titles, save_path=path)


def run_pair(feature, name, manual_fn, manual_args, library_fn, library_args, rows,
             quality_extras=None):
    """Run manual and OpenCV versions, then record quality and benchmark rows"""
    manual, manual_stats = measure_run(manual_fn, *manual_args)
    library, library_stats = measure_run(library_fn, *library_args)
    quality = compare_report(name, manual, library)
    if quality_extras:
        quality.update(quality_extras(manual, library))
    benchmark_report(name, manual_stats, library_stats)
    _append_row(rows, feature, name, "manual_numpy", manual_stats, quality)
    _append_row(rows, feature, name, "opencv", library_stats, quality)
    return manual, library


def run_smoothing_feature(img, rows):
    """Feature 1: smoothing filters"""
    print("\n=== Feature 1: Smoothing (Manual vs OpenCV) ===")
    ksize = SMOOTHING_KSIZE
    sigma = GAUSSIAN_SIGMA
    mean_m, mean_l = run_pair("smoothing", f"mean {ksize}x{ksize}",
                              mean_blur_manual, (img, ksize),
                              mean_blur_opencv, (img, ksize), rows)
    median_m, median_l = run_pair("smoothing", f"median {ksize}x{ksize}",
                                  median_blur_manual, (img, ksize),
                                  median_blur_opencv, (img, ksize), rows)
    gauss_m, gauss_l = run_pair("smoothing", f"gaussian {ksize}x{ksize} sigma={sigma:g}",
                                gaussian_blur_manual, (img, ksize, sigma),
                                gaussian_blur_opencv, (img, ksize, sigma), rows)
    bilateral_m, bilateral_l = run_pair("smoothing", f"bilateral d={ksize}",
                                        bilateral_manual, (img, ksize, BILATERAL_SIGMA_COLOR,
                                                           BILATERAL_SIGMA_SPACE),
                                        bilateral_opencv, (img, ksize, BILATERAL_SIGMA_COLOR,
                                                           BILATERAL_SIGMA_SPACE), rows)
    save_fig(
        "01_smoothing.png",
        [(f"Mean {ksize}x{ksize}", mean_m, mean_l, ("correlate2d", "cv2.blur")),
         (f"Median {ksize}x{ksize}", median_m, median_l, ("np.median", "cv2.medianBlur")),
         (f"Gaussian {ksize}x{ksize}", gauss_m, gauss_l, ("correlate2d",
                                                          "cv2.GaussianBlur")),
         (f"Bilateral d={ksize}", bilateral_m, bilateral_l, ("range weights",
                                                            "cv2.bilateralFilter"))],
        pair_cols=2,
    )


def run_edge_feature(img, rows):
    """Feature 2: edge detection filters and Canny pipeline"""
    print("\n=== Feature 2: Edge detection (Manual vs OpenCV) ===")
    gray = to_gray(img)
    roberts_m, roberts_l = run_pair("edge", "roberts", roberts_manual, (gray,),
                                    roberts_opencv, (gray,), rows)
    prewitt_m, prewitt_l = run_pair("edge", "prewitt", prewitt_manual, (gray,),
                                    prewitt_opencv, (gray,), rows)
    sobel_m, sobel_l = run_pair("edge", "sobel", sobel_manual, (gray,),
                                sobel_opencv, (gray,), rows)
    scharr_m, scharr_l = run_pair("edge", "scharr", scharr_manual, (gray,),
                                  scharr_opencv, (gray,), rows)
    save_fig(
        "02_edge_gradients.png",
        [("Roberts", roberts_m, roberts_l, ("correlate2d", "cv2.filter2D")),
         ("Prewitt", prewitt_m, prewitt_l, ("correlate2d", "cv2.filter2D")),
         ("Sobel", sobel_m, sobel_l, ("correlate2d", "cv2.Sobel")),
         ("Scharr", scharr_m, scharr_l, ("correlate2d", "cv2.Scharr"))],
        pair_cols=2,
    )

    lap_resp_m, lap_resp_l = run_pair("edge", "laplacian response",
                                      laplacian_response_map_manual,
                                      (gray, SECOND_ORDER_DISPLAY_MAX),
                                      laplacian_response_map_opencv,
                                      (gray, SECOND_ORDER_DISPLAY_MAX), rows)
    log_resp_m, log_resp_l = run_pair("edge", f"LoG response {LOG_KSIZE}x{LOG_KSIZE} sigma={LOG_SIGMA:g}",
                                      log_response_map_manual,
                                      (gray, LOG_KSIZE, LOG_SIGMA, SECOND_ORDER_DISPLAY_MAX),
                                      log_response_map_opencv,
                                      (gray, LOG_KSIZE, LOG_SIGMA, SECOND_ORDER_DISPLAY_MAX), rows)
    dog_resp_m, dog_resp_l = run_pair("edge", "DoG response",
                                      dog_response_map_manual, (gray, DOG_KSIZE, DOG_SIGMA1,
                                                               DOG_SIGMA2,
                                                               SECOND_ORDER_DISPLAY_MAX),
                                      dog_response_map_opencv, (gray, DOG_KSIZE, DOG_SIGMA1,
                                                               DOG_SIGMA2,
                                                               SECOND_ORDER_DISPLAY_MAX), rows)
    save_grid_fig(
        "03_edge_second_order.png",
        [("",
          [(lap_resp_m, "Manual (NumPy)", "correlate2d"),
           (log_resp_m, "Manual (NumPy)", "Gaussian+Laplacian"),
           (dog_resp_m, "Manual (NumPy)", "Gaussian difference")]),
         ("",
          [(lap_resp_l, "Library (OpenCV)", "cv2.Laplacian"),
           (log_resp_l, "Library (OpenCV)", "GaussianBlur+Laplacian"),
           (dog_resp_l, "Library (OpenCV)", "GaussianBlur difference")])],
        ("Laplacian", "LoG", "DoG"),
    )

    canny_rows = []
    for low, high in CANNY_PARAM_SETS:
        name = f"canny low={low} high={high}"
        canny_m, canny_l = run_pair("edge", name, canny_manual, (gray, low, high),
                                    canny_opencv, (gray, low, high), rows,
                                    quality_extras=lambda manual, library: {
                                        "edge_agreement_percent": edge_agreement(manual, library)
                                    })
        print(
            f"Canny edge-map pixel agreement low={low} high={high}: "
            f"{edge_agreement(canny_m, canny_l):.2f}%"
        )
        canny_rows.append((f"Canny {low}/{high}", canny_m, canny_l,
                           ("manual pipeline", "cv2.Canny")))
    save_fig("04_edge_canny.png", canny_rows, pair_cols=2)


def run_all(path=None, csv_name="benchmark_all.csv"):
    """Run all Lab 02 features and write one combined benchmark CSV"""
    setup_output_dirs()
    img = load_image(path or DEFAULT_IMAGE)
    rows = []
    run_smoothing_feature(img, rows)
    run_edge_feature(img, rows)
    write_benchmark_csv(csv_path(csv_name), rows)
    print(f"\nDone. Figures saved to:\n  {RESULTS}\n  {DOCFIG}")


def run_smoothing(path=None, csv_name="benchmark_smoothing.csv"):
    """Run only smoothing features"""
    setup_output_dirs()
    img = load_image(path or DEFAULT_IMAGE)
    rows = []
    run_smoothing_feature(img, rows)
    write_benchmark_csv(csv_path(csv_name), rows)


def run_edge(path=None, csv_name="benchmark_edge_detection.csv"):
    """Run only edge detection features"""
    setup_output_dirs()
    img = load_image(path or DEFAULT_IMAGE)
    rows = []
    run_edge_feature(img, rows)
    write_benchmark_csv(csv_path(csv_name), rows)


def main():
    """CLI entry point: python main.py [--feature ...] [path_to_image]"""
    args = parse_args()
    if args.feature == "all":
        run_all(args.image)
    elif args.feature == "smoothing":
        run_smoothing(args.image)
    elif args.feature == "edge":
        run_edge(args.image)


if __name__ == "__main__":
    main()
