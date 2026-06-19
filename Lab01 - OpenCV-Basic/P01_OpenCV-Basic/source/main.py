"""Run assignment features, save report figures, and write benchmark CSV files."""

import argparse
import csv
import os

import cv2

from image_io import imread_opencv
from scale import (resize_nearest_manual, resize_bilinear_manual, resize_opencv,
                   rotate_manual, rotate90_manual, rotate90_opencv, rotate_opencv)
from transform import (brightness_manual, brightness_opencv, contrast_manual, contrast_opencv,
                       grayscale_manual, grayscale_opencv, invert_manual, invert_opencv,
                       gamma_manual, gamma_opencv, saturation_manual, saturation_opencv,
                       hue_manual, hue_opencv)
from utils import (benchmark_report, compare_report, ensure_dir, measure_run,
                   show_compare_grid, show_grid)

SRC = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(SRC, "results")
DOCFIG = os.path.normpath(os.path.join(SRC, "..", "doc", "figures"))
DEFAULT_IMAGE = os.path.join(SRC, "images", "lena.jpg")

CSV_COLUMNS = (
    "feature",
    "operation",
    "implementation",
    "time_ms",
    "peak_kib",
    "mae_vs_opencv",
    "psnr_vs_opencv",
)


def output_path(name: str) -> str:
    """Return a path under source/results."""
    return os.path.join(RESULTS, name)


def csv_path(name: str) -> str:
    """Return a benchmark CSV path under source/results."""
    return output_path(name)


def setup_output_dirs() -> None:
    """Create output folders for generated figures and benchmark CSV files."""
    ensure_dir(RESULTS)
    ensure_dir(DOCFIG)


def parse_args():
    """Parse command-line options for the assignment runner."""
    parser = argparse.ArgumentParser(
        description="Run NumPy-vs-OpenCV assignment features and write figures/benchmark CSV files."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=DEFAULT_IMAGE,
        help="Input image path. Defaults to source/images/lena.jpg.",
    )
    parser.add_argument(
        "-f",
        "--feature",
        choices=("all", "io", "scale", "transform"),
        default="all",
        help="Feature group to run. Default: all.",
    )
    return parser.parse_args()


def load_image(path: str):
    """Read an image once with OpenCV; keep the original uint8 image data."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    print(f"Loading: {path}")
    img = imread_opencv(path)
    print("Image shape:", img.shape, "| dtype:", img.dtype)
    return img


def save_fig(name, rows, pair_cols=None, vertical_separator_rows=None):
    """Render a Manual-vs-OpenCV comparison grid into results/ and doc/figures/."""
    for p in (os.path.join(RESULTS, name), os.path.join(DOCFIG, name)):
        show_compare_grid(rows, save_path=p, pair_cols=pair_cols,
                          vertical_separator_rows=vertical_separator_rows)


def save_single_fig(name, images, titles):
    """Render a one-column figure into results/ and doc/figures/."""
    for p in (os.path.join(RESULTS, name), os.path.join(DOCFIG, name)):
        show_grid(images, titles, save_path=p, cols=1)


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
    })


def write_benchmark_csv(path: str, rows) -> None:
    """Write benchmark rows to CSV."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SAVED] {path}")


def run_pair(feature, name, manual_fn, manual_args, library_fn, library_args, rows,
             benchmark=False):
    """Run manual and OpenCV versions, optionally record benchmark CSV rows."""
    if benchmark:
        manual, manual_stats = measure_run(manual_fn, *manual_args)
        library, library_stats = measure_run(library_fn, *library_args)
    else:
        manual, library = manual_fn(*manual_args), library_fn(*library_args)
        manual_stats, library_stats = None, None
    quality = compare_report(name, manual, library)
    if benchmark:
        benchmark_report(name, manual_stats, library_stats)
        _append_row(rows, feature, name, "manual_numpy", manual_stats, quality)
        _append_row(rows, feature, name, "opencv", library_stats, quality)
    return manual, library


def run_io_feature(img):
    """Feature 1: read once with OpenCV and save a display figure."""
    print("\n=== Feature 1: Read & Display ===")
    print(f"[READ]    cv2.imread + RGB convert | shape = {img.shape} | dtype = {img.dtype}")
    height, width = img.shape[:2]
    channels = 1 if img.ndim == 2 else img.shape[2]
    info = f"Width: {width} px | Height: {height} px | Channels: {channels}"
    save_single_fig("01_io.png", [img], [("cv2.imread", info)])


def run_resize_feature(img, rows):
    """Feature 2a: nearest and bilinear resize."""
    print("\n=== Feature 2a: Resize ===")
    nw, nh = img.shape[1] // 3, img.shape[0] // 3
    near_m, near_l = run_pair("scale", "resize nearest", resize_nearest_manual, (img, nw, nh),
                              resize_opencv, (img, nw, nh, cv2.INTER_NEAREST_EXACT), rows,
                              benchmark=True)
    bil_m, bil_l = run_pair("scale", "resize bilinear", resize_bilinear_manual, (img, nw, nh),
                            resize_opencv, (img, nw, nh, cv2.INTER_LINEAR_EXACT), rows,
                            benchmark=True)
    save_fig("02_resize.png",
             [("Nearest", near_m, near_l),
              ("Bilinear", bil_m, bil_l)])


def run_rotate_feature(img, rows):
    """Feature 2b: arbitrary-angle and 90-degree rotation."""
    print("\n=== Feature 2b: Rotate ===")
    r30_m, r30_l = run_pair("scale", "rotate 30 (no expand)", rotate_manual, (img, 30, False),
                            rotate_opencv, (img, 30, False), rows,
                            benchmark=True)
    r90_m, r90_l = run_pair("scale", "rotate 90", rotate90_manual, (img, 1),
                            rotate90_opencv, (img, 1), rows,
                            benchmark=True)
    save_fig("03_rotate.png",
             [("Rotate 30 deg", r30_m, r30_l),
              ("Rotate 90 deg", r90_m, r90_l)])


def run_brightness_contrast_feature(img, rows):
    """Feature 3a: brightness and contrast."""
    print("\n=== Feature 3a: Brightness & Contrast ===")
    b_m, b_l = run_pair("transform", "brightness +60", brightness_manual, (img, 60),
                        brightness_opencv, (img, 60), rows,
                        benchmark=True)
    d_m, d_l = run_pair("transform", "brightness -60", brightness_manual, (img, -60),
                        brightness_opencv, (img, -60), rows)
    c_m, c_l = run_pair("transform", "contrast x1.6", contrast_manual, (img, 1.6),
                        contrast_opencv, (img, 1.6), rows,
                        benchmark=True)
    lo_m, lo_l = run_pair("transform", "contrast x0.5", contrast_manual, (img, 0.5),
                          contrast_opencv, (img, 0.5), rows)
    save_fig("04_brightness_contrast.png",
             [("Brightness +60", b_m, b_l), ("Brightness -60", d_m, d_l),
              ("Contrast x1.6", c_m, c_l), ("Contrast x0.5", lo_m, lo_l)])


def run_color_feature(img, rows):
    """Feature 3b: color transforms."""
    print("\n=== Feature 3b: Color ===")
    g_m, g_l = run_pair("transform", "grayscale", grayscale_manual, (img,),
                        grayscale_opencv, (img,), rows,
                        benchmark=True)
    sat_hi_m, sat_hi_l = run_pair("transform", "saturation x1.6", saturation_manual, (img, 1.6),
                                  saturation_opencv, (img, 1.6), rows,
                                  benchmark=True)
    sat_lo_m, sat_lo_l = run_pair("transform", "saturation x0.5", saturation_manual, (img, 0.5),
                                  saturation_opencv, (img, 0.5), rows)
    hue_pos_m, hue_pos_l = run_pair("transform", "hue +30 deg", hue_manual, (img, 30),
                                    hue_opencv, (img, 30), rows,
                                    benchmark=True)
    hue_neg_m, hue_neg_l = run_pair("transform", "hue -30 deg", hue_manual, (img, -30),
                                    hue_opencv, (img, -30), rows)
    inv_m, inv_l = run_pair("transform", "invert", invert_manual, (img,),
                            invert_opencv, (img,), rows,
                            benchmark=True)
    gamma_lo_m, gamma_lo_l = run_pair("transform", "gamma 0.5", gamma_manual, (img, 0.5),
                                      gamma_opencv, (img, 0.5), rows,
                                      benchmark=True)
    gamma_hi_m, gamma_hi_l = run_pair("transform", "gamma 2.0", gamma_manual, (img, 2.0),
                                      gamma_opencv, (img, 2.0), rows)
    save_fig("05_color.png",
             [("Grayscale", g_m, g_l),
              ("Negative", inv_m, inv_l),
              ("Gamma 0.5", gamma_lo_m, gamma_lo_l),
              ("Gamma 2.0", gamma_hi_m, gamma_hi_l),
              ("Saturation x1.6", sat_hi_m, sat_hi_l),
              ("Saturation x0.5", sat_lo_m, sat_lo_l),
              ("Hue +30 deg", hue_pos_m, hue_pos_l),
              ("Hue -30 deg", hue_neg_m, hue_neg_l)],
             pair_cols=2,
             vertical_separator_rows=[0])


def run_all(path=None, csv_name="benchmark_all.csv"):
    """Run all features and write one combined benchmark CSV."""
    setup_output_dirs()
    img = load_image(path or DEFAULT_IMAGE)
    rows = []
    run_io_feature(img)
    run_resize_feature(img, rows)
    run_rotate_feature(img, rows)
    run_brightness_contrast_feature(img, rows)
    run_color_feature(img, rows)
    write_benchmark_csv(csv_path(csv_name), rows)
    print(f"\nDone. Figures saved to:\n  {RESULTS}\n  {DOCFIG}")


def run_io(path=None):
    """Run only the read/show feature; no benchmark CSV is written for direct OpenCV I/O."""
    setup_output_dirs()
    img = load_image(path or DEFAULT_IMAGE)
    run_io_feature(img)


def run_scale(path=None, csv_name="benchmark_scale.csv"):
    """Run only scale features: resize and rotate."""
    setup_output_dirs()
    img = load_image(path or DEFAULT_IMAGE)
    rows = []
    run_resize_feature(img, rows)
    run_rotate_feature(img, rows)
    write_benchmark_csv(csv_path(csv_name), rows)


def run_transform(path=None, csv_name="benchmark_transform.csv"):
    """Run only transform features: brightness/contrast and color."""
    setup_output_dirs()
    img = load_image(path or DEFAULT_IMAGE)
    rows = []
    run_brightness_contrast_feature(img, rows)
    run_color_feature(img, rows)
    write_benchmark_csv(csv_path(csv_name), rows)


def main():
    """CLI entry point: python main.py [--feature ...] [path_to_image]."""
    args = parse_args()
    if args.feature == "all":
        run_all(args.image)
    elif args.feature == "io":
        run_io(args.image)
    elif args.feature == "scale":
        run_scale(args.image)
    elif args.feature == "transform":
        run_transform(args.image)


if __name__ == "__main__":
    main()
