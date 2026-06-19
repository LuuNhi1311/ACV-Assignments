"""Shared helpers for Lab 02: image I/O, metrics, benchmarking, and plotting
Convention: images are uint8 NumPy arrays in [0, 255], color in RGB channel order"""

import os
import time
import tracemalloc

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")
DOCFIG = os.path.normpath(os.path.join(HERE, "..", "doc", "figures"))
DEFAULT_IMAGE = os.path.join(HERE, "images", "lena.jpg")


def ensure_dir(path: str) -> None:
    """Create the directory and parents if they do not exist"""
    os.makedirs(path, exist_ok=True)


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """Clip to [0, 255], round and cast to uint8"""
    return np.clip(np.round(arr), 0, 255).astype(np.uint8)


def imread_pillow(path: str) -> np.ndarray:
    """Read an image as RGB uint8 with Pillow"""
    return np.array(Image.open(path).convert("RGB"))


def to_gray(img_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB image to grayscale with BT.601 luminance weights"""
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    # BT.601 luminance Y = 0.299R + 0.587G + 0.114B
    return to_uint8(0.299 * r + 0.587 * g + 0.114 * b)


def mae(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Absolute Error between two equally sized images"""
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio in dB"""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def compare_report(name: str, manual: np.ndarray, library: np.ndarray) -> dict:
    """Print and return MAE/PSNR between the manual and OpenCV result"""
    m, p = mae(manual, library), psnr(manual, library)
    print(f"[COMPARE] {name:<28s} | MAE = {m:8.4f} | PSNR = {p:7.2f} dB")
    return {"name": name, "mae": m, "psnr": p}


def measure_run(fn, *args, **kwargs):
    """Run one function and measure elapsed time plus Python-traced peak memory"""
    tracemalloc.start()
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    stats = {
        "time_ms": elapsed_ms,
        "peak_kib": peak / 1024.0,
    }
    return result, stats


def benchmark_report(name: str, manual: dict, library: dict) -> None:
    """Print timing and memory comparison for a manual/OpenCV operation pair"""
    print(
        f"[BENCH]   {name:<28s} | "
        f"time ms M/L = {manual['time_ms']:8.3f}/{library['time_ms']:8.3f} | "
        f"peak KiB M/L = {manual['peak_kib']:8.1f}/{library['peak_kib']:8.1f}"
    )


def gaussian_kernel(ksize: int, sigma: float) -> np.ndarray:
    """Build a normalized 2D Gaussian kernel from a separable 1D Gaussian"""
    center = (ksize - 1) * 0.5
    x = np.arange(ksize, dtype=np.float64) - center
    # 1D Gaussian g(x) = exp(-x^2 / 2 sigma^2), normalized to sum 1
    k1d = np.exp(-0.5 * (x / sigma) ** 2)
    k1d /= k1d.sum()
    # Separable 2D Gaussian kernel = outer product k k^T
    return np.outer(k1d, k1d)


def show_compare_grid(rows, suptitle=None, save_path=None, pair_cols=None,
                      vertical_separator_rows=None,
                      col_titles=("Manual (NumPy)", "Library (OpenCV)")):
    """Plot operation pairs in a compact report-friendly grid"""
    n = len(rows)
    if pair_cols is None:
        grid_rows = n
        pair_cols = 1
    else:
        pair_cols = min(pair_cols, n)
        grid_rows = int(np.ceil(n / pair_cols))
    grid_cols = pair_cols * 2

    fig_w = max(7.6, 3.0 * grid_cols)
    fig_h = 2.7 * grid_rows
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(fig_w, fig_h), squeeze=False)
    method_colors = ("tab:blue", "tab:red")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_edgecolor("0.55")

    for i, row in enumerate(rows):
        label, manual, library = row[0], row[1], row[2]
        captions = row[3] if len(row) > 3 else (None, None)
        grid_r = i // pair_cols
        pair_c = i % pair_cols
        base_c = pair_c * 2

        for j, (img, cap) in enumerate(((manual, captions[0]), (library, captions[1]))):
            ax = axes[grid_r, base_c + j]
            display_img = to_uint8(img) if img.ndim == 3 else img
            ax.imshow(display_img, cmap=("gray" if img.ndim == 2 else None), vmin=0, vmax=255)
            ax.set_title(label, fontsize=10, fontweight="bold")
            ax.set_xlabel(col_titles[j], fontsize=8, fontfamily="monospace",
                          color=method_colors[j], fontweight="bold")

    for i in range(n, grid_rows * pair_cols):
        base_c = (i % pair_cols) * 2
        grid_r = i // pair_cols
        axes[grid_r, base_c].axis("off")
        axes[grid_r, base_c + 1].axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    fig.tight_layout()

    used_rows = min(grid_rows, int(np.ceil(n / pair_cols)))
    if vertical_separator_rows is None:
        vertical_separator_rows = range(used_rows)
    for r in vertical_separator_rows:
        if r >= used_rows:
            continue
        pairs_in_row = min(pair_cols, n - r * pair_cols)
        for pair_c in range(1, pairs_in_row):
            left_pos = axes[r, pair_c * 2 - 1].get_position()
            right_pos = axes[r, pair_c * 2].get_position()
            x = (left_pos.x1 + right_pos.x0) / 2.0
            y0 = min(left_pos.y0, right_pos.y0) - 0.03
            y1 = max(left_pos.y1, right_pos.y1) + 0.03
            fig.add_artist(plt.Line2D([x, x], [y0, y1], transform=fig.transFigure,
                                      linewidth=1.4, color="0.6", zorder=10))

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"[SAVED] {save_path}")
    plt.close(fig)


def show_image_grid(rows, col_titles, save_path=None):
    """Plot a generic image grid with one algorithm per row"""
    nrows = len(rows)
    ncols = len(col_titles)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.2 * nrows),
                             squeeze=False)

    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=11, fontweight="bold")

    for r, (row_label, cells) in enumerate(rows):
        for c, cell in enumerate(cells):
            if len(cell) == 2:
                img, caption = cell
                method_label = row_label
            else:
                img, method_label, caption = cell
            ax = axes[r, c]
            display_img = to_uint8(img) if img.ndim == 3 else img
            ax.imshow(display_img, cmap=("gray" if img.ndim == 2 else None), vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(row_label, fontsize=11, fontweight="bold", color="black")
            method_color = "black"
            if method_label.lower().startswith("manual"):
                method_color = "tab:blue"
            elif method_label.lower().startswith(("opencv", "library")):
                method_color = "tab:red"
            ax.set_xlabel(method_label, fontsize=8, fontfamily="monospace",
                          color=method_color, fontweight="bold")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
                spine.set_edgecolor("0.55")

    fig.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"[SAVED] {save_path}")
    plt.close(fig)
