"""Shared helpers: manual-vs-library metrics, benchmarking, and image-grid display.
Convention: images are uint8 numpy arrays in [0,255], color in RGB channel order"""

import os
import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def ensure_dir(path: str) -> None:
    """Create the directory (and parents) if it does not exist"""
    os.makedirs(path, exist_ok=True)


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """Clip to [0,255], round and cast to uint8 (saturating cast)"""
    return np.clip(np.round(arr), 0, 255).astype(np.uint8)


def mae(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Absolute Error between two equally sized images"""
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio in dB; higher means the images are more alike"""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def compare_report(name: str, manual: np.ndarray, library: np.ndarray) -> dict:
    """Print and return MAE/PSNR between the manual and library result of one feature"""
    m, p = mae(manual, library), psnr(manual, library)
    print(f"[COMPARE] {name:<28s} | MAE = {m:8.4f} | PSNR = {p:7.2f} dB")
    return {"name": name, "mae": m, "psnr": p}


def measure_run(fn, *args, **kwargs):
    """Run one operation and measure elapsed time plus peak Python-traced memory.

    Note: tracemalloc tracks Python allocations, so native OpenCV allocations may be
    under-reported.
    """
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


def show_grid(images, titles, suptitle=None, save_path=None, cols=3, cmap=None):
    """Tile several images into a grid and optionally save it; 2D images use a gray cmap"""
    n = len(images)
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(4 * cols, 4 * rows))
    for i, (img, title) in enumerate(zip(images, titles)):
        ax = plt.subplot(rows, cols, i + 1)
        title_text, bottom_text = title if isinstance(title, tuple) else (title, None)
        this_cmap = "gray" if (img.ndim == 2 and cmap is None) else cmap
        display_img = to_uint8(img) if img.ndim == 3 else img
        ax.imshow(display_img, cmap=this_cmap, vmin=0, vmax=255)
        ax.set_title(title_text, fontsize=11)
        if bottom_text:
            ax.set_xlabel(bottom_text, fontsize=10, fontfamily="monospace")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    plt.tight_layout()
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"[SAVED] {save_path}")
    plt.close()


def show_compare_grid(rows, suptitle=None, save_path=None, pair_cols=None,
                      vertical_separator_rows=None,
                      col_titles=("Manual (NumPy)", "Library (OpenCV)")):
    """Plot operation pairs in a compact report-friendly grid.

    rows: list of (row_label, manual_img, library_img) or
          (row_label, manual_img, library_img, (manual_fn, library_fn))
    where the optional function names are printed under each image"""
    n = len(rows)
    if pair_cols is None:
        grid_rows = 1 if n <= 2 else 2
        pair_cols = int(np.ceil(n / grid_rows))
    else:
        pair_cols = min(pair_cols, n)
        grid_rows = int(np.ceil(n / pair_cols))
    grid_cols = pair_cols * 2

    fig_w = max(7.6, 3.0 * grid_cols)
    fig_h = 3.3 * grid_rows
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
            ax.set_title(label, fontsize=10, fontweight="bold", color="black")
            ax.set_xlabel(col_titles[j], fontsize=8, fontfamily="monospace",
                          color=method_colors[j], fontweight="bold")
    fig.tight_layout()

    # Add separators in the whitespace between feature groups. Lines are placed
    # between axes, so they do not cover the images themselves.
    used_rows = min(grid_rows, int(np.ceil(n / pair_cols)))
    used_positions = [ax.get_position() for ax in axes[:used_rows].ravel()]
    grid_y0 = min(pos.y0 for pos in used_positions)
    grid_y1 = max(pos.y1 for pos in used_positions)
    if vertical_separator_rows is None:
        for pair_c in range(1, pair_cols):
            left_pos = axes[0, pair_c * 2 - 1].get_position()
            right_pos = axes[0, pair_c * 2].get_position()
            x = (left_pos.x1 + right_pos.x0) / 2.0
            fig.add_artist(Line2D([x, x], [grid_y0 - 0.035, grid_y1 + 0.035],
                                  transform=fig.transFigure,
                                  linewidth=1.8, color="0.6", zorder=10))
    else:
        for r in vertical_separator_rows:
            if r >= used_rows:
                continue
            pairs_in_row = min(pair_cols, n - r * pair_cols)
            row_pos = [ax.get_position() for ax in axes[r, :pairs_in_row * 2]]
            y0 = min(pos.y0 for pos in row_pos) - 0.035
            y1 = max(pos.y1 for pos in row_pos) + 0.035
            for pair_c in range(1, pairs_in_row):
                left_pos = axes[r, pair_c * 2 - 1].get_position()
                right_pos = axes[r, pair_c * 2].get_position()
                x = (left_pos.x1 + right_pos.x0) / 2.0
                fig.add_artist(Line2D([x, x], [y0, y1], transform=fig.transFigure,
                                      linewidth=1.8, color="0.6", zorder=10))
    for r in range(1, used_rows):
        prev_row_pos = [ax.get_position() for ax in axes[r - 1]]
        next_row_pos = [ax.get_position() for ax in axes[r]]
        y = (min(pos.y0 for pos in prev_row_pos) + max(pos.y1 for pos in next_row_pos)) / 2.0
        x0 = min(pos.x0 for pos in prev_row_pos + next_row_pos)
        x1 = max(pos.x1 for pos in prev_row_pos + next_row_pos)
        fig.add_artist(Line2D([x0, x1], [y, y], transform=fig.transFigure,
                              linewidth=1.8, color="0.6", zorder=10))

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"[SAVED] {save_path}")
    plt.close(fig)
