"""Smoothing algorithms for Lab 02: manual NumPy implementations and OpenCV references"""

import cv2
import numpy as np

from ops import correlate2d, pad_border
from utils import gaussian_kernel, to_uint8


def mean_kernel(ksize: int) -> np.ndarray:
    """Return a normalized ksize x ksize mean kernel"""
    # Each tap = 1/k^2 so the kernel sums to 1
    return np.ones((ksize, ksize), dtype=np.float64) / (ksize * ksize)


def mean_blur_manual(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Mean blur implemented with the shared correlation operator"""
    return to_uint8(correlate2d(img, mean_kernel(ksize), padding="same", border="reflect"))


def mean_blur_opencv(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Mean blur using OpenCV"""
    return cv2.blur(img, (ksize, ksize))


def gaussian_blur_manual(img: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Gaussian blur implemented with a generated Gaussian kernel"""
    kernel = gaussian_kernel(ksize, sigma)
    return to_uint8(correlate2d(img, kernel, padding="same", border="reflect"))


def gaussian_blur_opencv(img: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Gaussian blur using OpenCV"""
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def median_blur_manual(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Median blur implemented by stacking shifted neighbourhood windows"""
    py = px = ksize // 2
    # Replicate padding matches cv2.medianBlur borders
    padded = pad_border(img, (py, py), (px, px), border="replicate")
    h, w = img.shape[:2]
    # Collect the k*k shifted neighbourhood windows
    windows = [padded[i:i + h, j:j + w] for i in range(ksize) for j in range(ksize)]
    # Per-pixel median over stacked windows
    return np.median(np.stack(windows, axis=0), axis=0).astype(np.uint8)


def median_blur_opencv(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Median blur using OpenCV"""
    return cv2.medianBlur(img, ksize)


def bilateral_manual(img: np.ndarray, d: int = 5, sigma_color: float = 50.0, sigma_space: float = 5.0) -> np.ndarray:
    """Bilateral filter using spatial and range Gaussian weights"""
    radius = d // 2
    f = img.astype(np.float64)
    padded = pad_border(f, (radius, radius), (radius, radius), border="reflect")
    h, w = f.shape[:2]
    out = np.zeros_like(f)
    weight_sum = np.zeros((h, w), dtype=np.float64)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = padded[dy + radius:dy + radius + h, dx + radius:dx + radius + w]
            # Spatial weight = exp(-(dx^2+dy^2) / 2 sigma_space^2)
            spatial = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma_space ** 2))
            diff = shifted - f
            # Range distance is summed across RGB channels
            dist2 = np.sum(diff * diff, axis=2) if f.ndim == 3 else diff * diff
            weight = spatial * np.exp(-dist2 / (2.0 * sigma_color ** 2))
            expanded = weight[..., None] if f.ndim == 3 else weight
            out += expanded * shifted
            weight_sum += weight

    # Normalize by total weight per pixel
    divisor = weight_sum[..., None] if f.ndim == 3 else weight_sum
    return to_uint8(out / divisor)


def bilateral_opencv(img: np.ndarray, d: int = 5, sigma_color: float = 50.0,
                     sigma_space: float = 5.0) -> np.ndarray:
    """Bilateral filter using OpenCV"""
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)