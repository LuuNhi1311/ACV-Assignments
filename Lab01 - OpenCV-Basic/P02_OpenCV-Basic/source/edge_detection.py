"""Edge detection algorithms for Lab 02: manual NumPy implementations and OpenCV references"""

import cv2
import numpy as np

from ops import correlate2d, pad_border
from utils import gaussian_kernel, to_uint8

ROBERTS_X = np.array([[1, 0], [0, -1]], dtype=np.float64)
ROBERTS_Y = np.array([[0, 1], [-1, 0]], dtype=np.float64)
PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
PREWITT_Y = PREWITT_X.T
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = SOBEL_X.T
SCHARR_X = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float64)
SCHARR_Y = SCHARR_X.T
LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)


def _grad_magnitude(gray: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    gx = correlate2d(gray, kx, padding="same", border="reflect")
    gy = correlate2d(gray, ky, padding="same", border="reflect")
    # Edge strength |grad I| = sqrt(Gx^2 + Gy^2)
    return np.sqrt(gx * gx + gy * gy)


def _grad_magnitude_cv(gray: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float64)
    gx = cv2.filter2D(g, cv2.CV_64F, kx)
    gy = cv2.filter2D(g, cv2.CV_64F, ky)
    return np.sqrt(gx * gx + gy * gy)


def roberts_manual(gray: np.ndarray) -> np.ndarray:
    """Roberts gradient magnitude from two 2x2 masks"""
    return to_uint8(_grad_magnitude(gray, ROBERTS_X, ROBERTS_Y))


def roberts_opencv(gray: np.ndarray) -> np.ndarray:
    """Roberts reference using cv2.filter2D with the same masks"""
    return to_uint8(_grad_magnitude_cv(gray, ROBERTS_X, ROBERTS_Y))


def prewitt_manual(gray: np.ndarray) -> np.ndarray:
    """Prewitt gradient magnitude from horizontal and vertical masks"""
    return to_uint8(_grad_magnitude(gray, PREWITT_X, PREWITT_Y))


def prewitt_opencv(gray: np.ndarray) -> np.ndarray:
    """Prewitt reference using cv2.filter2D with the same masks"""
    return to_uint8(_grad_magnitude_cv(gray, PREWITT_X, PREWITT_Y))


def sobel_manual(gray: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude from horizontal and vertical masks"""
    return to_uint8(_grad_magnitude(gray, SOBEL_X, SOBEL_Y))


def sobel_opencv(gray: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude using OpenCV"""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return to_uint8(np.sqrt(gx * gx + gy * gy))


def scharr_manual(gray: np.ndarray) -> np.ndarray:
    """Scharr gradient magnitude from horizontal and vertical masks"""
    return to_uint8(_grad_magnitude(gray, SCHARR_X, SCHARR_Y))


def scharr_opencv(gray: np.ndarray) -> np.ndarray:
    """Scharr gradient magnitude using OpenCV"""
    gx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    return to_uint8(np.sqrt(gx * gx + gy * gy))


def response_to_display(response: np.ndarray, display_max: float | None = None) -> np.ndarray:
    """Map an edge response magnitude to [0, 255] for visualization"""
    response_abs = np.abs(response.astype(np.float64))
    max_value = np.max(response_abs) if display_max is None else display_max
    if max_value == 0:
        return np.zeros_like(response_abs, dtype=np.uint8)
    return to_uint8(response_abs * 255.0 / max_value)


def laplacian_response_manual(gray: np.ndarray) -> np.ndarray:
    """Signed Laplacian response using a 4-neighbour kernel"""
    return correlate2d(gray, LAPLACIAN, padding="same", border="reflect")


def laplacian_response_opencv(gray: np.ndarray) -> np.ndarray:
    """Signed Laplacian response using OpenCV"""
    return cv2.Laplacian(gray, cv2.CV_64F, ksize=1)


def laplacian_response_map_manual(gray: np.ndarray, display_max: float | None = None) -> np.ndarray:
    """Display-ready Laplacian response magnitude"""
    return response_to_display(laplacian_response_manual(gray), display_max)


def laplacian_response_map_opencv(gray: np.ndarray, display_max: float | None = None) -> np.ndarray:
    """Display-ready OpenCV Laplacian response magnitude"""
    return response_to_display(laplacian_response_opencv(gray), display_max)


def log_response_manual(gray: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Signed Laplacian of Gaussian response"""
    smooth = correlate2d(gray, gaussian_kernel(ksize, sigma), padding="same", border="reflect")
    return correlate2d(smooth, LAPLACIAN, padding="same", border="reflect")


def log_response_opencv(gray: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Signed Laplacian of Gaussian response using OpenCV"""
    smooth = cv2.GaussianBlur(gray.astype(np.float64), (ksize, ksize), sigmaX=sigma)
    return cv2.Laplacian(smooth, cv2.CV_64F, ksize=1)


def log_response_map_manual(gray: np.ndarray, ksize: int = 5, sigma: float = 1.0,
                            display_max: float | None = None) -> np.ndarray:
    """Display-ready LoG response magnitude"""
    return response_to_display(log_response_manual(gray, ksize, sigma), display_max)


def log_response_map_opencv(gray: np.ndarray, ksize: int = 5, sigma: float = 1.0,
                            display_max: float | None = None) -> np.ndarray:
    """Display-ready OpenCV LoG response magnitude"""
    return response_to_display(log_response_opencv(gray, ksize, sigma), display_max)


def dog_response_manual(gray: np.ndarray, ksize: int = 5, sigma1: float = 1.0,
                        sigma2: float = 2.0) -> np.ndarray:
    """Signed Difference of Gaussian response"""
    g1 = correlate2d(gray, gaussian_kernel(ksize, sigma1), padding="same", border="reflect")
    g2 = correlate2d(gray, gaussian_kernel(ksize, sigma2), padding="same", border="reflect")
    # Difference of two Gaussian blurs approximates LoG response shape
    return g1 - g2


def dog_response_opencv(gray: np.ndarray, ksize: int = 5, sigma1: float = 1.0,
                        sigma2: float = 2.0) -> np.ndarray:
    """Signed Difference of Gaussian response using OpenCV"""
    g = gray.astype(np.float64)
    g1 = cv2.GaussianBlur(g, (ksize, ksize), sigmaX=sigma1)
    g2 = cv2.GaussianBlur(g, (ksize, ksize), sigmaX=sigma2)
    return g1 - g2


def dog_response_map_manual(gray: np.ndarray, ksize: int = 5, sigma1: float = 1.0,
                            sigma2: float = 2.0,
                            display_max: float | None = None) -> np.ndarray:
    """Display-ready DoG response magnitude"""
    return response_to_display(dog_response_manual(gray, ksize, sigma1, sigma2), display_max)


def dog_response_map_opencv(gray: np.ndarray, ksize: int = 5, sigma1: float = 1.0,
                            sigma2: float = 2.0,
                            display_max: float | None = None) -> np.ndarray:
    """Display-ready OpenCV DoG response magnitude"""
    return response_to_display(dog_response_opencv(gray, ksize, sigma1, sigma2), display_max)


def _gradient_direction(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Return gradient direction in degrees in [0, 180)"""
    return (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0


def _quantize_direction(theta: np.ndarray) -> np.ndarray:
    """Quantize gradient direction into 0, 45, 90, and 135 degrees"""
    direction = np.zeros(theta.shape, dtype=np.uint8)
    direction[((22.5 <= theta) & (theta < 67.5))] = 45
    direction[((67.5 <= theta) & (theta < 112.5))] = 90
    direction[((112.5 <= theta) & (theta < 157.5))] = 135
    return direction


def _non_max_suppression(mag: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Keep only local maxima along the quantized gradient direction"""
    padded = pad_border(mag, (1, 1), (1, 1), border="replicate")
    center = padded[1:-1, 1:-1]
    east, west = padded[1:-1, 2:], padded[1:-1, 0:-2]
    north, south = padded[0:-2, 1:-1], padded[2:, 1:-1]
    north_east, north_west = padded[0:-2, 2:], padded[0:-2, 0:-2]
    south_east, south_west = padded[2:, 2:], padded[2:, 0:-2]

    keep_0 = (direction == 0) & (center >= east) & (center >= west)
    keep_45 = (direction == 45) & (center >= north_west) & (center >= south_east)
    keep_90 = (direction == 90) & (center >= north) & (center >= south)
    keep_135 = (direction == 135) & (center >= north_east) & (center >= south_west)
    keep = keep_0 | keep_45 | keep_90 | keep_135
    return np.where(keep, center, 0.0)


def _hysteresis(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    edges = strong.copy()
    h, w = edges.shape
    while True:
        padded = pad_border(edges.astype(np.uint8), (1, 1), (1, 1), border="replicate")
        grown = np.zeros((h, w), dtype=bool)
        for di in range(3):
            for dj in range(3):
                # Dilate the edge set by one pixel using 8-connectivity
                grown |= padded[di:di + h, dj:dj + w].astype(bool)
        # Promote weak pixels that touch an existing edge
        new_edges = grown & weak & (~edges)
        if not new_edges.any():
            break
        edges |= new_edges
    return edges


def canny_manual(gray: np.ndarray, low: int = 50, high: int = 150, ksize: int = 5, sigma: float = 1.4) -> np.ndarray:
    """Canny pipeline: smooth, gradients, NMS, double threshold, hysteresis"""
    smooth = correlate2d(gray, gaussian_kernel(ksize, sigma), padding="same", border="reflect")
    gx = correlate2d(smooth, SOBEL_X, padding="same", border="reflect")
    gy = correlate2d(smooth, SOBEL_Y, padding="same", border="reflect")
    mag = np.sqrt(gx * gx + gy * gy)
    theta = _gradient_direction(gx, gy)
    direction = _quantize_direction(theta)
    thin = _non_max_suppression(mag, direction)
    strong = thin >= high
    weak = (thin >= low) & (thin < high)
    edges = _hysteresis(strong, weak)
    return (edges * 255).astype(np.uint8)


def canny_opencv(gray: np.ndarray, low: int = 50, high: int = 150, ksize: int = 5, sigma: float = 1.4) -> np.ndarray:
    """Canny edge detector using OpenCV"""
    smooth = cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    return cv2.Canny(smooth, low, high, L2gradient=True)


def edge_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Percentage of pixels with the same edge/non-edge label"""
    return float(np.mean((a > 0) == (b > 0)) * 100.0)