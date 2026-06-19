"""Feature 2 - Scaling: resize and rotate.
Manual: nearest/bilinear interpolation and inverse-mapping rotation with numpy.
Library: cv2.resize, cv2.getRotationMatrix2D + cv2.warpAffine"""

import math

import numpy as np
import cv2
from utils import to_uint8


def resize_nearest_manual(img: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """Resize by mapping each output pixel back to the nearest source pixel.

    OpenCV comparison: cv2.resize(..., interpolation=cv2.INTER_NEAREST_EXACT).
    """
    h, w = img.shape[:2]
    scale_x, scale_y = w / new_w, h / new_h

    # Build one source-coordinate lookup table for output columns and rows
    src_x = np.clip(np.floor((np.arange(new_w, dtype=np.float32) + 0.5) * scale_x).astype(int), 0, w - 1)
    src_y = np.clip(np.floor((np.arange(new_h, dtype=np.float32) + 0.5) * scale_y).astype(int), 0, h - 1)
    return img[src_y[:, None], src_x[None, :]]


def resize_bilinear_manual(img: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """Resize by bilinear interpolation from the four nearest source pixels.

    For each output pixel, inverse mapping finds a floating-point source coordinate. 
    The fractional part becomes the interpolation weight.
    """
    h, w = img.shape[:2]
    scale_x, scale_y = w / new_w, h / new_h

    # Pixel-center inverse mapping: output grid -> floating source coordinates
    xs = (np.arange(new_w, dtype=np.float32) + 0.5) * scale_x - 0.5
    ys = (np.arange(new_h, dtype=np.float32) + 0.5) * scale_y - 0.5

    # Neighbor indices: (x0,y0) is top-left, (x1,y1) is bottom-right
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    x1, y1 = x0 + 1, y0 + 1
    wx, wy = xs - x0, ys - y0
    x0, x1 = np.clip(x0, 0, w - 1), np.clip(x1, 0, w - 1)
    y0, y1 = np.clip(y0, 0, h - 1), np.clip(y1, 0, h - 1)

    # Gather four neighbor images for the whole output grid at once
    img_f = img.astype(np.float32)
    Ia, Ib = img_f[y0[:, None], x0[None, :]], img_f[y0[:, None], x1[None, :]]
    Ic, Id = img_f[y1[:, None], x0[None, :]], img_f[y1[:, None], x1[None, :]]

    # Broadcast x/y weights over channels, then blend horizontally and vertically
    wx2, wy2 = wx[None, :], wy[:, None]
    if img.ndim == 3:
        wx2, wy2 = wx2[..., None], wy2[..., None]
    top = Ia * (1 - wx2) + Ib * wx2
    bot = Ic * (1 - wx2) + Id * wx2
    return to_uint8(top * (1 - wy2) + bot * wy2)


def resize_opencv(img: np.ndarray, new_w: int, new_h: int,
                  interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    """Resize with OpenCV; dsize is (width, height)"""
    return cv2.resize(img, (new_w, new_h), interpolation=interpolation)


def rotate_manual(img: np.ndarray, angle_deg: float, expand: bool = True) -> np.ndarray:
    """Rotate around the image center using inverse mapping and bilinear interpolation.

    The output canvas can either expand to fit the rotated image (expand=True) or keep the original size (expand=False). 
    Pixels outside the source image are filled with black.
    """
    h, w = img.shape[:2]
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    # Choose the destination canvas size, then create the destination grid
    if expand:
        new_w = math.ceil(abs(w * cos_t) + abs(h * sin_t))
        new_h = math.ceil(abs(w * sin_t) + abs(h * cos_t))
    else:
        new_w, new_h = w, h
    ncx, ncy = (new_w - 1) / 2.0, (new_h - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(new_h, dtype=np.float32),
                         np.arange(new_w, dtype=np.float32), indexing="ij")

    # Move destination coordinates to the destination center, then rotate them backward into source coordinates
    # (avoids holes in the output image)
    xr, yr = xx - ncx, yy - ncy
    src_x = cos_t * xr - sin_t * yr + cx
    src_y = sin_t * xr + cos_t * yr + cy

    # Bilinear interpolation uses the surrounding source pixels
    x0, y0 = np.floor(src_x).astype(int), np.floor(src_y).astype(int)
    x1, y1 = x0 + 1, y0 + 1
    valid = (src_x >= 0) & (src_x <= w - 1) & (src_y >= 0) & (src_y <= h - 1)
    wx, wy = src_x - x0, src_y - y0
    x0c, x1c = np.clip(x0, 0, w - 1), np.clip(x1, 0, w - 1)
    y0c, y1c = np.clip(y0, 0, h - 1), np.clip(y1, 0, h - 1)
    img_f = img.astype(np.float32)
    Ia, Ib = img_f[y0c, x0c], img_f[y0c, x1c]
    Ic, Id = img_f[y1c, x0c], img_f[y1c, x1c]
    if img.ndim == 3:
        wx, wy, valid3 = wx[..., None], wy[..., None], valid[..., None]
    else:
        valid3 = valid
    top = Ia * (1 - wx) + Ib * wx
    bot = Ic * (1 - wx) + Id * wx
    return to_uint8(np.where(valid3, top * (1 - wy) + bot * wy, 0))


def rotate90_manual(img: np.ndarray, k: int = 1) -> np.ndarray:
    """Rotate by 90*k degrees counter-clockwise using transpose plus vertical flip"""
    k = k % 4
    out = img
    for _ in range(k):
        out = out.transpose(1, 0, 2)[::-1] if out.ndim == 3 else out.transpose(1, 0)[::-1]
    return out


def rotate90_opencv(img: np.ndarray, k: int = 1) -> np.ndarray:
    """Rotate by 90*k degrees counter-clockwise with OpenCV's exact rotate helper"""
    codes = {
        0: None,
        1: cv2.ROTATE_90_COUNTERCLOCKWISE,
        2: cv2.ROTATE_180,
        3: cv2.ROTATE_90_CLOCKWISE,
    }
    k = k % 4
    return img.copy() if k == 0 else cv2.rotate(img, codes[k])


def rotate_opencv(img: np.ndarray, angle_deg: float, expand: bool = True) -> np.ndarray:
    """Rotate with OpenCV via an affine matrix; adjusts translation to expand the canvas"""
    h, w = img.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    if expand:
        cos_t, sin_t = abs(M[0, 0]), abs(M[0, 1])
        new_w = int(np.ceil(h * sin_t + w * cos_t))
        new_h = int(np.ceil(h * cos_t + w * sin_t))
        M[0, 2] += (new_w - 1) / 2.0 - cx
        M[1, 2] += (new_h - 1) / 2.0 - cy
    else:
        new_w, new_h = w, h
    return cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR)
