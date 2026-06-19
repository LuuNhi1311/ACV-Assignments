"""Spatial image operators for Lab 02"""

import numpy as np


def _as_pair(value, name: str) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, int) for v in value):
        return value
    raise ValueError(f"{name} must be an int or a pair of ints")


def _normalize_padding(padding, kernel_shape: tuple[int, int]):
    kh, kw = kernel_shape
    if padding == "same":
        top, left = kh // 2, kw // 2
        return (top, kh - 1 - top), (left, kw - 1 - left)
    if padding == "valid":
        return (0, 0), (0, 0)
    if isinstance(padding, int):
        return (padding, padding), (padding, padding)
    if isinstance(padding, tuple) and len(padding) == 2:
        if all(isinstance(v, int) for v in padding):
            py, px = padding
            return (py, py), (px, px)
        if all(isinstance(v, tuple) and len(v) == 2 for v in padding):
            return padding
    raise ValueError("padding must be 'same', 'valid', int, (py, px), or ((top, bottom), (left, right))")


def _np_border_mode(border: str) -> str:
    mapping = {
        "constant": "constant",
        "reflect": "reflect",
        "replicate": "edge",
        "edge": "edge",
        "symmetric": "symmetric",
        "wrap": "wrap",
    }
    if border not in mapping:
        raise ValueError(f"Unsupported border mode: {border}")
    return mapping[border]


def pad_border(img: np.ndarray, padding_y, padding_x=None, border: str = "reflect", padding_value: float = 0.0) -> np.ndarray:
    """Pad an image with OpenCV/PyTorch-like border options"""
    if padding_x is None:
        padding_x = padding_y
    top, bottom = _as_pair(padding_y, "padding_y")
    left, right = _as_pair(padding_x, "padding_x")
    pad_width = [(top, bottom), (left, right)]
    if img.ndim == 3:
        pad_width.append((0, 0))
    mode = _np_border_mode(border)
    if mode == "constant":
        return np.pad(img, pad_width, mode=mode, constant_values=padding_value)
    return np.pad(img, pad_width, mode=mode)


def correlate2d(img: np.ndarray, kernel: np.ndarray, stride=1, padding="same", border: str = "reflect", padding_value: float = 0.0) -> np.ndarray:
    """Apply 2D correlation with configurable stride, padding, and border mode"""
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 2:
        raise ValueError("kernel must be 2D")
    sy, sx = _as_pair(stride, "stride")
    if sy <= 0 or sx <= 0:
        raise ValueError("stride values must be positive")

    (top, bottom), (left, right) = _normalize_padding(padding, kernel.shape)
    f = img.astype(np.float64)
    # Pad the input image
    padded = pad_border(f, (top, bottom), (left, right), border=border, padding_value=padding_value)
    # Specify the output shape
    kh, kw = kernel.shape
    out_h = (padded.shape[0] - kh) // sy + 1
    out_w = (padded.shape[1] - kw) // sx + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("kernel is larger than the padded image")

    out_shape = (out_h, out_w) + (() if f.ndim == 2 else (f.shape[2],))
    out = np.zeros(out_shape, dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            weight = kernel[i, j]
            if weight != 0.0:
                # Quickly find the slice of related position in output that affected by this kernel value
                window = padded[i:i + sy * out_h:sy, j:j + sx * out_w:sx]
                out += weight * window
    return out


def convolve2d(img: np.ndarray, kernel: np.ndarray, stride=1, padding="same", border: str = "reflect", padding_value: float = 0.0) -> np.ndarray:
    """Apply true 2D convolution by flipping the kernel before correlation"""
    flipped = np.flip(np.asarray(kernel, dtype=np.float64), axis=(0, 1))
    return correlate2d(img, flipped, stride=stride, padding=padding, border=border, padding_value=padding_value)
