"""Feature 1 - Reading and displaying images.
The assignment uses OpenCV to read images into numpy arrays, then matplotlib to show or save the loaded result"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils import to_uint8


def imread_opencv(path: str) -> np.ndarray:
    """Read an image with OpenCV and convert BGR to RGB, preserving uint8 dtype."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def imshow(img: np.ndarray, title: str = "image", save_path: str = None) -> None:
    """Show one image with matplotlib; if save_path is given, save instead of opening a window"""
    plt.figure(figsize=(5, 5))
    if img.ndim == 2:
        plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    else:
        plt.imshow(to_uint8(img), vmin=0, vmax=255)
    plt.title(title)
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
