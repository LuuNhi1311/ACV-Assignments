"""Feature 3 - Point transforms: brightness, contrast and color.
Each output pixel depends only on the input pixel, so manual numpy versions are vectorized.
Library: OpenCV (addWeighted, cvtColor, bitwise_not, LUT)"""

import numpy as np
import cv2
from utils import to_uint8


def brightness_manual(img: np.ndarray, beta: int) -> np.ndarray:
    """Manual brightness: add constant beta to every pixel then clip (out = in + beta)."""
    return to_uint8(img.astype(np.int16) + beta)


def brightness_opencv(img: np.ndarray, beta: int) -> np.ndarray:
    """Brightness with OpenCV addWeighted (saturating add, clips like the manual version)"""
    return cv2.addWeighted(img, 1.0, np.zeros_like(img), 0, float(beta))


def contrast_manual(img: np.ndarray, alpha: float) -> np.ndarray:
    """Manual contrast around mid-gray: out = alpha*(in-128)+128 (alpha>1 boosts contrast)"""
    return to_uint8(alpha * (img.astype(np.float32) - 128.0) + 128.0)


def contrast_opencv(img: np.ndarray, alpha: float) -> np.ndarray:
    """Contrast with OpenCV addWeighted (saturating); beta=128*(1-alpha) matches the manual formula"""
    return cv2.addWeighted(img, alpha, np.zeros_like(img), 0, 128.0 * (1.0 - alpha))


def grayscale_manual(img: np.ndarray) -> np.ndarray:
    """Manual RGB-to-gray with BT.601 luminance: Y = 0.299R + 0.587G + 0.114B"""
    img_f = img.astype(np.float32)
    r, g, b = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]
    return to_uint8(0.299 * r + 0.587 * g + 0.114 * b)


def grayscale_opencv(img_rgb: np.ndarray) -> np.ndarray:
    """RGB-to-gray with OpenCV cvtColor"""
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


def rgb_to_hsv_manual(img: np.ndarray):
    """Convert RGB uint8 image to float HSV channels: H in degrees, S/V in [0,1]"""
    rgb = img.astype(np.float32) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    # Value is the strongest RGB channel (maxc)
    maxc = np.max(rgb, axis=2)
    minc = np.min(rgb, axis=2)
    delta = maxc - minc     # Chroma

    # Hue depends on which channel is strongest
    hue = np.zeros_like(maxc)
    nonzero = delta > 0     # Avoid division by 0 error
    red_is_max = nonzero & (maxc == r)
    green_is_max = nonzero & (maxc == g)
    blue_is_max = nonzero & (maxc == b)
    hue[red_is_max] = (60.0 * ((g[red_is_max] - b[red_is_max]) / delta[red_is_max])) % 360.0
    hue[green_is_max] = 60.0 * ((b[green_is_max] - r[green_is_max]) / delta[green_is_max] + 2.0)
    hue[blue_is_max] = 60.0 * ((r[blue_is_max] - g[blue_is_max]) / delta[blue_is_max] + 4.0)

    # Saturation is chroma relative to value. Black pixels keep saturation=0 to avoid division by zero
    saturation = np.zeros_like(maxc)
    saturation[maxc > 0] = delta[maxc > 0] / maxc[maxc > 0]
    return hue, saturation, maxc


def hsv_to_rgb_manual(hue: np.ndarray, saturation: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Convert float HSV channels back to an RGB uint8 image"""
    hue = hue.astype(np.float32, copy=False) % 360.0
    saturation = saturation.astype(np.float32, copy=False)
    value = value.astype(np.float32, copy=False)

    # Chroma is the amount of color. X is the second-largest temporary channel inside the current 60-degree hue sector
    chroma = value * saturation
    x = chroma * (1.0 - np.abs((hue / 60.0) % 2.0 - 1.0))
    m = value - chroma

    # Build RGB from the hue sector. These temporary values are in [0,chroma]
    z = np.zeros_like(hue)
    rp, gp, bp = z.copy(), z.copy(), z.copy()
    masks = [
        (hue < 60.0),
        (hue >= 60.0) & (hue < 120.0),
        (hue >= 120.0) & (hue < 180.0),
        (hue >= 180.0) & (hue < 240.0),
        (hue >= 240.0) & (hue < 300.0),
        (hue >= 300.0),
    ]
    rp[masks[0]], gp[masks[0]] = chroma[masks[0]], x[masks[0]]
    rp[masks[1]], gp[masks[1]] = x[masks[1]], chroma[masks[1]]
    gp[masks[2]], bp[masks[2]] = chroma[masks[2]], x[masks[2]]
    gp[masks[3]], bp[masks[3]] = x[masks[3]], chroma[masks[3]]
    rp[masks[4]], bp[masks[4]] = x[masks[4]], chroma[masks[4]]
    rp[masks[5]], bp[masks[5]] = chroma[masks[5]], x[masks[5]]

    # Add m to restore the correct brightness, then convert back to uint8
    rgb = np.stack((rp + m, gp + m, bp + m), axis=2)
    return to_uint8(rgb * 255.0)


def saturation_manual(img: np.ndarray, factor: float) -> np.ndarray:
    """Manual saturation adjustment in HSV space; factor>1 increases color intensity"""
    hue, saturation, value = rgb_to_hsv_manual(img)
    saturation = np.clip(saturation * factor, 0.0, 1.0)
    return hsv_to_rgb_manual(hue, saturation, value)


def saturation_opencv(img: np.ndarray, factor: float) -> np.ndarray:
    """Saturation adjustment with OpenCV HSV conversion"""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = np.clip(np.round(hsv[:, :, 1].astype(np.float64) * factor), 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def hue_manual(img: np.ndarray, shift_deg: float) -> np.ndarray:
    """Manual hue rotation in HSV space; shift_deg wraps around the 360 degree color wheel"""
    hue, saturation, value = rgb_to_hsv_manual(img)
    return hsv_to_rgb_manual(hue + shift_deg, saturation, value)


def hue_opencv(img: np.ndarray, shift_deg: float) -> np.ndarray:
    """Hue rotation with OpenCV HSV conversion; OpenCV stores hue in [0,179]"""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    shift = int(round(shift_deg / 2.0))
    hsv[:, :, 0] = ((hsv[:, :, 0].astype(np.int16) + shift) % 180).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def invert_manual(img: np.ndarray) -> np.ndarray:
    """Manual photographic negative: out = 255 - in"""
    return to_uint8(255.0 - img.astype(np.float32))


def invert_opencv(img: np.ndarray) -> np.ndarray:
    """Negative with OpenCV bitwise_not (equivalent to 255 - x)"""
    return cv2.bitwise_not(img)


def gamma_manual(img: np.ndarray, gamma: float) -> np.ndarray:
    """Manual gamma correction: out = 255*(in/255)^gamma (gamma<1 brightens shadows)"""
    return to_uint8(255.0 * np.power(img.astype(np.float32) / 255.0, gamma))


def gamma_opencv(img: np.ndarray, gamma: float) -> np.ndarray:
    """Gamma correction with OpenCV cv2.LUT; the 256-entry table is rounded to match the manual version"""
    table = np.clip(np.round((np.arange(256) / 255.0) ** gamma * 255), 0, 255).astype(np.uint8)
    return cv2.LUT(img, table)


def keep_channel(img: np.ndarray, channel: int) -> np.ndarray:
    """Keep a single color channel (0=R,1=G,2=B) and zero out the other two"""
    out = np.zeros_like(img)
    out[:, :, channel] = img[:, :, channel]
    return out
