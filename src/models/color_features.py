"""Colour feature extraction - the 'colour' signal in multi-attribute similarity.

CLIP captures overall style well but is only loosely sensitive to exact colour.
An explicit HSV colour histogram gives the ranker a dedicated colour channel so a
black frame ranks other black frames above visually-similar tortoise ones.

Product shots sit on a near-white/grey studio background that would otherwise
dominate every histogram (making all images look colour-similar). We therefore
build a foreground mask that drops low-saturation, high-value background pixels and
compute the histogram over the frame pixels only. If masking removes almost
everything (e.g. a genuinely white/transparent frame), we fall back to the full
image so the descriptor is never empty.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

# 3-D HSV histogram bins (H, S, V). Kept small so the descriptor is cheap.
_H_BINS, _S_BINS, _V_BINS = 8, 4, 4
COLOR_DIM = _H_BINS * _S_BINS * _V_BINS

# Background = low saturation AND high value (white / light-grey studio backdrop).
_BG_SAT_MAX = 35        # 0-255
_BG_VAL_MIN = 200       # 0-255
_MIN_FG_FRACTION = 0.02  # if less foreground than this, use the whole image


def _foreground_mask(hsv: np.ndarray) -> np.ndarray | None:
    """Return a uint8 mask of frame (non-background) pixels, or None to use all pixels."""
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    background = (s < _BG_SAT_MAX) & (v > _BG_VAL_MIN)
    fg = (~background).astype("uint8") * 255
    if fg.mean() / 255.0 < _MIN_FG_FRACTION:
        return None
    return fg


def color_histogram(image: Image.Image) -> np.ndarray:
    """Return an L1-normalized, background-suppressed HSV colour histogram."""
    rgb = np.asarray(image.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = _foreground_mask(hsv)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], mask,
        [_H_BINS, _S_BINS, _V_BINS],
        [0, 180, 0, 256, 0, 256],
    ).flatten().astype("float32")
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def color_similarity(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """Histogram-intersection similarity in [0, 1] (1 == identical colour distribution)."""
    a = np.asarray(hist_a, dtype="float32")
    b = np.asarray(hist_b, dtype="float32")
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.minimum(a, b).sum())
