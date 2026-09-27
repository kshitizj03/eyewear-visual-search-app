"""Robust image loading - a single choke point used by ingest, search and the
conversion script so every code path handles real-world images identically.

Handles three things that silently corrupt embeddings otherwise:
  * EXIF orientation (phone photos are often rotated) -> auto-transpose,
  * transparency (RGBA / palette PNGs, WebP) -> flatten onto a white background
    (matches product-shot backdrops rather than PIL's default black),
  * odd modes (grayscale, CMYK, palette) -> normalise to RGB.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

_WHITE = (255, 255, 255)


def to_rgb(image: Image.Image) -> Image.Image:
    """Normalise any PIL image to an upright, opaque RGB image."""
    image = ImageOps.exif_transpose(image)          # honour camera orientation
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, _WHITE + (255,))
        image = Image.alpha_composite(bg, rgba)
    return image.convert("RGB")


def color_correct(image: Image.Image, strength: float = 1.0) -> Image.Image:
    """Gray-world white-balance - the 'color correction' preprocessing step (req 3.1).

    Neutralises colour casts from studio lighting so the same frame photographed under
    warm vs cool light lands in the same colour region. Applied identically to catalog
    and query images, it keeps the matching space consistent while genuinely helping
    cross-lighting queries (e.g. an on-model selfie vs a clean product shot). It is
    near-identity on already-neutral white-background product shots.
    """
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return image
    means = arr.reshape(-1, 3).mean(axis=0)
    gray = float(means.mean())
    scale = gray / np.clip(means, 1.0, None)
    scale = 1.0 + strength * (scale - 1.0)            # dial correction strength
    corrected = np.clip(arr * scale, 0, 255).astype(np.uint8)
    return Image.fromarray(corrected)


def load_rgb(path: str | Path) -> Image.Image:
    """Open an image file and return a normalised RGB image."""
    with Image.open(path) as im:
        return to_rgb(im)
