"""Smart cropping (BONUS): locate eyewear in a busy photo before searching.

When a user uploads a selfie or a celebrity photo, the frame occupies a small part
of the image and the background dominates the CLIP embedding. We detect the face,
then crop to the eye band (upper-middle of the face box) so the query embedding
focuses on the glasses. On plain product shots (no face detected) we fall back to
the full image.

Face detection uses OpenCV's bundled Haar cascade - no extra dependency, always
available wherever OpenCV is installed, and it degrades to a no-op if for any reason
the cascade can't load.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from src.utils.logging import get_logger

log = get_logger()

# Region of the face box to keep, as fractions of face height/width. Tuned so the
# crop spans brow-to-mid-nose (the whole frame) with a roughly 2:1 aspect ratio that
# matches clean product shots - a thin sliver would be distorted by CLIP's square
# preprocessing and retrieve poorly.
_EYE_BAND_TOP = 0.14
_EYE_BAND_BOTTOM = 0.66
_PAD = 0.04  # horizontal padding beyond the face box (fraction of face width)
# Reject detections smaller than this fraction of the image's shorter side
# (kills spurious tiny "faces" in busy backgrounds).
_MIN_FACE_FRACTION = 0.18

_cascade: cv2.CascadeClassifier | None = None


def _get_cascade() -> cv2.CascadeClassifier | None:
    """Lazily load the frontal-face Haar cascade shipped with OpenCV."""
    global _cascade
    if _cascade is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        clf = cv2.CascadeClassifier(path)
        _cascade = clf if not clf.empty() else None
        if _cascade is None:
            log.warning("smart_crop: Haar cascade unavailable; smart-crop disabled")
    return _cascade


def _detect_face_box(rgb: np.ndarray):
    """Return (x, y, w, h) of the largest plausibly-sized face, or None."""
    clf = _get_cascade()
    if clf is None:
        return None
    h, w = rgb.shape[:2]
    min_side = int(min(h, w) * _MIN_FACE_FRACTION)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                 minSize=(min_side, min_side))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda b: b[2] * b[3])  # largest by area


def smart_crop(image: Image.Image) -> tuple[Image.Image, bool]:
    """Crop to the eyewear region if a face is found.

    Returns (possibly-cropped image, cropped: bool).
    """
    try:
        rgb = np.asarray(image.convert("RGB"))
        box = _detect_face_box(rgb)
    except Exception as exc:  # never let the bonus break a search
        log.warning(f"smart_crop failed, using full image: {exc}")
        return image, False

    if box is None:
        return image, False

    x, y, w, h = (int(v) for v in box)
    img_w = image.width
    x0 = max(0, x - int(_PAD * w))
    x1 = min(img_w, x + w + int(_PAD * w))
    top = y + int(_EYE_BAND_TOP * h)
    bottom = y + int(_EYE_BAND_BOTTOM * h)
    crop = image.crop((x0, top, x1, bottom))
    if crop.width < 20 or crop.height < 10:      # sanity guard
        return image, False
    log.info("smart_crop: face detected, cropped to eye region")
    return crop, True
