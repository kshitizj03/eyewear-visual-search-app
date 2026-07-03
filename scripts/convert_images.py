"""Convert a folder of images to a standard format (JPEG or PNG).

The assignment specifies JPG/PNG input; catalogs are often WebP. This normalises
everything to one standard format using the same robust loader the pipeline uses
(EXIF-corrected, transparency flattened onto white). Originals in other formats are
moved to a backup folder so they are not indexed twice.

Usage:
    python -m scripts.convert_images --to jpeg          # default: WebP/etc -> JPEG
    python -m scripts.convert_images --to png
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src import config
from src.utils.image import load_rgb
from src.utils.logging import get_logger

log = get_logger()

_ALL_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
_TARGET = {"jpeg": (".jpg", "JPEG"), "png": (".png", "PNG")}


def convert(target: str = "jpeg", src_dir: Path = config.IMAGE_DIR) -> dict:
    ext, pil_fmt = _TARGET[target]
    backup = src_dir.parent / "images_original"
    backup.mkdir(exist_ok=True)

    sources = [p for p in src_dir.iterdir()
               if p.suffix.lower() in _ALL_EXTS and p.suffix.lower() != ext]
    converted, failed = 0, 0
    for path in sources:
        out = path.with_suffix(ext)
        try:
            img = load_rgb(path)
            save_kwargs = {"quality": 95} if pil_fmt == "JPEG" else {}
            img.save(out, pil_fmt, **save_kwargs)
            shutil.move(str(path), str(backup / path.name))   # keep the original safe
            converted += 1
        except Exception as exc:
            failed += 1
            log.error(f"convert failed for {path.name}: {exc}")

    summary = {"converted": converted, "failed": failed,
               "format": pil_fmt, "originals_backed_up_to": str(backup)}
    log.info(f"Conversion complete: {summary}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert images to JPEG/PNG.")
    ap.add_argument("--to", choices=["jpeg", "png"], default="jpeg")
    ap.add_argument("--src", default=str(config.IMAGE_DIR))
    args = ap.parse_args()
    convert(args.to, Path(args.src))


if __name__ == "__main__":
    main()
