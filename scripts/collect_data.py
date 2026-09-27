"""Build the sample dataset + catalog.csv.

Two modes, so you're never blocked on scraping:

  1) --download   Download images listed in data/seeds.csv (id,url,name,brand,price,
                  material,style,product_url). Fill this with Lenskart product image
                  URLs (right-click a product image -> Copy image address) or any
                  eyewear image URLs from the web.

  2) --from-folder  You already dropped images into data/images/. This scans them and
                    writes catalog.csv. Metadata is resolved in priority order:
                      (a) explicit `key=value` parts joined by `__`, e.g.
                          brand=RayBan__style=aviator__price=5999.jpg
                      (b) descriptive Lenskart filenames (brand/shape/material/colour
                          tokens are recognised automatically), e.g.
                          black-full-rim-square-vincent-chase-acetate-vc-e18174-eyeglasses.webp
                      (c) CLIP zero-shot tags fill any remaining style/material.

Examples:
    python -m scripts.collect_data --from-folder
    python -m scripts.collect_data --download
"""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

from src import config
from src.utils.logging import get_logger

log = get_logger()

SEEDS_CSV = config.DATA_DIR / "seeds.csv"
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Lenskart-style descriptive filenames encode brand/shape/material/rim/colour.
# Longer keys first so e.g. "lenskart-air" matches before "lenskart".
_BRANDS = {
    "john-jacobs": "John Jacobs", "vincent-chase": "Vincent Chase",
    "lenskart-air": "Lenskart Air", "lenskart": "Lenskart", "hustlr": "Hustlr",
    "ray-ban": "Ray-Ban", "rayban": "Ray-Ban", "oakley": "Oakley", "fossil": "Fossil",
}
_STYLES = ["cat-eye", "clubmaster", "wayfarer", "aviator", "geometric", "hexagonal",
           "rectangle", "square", "round", "oval", "pilot", "wrap"]
# Normalize raw material tokens (steel is a metal) to the filter vocabulary.
_MATERIALS = {"acetate": "acetate", "titanium": "titanium", "steel": "metal",
              "metal": "metal", "tr90": "tr90"}
_RIMS = ["full-rim", "half-rim", "rimless", "supra"]
_COLORS = ["gunmetal", "transparent", "tortoise", "black", "blue", "brown", "grey",
           "gray", "gold", "silver", "green", "red", "pink", "purple", "demi"]
# Plausible price bands per brand so the price filter demos meaningfully.
_PRICE_BANDS = {
    "John Jacobs": (2500, 6000), "Vincent Chase": (1500, 3500),
    "Lenskart Air": (2000, 4500), "Hustlr": (1000, 2500),
    "Lenskart": (1500, 4000), "Ray-Ban": (5000, 9000),
}


def _first(tokens: list[str], text: str) -> str | None:
    return next((t for t in tokens if t in text), None)


def _parse_lenskart_name(stem: str) -> dict:
    """Parse a descriptive Lenskart filename -> brand/style/material/rim/colour/name/code."""
    s = stem.lower()
    meta: dict = {}
    for key, label in _BRANDS.items():
        if key in s:
            meta["brand"] = label
            break
    if (style := _first(_STYLES, s)):
        meta["style"] = style
    if (mat := _first(list(_MATERIALS), s)):
        meta["material"] = _MATERIALS[mat]
    if (rim := _first(_RIMS, s)):
        meta["rim"] = rim
    if (color := _first(_COLORS, s)):
        meta["color"] = color
    # product code like vc-e18174 / jj-e70293 / la-e13069 / ls-e15666
    m = re.search(r"\b([a-z]{2}-?e\d{3,6}(?:-c\d)?)\b", s)
    if m:
        meta["code"] = m.group(1).upper()
    # human-readable name from the parts we recognised
    bits = [meta.get("brand"), (meta.get("color") or "").title() or None,
            (meta.get("style") or "").replace("-", " ").title() or None]
    name = " ".join(b for b in bits if b)
    if meta.get("code"):
        name = f"{name} ({meta['code']})" if name else meta["code"]
    if name:
        meta["name"] = name
    return meta


# --------------------------------------------------------------------------- #
def _valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def download() -> None:
    if not SEEDS_CSV.exists():
        log.error(f"{SEEDS_CSV} not found. Create it with columns: "
                  "id,url,name,brand,price,material,style,product_url")
        return
    seeds = pd.read_csv(SEEDS_CSV)
    rows, ok, failed = [], 0, 0
    headers = {"User-Agent": "Mozilla/5.0 (eyewear-visual-search demo)"}
    for _, s in seeds.iterrows():
        ext = Path(str(s["url"]).split("?")[0]).suffix or ".jpg"
        fname = f"{s['id']}{ext}"
        dest = config.IMAGE_DIR / fname
        try:
            r = requests.get(str(s["url"]), timeout=20, headers=headers)
            r.raise_for_status()
            dest.write_bytes(r.content)
            if not _valid_image(dest):
                raise ValueError("not a valid image")
            ok += 1
            rows.append({
                "id": s["id"], "image": fname, "name": s.get("name", ""),
                "brand": s.get("brand", ""), "price": s.get("price", 0),
                "material": s.get("material", ""), "style": s.get("style", ""),
                "product_url": s.get("product_url", ""),
            })
        except Exception as exc:
            failed += 1
            log.error(f"download failed for {s['id']}: {exc}")
            if dest.exists():
                dest.unlink(missing_ok=True)
    pd.DataFrame(rows).to_csv(config.CATALOG_CSV, index=False)
    log.info(f"Downloaded {ok} images ({failed} failed) -> catalog.csv written")


# --------------------------------------------------------------------------- #
def _parse_filename(stem: str) -> dict:
    """Extract metadata explicitly encoded as `key=value` parts joined by '__'.

    e.g. `brand=RayBan__style=aviator__price=5999.jpg`. Returns {} if no key=value
    parts are present, so descriptive filenames fall through to the Lenskart parser.
    """
    meta: dict = {}
    for p in stem.split("__"):
        if "=" in p:
            k, v = p.split("=", 1)
            meta[k.strip().lower()] = v.strip()
    return meta


def _price_for(brand: str, seed: str) -> float:
    lo, hi = _PRICE_BANDS.get(brand, (1500, 4000))
    rng = random.Random(seed)                       # deterministic per product
    return float(rng.randrange(lo, hi + 1, 100))


def from_folder() -> None:
    images = sorted(
        p for p in config.IMAGE_DIR.iterdir()
        if p.suffix.lower() in _IMG_EXTS
    )
    if not images:
        log.error(f"No images in {config.IMAGE_DIR}. Add image files and retry.")
        return

    # Lazy import so --download path doesn't need the model.
    from src.models.attribute_tagger import tag_image_embedding
    from src.models.embedder import get_embedder

    embedder = get_embedder()
    rows = []
    for i, path in enumerate(images):
        if not _valid_image(path):
            log.error(f"skipping unreadable image {path.name}")
            continue

        # First honour the explicit `key=value`/positional convention, then fall back
        # to parsing descriptive Lenskart filenames.
        meta = _parse_filename(path.stem) or {}
        for k, v in _parse_lenskart_name(path.stem).items():
            meta.setdefault(k, v)

        # Fill any still-missing style/material with CLIP zero-shot tags.
        if "style" not in meta or "material" not in meta:
            tags = tag_image_embedding(embedder.encode_image(Image.open(path))[0])
            meta.setdefault("style", tags["style"]["label"])
            meta.setdefault("material", tags["material"]["label"].replace(" frame", ""))

        brand = meta.get("brand", "Unknown")
        pid = f"p{i:04d}"
        rows.append({
            "id": pid,
            "image": path.name,
            "name": meta.get("name") or path.stem.replace("-", " ")[:60],
            "brand": brand,
            "price": float(meta.get("price", _price_for(brand, pid))),
            "material": meta.get("material", "acetate"),
            "style": meta.get("style", "round"),
            "product_url": meta.get("product_url", ""),
        })

    pd.DataFrame(rows).to_csv(config.CATALOG_CSV, index=False)
    n_brands = len({r["brand"] for r in rows})
    log.info(f"Wrote catalog.csv with {len(rows)} products across {n_brands} brands "
             f"(filename-parsed + CLIP-tagged).")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Build the eyewear sample dataset.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--download", action="store_true", help="download from data/seeds.csv")
    g.add_argument("--from-folder", action="store_true",
                   help="build catalog.csv from images already in data/images/")
    args = ap.parse_args()
    if args.download:
        download()
    else:
        from_folder()


if __name__ == "__main__":
    main()
