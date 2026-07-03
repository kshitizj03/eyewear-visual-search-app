"""Reusable image-ingestion pipeline (requirement 3.1).

For each catalog image:  load -> preprocess (via CLIP transform) -> CLIP embedding ->
colour histogram -> zero-shot tags -> persist (vector to FAISS, metadata to SQLite).

The same `embed_one` helper is reused by the query path, so query and index images
are processed identically - a common source of retrieval bugs, avoided by design.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from src import config
from src.models import color_features
from src.models.attribute_tagger import tag_image_embedding
from src.models.embedder import get_embedder
from src.storage.metadata_db import MetadataDB
from src.storage.vector_store import VectorStore
from src.utils.image import color_correct, load_rgb, to_rgb
from src.utils.logging import get_logger

log = get_logger()


@dataclass
class ImageFeatures:
    embedding: np.ndarray          # (D,) L2-normalized CLIP vector
    color_hist: np.ndarray         # (COLOR_DIM,)
    tags: dict                     # per-group predicted attributes


def embed_one(image: Image.Image) -> ImageFeatures:
    """Extract all features for a single PIL image (shared by ingest + search).

    Preprocessing per requirement 3.1: normalise orientation/mode (to_rgb), colour-correct
    (gray-world white balance), then the embedder resizes + centre-crops. Colour histogram
    is computed on the corrected image so the colour signal is lighting-consistent too.
    """
    image = color_correct(to_rgb(image))              # orientation-safe + colour-corrected
    emb = get_embedder().encode_image(image)[0]
    hist = color_features.color_histogram(image)
    tags = tag_image_embedding(emb)
    return ImageFeatures(embedding=emb, color_hist=hist, tags=tags)


def build_index(catalog_csv: Path = config.CATALOG_CSV,
                image_dir: Path = config.IMAGE_DIR) -> dict:
    """Index every row of the catalog. Returns a small summary dict for logging/tests."""
    df = pd.read_csv(catalog_csv)
    log.info(f"Ingesting {len(df)} catalog rows from {catalog_csv.name}")

    store = VectorStore()
    db = MetadataDB()
    ids: list[str] = []
    vectors: list[np.ndarray] = []
    ok, failed = 0, 0

    for _, row in df.iterrows():
        img_path = image_dir / str(row["image"])
        try:
            image = load_rgb(img_path)
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            failed += 1
            log.error(f"[ingest] failed to load {img_path.name}: {exc}")
            continue

        feats = embed_one(image)
        pid = str(row["id"])
        ids.append(pid)
        vectors.append(feats.embedding)

        db.upsert_product({
            "product_id": pid,
            "image_path": str(img_path),
            "name": row.get("name", ""),
            "brand": row.get("brand", ""),
            "price": float(row.get("price", 0) or 0),
            "material": row.get("material", ""),
            "style": row.get("style", ""),
            "product_url": row.get("product_url", ""),
            "color_hist": feats.color_hist.tolist(),
            "tags": feats.tags,
        })
        ok += 1

    if vectors:
        store.add(ids, np.vstack(vectors))
        store.save()

    summary = {"indexed": ok, "failed": failed, "total_vectors": len(store)}
    log.info(f"Ingestion complete: {summary}")
    db.close()
    return summary
