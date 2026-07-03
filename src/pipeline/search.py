"""Visual search + multi-attribute fusion ranking (requirements 3.2, and the
heart of the 'Search Accuracy & Visual Relevance' score).

Flow:
  1. Process the query image identically to catalog images (optional smart-crop).
  2. Optionally blend in a text modifier for multi-modal search (bonus).
  3. ANN retrieve an over-fetched candidate set from FAISS (cosine).
  4. Apply structured filters (price / brand / material) in SQLite.
  5. Re-rank candidates with a weighted fusion of CLIP + colour + attribute overlap,
     plus a per-style feedback boost learned from user clicks.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from src import config
from src.models import color_features
from src.models.attribute_tagger import flat_tags
from src.models.embedder import get_embedder, normalize
from src.models.smart_crop import smart_crop
from src.pipeline.ingest import ImageFeatures, embed_one
from src.utils.image import to_rgb
from src.storage.metadata_db import MetadataDB
from src.storage.vector_store import VectorStore
from src.utils.logging import get_logger, timed

log = get_logger()


@dataclass
class SearchResult:
    product_id: str
    similarity_score: float
    components: dict           # per-signal breakdown (clip/color/attr/boost) for transparency
    metadata: dict


@dataclass
class SearchResponse:
    query_id: str
    query_tags: dict
    cropped: bool
    latency_ms: float
    results: list[SearchResult] = field(default_factory=list)


class SearchEngine:
    """Loads the index + db once and answers queries. Used by the API and Streamlit."""

    def __init__(self) -> None:
        self.store = VectorStore.load()
        self.db = MetadataDB()
        self.embedder = get_embedder()
        log.info(f"SearchEngine ready: {len(self.store)} indexed products")

    # ------------------------------------------------------------------ #
    def _query_features(self, image: Image.Image, text: str | None,
                        use_smart_crop: bool) -> tuple[ImageFeatures, np.ndarray, bool]:
        """Return (image features, final query vector, cropped?)."""
        image = to_rgb(image)                          # orientation + transparency safe
        cropped = False
        if use_smart_crop:
            image, cropped = smart_crop(image)

        feats = embed_one(image)
        query_vec = feats.embedding

        if text:  # multi-modal blend (bonus)
            text_vec = self.embedder.encode_text(text)[0]
            a = config.MULTIMODAL_IMAGE_WEIGHT
            query_vec = normalize(a * feats.embedding + (1 - a) * text_vec)
        return feats, query_vec, cropped

    # ------------------------------------------------------------------ #
    def search(self, image: Image.Image, *, text: str | None = None,
               top_k: int = config.TOP_K, price_min: float | None = None,
               price_max: float | None = None, brand: str | None = None,
               material: str | None = None, use_smart_crop: bool = True) -> SearchResponse:
        query_id = uuid.uuid4().hex[:12]
        with timed(f"search[{query_id}]") as t:
            feats, query_vec, cropped = self._query_features(image, text, use_smart_crop)
            query_style = feats.tags.get("style", {}).get("label", "")

            # 1) ANN over-fetch
            candidates = self.store.search(query_vec, top_k * config.CANDIDATE_MULTIPLIER)
            cand_ids = [pid for pid, _ in candidates]
            clip_scores = {pid: s for pid, s in candidates}

            # 2) structured filters
            allowed = self.db.filter_ids(
                cand_ids, price_min=price_min, price_max=price_max,
                brand=brand, material=material,
            )
            cand_ids = [pid for pid in cand_ids if pid in allowed]

            # 3) fusion re-rank
            products = self.db.get_products(cand_ids)
            boosts = self.db.get_boosts(query_style)
            query_tags = flat_tags(feats.tags)
            w = config.FUSION_WEIGHTS

            results: list[SearchResult] = []
            for pid in cand_ids:
                meta = products.get(pid)
                if not meta:
                    continue
                clip = clip_scores.get(pid, 0.0)
                color = color_features.color_similarity(
                    feats.color_hist, np.asarray(meta["color_hist"], dtype="float32")
                )
                cand_tags = {v["label"] for v in meta["tags"].values()} if meta["tags"] else set()
                attr = len(query_tags & cand_tags) / max(1, len(query_tags))
                boost = boosts.get(pid, 0.0)

                score = w["clip"] * clip + w["color"] * color + w["attr"] * attr + boost
                results.append(SearchResult(
                    product_id=pid,
                    similarity_score=round(float(score), 4),
                    components={"clip": round(clip, 4), "color": round(color, 4),
                                "attr": round(attr, 4), "boost": round(boost, 4)},
                    metadata={k: meta[k] for k in
                              ("name", "brand", "price", "material", "style",
                               "image_path", "product_url", "tags")},
                ))

            results.sort(key=lambda r: r.similarity_score, reverse=True)
            results = results[:top_k]

        return SearchResponse(
            query_id=query_id, query_tags=feats.tags, cropped=cropped,
            latency_ms=t["ms"], results=results,
        )
