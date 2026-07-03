"""Central configuration for the Eyewear Visual Search platform.

Every tunable knob lives here so the pipeline is config-driven and the demo can
show/adjust the fusion weights live. Paths are resolved relative to the repo root
so the project runs unchanged on any machine.
"""
from __future__ import annotations

from pathlib import Path


def _detect_device() -> str:
    """Best available torch device, without making torch a hard import-time dependency."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
CATALOG_CSV = DATA_DIR / "catalog.csv"

ARTIFACT_DIR = ROOT_DIR / "artifacts"          # generated indexes / db live here
FAISS_INDEX_PATH = ARTIFACT_DIR / "eyewear.faiss"
FAISS_IDMAP_PATH = ARTIFACT_DIR / "faiss_idmap.json"
SQLITE_PATH = ARTIFACT_DIR / "eyewear.db"
LOG_DIR = ROOT_DIR / "logs"

for _d in (DATA_DIR, IMAGE_DIR, ARTIFACT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
DEVICE = _detect_device()

# open_clip model. ViT-B-32 is fast and CPU-friendly; upgrade to "ViT-L-14" on GPU
# for a few points of accuracy at higher latency.
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
EMBED_DIM = 512                                 # ViT-B-32 output dim

# --------------------------------------------------------------------------- #
# Retrieval + multi-attribute fusion ranking
# --------------------------------------------------------------------------- #
TOP_K = 12                                      # results returned to the user
CANDIDATE_MULTIPLIER = 6                         # ANN over-fetch before re-ranking

# Fusion weights: final_score = w_clip*clip + w_color*color + w_attr*attr_overlap + boost
FUSION_WEIGHTS = {
    "clip": 0.70,       # overall semantic / style similarity (CLIP cosine)
    "color": 0.18,      # HSV colour-histogram similarity
    "attr": 0.12,       # shared predicted attribute tags (shape / material)
}

# Multi-modal blend: query = alpha*image_emb + (1-alpha)*text_emb  (bonus)
MULTIMODAL_IMAGE_WEIGHT = 0.6

# Feedback loop: each net-positive click adds this much, capped, to a product's
# score for the matching visual style.
FEEDBACK_BOOST_PER_CLICK = 0.03
FEEDBACK_BOOST_CAP = 0.15

# --------------------------------------------------------------------------- #
# Attribute vocabulary (zero-shot tagging via CLIP text prompts)
# --------------------------------------------------------------------------- #
ATTRIBUTE_PROMPTS: dict[str, list[str]] = {
    # Frame shapes - aligned to the actual catalog vocabulary. Kept to genuine
    # shape terms (not sizes like "oversized") for cleaner zero-shot separation.
    "style": [
        "aviator", "wayfarer", "round", "oval", "cat-eye",
        "rectangle", "square", "geometric", "hexagonal", "clubmaster",
    ],
    "rim": ["full-rim", "half-rim", "rimless"],
    "material": ["acetate frame", "metal frame", "plastic frame", "titanium frame"],
    "transparency": ["transparent frame", "solid opaque frame"],
}

# Performance / observability
SLOW_QUERY_MS = 800.0    # log a warning above this latency
