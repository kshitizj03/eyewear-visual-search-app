"""FastAPI application - the service boundary over the AI + storage layers.

Auto-generated Swagger docs at /docs directly serve the 'API Design & Documentation'
criterion. The heavy SearchEngine (model + index) is loaded once at startup and
reused across requests. Failed uploads and slow queries are logged (observability).
"""
from __future__ import annotations

import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from src import config
from src.feedback.feedback import FeedbackService
from src.pipeline.search import SearchEngine
from src.storage.metadata_db import MetadataDB
from src.utils.logging import get_logger

log = get_logger()

_engine: SearchEngine | None = None
_feedback: FeedbackService | None = None
_db: MetadataDB | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model + index once at startup and reuse across all requests."""
    global _engine, _feedback, _db
    _db = MetadataDB()
    _feedback = FeedbackService(_db)
    try:
        _engine = SearchEngine()
    except FileNotFoundError:
        log.error("FAISS index missing. Run `python -m scripts.build_index` first.")
    yield


app = FastAPI(
    title="Eyewear Visual Search API",
    description="Upload an eyewear image to find visually similar products. "
                "Ranking fuses CLIP style similarity, colour, attribute tags and "
                "click feedback.",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class SearchResultOut(BaseModel):
    product_id: str
    similarity_score: float
    components: dict
    metadata: dict


class SearchResponseOut(BaseModel):
    query_id: str
    query_tags: dict
    cropped: bool
    latency_ms: float
    results: list[SearchResultOut]


class FeedbackIn(BaseModel):
    query_id: str
    product_id: str
    style: str
    relevant: bool


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "device": config.DEVICE,
        "indexed_products": len(_engine.store) if _engine else 0,
    }


@app.post("/search", response_model=SearchResponseOut)
async def search(
    file: UploadFile = File(..., description="Query image (JPG/PNG)"),
    text: str | None = Form(None, description="Optional text modifier, e.g. 'tortoise shell'"),
    top_k: int = Form(config.TOP_K),
    price_min: float | None = Form(None),
    price_max: float | None = Form(None),
    brand: str | None = Form(None),
    material: str | None = Form(None),
    smart_crop: bool = Form(True),
):
    if _engine is None:
        raise HTTPException(503, "Index not built. Run scripts/build_index.py.")
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        log.error(f"[upload] rejected {file.filename}: {exc}")
        raise HTTPException(400, "Invalid or unreadable image file.")

    resp = _engine.search(
        image, text=text, top_k=top_k, price_min=price_min, price_max=price_max,
        brand=brand, material=material, use_smart_crop=smart_crop,
    )
    return SearchResponseOut(
        query_id=resp.query_id, query_tags=resp.query_tags, cropped=resp.cropped,
        latency_ms=resp.latency_ms,
        results=[SearchResultOut(**r.__dict__) for r in resp.results],
    )


@app.post("/feedback")
def feedback(payload: FeedbackIn) -> dict:
    if _feedback is None:
        raise HTTPException(503, "Service not ready.")
    return _feedback.record(payload.query_id, payload.product_id,
                            payload.style, payload.relevant)


@app.get("/product/{product_id}")
def product(product_id: str) -> dict:
    row = _db.get_products([product_id]).get(product_id) if _db else None
    if not row:
        raise HTTPException(404, "Product not found.")
    return row


@app.get("/filters")
def filters() -> dict:
    """Expose available filter values for the UI."""
    if _db is None:
        raise HTTPException(503, "Service not ready.")
    lo, hi = _db.price_bounds()
    return {
        "brands": _db.distinct("brand"),
        "materials": _db.distinct("material"),
        "styles": _db.distinct("style"),
        "price_min": lo,
        "price_max": hi,
    }
