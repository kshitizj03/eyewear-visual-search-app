"""Fast unit tests for the components that don't require downloading CLIP.

Covers: colour similarity bounds, FAISS vector store round-trip + cosine ordering,
SQLite metadata filtering, and feedback-boost accumulation with capping. An optional
CLIP smoke test is skipped automatically if the model/weights aren't available.

Run:  pytest -q
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src import config
from src.models import color_features
from src.models.embedder import normalize
from src.storage.metadata_db import MetadataDB
from src.storage.vector_store import VectorStore
from src.utils.image import color_correct, to_rgb


# --------------------------------------------------------------------------- #
def test_color_similarity_bounds():
    a = np.random.rand(color_features.COLOR_DIM).astype("float32"); a /= a.sum()
    b = np.random.rand(color_features.COLOR_DIM).astype("float32"); b /= b.sum()
    assert color_features.color_similarity(a, a) == pytest.approx(1.0, abs=1e-5)
    s = color_features.color_similarity(a, b)
    assert 0.0 <= s <= 1.0


def test_to_rgb_flattens_transparency_to_white():
    im = Image.new("RGBA", (8, 8), (0, 0, 0, 0))     # fully transparent
    out = to_rgb(im)
    assert out.mode == "RGB"
    assert out.getpixel((0, 0)) == (255, 255, 255)   # transparent -> white, not black


def test_color_correct_neutralises_cast_and_preserves_neutral():
    # a neutral grey image should be (nearly) unchanged by gray-world balance
    grey = Image.new("RGB", (8, 8), (120, 120, 120))
    out = np.asarray(color_correct(grey)).astype(int)
    assert abs(out.mean() - 120) <= 2
    # a blue-cast image should have its channel means pulled toward each other
    cast = Image.new("RGB", (8, 8), (60, 90, 200))
    before = np.asarray(cast, dtype=float).reshape(-1, 3).mean(0)
    after = np.asarray(color_correct(cast), dtype=float).reshape(-1, 3).mean(0)
    assert after.std() < before.std()          # channels more balanced after correction


def test_color_histogram_ignores_white_background():
    """A red frame on a large white background should read as red, not white."""
    def red_on_white(bg_size):
        img = Image.new("RGB", (bg_size, bg_size), (255, 255, 255))
        img.paste((220, 20, 20), (0, 0, 24, 24))     # same red patch, different bg amount
        return img

    h_small = color_features.color_histogram(red_on_white(48))
    h_large = color_features.color_histogram(red_on_white(96))
    # Background masking makes the descriptor robust to how much white surrounds it.
    assert color_features.color_similarity(h_small, h_large) > 0.8
    assert h_small.sum() == pytest.approx(1.0, abs=1e-5)


def test_normalize_unit_length():
    v = np.array([3.0, 4.0], dtype="float32")
    assert np.linalg.norm(normalize(v)) == pytest.approx(1.0)
    assert np.linalg.norm(normalize(np.zeros(4, dtype="float32"))) == pytest.approx(0.0)


def test_vector_store_cosine_ordering():
    store = VectorStore(dim=4)
    vecs = normalize(np.array([
        [1, 0, 0, 0],
        [0.9, 0.1, 0, 0],
        [0, 1, 0, 0],
    ], dtype="float32"))
    store.add(["a", "b", "c"], vecs)
    results = store.search(normalize(np.array([1, 0, 0, 0], dtype="float32")), k=3)
    assert [pid for pid, _ in results] == ["a", "b", "c"]      # nearest first
    assert results[0][1] == pytest.approx(1.0, abs=1e-4)


def test_metadata_filtering(tmp_path):
    db = MetadataDB(path=tmp_path / "t.db")
    for i, (brand, price, mat) in enumerate(
        [("Ray-Ban", 5999, "metal"), ("Vincent Chase", 1999, "acetate"),
         ("Ray-Ban", 999, "acetate")]
    ):
        db.upsert_product({
            "product_id": f"p{i}", "image_path": "x.jpg", "name": "n",
            "brand": brand, "price": price, "material": mat, "style": "round",
            "product_url": "", "color_hist": [], "tags": {},
        })
    ids = ["p0", "p1", "p2"]
    assert db.filter_ids(ids, brand="Ray-Ban") == {"p0", "p2"}
    assert db.filter_ids(ids, price_min=1500, price_max=6000) == {"p0", "p1"}
    assert db.filter_ids(ids, material="acetate", price_max=1500) == {"p2"}
    db.close()


def test_feedback_boost_accumulates_and_caps(tmp_path):
    db = MetadataDB(path=tmp_path / "t.db")
    for _ in range(100):                       # way past the cap
        db.record_feedback("q1", "p0", "aviator", relevant=True)
    boost = db.get_boosts("aviator")["p0"]
    assert boost == pytest.approx(config.FEEDBACK_BOOST_CAP, abs=1e-6)

    for _ in range(100):
        db.record_feedback("q1", "p1", "aviator", relevant=False)
    assert db.get_boosts("aviator")["p1"] == pytest.approx(-config.FEEDBACK_BOOST_CAP, abs=1e-6)
    db.close()


@pytest.mark.parametrize("text", ["a photo of aviator sunglasses"])
def test_clip_optional_smoke(text):
    pytest.importorskip("open_clip")
    try:
        from src.models.embedder import get_embedder
        emb = get_embedder().encode_text(text)
    except Exception as exc:                    # no network / weights unavailable
        pytest.skip(f"CLIP weights unavailable: {exc}")
    assert emb.shape == (1, config.EMBED_DIM)
    assert np.linalg.norm(emb[0]) == pytest.approx(1.0, abs=1e-3)
