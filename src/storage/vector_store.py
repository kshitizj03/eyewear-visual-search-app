"""FAISS vector store - the vector-DB half of the data-storage layer.

We use `IndexFlatIP` (exact inner-product). Because every embedding is L2-normalized
upstream, inner product equals cosine similarity, giving exact cosine nearest-neighbour
search. Flat is the right call at this catalog size (hundreds-low thousands): zero
recall loss and sub-millisecond queries. The design note in the README explains how to
swap to `IndexHNSWFlat`/`IndexIVFFlat` for millions of vectors without touching callers.

FAISS stores only vectors + integer ids; all human-readable metadata lives in SQLite.
A JSON id-map keeps the FAISS row order ↔ product_id mapping stable across reloads.
"""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from src import config
from src.utils.logging import get_logger

log = get_logger()


class VectorStore:
    def __init__(self, dim: int = config.EMBED_DIM) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.ids: list[str] = []                 # row i in the index -> product_id

    # ------------------------------------------------------------------ #
    # Build / persist
    # ------------------------------------------------------------------ #
    def add(self, product_ids: list[str], vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")
        self.index.add(vectors)
        self.ids.extend(product_ids)

    def save(self, index_path: Path = config.FAISS_INDEX_PATH,
             idmap_path: Path = config.FAISS_IDMAP_PATH) -> None:
        faiss.write_index(self.index, str(index_path))
        idmap_path.write_text(json.dumps(self.ids), encoding="utf-8")
        log.info(f"Saved FAISS index ({self.index.ntotal} vectors) -> {index_path.name}")

    @classmethod
    def load(cls, index_path: Path = config.FAISS_INDEX_PATH,
             idmap_path: Path = config.FAISS_IDMAP_PATH) -> "VectorStore":
        if not index_path.exists():
            raise FileNotFoundError(
                f"No FAISS index at {index_path}. Run scripts/build_index.py first."
            )
        store = cls()
        store.index = faiss.read_index(str(index_path))
        store.ids = json.loads(idmap_path.read_text(encoding="utf-8"))
        store.dim = store.index.d
        return store

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #
    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Return up to k (product_id, cosine_similarity) pairs, best first."""
        if self.index.ntotal == 0:
            return []
        q = np.ascontiguousarray(np.asarray(query, dtype="float32").reshape(1, -1))
        k = min(k, self.index.ntotal)
        scores, idxs = self.index.search(q, k)
        results: list[tuple[str, float]] = []
        for score, row in zip(scores[0], idxs[0]):
            if row == -1:
                continue
            results.append((self.ids[row], float(score)))
        return results

    def __len__(self) -> int:
        return self.index.ntotal
