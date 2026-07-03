"""CNN vs ViT: quantify why CLIP-ViT is the backbone over a plain ImageNet CNN.

Metric: leave-one-out **style-precision@5** - for every catalog image, retrieve the
5 nearest neighbours (cosine) under each model and measure the fraction whose
catalog frame-shape matches the query's. A higher score means the embedding space
groups visually-similar eyewear more tightly. We also print a qualitative side-by-side
for one query.

Usage:  python -m scripts.compare_models
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.utils.image import load_rgb
from src.utils.logging import get_logger

log = get_logger()
TOP_K = 5


def _embed_all(images, encoder) -> np.ndarray:
    vecs = [encoder.encode_image(im)[0] for im in images]
    return np.vstack(vecs).astype("float32")


def _style_precision_at_k(emb: np.ndarray, styles: list[str], k: int = TOP_K) -> float:
    sims = emb @ emb.T                      # cosine (rows are L2-normalized)
    np.fill_diagonal(sims, -1.0)            # exclude self
    hits = 0
    for i, row in enumerate(sims):
        top = np.argsort(row)[::-1][:k]
        hits += sum(styles[j] == styles[i] for j in top) / k
    return hits / len(styles)


def main() -> None:
    df = pd.read_csv(config.CATALOG_CSV)
    images = [load_rgb(config.IMAGE_DIR / img) for img in df["image"]]
    styles = df["style"].astype(str).tolist()
    log.info(f"Comparing models on {len(images)} catalog images ...")

    from src.models.embedder import get_embedder, get_resnet_embedder
    clip_emb = _embed_all(images, get_embedder())
    resnet_emb = _embed_all(images, get_resnet_embedder())

    clip_p = _style_precision_at_k(clip_emb, styles)
    resnet_p = _style_precision_at_k(resnet_emb, styles)

    print("\n" + "=" * 56)
    print(f"  Leave-one-out style-precision@{TOP_K} (higher = better)")
    print("=" * 56)
    print(f"  CLIP ViT-B/32 : {clip_p:.3f}")
    print(f"  ResNet50 (CNN): {resnet_p:.3f}")
    delta = (clip_p - resnet_p) / max(resnet_p, 1e-6) * 100
    print(f"  CLIP is {delta:+.1f}% relative to the CNN baseline\n")

    # qualitative: one query, top-3 under each model
    qi = 0
    for name, emb in (("CLIP", clip_emb), ("ResNet50", resnet_emb)):
        sims = emb @ emb[qi]
        sims[qi] = -1
        top = np.argsort(sims)[::-1][:3]
        print(f"  [{name}] query = {df.iloc[qi]['name']} ({styles[qi]})")
        for j in top:
            print(f"      {sims[j]:.3f}  {df.iloc[j]['style']:<10} {df.iloc[j]['name'][:40]}")
    print()


if __name__ == "__main__":
    main()
