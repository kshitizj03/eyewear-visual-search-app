"""Zero-shot attribute tagging (mandatory requirement 3.3).

Rather than training a classifier (which would need a labelled dataset), we exploit
CLIP's shared image/text space: for each attribute group we embed a set of text
prompts once, then score an image embedding against them via cosine similarity and
softmax. This labels frames as e.g. Aviator / Rimless / Metal / Transparent with a
confidence - no training data required, and new labels are added by editing config.
"""
from __future__ import annotations

import functools

import numpy as np

from src import config
from src.models.embedder import get_embedder


# Prompt ensembling: encoding several phrasings per label and averaging the text
# embeddings yields a more robust class prototype than any single caption (the
# standard CLIP zero-shot trick). Templates differ slightly per attribute group.
_TEMPLATES = {
    "style": [
        "a photo of {v} eyeglasses",
        "{v} shaped glasses",
        "a close-up of {v} eyewear frames",
        "spectacles with a {v} frame shape",
    ],
    "rim": [
        "a photo of {v} eyeglasses",
        "{v} spectacle frames",
        "glasses with a {v} design",
    ],
    "material": [
        "eyeglasses with a {v}",
        "spectacles made of {v} material".replace(" frame", ""),
        "a {v}",
    ],
    "transparency": [
        "eyeglasses with a {v}",
        "spectacles that are {v}".replace(" frame", ""),
    ],
}


def _prompts(group: str, value: str) -> list[str]:
    """All prompt phrasings for a (group, label) pair."""
    templates = _TEMPLATES.get(group, ["a photo of {v} eyewear"])
    return [t.format(v=value) for t in templates]


@functools.lru_cache(maxsize=1)
def _group_text_banks() -> dict[str, tuple[list[str], np.ndarray]]:
    """Pre-encode ensembled attribute prompts once -> {group: (labels, (n, D) protos)}."""
    emb = get_embedder()
    banks: dict[str, tuple[list[str], np.ndarray]] = {}
    for group, labels in config.ATTRIBUTE_PROMPTS.items():
        protos = []
        for value in labels:
            vecs = emb.encode_text(_prompts(group, value))     # (t, D), L2-normalized
            proto = vecs.mean(axis=0)
            proto /= np.linalg.norm(proto) or 1.0              # re-normalize the mean
            protos.append(proto)
        banks[group] = (list(labels), np.vstack(protos).astype("float32"))
    return banks


def _softmax(x: np.ndarray, temp: float = 0.01) -> np.ndarray:
    z = x / temp
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


def tag_image_embedding(image_emb: np.ndarray) -> dict[str, dict]:
    """Classify a single (D,) image embedding into each attribute group.

    Returns {group: {"label": str, "confidence": float, "scores": {label: prob}}}.
    """
    img = np.asarray(image_emb, dtype="float32").reshape(-1)
    out: dict[str, dict] = {}
    for group, (labels, bank) in _group_text_banks().items():
        cos = bank @ img                      # both L2-normalized -> cosine
        probs = _softmax(cos)
        best = int(np.argmax(probs))
        out[group] = {
            "label": labels[best],
            "confidence": round(float(probs[best]), 3),
            "scores": {lbl: round(float(p), 3) for lbl, p in zip(labels, probs)},
        }
    return out


def flat_tags(tags: dict[str, dict]) -> set[str]:
    """Flatten the per-group tag dict into a set of predicted labels for overlap scoring."""
    return {grp_result["label"] for grp_result in tags.values()}
