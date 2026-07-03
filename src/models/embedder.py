"""CLIP (ViT) embedding backbone - the heart of the AI inference layer.

One model gives us three capabilities the assignment needs:
  * image embeddings for visual similarity search,
  * text embeddings for zero-shot attribute tagging and multi-modal search,
  * a shared space so image and text vectors are directly comparable.

All embeddings are L2-normalized, so a dot product == cosine similarity. That lets
FAISS use a simple inner-product index (`IndexFlatIP`) as an exact cosine index.

A ResNet50 CNN baseline (`ResNetEmbedder` / `get_resnet_embedder`) is provided to
support the CNN-vs-ViT comparison - see `scripts/compare_models.py`.
"""
from __future__ import annotations

import functools
from typing import Sequence

import numpy as np
from PIL import Image

from src import config
from src.utils.logging import get_logger

log = get_logger()


class ClipEmbedder:
    """Thin, lazy wrapper around an open_clip ViT model."""

    def __init__(self, model_name: str = config.CLIP_MODEL_NAME,
                 pretrained: str = config.CLIP_PRETRAINED,
                 device: str = config.DEVICE) -> None:
        import open_clip  # imported lazily so the module imports without torch installed
        import torch

        self._torch = torch
        self.device = device
        log.info(f"Loading CLIP {model_name} ({pretrained}) on {device} ...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.embed_dim = config.EMBED_DIM

    # ------------------------------------------------------------------ #
    # Image
    # ------------------------------------------------------------------ #
    def encode_image(self, images: Image.Image | Sequence[Image.Image]) -> np.ndarray:
        """Encode one or more PIL images -> (N, D) float32 L2-normalized array."""
        if isinstance(images, Image.Image):
            images = [images]
        with self._torch.inference_mode():
            batch = self._torch.stack(
                [self.preprocess(im.convert("RGB")) for im in images]
            ).to(self.device)
            feats = self.model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.cpu().numpy().astype("float32")

    # ------------------------------------------------------------------ #
    # Text
    # ------------------------------------------------------------------ #
    def encode_text(self, texts: str | Sequence[str]) -> np.ndarray:
        """Encode one or more strings -> (N, D) float32 L2-normalized array."""
        if isinstance(texts, str):
            texts = [texts]
        with self._torch.inference_mode():
            tokens = self.tokenizer(list(texts)).to(self.device)
            feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.cpu().numpy().astype("float32")


class ResNetEmbedder:
    """ResNet50 (ImageNet) penultimate-layer features - a CNN baseline for the
    CNN-vs-ViT comparison. Produces 2048-d global-average-pooled, L2-normalized
    vectors. Not used by the live search path (CLIP is the backbone); this exists
    to demonstrate *why* CLIP-ViT retrieves better than generic ImageNet features.
    """

    embed_dim = 2048

    def __init__(self, device: str = config.DEVICE) -> None:
        import torch
        import torchvision

        self._torch = torch
        self.device = device
        log.info(f"Loading ResNet50 (ImageNet) baseline on {device} ...")
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        net = torchvision.models.resnet50(weights=weights)
        net.fc = torch.nn.Identity()          # keep the 2048-d pooled features
        self.model = net.to(device).eval()
        self.preprocess = weights.transforms()

    def encode_image(self, images: Image.Image | Sequence[Image.Image]) -> np.ndarray:
        if isinstance(images, Image.Image):
            images = [images]
        with self._torch.inference_mode():
            batch = self._torch.stack(
                [self.preprocess(im.convert("RGB")) for im in images]
            ).to(self.device)
            feats = self.model(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.cpu().numpy().astype("float32")


@functools.lru_cache(maxsize=1)
def get_embedder() -> ClipEmbedder:
    """Process-wide singleton so the model loads exactly once."""
    return ClipEmbedder()


@functools.lru_cache(maxsize=1)
def get_resnet_embedder() -> "ResNetEmbedder":
    """Process-wide ResNet50 baseline singleton."""
    return ResNetEmbedder()


def normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a 1-D or 2-D array (row-wise), guarding against zero vectors."""
    vec = np.asarray(vec, dtype="float32")
    if vec.ndim == 1:
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else vec
    n = np.linalg.norm(vec, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return vec / n
