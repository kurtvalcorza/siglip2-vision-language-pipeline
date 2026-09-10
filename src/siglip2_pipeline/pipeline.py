from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import DEFAULT_PROMPT_TEMPLATE, TEXT_MAX_LENGTH
from .model import load_components

ImageInput = str | Path | bytes | Image.Image


@dataclass(frozen=True, slots=True)
class ClassificationScore:
    label: str
    score: float


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    index: int
    score: float


def _coerce_image(value: ImageInput) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, bytes):
        with Image.open(BytesIO(value)) as image:
            return image.convert("RGB")
    path = Path(value)
    raw = str(value)
    if raw.lower().startswith(("http://", "https://")):
        raise ValueError("Remote image URLs are not accepted; provide local bytes or a local path")
    with Image.open(path) as image:
        return image.convert("RGB")


def _require_texts(values: Sequence[str], *, name: str) -> list[str]:
    result = [value.strip() for value in values]
    if not result or any(not value for value in result):
        raise ValueError(f"{name} must contain at least one non-empty string")
    return result


def _siglip2_texts(values: Sequence[str]) -> list[str]:
    """Match SigLIP 2 training-time text lowercasing before tokenization."""

    return [value.lower() for value in values]


def _move_batch(batch: Any, device: torch.device) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


class Siglip2Pipeline:
    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        device: str | torch.device = "cpu",
        checkpoint_path: Path | str | None = None,
        checkpoint_source: str | None = None,
        manifest_verified: bool = False,
        weight_sha256: str | None = None,
        weight_size_bytes: int | None = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.checkpoint_source = checkpoint_source
        self.manifest_verified = manifest_verified
        self.weight_sha256 = weight_sha256
        self.weight_size_bytes = weight_size_bytes

    @classmethod
    def from_pretrained(
        cls,
        *,
        device: str | torch.device | None = None,
        cache_dir: str | Path | None = None,
        weights_path: str | Path | None = None,
    ) -> Siglip2Pipeline:
        model, processor, target_device, _, metadata = load_components(
            device=device,
            cache_dir=cache_dir,
            weights_path=weights_path,
            return_metadata=True,
        )
        return cls(
            model,
            processor,
            device=target_device,
            checkpoint_path=metadata.get("checkpoint_path"),
            checkpoint_source=metadata.get("checkpoint_source"),
            manifest_verified=metadata.get("manifest_verified", False),
            weight_sha256=metadata.get("weight_sha256"),
            weight_size_bytes=metadata.get("weight_size_bytes"),
        )

    def zero_shot_classify(
        self,
        image: ImageInput,
        labels: Sequence[str],
        *,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ) -> list[ClassificationScore]:
        if not labels:
            raise ValueError("labels must contain at least one non-empty string")
        original_labels = list(labels)
        cleaned_labels = [label.strip() for label in original_labels]
        if any(not label for label in cleaned_labels):
            raise ValueError("labels must contain at least one non-empty string")
        if "{label}" not in prompt_template:
            raise ValueError("prompt_template must contain the literal {label} placeholder")

        prompts = [prompt_template.format(label=label) for label in cleaned_labels]
        batch = self.processor(
            text=_siglip2_texts(prompts),
            images=[_coerce_image(image)],
            padding="max_length",
            max_length=TEXT_MAX_LENGTH,
            return_tensors="pt",
        )
        batch = _move_batch(batch, self.device)
        with torch.inference_mode():
            logits = self.model(**batch).logits_per_image[0]
            scores = torch.sigmoid(logits).detach().cpu().tolist()

        ranked = [
            ClassificationScore(label=orig_label, score=float(score))
            for orig_label, score in zip(original_labels, scores, strict=True)
        ]
        return sorted(ranked, key=lambda item: item.score, reverse=True)

    def embed_image(self, images: Sequence[ImageInput]) -> np.ndarray:
        if not images:
            raise ValueError("images must contain at least one image")
        batch = self.processor(
            images=[_coerce_image(image) for image in images],
            return_tensors="pt",
        )
        batch = _move_batch(batch, self.device)
        with torch.inference_mode():
            features = self.model.get_image_features(**batch)
            features = F.normalize(features, p=2, dim=-1)
        return features.detach().cpu().numpy().astype(np.float32, copy=False)

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        texts_list = _require_texts(texts, name="texts")
        batch = self.processor(
            text=_siglip2_texts(texts_list),
            padding="max_length",
            max_length=TEXT_MAX_LENGTH,
            return_tensors="pt",
        )
        batch = _move_batch(batch, self.device)
        with torch.inference_mode():
            features = self.model.get_text_features(**batch)
            features = F.normalize(features, p=2, dim=-1)
        return features.detach().cpu().numpy().astype(np.float32, copy=False)

    def similarity(self, images: Sequence[ImageInput], texts: Sequence[str]) -> np.ndarray:
        image_features = self.embed_image(images)
        text_features = self.embed_text(texts)
        return image_features @ text_features.T

    def retrieve(
        self,
        query: str,
        images: Sequence[ImageInput],
        *,
        top_k: int = 5,
    ) -> list[RetrievalHit]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not query.strip():
            raise ValueError("query must be a non-empty string")
        if not images:
            raise ValueError("images must contain at least one image")

        scores = self.similarity(images, [query])[:, 0]
        order = np.argsort(-scores, kind="stable")[: min(top_k, len(images))]
        return [RetrievalHit(index=int(index), score=float(scores[index])) for index in order]


def load_pipeline(
    *,
    device: str | torch.device | None = None,
    cache_dir: str | Path | None = None,
    weights_path: str | Path | None = None,
) -> Siglip2Pipeline:
    return Siglip2Pipeline.from_pretrained(
        device=device,
        cache_dir=cache_dir,
        weights_path=weights_path,
    )
