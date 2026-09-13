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

from .config import DEFAULT_PROMPT_TEMPLATE, MODEL_ID, MODEL_REVISION, TEXT_MAX_LENGTH
from .model import load_components, stage_missing_files, verify_snapshot

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
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> Siglip2Pipeline:
        """Load the one supported checkpoint.

        ``weights_dir`` names a fleet snapshot directory holding ``dimer-base-manifest.json``
        (normally ``weights/<DEFAULT_MODEL_KEY>/``): manifest entries that are absent are staged
        with :func:`stage_missing_files` (only when ``allow_download=True``), the directory is
        verified against the manifest and the pinned digests by :func:`verify_snapshot`, and the
        same :func:`load_components` call then loads it as an explicit path (source
        ``explicit_path``; nothing goes through ``snapshot_download``).
        """
        if weights_dir is not None:
            if weights_path is not None:
                raise ValueError("pass either weights_dir or weights_path, not both")
            stage_missing_files(weights_dir, allow_download=allow_download)
            verify_snapshot(weights_dir)
            weights_path = weights_dir
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

# --------------------------------------------------------------------------
# Role stages (DIMER NOTEBOOK_SPEC 1.1 DAT24 / EVAL21)
#
# `validate_inputs` is the public validation stage: it applies exactly the checks the four core
# methods apply (image coercion through `_coerce_image`, non-empty texts through `_require_texts`,
# the label / prompt-template / top_k checks of `zero_shot_classify` and `retrieve`) and reports
# what was proven as an input manifest. `evaluation_report` is the public evaluation stage: its
# metric ids are this module's own helpers `top1_accuracy` and `recall_at_1` (the tutorial's
# sanity metrics, extracted from the notebook), and it always produces a report.
# --------------------------------------------------------------------------

#: The input contract and every named ceiling, in one readable structure.
INPUT_SCHEMA: dict[str, Any] = {
    "images": (
        "PIL.Image.Image, raw bytes, or a local path decodable by Pillow; any mode, converted to "
        "RGB; remote URLs are refused"
    ),
    "texts": "non-empty strings (labels, queries, or free text); lowercased before tokenization",
    "text_max_length": TEXT_MAX_LENGTH,
    "prompt_template": DEFAULT_PROMPT_TEMPLATE,
    "top_k": [1, None],
    "preprocessing": (
        "the pinned processor resizes every image to 224x224 and normalizes it; text is tokenized "
        f"to max_length={TEXT_MAX_LENGTH} with padding, longer text is truncated"
    ),
}


def _check_inputs(
    images: Sequence[ImageInput] | ImageInput | None,
    texts: Sequence[str] | None,
    *,
    top_k: int | None,
    prompt_template: str,
) -> tuple[list[Image.Image], list[str] | None]:
    """The checks the core methods apply, in their order, raising exactly what they raise."""
    if images is None:
        coerced: list[Image.Image] = []
    else:
        if isinstance(images, str | Path | bytes | Image.Image):
            images = [images]
        if not images:
            raise ValueError("images must contain at least one image")
        coerced = [_coerce_image(image) for image in images]
    checked_texts: list[str] | None = None
    if texts is not None:
        if not texts:
            raise ValueError("labels must contain at least one non-empty string")
        checked_texts = _require_texts(texts, name="texts")
    if "{label}" not in prompt_template:
        raise ValueError("prompt_template must contain the literal {label} placeholder")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1")
    return coerced, checked_texts


def validate_inputs(
    images: Sequence[ImageInput] | ImageInput | None,
    texts: Sequence[str] | None = None,
    *,
    top_k: int | None = None,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, per-input observations, verdict).

    Rejection is reported by raising exactly as ``zero_shot_classify`` / ``embed_image`` /
    ``embed_text`` / ``retrieve`` would; a caller that wants the finding recorded catches the
    exception and stores ``str(exc)`` under ``findings``.
    """
    coerced, checked_texts = _check_inputs(
        images, texts, top_k=top_k, prompt_template=prompt_template
    )
    if names is not None and len(names) != len(coerced):
        raise ValueError("names must have one entry per image")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [
            {
                "id": names[i] if names else f"image-{i}",
                "mode": image.mode,
                "size": list(image.size),
            }
            for i, image in enumerate(coerced)
        ],
        "texts": checked_texts,
        "top_k": top_k,
        "prompt_template": prompt_template,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def top1_accuracy(
    classifications: Sequence[Sequence[ClassificationScore]], expected_labels: Sequence[str]
) -> float:
    """Fraction of images whose top-scoring label equals the expected label (tutorial sanity)."""
    if len(classifications) != len(expected_labels):
        raise ValueError("classifications and expected_labels must have the same length")
    if not classifications:
        raise ValueError("classifications must not be empty")
    pairs = zip(classifications, expected_labels, strict=True)
    hits = sum(int(scores[0].label == expected) for scores, expected in pairs)
    return hits / len(classifications)


def recall_at_1(
    retrievals: Sequence[Sequence[RetrievalHit]], expected_indices: Sequence[int]
) -> float:
    """Fraction of queries whose top-ranked hit is the expected image index (tutorial sanity)."""
    if len(retrievals) != len(expected_indices):
        raise ValueError("retrievals and expected_indices must have the same length")
    if not retrievals:
        raise ValueError("retrievals must not be empty")
    pairs = zip(retrievals, expected_indices, strict=True)
    hits = sum(int(ranked[0].index == expected) for ranked, expected in pairs)
    return hits / len(retrievals)


def evaluation_report(
    result: dict[str, Any],
    targets: dict[str, Any] | None = None,
    *,
    sample_kind: str = "synthetic",
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    ``result`` gathers what the notebook demonstrated: ``classifications`` (one
    ``zero_shot_classify`` list per image), ``retrievals`` (one ``retrieve`` list per query), and
    optionally ``embedding_shapes`` / ``similarity_shape``. ``targets`` supplies the ground truth:
    ``labels`` (the expected label per classified image) and/or ``retrieval_indices`` (the expected
    image index per query). With targets the report carries ``top1_accuracy`` (against the
    fixed-class baseline 1/n_labels) and/or ``recall_at_1`` with the verdict ``sample-sanity``;
    without them the verdict is ``not-measurable``. Embeddings and similarity matrices are
    representations with no intrinsic metric and are always reported as not measurable.
    """
    classifications = list(result.get("classifications") or [])
    retrievals = list(result.get("retrievals") or [])
    base: dict[str, Any] = {
        "task": "zero-shot image classification, image/text embeddings, similarity and retrieval",
        "score_semantics": (
            "classification and similarity scores are independent SigLIP sigmoids / cosine "
            "similarities, not calibrated probabilities and not summing to one; the decision rule "
            "used for the sanity metrics is argmax; no threshold is shipped"
        ),
        "sample_kind": sample_kind,
        "n_classified_images": len(classifications),
        "n_retrieval_queries": len(retrievals),
        "embeddings": (
            "representations, not predictions: no intrinsic accuracy metric; "
            f"shapes {result.get('embedding_shapes')}"
        ),
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    labels = (targets or {}).get("labels")
    indices = (targets or {}).get("retrieval_indices")
    metrics: list[dict[str, Any]] = []
    baselines: list[dict[str, Any]] = []
    estimation = "single synthetic sample set; no dispersion estimate"
    if labels is not None and classifications:
        n_candidates = len(classifications[0])
        metrics.append(
            {
                "id": "top1_accuracy",
                "value": top1_accuracy(classifications, list(labels)),
                "n_candidate_labels": n_candidates,
                "estimation": estimation,
            }
        )
        baselines.append(
            {
                "id": "fixed_class_baseline",
                "metric": "top1_accuracy",
                "value": 1.0 / n_candidates if n_candidates else None,
                "note": "always predicting one fixed candidate label",
            }
        )
    if indices is not None and retrievals:
        metrics.append(
            {
                "id": "recall_at_1",
                "value": recall_at_1(retrievals, list(indices)),
                "n_gallery_images": len(result.get("gallery_ids") or []) or None,
                "estimation": estimation,
            }
        )
    if not metrics:
        return {
            **base,
            "metrics": [],
            "verdict": "not-measurable",
            "reason": "no expected labels or expected retrieval indices were supplied",
            "needs": (
                "labelled images from the deployment domain (one expected label per image among "
                "the candidate labels) scored with top1_accuracy against the fixed-class baseline, "
                "and/or query-image relevance pairs scored with recall_at_1"
            ),
        }
    return {
        **base,
        "metrics": metrics,
        "baselines": baselines,
        "verdict": "sample-sanity",
        "reason": (
            f"{len(classifications)} labelled image(s) and {len(retrievals)} query(ies) from the "
            "tutorial sample; not a benchmark"
        ),
        "needs": (
            "a labelled evaluation set from the deployment domain, with prompt wording fixed in "
            "advance, for any generalisable accuracy or retrieval claim"
        ),
    }
