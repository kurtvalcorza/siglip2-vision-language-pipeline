from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoProcessor

from .config import (
    ALLOWED_CHECKPOINT_FILES,
    MODEL_FILENAME,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SHA256,
    MODEL_SIZE_BYTES,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(snapshot_path: str | Path) -> Path:
    root = Path(snapshot_path)
    weight_path = root / MODEL_FILENAME
    if not weight_path.is_file():
        raise RuntimeError(f"Pinned checkpoint is missing {MODEL_FILENAME}")

    unsafe = sorted(str(path.relative_to(root)) for path in root.rglob("*.bin"))
    if unsafe:
        raise RuntimeError(f"Refusing pickle-style checkpoint files: {unsafe}")

    size = weight_path.stat().st_size
    if size != MODEL_SIZE_BYTES:
        raise RuntimeError(
            f"Unexpected {MODEL_FILENAME} size: {size}; expected {MODEL_SIZE_BYTES}"
        )

    digest = _sha256(weight_path)
    if digest != MODEL_SHA256:
        raise RuntimeError(
            f"Unexpected {MODEL_FILENAME} SHA-256: {digest}; expected {MODEL_SHA256}"
        )
    return root


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_components(
    *,
    device: str | torch.device | None = None,
    cache_dir: str | Path | None = None,
) -> tuple[Any, Any, torch.device, Path]:
    """Acquire, verify, and load the one supported SigLIP 2 checkpoint."""

    snapshot_path = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        allow_patterns=list(ALLOWED_CHECKPOINT_FILES),
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    verified = verify_checkpoint(snapshot_path)
    target_device = _resolve_device(device)

    processor = AutoProcessor.from_pretrained(
        verified,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModel.from_pretrained(
        verified,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    )
    model = model.eval().to(target_device)
    return model, processor, target_device, verified
