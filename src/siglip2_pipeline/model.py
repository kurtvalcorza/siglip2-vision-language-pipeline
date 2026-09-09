from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoProcessor

from .config import (
    ALLOWED_CHECKPOINT_FILES,
    DEFAULT_MODEL_KEY,
    MODEL_FILENAME,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SHA256,
    MODEL_SIZE_BYTES,
    UNSAFE_WEIGHT_EXTENSIONS,
)

MANIFEST_NAME = "dimer-base-manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(
    snapshot_path: str | Path,
    *,
    require_configs: bool = False,
    return_manifest_verified: bool = False,
) -> Path | tuple[Path, bool]:
    root = Path(snapshot_path)
    if not root.is_dir():
        raise RuntimeError(f"Checkpoint directory does not exist: {root}")

    weight_path = root / MODEL_FILENAME
    if not weight_path.is_file():
        raise RuntimeError(f"Pinned checkpoint is missing {MODEL_FILENAME}")

    unsafe = sorted(
        p.name
        for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in UNSAFE_WEIGHT_EXTENSIONS
    )
    if unsafe:
        raise RuntimeError(f"Refusing unsafe weight files: {unsafe}")

    manifest_path = root / MANIFEST_NAME
    manifest_verified = False

    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Corrupt manifest {MANIFEST_NAME}: {exc}") from exc

        files = manifest.get("files") or []
        if not files:
            raise RuntimeError(f"Manifest {MANIFEST_NAME} contains no files")

        for entry in files:
            rel_path = entry.get("path")
            if not rel_path:
                continue
            target = root / rel_path
            if not target.is_file():
                raise RuntimeError(f"Manifest file missing: {rel_path}")
            exp_bytes = entry.get("bytes")
            if exp_bytes is not None and target.stat().st_size != exp_bytes:
                raise RuntimeError(
                    f"Size mismatch for {rel_path}: {target.stat().st_size} != {exp_bytes}"
                )
            exp_sha = entry.get("sha256")
            if exp_sha is not None and _sha256(target) != exp_sha:
                raise RuntimeError(f"SHA-256 mismatch for {rel_path}")

        manifest_verified = True

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

    if require_configs:
        for req in ("config.json", "preprocessor_config.json"):
            if not (root / req).is_file():
                raise RuntimeError(f"Missing required configuration file: {req}")

    if return_manifest_verified:
        return root, manifest_verified
    return root


def resolve_weights_path(
    weights_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> tuple[Path, str]:
    """Resolve weights path with precedence:

    1. Explicit argument `weights_path` -> 'explicit_path'
    2. Environment variable `SIGLIP2_WEIGHTS_DIR` -> 'env_var'
    3. Source checkout convention `weights/siglip2-base-patch16-224` -> 'repo_offline'
       (only if pyproject.toml exists at repo root and weights/ contains model.safetensors)
    4. Hugging Face Hub snapshot download -> 'hf_hub'
    """
    if weights_path is not None:
        return Path(weights_path), "explicit_path"

    env_dir = os.environ.get("SIGLIP2_WEIGHTS_DIR")
    if env_dir:
        return Path(env_dir), "env_var"

    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "pyproject.toml").is_file():
        repo_weights = repo_root / "weights" / DEFAULT_MODEL_KEY
        if (repo_weights / MODEL_FILENAME).is_file():
            return repo_weights, "repo_offline"

    hub_path = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            allow_patterns=list(ALLOWED_CHECKPOINT_FILES),
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
    )
    return hub_path, "hf_hub"


_resolve_weights_path = resolve_weights_path


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_components(
    *,
    device: str | torch.device | None = None,
    cache_dir: str | Path | None = None,
    weights_path: str | Path | None = None,
    return_metadata: bool = False,
) -> tuple[Any, Any, torch.device, Path] | tuple[Any, Any, torch.device, Path, dict[str, Any]]:
    """Acquire, verify, and load the one supported SigLIP 2 checkpoint."""

    candidate_path, source = resolve_weights_path(
        weights_path=weights_path,
        cache_dir=cache_dir,
    )

    verified, manifest_verified = verify_checkpoint(
        candidate_path,
        require_configs=True,
        return_manifest_verified=True,
    )
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

    weight_file = verified / MODEL_FILENAME
    metadata: dict[str, Any] = {
        "checkpoint_path": verified,
        "checkpoint_source": source,
        "manifest_verified": manifest_verified,
        "weight_sha256": _sha256(weight_file),
        "weight_size_bytes": weight_file.stat().st_size,
        "device": str(target_device),
    }

    if return_metadata:
        return model, processor, target_device, verified, metadata
    return model, processor, target_device, verified
