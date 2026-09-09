from __future__ import annotations

import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_PROMPT_TEMPLATE,
    MODEL_FILENAME,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SHA256,
    MODEL_SIZE_BYTES,
    TEXT_MAX_LENGTH,
)

_RUNTIME_PACKAGES = (
    "huggingface-hub",
    "numpy",
    "pillow",
    "safetensors",
    "torch",
    "transformers",
)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def build_provenance(
    *,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    include_runtime: bool = True,
) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "weight_file": MODEL_FILENAME,
            "weight_sha256": MODEL_SHA256,
            "weight_size_bytes": MODEL_SIZE_BYTES,
        },
        "processor": {
            "input_resolution": [224, 224],
            "text_max_length": TEXT_MAX_LENGTH,
            "lowercase_model_bound_text": True,
        },
        "inference": {
            "prompt_template": prompt_template,
            "zero_shot_score_semantics": "independent_sigmoid_not_calibrated_probability",
            "embedding_normalization": "l2",
            "similarity": "cosine_via_normalized_dot_product",
        },
    }
    if include_runtime:
        provenance["runtime"] = {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": sys.platform,
            "packages": {name: _package_version(name) for name in _RUNTIME_PACKAGES},
        }
    return provenance


def write_provenance(
    path: str | Path,
    *,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    include_runtime: bool = True,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            build_provenance(
                prompt_template=prompt_template,
                include_runtime=include_runtime,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return target
