from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import siglip2_pipeline.model as model_module
from siglip2_pipeline.config import (
    ALLOWED_CHECKPOINT_FILES,
    MODEL_FILENAME,
    MODEL_ID,
    MODEL_REVISION,
)


def test_model_identity_and_revision_are_explicit():
    assert MODEL_ID == "google/siglip2-base-patch16-224"
    assert MODEL_REVISION == "5ffaac51d5e2f3367f7dab0cad4be4cb07c0caa2"
    assert MODEL_FILENAME == "model.safetensors"
    assert not any(path.endswith(".bin") for path in ALLOWED_CHECKPOINT_FILES)


def test_verify_checkpoint_checks_size_and_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    payload = b"verified-siglip2-test-payload"
    weight = tmp_path / MODEL_FILENAME
    weight.write_bytes(payload)
    monkeypatch.setattr(model_module, "MODEL_SIZE_BYTES", len(payload))
    monkeypatch.setattr(model_module, "MODEL_SHA256", hashlib.sha256(payload).hexdigest())

    assert model_module.verify_checkpoint(tmp_path) == tmp_path


def test_verify_checkpoint_rejects_pickle_weight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    payload = b"verified-siglip2-test-payload"
    (tmp_path / MODEL_FILENAME).write_bytes(payload)
    (tmp_path / "pytorch_model.bin").write_bytes(b"pickle-style")
    monkeypatch.setattr(model_module, "MODEL_SIZE_BYTES", len(payload))
    monkeypatch.setattr(model_module, "MODEL_SHA256", hashlib.sha256(payload).hexdigest())

    with pytest.raises(RuntimeError, match="pickle-style"):
        model_module.verify_checkpoint(tmp_path)
