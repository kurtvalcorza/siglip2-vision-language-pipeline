from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fetch_weights import (  # noqa: E402
    DEFAULT_DEST_DIR,
    DEFAULT_MODEL_KEY,
    MANIFEST_FORMAT,
    MANIFEST_FORMAT_VERSION,
    MANIFEST_NAME,
    generate_manifest,
    matches_patterns,
    verify_snapshot,
)

from siglip2_pipeline.config import (  # noqa: E402
    MODEL_FILENAME,
    MODEL_ID,
    MODEL_REVISION,
)


def test_matches_patterns():
    # Allowed files
    assert matches_patterns("model.safetensors")
    assert matches_patterns("config.json")
    assert matches_patterns("preprocessor_config.json")
    assert matches_patterns("special_tokens_map.json")
    assert matches_patterns("tokenizer.json")
    assert matches_patterns("tokenizer.model")
    assert matches_patterns("tokenizer_config.json")
    assert matches_patterns("README.md")
    assert matches_patterns("LICENSE")

    # Forbidden / unwanted binaries
    assert not matches_patterns("pytorch_model.bin")
    assert not matches_patterns("model.pt")
    assert not matches_patterns("model.pth")
    assert not matches_patterns("model.onnx")
    assert not matches_patterns("subfolder/model.bin")
    assert not matches_patterns(".git/config")


def test_generate_manifest_and_verify_snapshot(tmp_path: Path):
    file_a = tmp_path / "config.json"
    file_a.write_text('{"test": true}', encoding="utf-8")

    file_b = tmp_path / "model.safetensors"
    file_b.write_bytes(b"dummy safetensors content")

    manifest = generate_manifest(
        tmp_path,
        model_key=DEFAULT_MODEL_KEY,
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
    )

    assert manifest["format"] == MANIFEST_FORMAT
    assert manifest["formatVersion"] == MANIFEST_FORMAT_VERSION
    assert manifest["modelKey"] == DEFAULT_MODEL_KEY
    assert manifest["modelId"] == MODEL_ID
    assert manifest["revision"] == MODEL_REVISION
    assert len(manifest["files"]) == 2
    assert manifest["totalBytes"] == file_a.stat().st_size + file_b.stat().st_size

    # Write manifest and verify
    manifest_path = tmp_path / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    ok, errors = verify_snapshot(tmp_path)
    assert ok is True
    assert errors == []


def test_verify_snapshot_catches_tampering(tmp_path: Path):
    file_a = tmp_path / "config.json"
    file_a.write_text('{"test": true}', encoding="utf-8")

    manifest = generate_manifest(tmp_path)
    manifest_path = tmp_path / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # Tamper with file
    file_a.write_text('{"tampered": true}', encoding="utf-8")
    ok, errors = verify_snapshot(tmp_path)
    assert ok is False
    assert any("mismatch" in err.lower() for err in errors)


def test_verify_snapshot_catches_missing_file(tmp_path: Path):
    file_a = tmp_path / "config.json"
    file_a.write_text('{"test": true}', encoding="utf-8")

    manifest = generate_manifest(tmp_path)
    manifest_path = tmp_path / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    file_a.unlink()
    ok, errors = verify_snapshot(tmp_path)
    assert ok is False
    assert any("missing file" in err.lower() for err in errors)


def test_committed_base_model_snapshot_manifest_matches_weights():
    """Verify that the repository snapshot directory conforms to dimer-base-manifest.json."""
    if not DEFAULT_DEST_DIR.is_dir():
        pytest.skip(f"Weights directory {DEFAULT_DEST_DIR} does not exist")

    manifest_path = DEFAULT_DEST_DIR / MANIFEST_NAME
    assert manifest_path.is_file(), f"Missing {MANIFEST_NAME} in {DEFAULT_DEST_DIR}"

    ok, errors = verify_snapshot(DEFAULT_DEST_DIR)
    if not (DEFAULT_DEST_DIR / MODEL_FILENAME).is_file():
        # In CI/fresh clone without weights downloaded,
        # only model.safetensors and total bytes should mismatch
        assert ok is False
        assert any("Missing file: model.safetensors" in err for err in errors)
        non_weight_errors = [
            err
            for err in errors
            if "model.safetensors" not in err and "Total bytes mismatch" not in err
        ]
        assert (
            non_weight_errors == []
        ), f"Unexpected config/tokenizer errors in manifest: {non_weight_errors}"
    else:
        assert ok is True, f"Snapshot verification failed: {errors}"
