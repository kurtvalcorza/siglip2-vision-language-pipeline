from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "siglip2_vision_language_colab.ipynb"
REGISTRY = ROOT / "tutorials" / "README.md"
LOCKFILE = ROOT / "requirements.lock.txt"

EXPECTED_OUTPUTS = {
    "classification.json",
    "image_embeddings.npz",
    "text_embeddings.npz",
    "similarity.csv",
    "retrieval.json",
    "metrics.json",
    "new_data_classification.json",
    "provenance.json",
}


def _load_notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source_text(notebook: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )


def _normalized_source(notebook: dict) -> str:
    return " ".join(_source_text(notebook).split())


def test_release_notebook_declares_multi_capability_profile() -> None:
    notebook = _load_notebook()
    dimer = notebook["metadata"]["dimer"]
    assert dimer["notebook_profile"] == "MULTI-CAPABILITY"
    assert dimer["notebook_spec_version"] == "1.0"

    registry = REGISTRY.read_text(encoding="utf-8")
    assert "siglip2_vision_language_colab.ipynb" in registry
    assert "`MULTI-CAPABILITY`" in registry


def test_release_notebook_has_required_learning_contract_markers() -> None:
    source = _normalized_source(_load_notebook())
    required = (
        "No gradient training, fine-tuning, in-context conditioning",
        "object detection",
        "semantic segmentation",
        "OCR",
        "caption generation",
        "not calibrated probabilities",
        "downstream application",
        "synthetic tutorial/smoke assets",
        "## Interpretation, limits, and next steps",
    )
    for marker in required:
        assert marker in source


def test_release_notebook_has_gated_byod_and_new_data_paths() -> None:
    source = _source_text(_load_notebook())
    assert "ENABLE_BYOD = False" in source
    assert "colab_files.upload()" in source
    assert "BYOD_PATH" in source
    assert "BYOD file is not a decodable image" in source
    assert "## 10. Default new-data inference" in source
    assert "new_data_scores = pipe.zero_shot_classify" in source


def test_release_notebook_exports_every_demonstrated_capability() -> None:
    source = _source_text(_load_notebook())
    for filename in EXPECTED_OUTPUTS:
        assert filename in source

    assert "image_ids=np.asarray" in source
    assert "text_ids=np.asarray" in source
    assert "write_provenance" in source


def test_release_notebook_prints_runtime_and_immutable_identity() -> None:
    source = _source_text(_load_notebook())
    required = (
        'print("Python:"',
        'print("PyTorch:"',
        'print("Transformers:"',
        'print("Device:"',
        'print("Model ID:"',
        'print("Revision:"',
        'print("Weight SHA-256:"',
    )
    for marker in required:
        assert marker in source


def test_release_notebook_matches_cpu_reference_lock() -> None:
    source = _source_text(_load_notebook())
    registry = REGISTRY.read_text(encoding="utf-8")
    lockfile = LOCKFILE.read_text(encoding="utf-8")

    assert "torch==2.14.0+cpu" in lockfile
    assert 'DEVICE = "cpu"' in source
    assert "CPU-only" in source
    assert "CPU-only reference" in registry
    assert "CUDA optional" not in registry
    assert "torch.cuda.is_available()" not in source


def test_release_notebook_source_is_clean() -> None:
    notebook = _load_notebook()
    source = _source_text(notebook)

    forbidden = ("TODO", "TBD", "FIXME")
    for marker in forbidden:
        assert marker not in source

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs", []) == []
