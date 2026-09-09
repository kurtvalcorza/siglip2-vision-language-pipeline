from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from siglip2_pipeline import MODEL_ID, MODEL_REVISION, MODEL_SHA256, build_provenance

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_SPEC = ROOT / "examples" / "sample-data"


def _load_generator():
    path = SAMPLE_SPEC / "generate_samples.py"
    spec = importlib.util.spec_from_file_location("siglip2_sample_generator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lock_parity_script_passes() -> None:
    subprocess.run([sys.executable, "scripts/check_lock.py"], cwd=ROOT, check=True)


def test_sample_generator_matches_manifest_and_pillow(tmp_path: Path) -> None:
    expected = {}
    for line in (SAMPLE_SPEC / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        expected[name] = digest

    generator = _load_generator()
    written = generator.generate(tmp_path)

    assert {path.name for path in written} == set(expected)
    for path in written:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected[path.name]
        with Image.open(path) as image:
            assert image.mode == "RGB"
            assert image.size == (32, 32)


def test_provenance_identity_and_semantics() -> None:
    record = build_provenance(include_runtime=False)
    assert record["model"]["id"] == MODEL_ID
    assert record["model"]["revision"] == MODEL_REVISION
    assert record["model"]["weight_sha256"] == MODEL_SHA256
    assert record["processor"]["lowercase_model_bound_text"] is True
    assert record["inference"]["zero_shot_score_semantics"].startswith("independent_sigmoid")
    assert record["inference"]["embedding_normalization"] == "l2"


def test_colab_notebook_is_json_and_python_cells_compile() -> None:
    path = ROOT / "tutorials" / "siglip2_vision_language_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    for index, cell in enumerate(code_cells):
        source = "".join(cell.get("source", []))
        compile(source, f"{path}#cell-{index}", "exec")
