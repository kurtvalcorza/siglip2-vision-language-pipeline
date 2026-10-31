"""DAT19/DAT20: a refused BYOD input renders one `Refused: <file>: <reason>` line, no traceback.

Executes the committed Section 4 BYOD block with the real loader (or stubs). Covers the scripted
hosted refusal input (`BYOD_PATH='/etc/hostname'`, a file not ending in .zip), a missing path, an
empty path outside Colab, a cancelled upload, a non-zip `.zip`, the in-cell invalidation, and
that a genuine error keeps its own type.
"""

from __future__ import annotations

import json
import os
import sys
import types
import zipfile
from pathlib import Path

import pytest

from siglip2_pipeline.samples import load_byod_dataset

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "siglip2_vision_language_colab.ipynb"
REFUSAL_CELL = "siglip2_vision_language-24"
STEM = "siglip2_vision_language"


def _byod_cell() -> dict:
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    found = [
        c for c in cells if c["cell_type"] == "code" and "def refuse_byod" in "".join(c["source"])
    ]
    assert len(found) == 1
    return found[0]


def _block() -> str:
    source = "".join(_byod_cell()["source"])
    return source[source.index("class BYODRefusal(") : source.index("else:\n    corpus_files")]


def _run(path: str, loader=load_byod_dataset):
    namespace = {
        "USE_BYOD": True,
        "BYOD_PATH": path,
        "Path": Path,
        "SPLIT_SEED": 42,
        "os": os,
        "json": json,
        "zipfile": zipfile,
        "validate_dataset": lambda part, min_records=1: None,
        "load_byod_dataset": loader,
        "split_dataset": lambda records, seed: {"train": records, "test": records},
        "splits": {"train": ["stale"]},
        "train_records": ["stale"],
    }
    exec(compile(_block(), "<byod>", "exec"), namespace)
    return namespace


def _refused(path: str, **kwargs) -> str:
    with pytest.raises(ValueError) as caught:
        _run(path, **kwargs)
    exc = caught.value
    assert getattr(exc, "dimer_refusal", False), exc
    rendered = exc._render_traceback_()
    assert len(rendered) == 1 and "\n" not in rendered[0], rendered
    assert rendered[0].startswith("Refused: ") and "Traceback" not in rendered[0]
    return rendered[0]


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / f"{STEM}_result.json").write_text("{}")
    monkeypatch.setitem(sys.modules, "google.colab", None)
    return tmp_path


def test_scripted_hosted_input_a_file_not_ending_in_zip(workdir):
    hostname = workdir / "hostname"
    hostname.write_text("colab-vm\n")
    line = _refused(str(hostname))
    assert line.startswith("Refused: hostname: expected a zip of images"), line


def test_missing_path_and_empty_path_off_colab(workdir):
    missing = workdir / "nope.zip"
    assert _refused(str(missing)).startswith(f"Refused: {missing}: BYOD path does not exist")
    assert _refused("").startswith("Refused: (no file): BYOD is on but BYOD_PATH is empty")


def test_cancelled_upload(workdir, monkeypatch):
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: {})
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    assert _refused("").startswith("Refused: (no file): upload exactly one zip")


def test_a_zip_suffix_that_is_not_a_zip(workdir):
    fake = workdir / "photos.zip"
    fake.write_bytes(b"not a zip")
    assert _refused(str(fake)).startswith("Refused: photos.zip: ")


def test_refusal_invalidates_earlier_results_in_the_cell(workdir):
    with pytest.raises(ValueError):
        _run(str(workdir / "nope.zip"))
    marker = json.loads((workdir / "outputs" / f"{STEM}_run_invalidated.json").read_text())
    assert marker["files_from_an_earlier_run"] == [f"{STEM}_result.json"]


def test_genuine_error_keeps_its_type(workdir):
    good = workdir / "ok.zip"
    good.write_bytes(b"PK")

    def broken(path):
        raise RuntimeError("internal bug")

    with pytest.raises(RuntimeError, match="internal bug") as caught:
        _run(str(good), loader=broken)
    assert not getattr(caught.value, "dimer_refusal", False)


def test_refusal_cell_id():
    assert _byod_cell()["id"] == REFUSAL_CELL
