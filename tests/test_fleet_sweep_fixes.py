"""Regression tests for the 2026-10-05 fleet-sweep fixes (SWP-R restart guard, SWP-G guided layer and the
repository-specific SWP-A / SWP-F / SWP-B fixes recorded in docs/reviews/2026-10-05-fleet-sweep/).

Every test needs only CI's dependencies. The notebooks' own cell sources are executed with stand-ins; no model, no
network and no torch are needed.
"""
# ruff: noqa: E501

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ['siglip2_vision_language_colab']
LOCK = ROOT / 'tutorials/requirements-colab.lock.txt'
MIN_PREDICT = {'siglip2_vision_language_colab': 6}


@functools.cache
def _nb_text(name: str) -> str:
    return (ROOT / "tutorials" / f"{name}.ipynb").read_text(encoding="utf-8")


def _nb(name: str) -> dict:
    return json.loads(_nb_text(name))


def _code_cells(notebook: dict) -> list[dict]:
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def _cell(notebook: dict, marker: str) -> str:
    found = [c["source"] for c in _code_cells(notebook) if marker in c["source"]]
    assert len(found) == 1, f"expected one code cell containing {marker!r}, found {len(found)}"
    return found[0]


def _build():
    spec = importlib.util.spec_from_file_location("_sweep_build_notebook", ROOT / "tools" / "build_notebook.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    return build


# --- SWP-R: no in-kernel install, no restart, idempotent Section 1 (shared by every notebook) -------------------------


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_r_nothing_is_pip_installed_into_the_kernel_and_no_restart_is_requested(name):
    notebook = _nb(name)
    code = "\n".join(c["source"] for c in _code_cells(notebook))
    assert "pip install" not in code and "'-m', 'pip'" not in code
    assert "restart the runtime" not in json.dumps(notebook).lower()
    kernel = [c for c in _code_cells(notebook) if "# dimer: kernel cell" in c["source"]]
    assert len(kernel) == 1, "exactly one cell may run in the kernel"
    source = kernel[0]["source"]
    for needed in ("'--require-hashes', '--only-binary', ':all:'", "'--managed-python'", "UV_SHA256", "LOCK_SHA256"):
        assert needed in source
    # The worker gets a clean interpreter environment and a non-interactive matplotlib backend.
    for needed in ('MPLBACKEND="Agg"', '"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"'):
        assert needed in source
    assert notebook["metadata"]["dimer"]["environment"].startswith("isolated hash-locked uv environment")


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_r_carried_lock_is_the_committed_lock_and_pins_every_runtime_pin(name):
    source = _cell(_nb(name), "# dimer: kernel cell")
    lock_text = LOCK.read_text(encoding="utf-8")
    digest = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    assert digest == hashlib.sha256(lock_text.encode("utf-8")).hexdigest()
    assert f"LOCK_TEXT = r'''{lock_text}'''" in source
    build = _build()
    build.check_lock(build._pins(ROOT), lock_text)  # raises SystemExit on any drift


class _Shell:
    def __init__(self) -> None:
        self.input_transformers_cleanup: list = []


def test_swp_r_section_1_is_idempotent_and_keeps_the_live_worker(tmp_path, monkeypatch, capsys):
    """Re-running the Section 1 cell reuses the matching environment (no download) and keeps the live worker, so the
    variables later cells created survive and the cells after it are not stranded."""
    source = _cell(_nb(NOTEBOOKS[0]), "# dimer: kernel cell")
    lock_sha = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "python").symlink_to(sys.executable)  # stand-in interpreter for the isolated environment
    (env / ".dimer-lock-sha256").write_text(lock_sha + "\n", encoding="utf-8")
    monkeypatch.setenv("DIMER_ISOLATED_ENV", str(env))
    monkeypatch.delenv("DIMER_NOTEBOOK_CI_PREINSTALLED", raising=False)
    shell = _Shell()
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: shell
    ipython_display = types.ModuleType("IPython.display")
    ipython_display.display = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)

    def no_download(*args, **kwargs):
        raise AssertionError("a matching environment must be reused, not downloaded again")

    monkeypatch.setattr("urllib.request.urlopen", no_download)
    namespace: dict = {"__name__": "__main__"}
    exec(compile(source, "<section 1>", "exec"), namespace)
    runtime = namespace["_DIMER_ISOLATED_RUNTIME"]
    try:
        assert "'reused': True" in capsys.readouterr().out
        runtime.run("learner_value = 41 + 1\n")
        exec(compile(source, "<section 1 again>", "exec"), namespace)  # the learner re-runs Section 1 on its own
        assert namespace["_DIMER_ISOLATED_RUNTIME"] is runtime and runtime.alive()
        assert [t.__name__ for t in shell.input_transformers_cleanup] == ["_route_to_isolated_runtime"]
        runtime.run("print('value', learner_value)\n")
        assert "value 42" in capsys.readouterr().out
        assert namespace["_route_to_isolated_runtime"](["x = 1\n"]) == ["_DIMER_ISOLATED_RUNTIME.run('x = 1\\n')\n"]
        assert namespace["_route_to_isolated_runtime"]([source]) == [source]  # the kernel cell itself stays in the kernel
        with pytest.raises(RuntimeError, match="ZeroDivisionError"):
            runtime.run("1 / 0\n")
    finally:
        runtime.close()


# --- SWP-G: the guided layer and infrastructure labelling (shared) ----------------------------------------------------


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_g_guided_layer_is_present(name):
    notebook = _nb(name)
    markdown = "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")
    for heading in (
        "**Who this notebook is for.**",
        "**Input → Model → Output.**",
        "**How to use this notebook.**",
        "**Roadmap:**",
        "## Troubleshooting",
        "## Glossary",
        "## Conclusion (your notes)",
        "## Change one thing (next experiments)",
    ):
        assert heading in markdown, heading
    assert markdown.count("**Predict:**") >= MIN_PREDICT[name]
    assert markdown.count("<details><summary>Check your reasoning</summary>") >= MIN_PREDICT[name]
    assert "Run all completes in one pass" in markdown


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_g_infrastructure_cells_are_labelled_and_collapsed(name):
    cells = _code_cells(_nb(name))
    infra = [c for c in cells if c["metadata"].get("cellView") == "form"]
    assert any("# dimer: kernel cell" in c["source"] for c in infra)
    assert any(c["metadata"].get("dimer", {}).get("embedded_module") for c in infra)
    assert any(c["source"].startswith("# @title Infrastructure: stage and digest-verify") for c in infra)
    learner = [c for c in cells if c["metadata"].get("cellView") != "form"]
    assert learner and all("# @title Infrastructure" not in c["source"] for c in learner)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_g_no_template_placeholders_leak(name):
    notebook = _nb(name)
    text = "\n".join(
        c["source"] for c in notebook["cells"] if not c.get("metadata", {}).get("dimer", {}).get("embedded_module")
    )
    for leftover in ("{{", "{MODEL_ID}", "{stem}", "@P:"):
        assert leftover not in text, leftover


def _colab(monkeypatch, upload) -> None:
    google = types.ModuleType("google")
    google.__path__ = []
    colab_mod = types.ModuleType("google.colab")
    files = types.ModuleType("google.colab.files")
    files.upload = upload
    colab_mod.files = files
    google.colab = colab_mod
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab_mod)
    monkeypatch.setitem(sys.modules, "google.colab.files", files)


def _no_colab(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "google.colab", None)  # import fails as it does on Kaggle / Jupyter


# --- repository-specific: SWP-G checkpoint numbers, SWP-A verdicts, SWP-F frozen reload, SWP-B BYOD path -------------

import os  # noqa: E402

NB = NOTEBOOKS[0]
RECORD = ROOT / "docs" / "release-verification.md"


def test_swp_g_checkpoint_answers_quote_the_recorded_runs():
    """Every number in a Check-your-reasoning answer names its run (R82-m1, REL13): the 2026-10-18 Colab CLI T4
    pre-registered evaluation of record (6df20c7 / blob 1f274277), the 2026-10-09 Colab CLI T4 run of the previous blob,
    both recorded in docs/release-verification.md, or the 2026-10-18 CPU build record."""
    markdown = "\n".join(c["source"] for c in _nb(NB)["cells"] if c["cell_type"] == "markdown")
    checks = re.findall(r"<details><summary>Check your reasoning</summary>(.*?)</details>", markdown, re.S)
    assert len(checks) == 7
    record = RECORD.read_text(encoding="utf-8")
    hosted = ("0.167", "0.26", "0.76", "0.792", "0.758", "0.793", "0.723", "0.871", "0.479", "0.51", "0.459", "0.453", "0.467", "0.492", "0.44", "0.75", "0.62", "0.94", "0.680", "0.874", "0.854", "0.863", "0.876", "0.872")
    for number in hosted:
        assert number in record, number
    assert all(n in checks[2] for n in ("0.167", "0.26", "0.76", "0.723", "0.479", "9 October 2026"))
    fresh = ("0.552", "0.677", "0.543", "0.68", "0.628", "0.771", "chipping_sparrow", "song_sparrow")
    for number in fresh:
        assert number in record, number
    assert all(n in checks[5] for n in ("0.552", "0.677", "0.543", "0.68", "0.628", "0.771", "n = 96", "Clopper", "[0.447, 0.654]", "[0.574, 0.769]", "2 to 12 of 16", "0.203", "tie-order artefact", "point gain (no pre-registered criterion)", "photographers", "Chipping Sparrow", "Song Sparrow", "0.76", "0.723", "0.871", "0.792"))
    assert "No fresh-test number" not in checks[5] and "pending the hosted run" not in checks[3]
    assert all(n in checks[3] for n in ("0.8758", "0.8767", "0.8667", "0.0009", "0.02"))
    assert "0.680" in checks[4] and "epoch 4" in checks[4]
    for answer in (checks[1], checks[2], checks[3], checks[4], checks[5], checks[6]):
        assert "pre-registered evaluation of record of 18 October 2026" in answer and "1f274277" in answer
    assert "@@" not in markdown
    assert "Kaggle T4 run" not in markdown and "pre-flight" not in markdown


def _compare_fn():
    s8 = _cell(_nb(NB), "def compare(majority, neighbour, frozen, adapted, frozen_sci, adapted_sci):")
    from siglip2_pipeline.protocol import point_verdict

    ns = {"point_verdict": point_verdict}
    exec(s8[s8.index("def fields(result):") : s8.index("\n\n\nadapted_val = ")], ns)
    return ns


def test_swp_a_no_quality_assert_remains_and_verdicts_reach_the_report():
    code = "\n".join(c["source"] for c in _code_cells(_nb(NB)))
    quality = [line for line in code.splitlines() if line.strip().startswith("assert ") and re.search(r"\['(t2i_map|accuracy|macro_f1)'\]", line)]
    assert quality == []
    assert "assert parity['identical_rows'] == parity['of']" in code  # reload parity stays a hard check
    ns = _compare_fn()

    def result(t2i_map, accuracy):
        return {"t2i_map": t2i_map, "accuracy": accuracy, "macro_f1": accuracy, "per_class": {"a": {"n": 1, "recall": accuracy, "ap": t2i_map}}}

    ns["classes"] = ["a"]
    ns["METRICS"] = ("accuracy", "macro_f1", "t2i_map")
    majority, neighbour = result(0.203, 0.167), result(0.2, 0.26)
    cases = (
        (result(0.723, 0.76), result(0.871, 0.792), ("above the baselines", "point gain (no pre-registered criterion)", "point gain (no pre-registered criterion)")),
        (result(0.2, 0.2), result(0.2, 0.1), ("not above the baselines", "no change", "point loss (no pre-registered criterion)")),
        (result(0.723, 0.76), result(0.6, 0.76), ("above the baselines", "point loss (no pre-registered criterion)", "no change")),
    )
    for frozen, adapted, expected in cases:
        table = ns["compare"](majority, neighbour, frozen, adapted, frozen, adapted)
        verdict = table["verdict"]
        assert (verdict["frozen_vs_baselines"], verdict["adapted_vs_frozen_t2i_map"], verdict["adapted_vs_frozen_accuracy"]) == expected
    kept_frozen = ns["compare"](majority, neighbour, result(0.7, 0.7), None, result(0.5, 0.5), None)
    assert "adaptation not kept" in kept_frozen["verdict"]["adapted_vs_frozen_t2i_map"]
    s8 = _cell(_nb(NB), "def compare(majority, neighbour, frozen, adapted, frozen_sci, adapted_sci):")
    assert s8.index("development_view = ") < s8.index("atomic_write_json(written, evaluation_report_payload)")
    assert "'comparison': comparison," in _cell(_nb(NB), "result_payload = {")


def test_swp_f_frozen_pipeline_restores_an_adapted_pipeline_and_keeps_a_frozen_one(capsys):
    s6 = _cell(_nb(NB), "def frozen_pipeline():")
    helper = s6[s6.index("def frozen_pipeline():") : s6.index("\n\n\nfrozen_pipeline()")]
    calls = []

    class Pipe:
        adapter = {"best_epoch": 4}

        def restore_base(self):
            calls.append("restore")
            self.adapter = None

    ns = {"pipe": Pipe()}
    exec(helper, ns)
    ns["frozen_pipeline"]()
    assert calls == ["restore"] and ns["pipe"].adapter is None
    assert "base tensors restored" in capsys.readouterr().out
    ns["frozen_pipeline"]()
    assert calls == ["restore"]  # a frozen pipeline is kept


def test_swp_f_sections_6_7_and_7b_restore_the_base_before_scoring_or_training():
    s6 = _cell(_nb(NB), "frozen_dev = pipe.evaluate(dev_records, classes=classes, class_names_map=display_names)")
    assert s6.index("\nfrozen_pipeline()\n") < s6.index("frozen_dev = pipe.evaluate(")
    grid = _cell(_nb(NB), "grid_selection = select_config(")
    assert grid.index("frozen_pipeline()") < grid.index("run = pipe.adapt(") < grid.index("pipe.restore_base()")
    s7 = _cell(_nb(NB), "adapt_result = pipe.adapt(")
    assert s7.index("frozen_pipeline()") < s7.index("adapt_result = pipe.adapt(")


def _byod_file(nb: dict):
    source = _cell(nb, "def byod_file(")
    helper = source[source.index("def byod_file(") : source.index("if USE_BYOD:")]
    ns = {"os": os, "Path": Path}
    exec(helper, ns)
    return ns["byod_file"]


def test_swp_b_byod_path_reads_a_zip_and_refusals_name_the_file(tmp_path, monkeypatch):
    helper = _byod_file(_nb(NB))
    good = tmp_path / "mine.zip"
    good.write_bytes(b"PK")
    assert helper(str(good), "zip", (".zip",)) == good
    with pytest.raises(FileNotFoundError, match=r"missing.zip: BYOD path does not exist"):
        helper(str(tmp_path / "missing.zip"), "zip")
    bad = tmp_path / "wrong.tar"
    bad.write_bytes(b"x")
    with pytest.raises(ValueError, match=r"wrong.tar: expected a"):
        helper(str(bad), "zip", (".zip",))
    _no_colab(monkeypatch)
    with pytest.raises(RuntimeError, match=r"^\(no file\): BYOD is on.*only in Google Colab"):
        helper("", "zip")


def test_swp_b_cancelled_upload_is_named_and_one_upload_is_saved(tmp_path, monkeypatch):
    helper = _byod_file(_nb(NB))
    _colab(monkeypatch, lambda: {})
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match=r"^\(no file\): upload exactly one"):
        helper("", "zip")
    _colab(monkeypatch, lambda: {"up.zip": b"1"})
    saved = helper("", "zip", (".zip",))
    assert saved.read_bytes() == b"1" and saved.parent.name == "work"


def test_swp_b_byod_gate_is_off_by_default_and_has_a_path_field():
    code = "\n".join(c["source"] for c in _code_cells(_nb(NB)))
    assert "USE_BYOD = False  # @param" in code and "BYOD_PATH = ''  # @param" in code
    assert "file_name, payload = next(iter(uploaded.items()))" not in code
