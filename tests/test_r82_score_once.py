"""Row 82 review fixes: pre-registered grid, fresh test, score-once wiring (SCORE_ONCE_RULE 1-8),
DAT9/REL13 prose, spec 2.5 and the epoch-selection fix. Offline: no checkpoint, no network."""
# ruff: noqa: E501  -- parametrised settings and probe strings are kept on one line

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
from pathlib import Path

import pytest

from siglip2_pipeline import pipeline as pipeline_module
from siglip2_pipeline import protocol, samples
from siglip2_pipeline.protocol import (
    GRID_CONFIGS,
    PREREGISTERED_SETTINGS,
    RECORDED_EVALUATION,
    TEST_LOOKS,
    adapter_mismatch,
    look_at_test,
    output_suffix,
    run_fingerprint,
    select_config,
    settings_differing,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "siglip2_vision_language_colab.ipynb"
FP = run_fingerprint(
    data_source="pinned sample",
    pin_manifest_sha256=samples.pin_manifest_digest(),
    split_seed=42,
    validation_digest="v" * 64,
    epochs=6,
    batch_size=16,
    seed=0,
    prompt_template=PREREGISTERED_SETTINGS["prompt_template"],
)


def _settings(**overrides):
    base = {**PREREGISTERED_SETTINGS, **GRID_CONFIGS["C0"]}
    base.update(overrides)
    return base


def _selection(name="C0"):
    scores = {"C0": 0.87, "C1": 0.87, "C2": 0.86}
    selection = select_config(scores, 0.68, fingerprint=FP)
    assert selection["selected"] == "C0"
    if name != "C0":
        selection = {**selection, "selected": name, "config": dict(GRID_CONFIGS[name])}
    return selection


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _notebook_text() -> str:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"])


# --- grid rule (preregistration section 3) ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scores", "frozen", "selected", "kept"),
    [
        ({"C0": 0.87, "C1": 0.88, "C2": 0.86}, 0.68, "C0", True),  # inside the margin: keep C0
        ({"C0": 0.85, "C1": 0.80, "C2": 0.88}, 0.68, "C2", True),  # C2 beats C0 by >= 0.02
        ({"C0": 0.85, "C1": 0.90, "C2": 0.80}, 0.68, "C1", True),  # C1 beats C0 and the best other config
        ({"C0": 0.85, "C1": 0.89, "C2": 0.88}, 0.68, "C2", True),  # C1 not 0.02 over C2: smaller scope
        ({"C0": 0.69, "C1": 0.69, "C2": 0.69}, 0.68, "C0", False),  # not 0.02 over frozen: keep frozen
    ],
)
def test_select_config_applies_the_preregistered_rule(scores, frozen, selected, kept):
    result = select_config(scores, frozen)
    assert result["selected"] == selected
    assert result["adaptation_kept"] is kept
    assert result["config"] == GRID_CONFIGS[selected]


def test_grid_is_the_preregistered_vision_only_grid():
    assert GRID_CONFIGS == {
        "C0": {"trainable_vision_layers": 2, "lr": 5e-5},
        "C1": {"trainable_vision_layers": 4, "lr": 5e-5},
        "C2": {"trainable_vision_layers": 2, "lr": 1e-5},
    }
    doc = _source("docs/siglip2-preregistration.md")
    assert samples.pin_manifest_digest() in doc and "0.02" in doc
    assert protocol.GRID_MARGIN == 0.02 and protocol.DEFAULT_CONFIG == "C0"
    with pytest.raises(ValueError):
        select_config({"C0": 0.8}, 0.5)


# --- SCORE_ONCE_RULE item 7: a changed setting never scores the fresh test; an identical re-run is a reproduction --


@pytest.mark.parametrize(
    "change",
    [
        {"epochs": 7},
        {"batch_size": 8},
        {"adapt_seed": 1},
        {"split_seed": 43},
        {"data_source": "BYOD"},
        {"pin_manifest_sha256": "0" * 64},
        {"fresh_manifest_sha256": None},
        {"fresh_test_masked": True},
        {"prompt_template": "A photo of a {label}."},
        {"grid_run": False},
        {"lr": 1e-5},
        {"trainable_vision_layers": 4},
    ],
)
def test_any_changed_setting_makes_the_run_exploratory(change):
    differing = settings_differing(_settings(**change), _selection(), FP, [])
    assert differing
    assert look_at_test(differing) == "not scored (exploratory)"


def test_identical_settings_are_preregistered_then_a_reproduction(tmp_path):
    differing = settings_differing(_settings(), _selection(), FP, [])
    assert differing == []
    first = look_at_test(differing, None, recorded=None)  # before the evaluation of record was carried
    assert first == TEST_LOOKS[0] == "pre-registered evaluation (this runtime)"
    assert look_at_test(differing, {"test_look": first}, recorded=None) == "reproduction (this runtime)"
    report = tmp_path / "x_evaluation_report.json"
    assert output_suffix(report, TEST_LOOKS[0]) == ""
    assert output_suffix(report, TEST_LOOKS[1]) == "_reproduction_1"
    (tmp_path / "x_evaluation_report_exploratory_1.json").write_text("{}")
    assert output_suffix(report, TEST_LOOKS[2]) == "_exploratory_2"


def test_grid_selection_must_come_from_this_run_and_the_model_state_must_match():
    other = {**FP, "validation_digest": "w" * 64}
    assert any("grid selection" in d for d in settings_differing(_settings(), _selection(), other, []))
    assert any("no grid selection" in d for d in settings_differing(_settings(), None, FP, []))
    assert any("config (used" in d for d in settings_differing(_settings(), _selection("C2"), FP, []))
    adapter = {"epochs": 6, "lr": 5e-5, "trainable_vision_layers": 2, "seed": 0, "best_epoch": 4, "n_train": 216, "n_val": 48, "batch_size": 16}
    assert adapter_mismatch(adapter, dict(adapter)) == []
    assert adapter_mismatch(None, adapter) and adapter_mismatch(adapter, {**adapter, "best_epoch": 3})
    assert RECORDED_EVALUATION is not None and RECORDED_EVALUATION["run"] == "Colab CLI T4, 2026-10-18, commit 6df20c7, blob 1f274277"
    assert RECORDED_EVALUATION["record"] == "docs/release-verification.md"


def test_a_later_identical_run_is_a_reproduction_of_the_recorded_evaluation(tmp_path):
    differing = settings_differing(_settings(), _selection(), FP, [])
    assert differing == []
    assert look_at_test(differing) == TEST_LOOKS[3] == "reproduction of the pre-registered evaluation"
    assert look_at_test(differing, {"test_look": TEST_LOOKS[3]}) == TEST_LOOKS[3]
    assert look_at_test(["epochs"]) == "not scored (exploratory)"  # the record never makes a changed run count
    report = tmp_path / "x_evaluation_report.json"
    assert output_suffix(report, TEST_LOOKS[3]) == ""  # fresh runtime: the documented paths
    report.write_text("{}")
    assert output_suffix(report, TEST_LOOKS[3]) == "_reproduction_1"  # never over an earlier file
    assert "reproduction of the pre-registered evaluation" in _notebook_text()


# --- fresh test pins (preregistration section 4) ----------------------------------------------------------------


def test_fresh_test_pins_are_balanced_new_and_disjoint():
    rows = samples.FRESH_TEST_RECORDS
    assert len(rows) == 96
    assert {k: sum(r[1] == k for r in rows) for k in samples.SPECIES} == dict.fromkeys(samples.SPECIES, 16)
    observers = [r[4] for r in rows]
    assert len(set(observers)) == 96
    assert not set(observers) & {r[4] for r in samples.SAMPLE_RECORDS}
    assert not {r[2] for r in rows} & {r[2] for r in samples.SAMPLE_RECORDS}
    assert not {r[8] for r in rows} & {r[6] for r in samples.SAMPLE_RECORDS}
    assert all(r[5] >= samples.FRESH_TEST_CREATED_FROM for r in rows)
    log = json.loads(_source("docs/siglip2-fresh-test-draw.json"))
    assert log["manifest_sha256"] == samples.fresh_manifest_digest()
    assert PREREGISTERED_SETTINGS["fresh_manifest_sha256"] == samples.fresh_manifest_digest()
    assert log["balanced_size_per_species"] == 16


def test_fetch_fresh_test_refuses_a_mismatch(tmp_path):
    with pytest.raises(ValueError, match="pinned"):
        samples.fetch_fresh_test(cache_dir=tmp_path, fetcher=lambda url: b"not the photo")


# --- pipeline corrections ---------------------------------------------------------------------------------------


def test_epoch_selection_uses_validation_t2i_map():
    source = inspect.getsource(pipeline_module.Siglip2Pipeline.adapt)
    assert 'entry["val"]["t2i_map"]' in source
    assert 'entry["val"]["accuracy"] if' not in source


def test_restore_base_puts_the_pinned_tensors_back():
    torch = pytest.importorskip("torch")
    pipe = object.__new__(pipeline_module.Siglip2Pipeline)
    pipe.model = torch.nn.Linear(3, 2)
    pipe.adapter = {"x": 1}
    original = pipe.model.weight.detach().clone()
    pipe._remember_base(["weight"])
    with torch.no_grad():
        pipe.model.weight.add_(1.0)
    pipe._remember_base(["weight"])  # a second adaptation never overwrites the remembered base
    pipe.restore_base()
    assert torch.equal(pipe.model.weight, original) and pipe.adapter is None


# --- notebook wiring and prose ----------------------------------------------------------------------------------


def test_notebook_scores_the_fresh_test_only_under_the_preregistered_settings():
    text = _notebook_text()
    for needle in ("preregistered = not differing", "test_look", "settings_differing", "_exploratory_", "RECORDED_EVALUATION", "adapter_mismatch", "FRESH_TEST_MASK_ENV"):
        assert needle in text, needle
    gate = text.index("if preregistered:\n    # THE fresh test")
    uses = [i for i in range(len(text)) if text.startswith("evaluate(fresh_records", i)]
    assert uses and all(i > gate for i in uses)
    assert "held-out mAP" not in text


@pytest.mark.parametrize(
    "stale",
    [
        "never told about",
        "recorded Kaggle T4 run",
        "recorded workstation pre-flight",
        "424 s",
        "are in the model card",
        "gained on the held-out split",
        "never used for training or epoch selection",
        "differ in the second decimal",
        "web-scale captions describe precisely",
    ],
)
def test_stale_claims_are_gone(stale):
    assert stale not in _notebook_text()
    assert stale not in _source("src/siglip2_pipeline/samples.py")


def test_dat9_statement_and_spec_25():
    text = _notebook_text()
    assert "cannot be ruled out" in text and "2025-02-20" in text and "exposed development test" in text
    meta = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["metadata"]["dimer"]
    assert meta["notebook_spec"] == "2.5" and meta["notebook_kind"] == "TUTORIAL"
    assert "Notice." in "".join(json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"][0]["source"])


def test_ci_and_the_local_runner_mask_the_fresh_test():
    assert "DIMER_FRESH_TEST_MASK: '1'" in _source(".github/workflows/ci.yml")
    assert "DIMER_FRESH_TEST_MASK" in _source("tools/run_notebook.py")


def test_stamp_names_a_commit_carrying_the_modules():
    meta = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["metadata"]["dimer"]["generated_from"]
    revision = meta["revision"]
    assert "src/siglip2_pipeline/protocol.py" in meta["modules"]
    for rel in meta["modules"]:
        try:
            blob = subprocess.run(["git", "show", f"{revision}:{rel}"], cwd=ROOT, capture_output=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            pytest.skip(f"revision {revision} is not in this checkout (shallow clone)")
        assert blob == (ROOT / rel).read_bytes(), rel
        assert hashlib.sha256(blob).hexdigest()
