"""Row 82 round 2 (peer review of PR #19, 2026-10-18): tie-aware AP, atomic report writes, keep-frozen export,
reproduction-vs-record check, post-hoc paired display. Offline: no checkpoint, no fresh-test photographs."""
# ruff: noqa: E501  -- probe strings are kept on one line

from __future__ import annotations

import itertools
import json
import random
import statistics
from pathlib import Path

import numpy as np
import pytest

from siglip2_pipeline import protocol
from siglip2_pipeline.metrics import average_precision, classification_metrics, majority_baseline

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "siglip2_vision_language_colab.ipynb"


def _code() -> str:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def _brute_expected_ap(scores, relevant):
    values = []
    for perm in itertools.permutations(range(len(scores))):
        order = sorted(perm, key=lambda i: -scores[i])  # Python's sort is stable: ties keep the permuted order
        hits = total = 0
        for rank, i in enumerate(order, start=1):
            if relevant[i]:
                hits += 1
                total += hits / rank
        values.append(total / sum(relevant))
    return statistics.mean(values)


# --- M3: tied scores ------------------------------------------------------------------------------------------


def test_tied_ap_is_the_expectation_over_tie_orders():
    rng = random.Random(7)
    for _ in range(25):
        scores = [rng.choice([0.0, 0.5, 1.0]) for _ in range(6)]
        relevant = [rng.random() < 0.5 for _ in range(6)]
        relevant[0] = True
        assert average_precision(scores, relevant) == pytest.approx(_brute_expected_ap(scores, relevant), abs=1e-12)


def test_untied_ap_is_unchanged():
    assert average_precision([0.9, 0.8, 0.7, 0.6], [True, False, True, False]) == pytest.approx((1 + 2 / 3) / 2)
    assert average_precision([0.1, 0.2, 0.3], [False, False, False]) == 0.0


def test_majority_floor_map_no_longer_depends_on_record_order():
    """Grouped by species (as FRESH_TEST_RECORDS is) the old metric gave 0.312; the floor is now 0.2030 in any order."""
    classes = [f"c{i}" for i in range(6)]
    grouped = [{"label": c} for c in classes for _ in range(16)]
    shuffled = list(grouped)
    random.Random(1).shuffle(shuffled)
    train = [{"label": c} for c in classes for _ in range(36)]
    floors = [majority_baseline(train, test, classes)["t2i_map"] for test in (grouped, shuffled)]
    assert floors[0] == pytest.approx(floors[1], abs=1e-12)
    assert floors[0] == pytest.approx(0.2030, abs=5e-4)
    constant = classification_metrics(np.ones((96, 6)), [i // 16 for i in range(96)], classes)
    assert constant["t2i_map"] == pytest.approx(floors[0])


# --- m1: atomic report writes ---------------------------------------------------------------------------------


def test_report_write_is_atomic_and_a_bad_report_never_breaks_a_later_run(tmp_path):
    report = tmp_path / "x_evaluation_report.json"
    protocol.atomic_write_json(report, {"test_look": "first"})
    with pytest.raises(TypeError):
        protocol.atomic_write_json(report, {"bad": object()})  # cannot be serialised: the file must survive intact
    assert json.loads(report.read_text(encoding="utf-8")) == {"test_look": "first"}
    assert not list(tmp_path.glob("*.tmp"))
    report.write_text("", encoding="utf-8")
    assert protocol.read_json_or_none(report) is None
    report.write_text("{not json", encoding="utf-8")
    assert protocol.read_json_or_none(report) is None
    assert protocol.read_json_or_none(tmp_path / "missing.json") is None
    code = _code()
    assert "prior_report = read_json_or_none(report_path)" in code
    assert "atomic_write_json(written, evaluation_report_payload)" in code
    assert "json.dump(evaluation_report_payload" not in code


def test_run_numbered_never_overwrites(tmp_path):
    path = tmp_path / "g.json"
    assert protocol.run_numbered(path) == path
    path.write_text("{}")
    assert protocol.run_numbered(path).name == "g_run2.json"


# --- M2: verdict wording, post-hoc display --------------------------------------------------------------------


def test_verdicts_name_no_criterion():
    assert protocol.point_verdict(0.1) == "point gain (no pre-registered criterion)"
    assert protocol.point_verdict(-0.1) == "point loss (no pre-registered criterion)"
    assert protocol.point_verdict(0.0) == "no change"
    assert "'improved' if delta > 0" not in _code()


def test_post_hoc_paired_is_labelled_and_display_only():
    assert protocol.exact_mcnemar(15, 3) == pytest.approx(0.007538, abs=1e-6)
    assert protocol.exact_mcnemar(0, 0) == 1.0
    frozen = {"ids": ["a", "b", "c", "d"], "gold": ["x", "x", "y", "y"], "predictions": ["x", "y", "x", "y"], "prediction_scores": [1, 1, 1, 1]}
    adapted = {**frozen, "predictions": ["x", "x", "y", "x"]}
    result = protocol.post_hoc_paired(frozen, adapted)
    assert result["label"] == "post-hoc analysis, not pre-registered; added 2026-10-18 after the record"
    assert (result["gained"], result["lost"], result["frozen_correct"], result["adapted_correct"]) == (2, 1, 2, 3)
    assert result["paired_bootstrap_95ci_accuracy_difference"] == list(protocol.paired_bootstrap_ci([1, 0, 0, 1], [1, 1, 1, 0]))
    rows = protocol.prediction_rows(frozen, adapted)
    assert rows[1] == {"id": "b", "gold": "x", "frozen_prediction": "y", "frozen_score": 1, "adapted_prediction": "x", "adapted_score": 1}
    code = _code()
    compare = code[code.index("def compare(") : code.index("\n\n\nadapted_val = ")]
    assert "post_hoc" not in compare  # the verdict fields never read the post-hoc analysis
    assert code.index("table['verdict']") < code.index("post_hoc = post_hoc_paired(")


# --- m2: keep-frozen; m3: record check; m4: model identity; s1: mask parsing -------------------------------------


def test_keep_frozen_never_puts_the_adapter_back_or_exports_it():
    code = _code()
    assert "    if adaptation_kept:\n        pipe.load_artifact(parked)" in code
    assert "if not adaptation_kept and pipe.adapter is not None:\n    pipe.restore_base()" in code
    assert "if pipe.adapter is None:\n    print('Adaptation was not kept (keep-frozen rule)" in code


def test_reproduction_is_checked_against_the_recorded_numbers():
    recorded = protocol.RECORDED_EVALUATION["fresh_test"]
    same = {m: dict(recorded[m]) for m in ("accuracy", "macro_f1", "t2i_map")}
    same["t2i_map"]["majority"] = 0.203  # the corrected floor is not compared with the artefact
    same["scientific_name_prompts"] = recorded["scientific_name_prompts"]
    check = protocol.record_match(same)
    assert check["matches_record"] is True and "majority.t2i_map" not in check["metrics"]
    other = {**same, "accuracy": {**same["accuracy"], "adapted": 0.667}}
    assert protocol.record_match(other)["matches_record"] is False
    assert "record_check = record_match(fresh_test) if test_look == REPRODUCTION_OF_RECORD else None" in _code()


def test_model_identity_is_checked_and_no_setting_value_changed():
    s = protocol.PREREGISTERED_SETTINGS
    assert s["model_id"] == "google/siglip2-base-patch16-224"
    assert (s["epochs"], s["batch_size"], s["adapt_seed"], s["split_seed"], s["grid_run"], s["fresh_test_masked"]) == (6, 16, 0, 42, True, False)
    assert s["split_sizes"] == {"train": 36, "validation": 8, "test": 16}
    assert protocol.GRID_CONFIGS["C0"] == {"trainable_vision_layers": 2, "lr": 5e-5} and protocol.GRID_MARGIN == 0.02
    assert "run_settings = {'model_id': MODEL_ID, 'model_revision': MODEL_REVISION," in _code()


@pytest.mark.parametrize(("value", "masked"), [(None, False), ("", False), ("0", False), ("1", True), (" 1 ", True)])
def test_mask_values(value, masked):
    assert protocol.parse_mask(value) is masked


def test_unknown_mask_value_is_refused():
    with pytest.raises(ValueError, match="DIMER_FRESH_TEST_MASK"):
        protocol.parse_mask("yes")
