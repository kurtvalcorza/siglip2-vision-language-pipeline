"""The tutorial's pre-registered evaluation protocol (docs/siglip2-preregistration.md, 2026-10-18).

A three-config vision-only grid is scored on validation only and a challenger replaces the default only by a margin;
the fresh test is scored only when every pre-registered setting holds (SCORE_ONCE_RULE). Any other setting is
exploratory: it reads validation and the labelled exposed development test, and writes `*_exploratory_<n>.json`.
"""
# ruff: noqa: E501  -- written at the 110-column fleet width, like samples.py

from __future__ import annotations

import json
import math
import os
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config import DEFAULT_PROMPT_TEMPLATE, MODEL_ID, MODEL_REVISION
from .samples import SAMPLE_SEED, SAMPLE_SPLIT, fresh_manifest_digest, pin_manifest_digest

PREREGISTRATION = "docs/siglip2-preregistration.md"
GRID_MARGIN = 0.02  # validation text-to-image mAP on 48 photographs; a practical tolerance, not a significance threshold
DEFAULT_CONFIG = "C0"
GRID_CONFIGS: dict[str, dict[str, Any]] = {
    "C0": {"trainable_vision_layers": 2, "lr": 5e-5},
    "C1": {"trainable_vision_layers": 4, "lr": 5e-5},
    "C2": {"trainable_vision_layers": 2, "lr": 1e-5},
}
LARGER_SCOPE = "C1"  # must also beat the best other configuration by the margin
# The environment variable that masks the fresh test (CI and CPU development checks; SCORE_ONCE_RULE item 8).
FRESH_TEST_MASK_ENV = "DIMER_FRESH_TEST_MASK"
# Every setting the pre-registration fixes; `config` must also equal the grid's selection in the same run.
PREREGISTERED_SETTINGS: dict[str, Any] = {
    "data_source": "pinned sample",
    "pin_manifest_sha256": pin_manifest_digest(),
    "fresh_manifest_sha256": fresh_manifest_digest(),
    "fresh_test_masked": False,
    "split_seed": SAMPLE_SEED,
    "split_sizes": dict(SAMPLE_SPLIT),
    "adapt_seed": 0,
    "epochs": 6,
    "batch_size": 16,
    "prompt_template": DEFAULT_PROMPT_TEMPLATE,
    "class_names": "common names",
    "grid_run": True,
    # Added 2026-10-18 after the record (peer review m4): the model identity was fixed by the pre-registration's
    # scope but not checked. Adding it changes no setting; the record and its reproduction used exactly these.
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
}
# SCORE_ONCE_RULE item 8: a runtime cannot know about other runtimes, so the first two labels say "(this runtime)"; the
# evaluation of record (the first hosted pre-registered run) is stated in docs/release-verification.md.
# Once the evaluation of record is carried here (RECORDED_EVALUATION), every later identical run is labelled
# "reproduction of the pre-registered evaluation" (SCORE_ONCE_RULE item 3), in any runtime.
REPRODUCTION_OF_RECORD = "reproduction of the pre-registered evaluation"
TEST_LOOKS = ("pre-registered evaluation (this runtime)", "reproduction (this runtime)", "not scored (exploratory)", REPRODUCTION_OF_RECORD)
# The first hosted pre-registered run, as recorded in docs/release-verification.md.
RECORDED_EVALUATION: dict[str, Any] | None = {
    "run": "Colab CLI T4, 2026-10-18, commit 6df20c7, blob 1f274277",
    "record": "docs/release-verification.md",
    "evidence": "docs/execution-evidence/2026-10-18-6df20c7/",
    # The recorded fresh-test numbers (as printed, 3 decimals), so a reproduction can say whether it matches (m3).
    # The majority mAP 0.312 is a tie-order artefact of the old metric (M3); later runs print the corrected floor.
    "fresh_test": {
        "accuracy": {"majority": 0.167, "neighbour": 0.208, "frozen": 0.552, "adapted": 0.677},
        "macro_f1": {"majority": 0.048, "neighbour": 0.21, "frozen": 0.543, "adapted": 0.68},
        "t2i_map": {"majority": 0.312, "neighbour": 0.245, "frozen": 0.628, "adapted": 0.771},
        "scientific_name_prompts": {
            "accuracy": {"frozen": 0.396, "adapted": 0.458},
            "macro_f1": {"frozen": 0.35, "adapted": 0.415},
            "t2i_map": {"frozen": 0.4, "adapted": 0.399},
        },
        "verdict_as_printed": "improved (a zero-threshold label; no success criterion was pre-registered)",
    },
}
MODEL_STATE_MISMATCH = "model state is not the selected adapted model"
_ADAPTER_KEYS = ("epochs", "lr", "trainable_vision_layers", "seed", "best_epoch", "n_train", "n_val", "batch_size")


def _scope(name: str) -> tuple[int, float]:
    config = GRID_CONFIGS[name]
    return (int(config["trainable_vision_layers"]), float(config["lr"]))


def run_fingerprint(
    *,
    data_source: str,
    pin_manifest_sha256: str | None,
    split_seed: int,
    validation_digest: str,
    epochs: int,
    batch_size: int,
    seed: int,
    prompt_template: str,
) -> dict[str, Any]:
    """The inputs a grid selection is only valid for."""
    return {
        "data_source": data_source,
        "pin_manifest_sha256": pin_manifest_sha256,
        "split_seed": split_seed,
        "validation_digest": validation_digest,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "prompt_template": prompt_template,
    }


def adapter_mismatch(adapter: Mapping[str, Any] | None, adapt_result: Mapping[str, Any] | None) -> list[str]:
    """Why the model in memory is not the one Section 7b trained (empty = it is)."""
    if adapter is None or adapt_result is None:
        return [MODEL_STATE_MISMATCH + " (no adapter on the pipeline, or no Section 7b result)"]
    bad = [k for k in _ADAPTER_KEYS if adapter.get(k) != adapt_result.get(k)]
    return [f"{MODEL_STATE_MISMATCH} (differs in {bad})"] if bad else []


def select_config(
    scores: Mapping[str, float],
    frozen_val_map: float,
    *,
    margin: float = GRID_MARGIN,
    fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the pre-registered rule (preregistration section 3) to best-epoch validation mAP per config.

    A challenger qualifies only if it beats C0 by >= margin; the larger scope C1 must also beat the best other config
    by >= margin; otherwise the best qualifying smaller config, else C0. The selected config must beat the frozen
    model's validation mAP by the margin, else the frozen model is kept."""
    if set(scores) != set(GRID_CONFIGS):
        raise ValueError(f"grid scores must cover exactly {sorted(GRID_CONFIGS)}")
    eps = 1e-12
    base = float(scores[DEFAULT_CONFIG])
    qualify = [n for n in GRID_CONFIGS if n != DEFAULT_CONFIG and float(scores[n]) >= base + margin - eps]
    smaller = sorted((n for n in qualify if n != LARGER_SCOPE), key=lambda n: (-float(scores[n]), _scope(n)))
    selected, reason = DEFAULT_CONFIG, f"no challenger beat {DEFAULT_CONFIG} by {margin}"
    if smaller:
        selected, reason = smaller[0], f"{smaller[0]} beat {DEFAULT_CONFIG} by >= {margin}"
    if LARGER_SCOPE in qualify:
        best_other = max(float(scores[n]) for n in GRID_CONFIGS if n != LARGER_SCOPE)
        if float(scores[LARGER_SCOPE]) >= best_other + margin - eps:
            selected, reason = LARGER_SCOPE, f"{LARGER_SCOPE} beat {DEFAULT_CONFIG} and the best other config by >= {margin}"
        else:
            reason += f"; {LARGER_SCOPE} beat {DEFAULT_CONFIG} but not the best other config by {margin}, so the smaller scope stays"
    kept = float(scores[selected]) >= float(frozen_val_map) + margin - eps
    return {
        "selected": selected,
        "config": dict(GRID_CONFIGS[selected]),
        "adaptation_kept": kept,
        "reason": reason
        + ("" if kept else f"; adaptation did not beat frozen on validation by {margin}: the frozen model is kept"),
        "scores": {k: round(float(v), 4) for k, v in scores.items()},
        "frozen_val_map": round(float(frozen_val_map), 4),
        "margin": margin,
        "fingerprint": dict(fingerprint) if fingerprint is not None else None,
    }


def settings_differing(
    settings: Mapping[str, Any],
    selection: Mapping[str, Any] | None,
    fingerprint: Mapping[str, Any] | None = None,
    model_state: list[str] | None = None,
) -> list[str]:
    """The settings that differ from the pre-registration (empty = pre-registered): every fixed setting, the grid's
    selection and the inputs it ran on (`fingerprint`), and the model state (`model_state`, from `adapter_mismatch`)."""
    differing = [k for k, v in PREREGISTERED_SETTINGS.items() if settings.get(k) != v]
    differing += list(model_state or [])
    if selection is None:
        differing.append("config (no grid selection in this run)")
    else:
        if fingerprint is None or selection.get("fingerprint") != dict(fingerprint):
            differing.append("grid selection (it ran on other data or settings than this run; re-run Section 7)")
        used = {"trainable_vision_layers": settings.get("trainable_vision_layers"), "lr": settings.get("lr")}
        if used != selection["config"]:
            differing.append(f"config (used {used}, grid selected {selection['selected']} {selection['config']})")
    return differing


def look_at_test(
    differing: list[str],
    prior: Mapping[str, Any] | None = None,
    recorded: Mapping[str, Any] | None = RECORDED_EVALUATION,
) -> str:
    """`not scored (exploratory)` when any setting differs; `reproduction of the pre-registered evaluation` when the
    evaluation of record is carried (`recorded`); else `reproduction (this runtime)` when an earlier report in this
    runtime carries a pre-registered scoring; else `pre-registered evaluation (this runtime)`."""
    if differing:
        return TEST_LOOKS[2]
    if recorded is not None:
        return TEST_LOOKS[3]
    if (prior or {}).get("test_look") in TEST_LOOKS[:2]:
        return TEST_LOOKS[1]
    return TEST_LOOKS[0]


def output_suffix(path: str | Path, test_look: str) -> str:
    """One suffix per run for every file it writes: '' for the first pre-registered scoring, or the first reproduction
    of the record, in this runtime; `_reproduction_<n>` or `_exploratory_<n>` (first free n beside `path`) otherwise, so
    no run overwrites another."""
    path = Path(path)
    if test_look == TEST_LOOKS[0] or (test_look == TEST_LOOKS[3] and not path.exists()):
        return ""  # a reproduction of the record writes the documented paths in a fresh runtime, never over a file
    kind = "exploratory" if test_look == TEST_LOOKS[2] else "reproduction"
    n = 1
    while path.with_name(f"{path.stem}_{kind}_{n}{path.suffix}").exists():
        n += 1
    return f"_{kind}_{n}"


# --- Round 2 (peer review, 2026-10-18): wording, record check, safe writes, post-hoc display ---------------------

POST_HOC_LABEL = "post-hoc analysis, not pre-registered; added 2026-10-18 after the record"
NO_CRITERION = "no pre-registered criterion"
RECORD_TOLERANCE = 0.0005  # the record is printed to 3 decimals
TIE_ORDER_ARTEFACT = ("majority", "t2i_map")  # recorded under the old, order-dependent AP


def point_verdict(delta: float) -> str:
    """Describe a point difference without a decision criterion (none was pre-registered; peer review M2)."""
    if delta > 0:
        return f"point gain ({NO_CRITERION})"
    if delta < 0:
        return f"point loss ({NO_CRITERION})"
    return "no change"


def parse_mask(value: str | None) -> bool:
    """`DIMER_FRESH_TEST_MASK`: '1' masks, unset/''/'0' does not; anything else is refused, never read as unmasked."""
    if value is None or value.strip() in ("", "0"):
        return False
    if value.strip() == "1":
        return True
    raise ValueError(f"{FRESH_TEST_MASK_ENV}={value!r}: use 1 to mask the fresh test or 0 / unset to leave it unmasked")


def record_match(fresh_test: Mapping[str, Any] | None, recorded: Mapping[str, Any] | None = RECORDED_EVALUATION) -> dict[str, Any] | None:
    """Compare a reproduction's fresh-test numbers with the record, metric by metric (peer review m3)."""
    if not fresh_test or not recorded or "fresh_test" not in recorded:
        return None
    rows, ref = {}, recorded["fresh_test"]
    for metric in ("accuracy", "macro_f1", "t2i_map"):
        for system, want in ref[metric].items():
            if (system, metric) == TIE_ORDER_ARTEFACT or system not in fresh_test.get(metric, {}):
                continue
            got = float(fresh_test[metric][system])
            rows[f"{system}.{metric}"] = {"record": want, "this_run": round(got, 3), "match": abs(got - want) <= RECORD_TOLERANCE + 1e-9}
    for metric, by_system in ref.get("scientific_name_prompts", {}).items():
        for system, want in by_system.items():
            got = fresh_test.get("scientific_name_prompts", {}).get(metric, {}).get(system)
            if got is not None:
                rows[f"scientific.{system}.{metric}"] = {"record": want, "this_run": round(float(got), 3), "match": abs(float(got) - want) <= RECORD_TOLERANCE + 1e-9}
    return {
        "matches_record": bool(rows) and all(r["match"] for r in rows.values()),
        "metrics": rows,
        "note": "majority t2i_map is not compared: the record's 0.312 is a tie-order artefact of the old metric",
    }


def read_json_or_none(path: str | Path) -> Any:
    """An earlier report, or None when it is missing, empty or corrupt (never crashes a later run; peer review m1)."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    """Write JSON to a temporary file beside `path`, then rename it into place: `path` is never truncated."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def run_numbered(path: str | Path) -> Path:
    """`path` when it is free, else `<stem>_run<n><suffix>` with the first free n (grid and training files, m13)."""
    path = Path(path)
    if not path.exists():
        return path
    n = 2
    while path.with_name(f"{path.stem}_run{n}{path.suffix}").exists():
        n += 1
    return path.with_name(f"{path.stem}_run{n}{path.suffix}")


def exact_mcnemar(gained: int, lost: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pairs (binomial, p = 0.5)."""
    n = gained + lost
    if n == 0:
        return 1.0
    k = min(gained, lost)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_bootstrap_ci(frozen_correct: Sequence[bool], adapted_correct: Sequence[bool], *, n_boot: int = 10_000, seed: int = 20261018, level: float = 0.95) -> tuple[float, float]:
    """Percentile CI of the paired accuracy difference (adapted - frozen), resampling images with replacement."""
    if len(frozen_correct) != len(adapted_correct) or not frozen_correct:
        raise ValueError("paired correctness vectors must be non-empty and the same length")
    diffs = [int(a) - int(f) for f, a in zip(frozen_correct, adapted_correct, strict=True)]
    rng, n = random.Random(seed), len(diffs)
    stats = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo, hi = stats[int((1 - level) / 2 * n_boot)], stats[min(n_boot - 1, int((1 + level) / 2 * n_boot))]
    return (round(lo, 4), round(hi, 4))


def post_hoc_paired(frozen: Mapping[str, Any], adapted: Mapping[str, Any]) -> dict[str, Any]:
    """Display-only paired analysis of one set's per-image predictions. It never alters a verdict field."""
    if frozen["ids"] != adapted["ids"]:
        raise ValueError("frozen and adapted predictions are not over the same images in the same order")
    f_ok = [p == g for p, g in zip(frozen["predictions"], frozen["gold"], strict=True)]
    a_ok = [p == g for p, g in zip(adapted["predictions"], adapted["gold"], strict=True)]
    gained = sum(1 for f, a in zip(f_ok, a_ok, strict=True) if a and not f)
    lost = sum(1 for f, a in zip(f_ok, a_ok, strict=True) if f and not a)
    return {
        "label": POST_HOC_LABEL,
        "n": len(f_ok),
        "frozen_correct": sum(f_ok),
        "adapted_correct": sum(a_ok),
        "gained": gained,
        "lost": lost,
        "exact_mcnemar_p": round(exact_mcnemar(gained, lost), 6),
        "paired_bootstrap_95ci_accuracy_difference": list(paired_bootstrap_ci(f_ok, a_ok)),
    }


def prediction_rows(frozen: Mapping[str, Any], adapted: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Per-image rows: photo id, gold, frozen prediction and score, adapted prediction and score."""
    rows = []
    for i, rid in enumerate(frozen["ids"]):
        row = {"id": rid, "gold": frozen["gold"][i], "frozen_prediction": frozen["predictions"][i], "frozen_score": round(frozen["prediction_scores"][i], 6)}
        if adapted is not None:
            row.update({"adapted_prediction": adapted["predictions"][i], "adapted_score": round(adapted["prediction_scores"][i], 6)})
        rows.append(row)
    return rows
