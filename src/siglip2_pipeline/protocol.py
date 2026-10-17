"""The tutorial's pre-registered evaluation protocol (docs/siglip2-preregistration.md, 2026-10-18).

A three-config vision-only grid is scored on validation only and a challenger replaces the default only by a margin;
the fresh test is scored only when every pre-registered setting holds (SCORE_ONCE_RULE). Any other setting is
exploratory: it reads validation and the labelled exposed development test, and writes `*_exploratory_<n>.json`.
"""
# ruff: noqa: E501  -- written at the 110-column fleet width, like samples.py

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import DEFAULT_PROMPT_TEMPLATE
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
}
# SCORE_ONCE_RULE item 8: a runtime cannot know about other runtimes, so the labels say "(this runtime)"; which run is the
# evaluation of record (the first hosted pre-registered run) is stated only in docs/release-verification.md.
TEST_LOOKS = ("pre-registered evaluation (this runtime)", "reproduction (this runtime)", "not scored (exploratory)")
# The first hosted pre-registered run, once recorded in docs/release-verification.md (None until then).
RECORDED_EVALUATION: dict[str, Any] | None = None
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


def look_at_test(differing: list[str], prior: Mapping[str, Any] | None = None) -> str:
    """`not scored (exploratory)` when any setting differs; `reproduction (this runtime)` when an earlier report in this
    runtime carries a pre-registered scoring; else `pre-registered evaluation (this runtime)`."""
    if differing:
        return TEST_LOOKS[2]
    if (prior or {}).get("test_look") in TEST_LOOKS[:2]:
        return TEST_LOOKS[1]
    return TEST_LOOKS[0]


def output_suffix(path: str | Path, test_look: str) -> str:
    """One suffix per run for every file it writes: '' for the first pre-registered scoring in this runtime,
    `_reproduction_<n>` or `_exploratory_<n>` (first free n beside `path`) otherwise, so no run overwrites another."""
    if test_look == TEST_LOOKS[0]:
        return ""
    kind = "reproduction" if test_look == TEST_LOOKS[1] else "exploratory"
    path = Path(path)
    n = 1
    while path.with_name(f"{path.stem}_{kind}_{n}{path.suffix}").exists():
        n += 1
    return f"_{kind}_{n}"
