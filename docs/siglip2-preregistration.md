# Pre-registration: SigLIP 2 tutorial configuration grid and fresh test

- **Date:** 2026-10-18, committed on its own before any grid run, any fresh-test download and any code change.
- **Scope:** `tutorials/siglip2_vision_language_colab.ipynb` (row 82 tutorial review, findings R82-M1 and R82-M3,
  owner rulings R82-D1 = fresh test from new iNaturalist photographs and R82-D2 = vision-only grid plus disclosure).
- **Status:** binding for the notebook's default path. A change to anything below needs a dated amendment file
  committed before the run it affects.

## 1. Why

The trainable scope of the fine-tune (the vision tower's last blocks, its post-layernorm and attention-pool head,
with the text tower frozen) was kept at build time (2026-09-19/20) over "adapting the text tower instead, or both
towers" because it "gained on the held-out split", an earlier 48-photograph **test** split. No record of that
comparison survives. The 96 test photographs of the current split have been scored in every recorded run.
From this revision:

- The 96 become the **exposed development test**. They are reported for continuity and inform no decision.
- The configuration is chosen on **validation** by the rule below.
- A **fresh test** of new photographs (section 4) is scored only under the settings fixed here.

The frozen text tower is **kept by design and not re-tested** (ruling R82-D2). The tutorial's prompt-transfer
lesson (the scientific-name prompts the tuned tower never saw) depends on a frozen text tower. The disclosure in
the notebook says the choice was made on an exposed split.

## 2. Fixed settings (every one is part of the `preregistered` check)

| Setting | Value |
|---|---|
| Data source | the pinned sample (not BYOD) |
| Corpus pins | `SAMPLE_RECORDS`, 360 photographs, 39,223,447 bytes; SHA-256 of the compact JSON of the records = `92debe6627946d2d6d22368c42e83c3fac4e0085b202c687945b99fbca3e7de7` |
| Fresh-test pins | `FRESH_TEST_RECORDS`, drawn by section 4; their manifest digest is fixed in code by the draw commit and checked at run time |
| Split | `SPLIT_SEED = 42`, 36 / 8 / 16 per species (216 train / 48 validation / 96 exposed development test) |
| Fine-tune | `EPOCHS = 6`, `BATCH_SIZE = 16`, adapt seed 0, AdamW (weight decay 0.01), gradient clipping 1.0, no scheduler, sigmoid loss over the class prompts |
| Prompts | `DEFAULT_PROMPT_TEMPLATE` filled with the common names; the text tower, the embeddings, `logit_scale` and `logit_bias` frozen |
| Epoch selection | the epoch with the highest **validation text-to-image mAP**; the earliest wins a tie; epoch 0 is the frozen model |
| Grid | run in the same runtime, on this run's data and settings (`RUN_GRID = True`, the default) |
| Configuration | the grid's selection (section 3) |
| Model state | the model in memory is the one Section 7b trained (or the frozen model when the rule keeps it) |
| Fresh test | fetched and not masked (`DIMER_FRESH_TEST_MASK` unset) |

**Code correction recorded here.** At `4fe3348` `Siglip2Pipeline.adapt` kept the epoch with the highest validation
**accuracy**, although its docstring, its `selection` field and the notebook all say text-to-image mAP. The fix
commit makes the code select on validation t2i mAP, as written above, before any grid run.

## 3. Grid (validation only)

| Config | Trainable vision blocks | Learning rate |
|---|---|---|
| **C0** (default) | last 2 + post-layernorm + head | 5e-5 |
| C1 | last 4 + post-layernorm + head | 5e-5 |
| C2 | last 2 + post-layernorm + head | 1e-5 |

**Score.** Each configuration is scored by its best-epoch validation text-to-image mAP on the 48 validation
photographs. Neither test set is read.

**Selection rule.** The margin is **0.02** mAP. It is a practical tolerance, not a significance threshold.

1. A challenger qualifies only if it beats C0 by at least the margin.
2. If no challenger qualifies, C0 is kept.
3. If only C2 qualifies, C2 is selected.
4. C1 is the larger scope. It is selected only if it qualifies and also beats the best other configuration (C0 or
   C2) by at least the margin. Otherwise the best smaller configuration that qualified is selected, or C0.
5. Ties and differences inside the margin go to C0, then to the smaller scope, then to the lower learning rate.

**Keep-frozen rule.** The selected configuration must beat the frozen model's validation mAP (epoch 0) by the
margin. Otherwise the **frozen model** is the model of record, and the fresh test reports the frozen model and the
baselines only.

**Time.** C0 took 32.9 s on a Colab T4 (2026-10-09 evidence). The grid is estimated at about 2 minutes on a T4, so
it is **on by default**. Turning it off makes the run exploratory.

## 4. Fresh test (ruling R82-D1)

**Size.** 96 photographs, 16 per species.

**Source.** The iNaturalist API v1 `observations` endpoint.

- Filters: the species' taxon id (resolved by exact scientific name), `quality_grade=research`,
  `photo_license=cc0`, `photos=true`.
- Creation window: `created_d1=2025-03-01`, `created_d2=2026-10-17`. The end date fixes the pool.
- Paging: `order_by=id`, ascending, 200 per page, until the pool is exhausted.
- Requests are paced to at least 1 s apart and use a generic User-Agent. No personal identifier is sent.

**Eligibility.** An observation is eligible when all of these hold:

- Its observer is **not** one of the 222 observers of the pinned 360.
- Its observer has not already been selected. There is one photograph per observer across the whole fresh test,
  and species are processed in the `SPECIES` order.
- It has `observed_on` and coordinates. A candidate that cannot be screened for place and day is ineligible.
- It has at least one CC0 photograph. The first CC0 photograph by position is the candidate.

**Order.** The eligible pool of each species is shuffled with `random.Random(20261018)`, seeded per species as
`20261018 + index in SPECIES`. Candidates are walked in that order until 16 pass the screens below. Images and model
scores are never looked at to choose.

**Screens.** A candidate is excluded when any of these holds:

1. **Same place, same day.** It has the same `observed_on` date as any of the 360 or any fresh photograph already
   selected, and coordinates within 0.01° in both latitude and longitude.
2. **Exact copy.** Its served `medium` bytes have the same SHA-256 as any of the 360.
3. **Near-duplicate, confirmed.** Its perceptual hash (64-bit DCT pHash of the 32×32 grayscale image) is within
   threshold **T** of any of the 360 (train, validation or exposed development test) or of an earlier selected fresh
   photograph, **and** the hit is confirmed by content. Confirmation is a second, independent measure: the Pearson
   correlation of the two 64×64 grayscale images is at least 0.90, plus a visual check of the pair recorded in the
   draw log. An unconfirmed hit is logged and the candidate is kept.

**Threshold T.** T is set on synthetic copies before any fresh candidate is downloaded, never on outcomes.

- Copies: each of the 360 photographs gets four synthetic copies: a 50 % downscale, JPEG re-encoding at quality
  70, brightness ×1.1, and a 95 % centre crop.
- T is the 99th percentile of the original-to-copy pHash Hamming distances, rounded up.
- The false-positive rate at T over all distinct pairs of the 360 is reported in the draw log.

**Pins.** Each photograph is pinned in `FRESH_TEST_RECORDS` with:

- photo id, observation id and observer login;
- `created_at` and `observed_on`;
- byte size, SHA-256 and extension of the served `medium` file.

The draw script (`tools/draw_fresh_test.py`) and its log (`docs/siglip2-fresh-test-draw.json`) are committed with
the pins.

**Stop rule.** If the screens leave fewer than 16 eligible photographs for any species, the largest balanced size
is reported and **no pins are committed**. The draw stops for the owner.

**Pretraining overlap (DAT9).** SigLIP 2 was released on 2025-02-20 (arXiv:2502.14786) and trained on WebLI, a web
crawl without an instance manifest.

- 255 of the 360 pinned photographs (74 of the 96 exposed) were created on iNaturalist before that date.
- The fresh test's creation cutoff, 2025-03-01, **narrows but does not remove** the overlap risk. A photograph can
  be older than its upload, and it can have been posted elsewhere before.

## 5. Score-once wiring (SCORE_ONCE_RULE, 2026-10-17)

- **`preregistered`** is computed from every setting in section 2, not from one field.
- **Pre-registered run.** The fresh test is scored, labelled `pre-registered evaluation (this runtime)`. A later
  identical run in the same runtime is labelled `reproduction (this runtime)` and written to
  `*_reproduction_<n>.json`.
- **Evaluation of record.** It is the **first hosted pre-registered run**, recorded in
  `docs/release-verification.md`. Until then `RECORDED_EVALUATION = None`.
- **Exploratory runs.** These include any form change, BYOD, the grid turned off, a smaller batch for memory, the
  optional experiments, and the CI integration job (which sets `DIMER_FRESH_TEST_MASK=1`). They do **not** score the
  fresh test. They read validation and the exposed development test, and write `*_exploratory_<n>.json`.
- **CPU checks.** Checks by the fixer on CPU run with the fresh test masked. The fresh test is first scored on the
  hosted run.
- **Output fields.** Every output carries `test_look` and `settings_differing`.
