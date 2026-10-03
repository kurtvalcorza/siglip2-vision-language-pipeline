# DIMER Vision-Language Retrieval Notebook — Review (SigLIP2 entry point)

**Verdict: Needs revision**
**Review date:** 2 October 2026 (written 3 October 2026)
**Repository:** `kurtvalcorza/siglip2-vision-language-pipeline`
**Notebook:** `tutorials/DIMER_MultiModel_Vision_Language_Retrieval_Workshop.ipynb`
**Reviewed commit:** `80500e1c2982ca5a34ebe3fd8f348eefb18f1e84` (`main`, confirmed against the GitHub API at review time)
**Notebook Git blob:** `e5651bb143d8576d289bacad53976b474c52bb6f` (sha256 `38b42048…eba0b4`)
**Finding prefix:** `SVR`
**Companion review:** `blip-itm-pipeline` `docs/reviews/2026-10-02-notebook-review/` (prefix `BVR`, reviewed `a25fdf6`, fixed on branch `fix/bvr-workshop-review`, draft PR #9)

## Executive assessment

This notebook is one learning unit with two repository entry points. The copy reviewed here is **the same Git blob** (`e5651bb`) as the copy in `blip-itm-pipeline` at `a25fdf6` that the BVR review examined, and the archived 2026-09-26 Colab T4 run (`docs/execution-evidence/2026-09-26/…`, sha256 `b76acfad…`) is the same file in both repositories. The notebook-level findings therefore transfer unchanged: this review re-ran every BVR probe against this repository's copy (probe ZIP alongside) and maps each BVR finding to an SVR ID one-to-one. It does not repeat the BVR narrative; read the BVR report for the full observed-issue text and the hosted-evidence numbers.

What this review adds is the repository context. Three things differ here: (1) this repository's tutorial registry has no row and no Colab link for the workshop notebook, and its README and model card never mention it, so the SigLIP2 entry point is hard to find (new finding **SVR-m10**); (2) `docs/release-verification.md` repeats the same not-like-for-like reranking sentence ("raises these to … 0.731"), so SVR-M1 also needs a correction note in this repository's release record; (3) CI installs the full frozen reference environment (including CPU `torch` and `pandas`), a larger budget than blip's, so the BVR regression tests run here unchanged.

Readiness is **Needs revision**: three open Majors (SVR-M1..M3) and unresolved `MUST`s DAT19, DAT9 and EVAL3.

## 1. Review contract and evidence

| Item | Scope |
|---|---|
| Declared profile and mode | `MULTI-CAPABILITY` / `WORKSHOP`; notebook specification `2.1` (`metadata.dimer`) |
| Specification baseline | Declared 2.1; requirements checked against fleet `NOTEBOOK_SPEC.md` 2.2 at `ml-worker` `origin/main` `b1cfe13` |
| Design specification | `docs/vision-language-retrieval-workshop-spec.md` (content identical to the blip copy) |
| Stated learner | Can run Python cells in Colab/Jupyter; new to multimodal embeddings, retrieval or reranking |
| Supported runtime | NVIDIA Tesla T4 or equivalent; `STANDARD` default tier |
| Promised outcomes | A: dual-encoder retrieval (SigLIP 2, SigLIP v1, BLIP ITC); B: BLIP ITM reranking; C (`FULL`): bounded BLIP adaptation; 12 objectives; BYOD; exported indexes and provenance |
| Optional paths | `WORKSHOP_TIER="FULL"`; `USE_BYOD` + `BYOD_ZIP_PATH`; candidate-oracle scratch activity |
| Repository role | Secondary entry point. The repository's own product is the generated `E2E` notebook `siglip2_vision_language_colab.ipynb` (Release-grade); the workshop is a supplemental, hand-maintained `Candidate` notebook. The validator (`validate_notebooks`) checks only its profile, standalone/candidate metadata, syntax, saved errors, three runtime pins and five markers. |
| Model pin consistency | The workshop's SigLIP 2 revision `5ffaac51…` equals the package's pinned revision (non-finding) |

### Evidence actually obtained

- **Source inspection:** all 72 cells at `80500e1`; `tutorials/README.md`, `README.md`, `MODEL_CARD.md`, `docs/release-verification.md`, the design specification, `tests/test_retrieval_workshop_paths.py` (identical to blip's), `tools/validate_release_assets.py`, `.github/workflows/ci.yml`, `requirements.lock.txt`.
- **Documented execution evidence:** `docs/execution-evidence/2026-09-26/DIMER_MultiModel_Vision_Language_Retrieval_Workshop.ipynb` (Colab, Tesla T4, `STANDARD` only). Its 33 code cells equal the reviewed notebook's code cells after removing Colab `# @title` lines and trailing whitespace (0 of 33 differ; checked in this review). The later commit `9327e2b` changed markdown only. `FULL` and BYOD were not run. Saved outputs were inspected; execution was not repeated.
- **Direct execution (this review):** CPU only, Windows, anaconda Python 3.13 (NumPy 2.3.5, pandas 2.3.3). The notebook's own helpers were executed on synthetic NumPy data with inert stand-in models; no weights, no GPU. Probes P01–P16 in `DIMER_MultiModel_Vision_Language_Retrieval_Workshop_Review_Probes.zip`.
- **Learner observation:** none.
- **Shared unit (P16):** this copy's blob `e5651bb` equals blip `a25fdf6` and blip `origin/main`; blip `fix/bvr-workshop-review` carries `571cb2f` (revision 0.2.0-candidate). Until both fix PRs merge, the two entry points diverge.

### Journeys

| Journey | Basis | Outcome |
|---|---|---|
| First-time learner | Source inspection | As BVR, plus SVR-m10: a learner arriving at this repository is not pointed to the workshop at all |
| Clean default (`STANDARD`) | Documented execution evidence, 2026-09-26 (same file as blip) | Completed with zero saved errors; two displayed comparisons are not like-for-like (M1, M2) |
| Active learning | Direct execution of `candidate_oracle` (synthetic); `FULL` not verified | Activity works on saved validation scores; `FULL` has no hosted run |
| Reuse and recovery | Direct execution of `load_byod` on crafted archives | Invalid archives refused without naming the offender (m1); BYOD results never shown (M3); hosted BYOD not verified |

## 2. Separate judgments

- **Technical correctness:** strong, as in BVR §2 (digest pins, `trust_remote_code=False`, decoded-pixel leakage checks, ITM range checks, full ranking-parity reload, safe ZIP extraction). Defect: bundle includes earlier runs (m7).
- **Promise fulfilment:** A and B run by default; caption perturbation promised but absent (m3); BYOD stops short of a visible result (M3). In this repository, the notebook is also under-advertised (m10).
- **Scientific validity:** reranking and category comparisons are not like-for-like (M1, M2); this repository's release record restates the M1 comparison as a reranking gain.
- **Learner experience:** good scaffolding; several reported quantities unexplained or unanswered (m4, m5, m6, m9).
- **Spec conformance:** unresolved `MUST`s DAT19 (m1), DAT9 (m2), EVAL3 for ITM pair accuracy (m4); REL1/REL12 for the optional paths (no `FULL`/BYOD hosted run); RUN8 partial (m8). SRC10 (`SHOULD`, links point to current resources) partially unmet (m10).

## 3. BVR ↔ SVR cross-reference

| SVR | BVR | Severity | Applies here? | Probe (this repo) | Repository-specific note |
|---|---|---|---|---|---|
| SVR-M1 | BVR-M1 | Major | Yes, identical cells | P01, P13 defect | `docs/release-verification.md` here carries the same "raises these to … 0.731" sentence; needs the same correction note |
| SVR-M2 | BVR-M2 | Major | Yes | P02 defect | — |
| SVR-M3 | BVR-M3 | Major | Yes | P03 defect | — |
| SVR-m1 | BVR-m1 | Minor (DAT19) | Yes | P04 defect (6/6 messages anonymous, 2 typos) | — |
| SVR-m2 | BVR-m2 | Minor (DAT9) | Yes | P06 defect | — |
| SVR-m3 | BVR-m3 | Minor | Yes | P05 defect | The perturbation uses SigLIP 2, whose lowercasing behaviour this repository's README documents; the expected "lowercase changes nothing" result is consistent with it |
| SVR-m4 | BVR-m4 | Minor (EVAL3) | Yes | P07 defect | — |
| SVR-m5 | BVR-m5 | Minor (GDL9) | Yes | P08 defect | — |
| SVR-m6 | BVR-m6 | Minor (GDL8) | Yes | P09 defect | — |
| SVR-m7 | BVR-m7 | Minor | Yes | P10 defect | — |
| SVR-m8 | BVR-m8 | Minor (RUN8) | Yes | P11 defect | — |
| SVR-m9 | BVR-m9 | Minor (UX1) | Yes | P12 defect | — |
| SVR-m10 | — | Minor (SRC10, §27 registry) | New | P15 defect | Registry row, Colab link and README pointer missing; shared-unit sentence |
| SVR-S1..S4 | BVR-S1..S4 | Suggestion | Yes | — | — |

## 4. Findings

Shared findings are given in the required six-part form, abbreviated; the BVR report (same cell ids) holds the full evidence text. Acceptance checks are restated in full because the fix is judged against them.

### SVR-M1 — Major: the reranking table compares different query sets

- **Cell/section:** §18, cell `e5928ea5`; helper `rerank_with_blip` (`8477ecb0`); BYOD `reranking.csv` (`05f2bcd3`); `FULL` rows (`aaf0e9d5`). Also `docs/release-verification.md` "Maintainer-supplied successful Colab run — 2026-09-26".
- **Observed issue:** `coarse_t2i_r1` averages all 1,737 caption queries; `itm_t2i_recall_at_1` covers the 391 canonical captions; no same-query baseline or Δ.
- **Consequence:** the change in T2I R@1 is read as the reranker's effect; this repository's release record states it as a gain ("raises these to … 0.731").
- **Evidence:** direct execution P01 (order-preserving stand-in reranker: coarse all-caption 0.400 vs reranked 0.133; coarse on the same canonical queries 0.133); documented execution P13 (saved SigLIP 2 row 0.7214 vs 0.7315).
- **Recommended correction:** as BVR-M1 (same-query `coarse_i2t_r1`, `coarse_t2i_r1_canonical`, deltas in every reranking table; rename the all-caption column; how-to-read note), plus a dated note in this repository's `docs/release-verification.md` explaining that the 2026-09-26 T2I comparison is not like-for-like.
- **Acceptance check:** with an order-preserving stand-in reranker, `delta_i2t_r1 == 0` and `delta_t2i_r1 == 0`, and `coarse_t2i_r1_canonical` equals the argmax-over-images R@1 of the canonical captions; every reranking table (default, `FULL`, BYOD) carries the same-query coarse columns; the markdown before the table states each column's query set; `docs/release-verification.md` carries the correction note.

### SVR-M2 — Major: category diagnostics compare sub-galleries of different sizes

- **Cell/section:** §21, cell `58332fc5`.
- **Observed issue:** each category scored in its own sub-gallery (160 vs 231 photographs), no chance reference, no prose.
- **Consequence:** a gallery-size effect is presented as a category effect.
- **Evidence:** direct execution P02 (same-quality synthetic scores: 0.250 at 160 vs 0.225 at 231); documented category sizes.
- **Recommended correction:** as BVR-M2.
- **Acceptance check:** the category table has `full_gallery_i2t_r1` and `full_gallery_t2i_r1` equal to the full-gallery per-query hits restricted to that category (synthetic test); a sub-gallery chance column is present; the markdown above the cell states which columns are comparable across categories.

### SVR-M3 — Major: the BYOD branch never shows its results

- **Cell/section:** §26 (`07a272b2`), BYOD cell `05f2bcd3`.
- **Observed issue:** a successful BYOD run prints only a directory; no ZIP placement steps; the empty-path error implies a non-existent interactive mode.
- **Consequence:** the promised user-data path ends without an interpretable result.
- **Evidence:** source inspection P03.
- **Recommended correction:** as BVR-M3.
- **Acceptance check:** executing the BYOD cell with stand-in models and a valid two-image archive displays a retrieval table with three model rows and a reranking table with three candidate-source rows from that run's directory; `USE_BYOD=True` with an empty path raises an error naming `BYOD_ZIP_PATH` and the upload step; the BYOD markdown contains the placement steps.

### SVR-m1 — Minor (DAT19 MUST): BYOD validation errors do not name the offending record

- **Cell/section:** `load_byod`, `05f2bcd3`. **Issue:** six messages name no line/id/value/file; typos "outside16", "At most100". **Consequence:** the learner must bisect a large archive. **Evidence:** direct execution P04 (6/6 crafted archives refused, none naming the offender). **Correction:** as BVR-m1.
- **Acceptance check:** each of the six P04 archives raises `ValueError` whose message contains the offending id/value/file; the existing invalid-BYOD tests still raise.

### SVR-m2 — Minor (DAT9 MUST): no pretraining-overlap statement; corpus licence not stated

- **Cell/section:** §4 (`472e7ac5`), §27 (`63ebfe7c`). **Evidence:** source inspection P06. **Correction:** as BVR-m2.
- **Acceptance check:** the notebook markdown contains a pretraining-overlap statement and "CC BY 4.0".

### SVR-m3 — Minor: caption perturbation promised but not run

- **Cell/section:** §25 (`73e42590`, `797b6acd`). **Evidence:** source inspection P05; design spec §68. **Correction:** as BVR-m3.
- **Acceptance check:** the cell computes validation T2I R@1 for original / lowercase / first-five-words captions with the notebook's own `retrieval_metrics`, writes `caption_perturbation.csv`, touches no test data, and the markdown explains the expected lowercase result.

### SVR-m4 — Minor (EVAL3): ITM pair accuracy and cost columns unexplained

- **Cell/section:** `8477ecb0`, `e5928ea5`, glossary `a46a54b6`. **Evidence:** P07. **Correction:** as BVR-m4.
- **Acceptance check:** markdown and glossary define "ITM pair accuracy" (with its chance level) and `pair_evaluations`.

### SVR-m5 — Minor (GDL9): exercises have no sample answers; heading typo

- **Cell/section:** §28 (`3b012ccb`). **Evidence:** P08. **Correction:** as BVR-m5.
- **Acceptance check:** each exercise A–F is followed by a `<details>` sample answer; the heading reads "Try it yourself".

### SVR-m6 — Minor (GDL8): gallery-size section has no guidance

- **Cell/section:** §20 (`5333fe70`, `6ae280da`). **Evidence:** P09 (hosted table: model ordering flips between 70 and 391 images). **Correction:** as BVR-m6.
- **Acceptance check:** §20 markdown has a prediction prompt and a reading note; the table has a chance column equal to `1/n_images`.

### SVR-m7 — Minor: the report bundle can include files from earlier runs

- **Cell/section:** §29 export (`005e710a`). **Evidence:** P10. **Correction:** as BVR-m7.
- **Acceptance check:** a file older than the session start is excluded from the bundle and reported; files written in the session are included.

### SVR-m8 — Minor (RUN8): completion summary reports no results

- **Cell/section:** `76b36372`. **Evidence:** P11. **Correction:** as BVR-m8.
- **Acceptance check:** the summary includes per-model test I2T/T2I R@1, reranked R@1 with Δ, adaptation status and BYOD status.

### SVR-m9 — Minor (UX1): objectives 11–12 are `FULL`-only but unmarked

- **Cell/section:** §0 objectives (`e4c7f831`). **Evidence:** P12. **Correction:** as BVR-m9.
- **Acceptance check:** objectives 11–12 state the tier that exercises them.

### SVR-m10 — Minor (SRC10 `SHOULD`; NOTEBOOK_SPEC §27 registry): the SigLIP2 entry point is not discoverable

- **Cell/section:** repository docs, not the notebook: `tutorials/README.md` registry table and its "Shared Vision-Language Retrieval learning unit" section; `README.md`.
- **Observed issue:** the registry table lists only `siglip2_vision_language_colab.ipynb`; the workshop appears only in a trailing prose section, with no profile/mode/runtime/status row and no Colab link (blip's registry has a row). `README.md` and `MODEL_CARD.md` never name the notebook. The section says "Both carry identical executable cells", which becomes false the moment blip PR #9 merges without this repository's counterpart.
- **Consequence:** a learner who arrives at the SigLIP2 repository (the model the workshop leads with) cannot find or open the workshop; a maintainer reading the registry cannot see its status or its sync state with the blip copy.
- **Evidence:** source inspection P15 (`workshop_row_present: false`, `workshop_colab_link_present: false`, README/model-card mentions false).
- **Recommended correction:** add a registry row (`MULTI-CAPABILITY` / `WORKSHOP` / standalone (supplemental) / T4 / VizWiz-Captions / BYOD zip / Candidate, revision 0.2.0-candidate) with a Colab link; add a one-paragraph pointer in `README.md`'s tutorial section; reword the shared-unit sentence to name the revision both copies carry and that they must change together.
- **Acceptance check:** the registry table has a row naming `DIMER_MultiModel_Vision_Language_Retrieval_Workshop.ipynb` with status Candidate; `tutorials/README.md` contains a Colab link to it; `README.md` names it; the shared-unit section names the revision and the sync rule; `tools/validate_release_assets.py` still passes.

### Suggestions (optional, not release requirements)

- **SVR-S1** (= BVR-S1): bootstrap intervals for test R@1 differences.
- **SVR-S2** (= BVR-S2): validation `RERANK_TOP_K` sweep and median-rank table from the design spec.
- **SVR-S3** (= BVR-S3): "no advantage" label for disagreement examples with rank difference ≤ 0.
- **SVR-S4** (= BVR-S4): no-match queries on the validation gallery with ITM scores.

## 5. Positive findings and non-findings

As BVR §4: freeze-before-test, full-ranking index reload, isolated candidate-oracle activity, ITM output checks and safe extraction all hold for this copy (same cells). Non-findings specific to this repository: the workshop's SigLIP 2 revision equals the package pin; the validator's relaxed rules for the supplemental notebook are deliberate (recorded in `docs/release-verification.md` 2026-09-26) and do not hide a defect; the archived hosted run is byte-identical to blip's.

## 6. Readiness

**Needs revision.** Open Majors SVR-M1, M2, M3; unresolved `MUST`s DAT19, DAT9, EVAL3. After fixes, readiness becomes **Verification pending** until a hosted T4 run of the fixed notebook covers `STANDARD` Run all, `FULL` Run all, and BYOD with a valid plus one rejected archive (REL12). Because the two entry points carry one notebook, one hosted run of the shared revision counts for both only if the two blobs are identical at the time of the run. Status stays Candidate.

## 7. Verified versus inferred

- Verified by direct execution (this repository's copy): M1 mechanism (P01), M2 mechanism (P02), m1 messages (P04), positive controls (P14).
- Verified from source/git: blob identity with blip (P16), hosted-run code-cell coverage (33/33), registry gaps (P15).
- Inferred from source: M3 learner consequence, m3–m10.
- Not verified: `FULL` and BYOD on a real runtime; any real-model numbers after fixes.
- Finding most likely to be wrong: **SVR-m10's severity** — the registry template is a `SHOULD`/suggested format and the notebook is reachable from the blip repository; it could reasonably be a Suggestion.
