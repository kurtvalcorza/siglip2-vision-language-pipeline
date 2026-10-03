# Release verification

`tutorials/siglip2_vision_language_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate
record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`config.py`, `metrics.py`, `model.py`, `provenance.py`, `samples.py`,
  `pipeline.py`, in dependency order), each equal to its source after the generator's documented rewrites (the
  `DEFAULT_WEIGHTS_DIR` rule, the `resolve_weights_path` checkout-convenience line, and the removal of
  package-relative imports); the inline `MANIFEST` equal to the committed 8-entry snapshot manifest and the inline
  `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cells (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md` and `MODEL_CARD.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `Siglip2Pipeline.from_pretrained(weights_dir=...)`, `fetch_corpus` from the pinned cache path, `read_corpus`,
  `build_sample_dataset(corpus, seed=SPLIT_SEED)` / `load_byod_dataset` + `split_dataset`, `validate_dataset` per
  split, `check_split_disjoint`, `observer_overlap`, `class_names`, `write_dataset_csv`, the four dataset refusal
  probes, the ceiling print, the digest-asserted synthetic shapes, `validate_inputs` with the remote-URL refusal
  probe, `zero_shot_classify` / `retrieve` / `embed_image` with the sanity checks and the per-grid
  `evaluation_report` on the drawn shapes, `majority_baseline`, `colour_neighbour_baseline`, `pipe.evaluate` on the
  frozen model with the display-name and scientific-name prompt sets and the baseline assertion, `pipe.adapt` with
  its explicit hyperparameters, `pipe.evaluate` on the validation and test splits after adaptation with the mAP
  assertion, `evaluation_report` on the shapes after adaptation, `pipe.save_artifact`,
  `Siglip2Pipeline.from_artifact` and the reload-parity assertion, `write_provenance`, and the result fields
  `weight_file` / `weight_format` / `weight_sha256` and the `corpus` block), the seven expected `outputs/` paths, the
  learner-facing statements (Apache-2.0 weights, uncalibrated and prompt-dependent sigmoid scores, adaptation with
  labelled photographs, the CC0 corpus, text-to-image mAP, the two non-neural baselines, no dispersion estimate, the
  leakage and prompt guidance, the excluded tasks, the snapshot note) and the gated-off BYOD default; forbidden
  patterns (credential-in-URL, any `git clone` / `github.com` / repository import on the primary path, a mutable
  `revision='main'`, direct `from transformers import` / `AutoModel` / `AutoProcessor` / `get_image_features` /
  `torch.sigmoid` / `from huggingface_hub import` / `snapshot_download` / `urllib.request` / `safetensors` imports /
  `torch.optim` / `.backward(` / `requires_grad` / `logit_scale` / `pipe.model.` / `extractall(` use **outside the
  carried module cells**, `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  checkpoint-invariants section.

CI also installs the frozen CPU reference environment (`requirements.lock.txt`), runs `ruff`, `scripts/check_lock.py`,
`tools/build_notebook.py --check`, and the offline unit suite (`tests/`, including `test_adaptation.py`,
`test_role_helpers.py`, `test_notebook_parity.py`; no weights, injected downloader and photo fetcher —
`tests/test_model_backed.py` is skipped without the snapshot). These are source/provenance and unit checks. They are
**not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Repository CI integration job (`tools/run_notebook.py`, manual `workflow_dispatch` or push to `main`) | GitHub-hosted Ubuntu runner, the frozen CPU reference environment with `DIMER_NOTEBOOK_CI_PREINSTALLED=1` | Executes the standalone notebook's code cells sequentially against the real pinned weights; a **pre-flight** on the locked stack, not a fresh-boundary run of the inline `PINS` and not promotion evidence on its own |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/siglip2-base-patch16-224/` or the photograph cache `weights/inat-birds/` (the standalone path
   writes the manifest itself, stages the missing files from the Hub and fetches the pinned photographs from the
   iNaturalist open-data bucket, so neither directory may be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `EPOCHS = 6`, `LEARNING_RATE = 5e-5`, `BATCH_SIZE = 16`,
   `TRAINABLE_VISION_LAYERS = 2`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`,
   `pillow==11.3.0`, `huggingface-hub==0.36.2` (an interpreter restart after the install is expected where the
   runtime's preinstalled torch or numpy differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the six carried module cells execute (defining `Siglip2Pipeline`, `verify_snapshot`, `stage_missing_files`,
     `validate_inputs`, `evaluation_report`, `classification_metrics`, `majority_baseline`,
     `colour_neighbour_baseline`, `fetch_corpus`, `read_corpus`, `build_sample_dataset`, `validate_dataset`,
     `check_split_disjoint`, `observer_overlap`, `split_dataset`, `load_byod_dataset`, `write_dataset_csv`,
     `write_provenance` and the ceilings) with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting all 8 manifest entries fetched from `google/siglip2-base-patch16-224` at the
     immutable revision on a clean runtime, `verify_snapshot` returning its dict (8 files, the 1.5 GB
     `model.safetensors` re-hashed), and `from_pretrained(weights_dir=WEIGHTS_DIR)` loading from the verified
     directory;
   - Section 4: `fetch_corpus` fetching the 360 pinned photographs with every byte count and SHA-256 matching; the
     seeded split into 216 / 48 / 96 (36 / 8 / 16 per species) with `check_split_disjoint` reporting no shared
     image, the observer overlap and the three dataset digests printed; `outputs/…_train.csv` written; the four
     dataset refusal probes each raising `ValueError`;
   - Section 5: the ceilings (`TEXT_MAX_LENGTH` 64, `DEFAULT_PROMPT_TEMPLATE`, `MAX_IMAGE_SIDE` 4096,
     `MIN_RECORDS` 8, `MAX_RECORDS` 20000, `MIN_CLASSES` 2) surfaced; the three synthetic PPM images generated in
     code with SHA-256 `b38ff0c9…` / `2e3e6576…` / `f0f4c38c…` (equal to `examples/sample-data/SHA256SUMS`);
     `validate_inputs` writing `outputs/…_input_manifest.json` (verdict `accepted`, one recorded rejection finding
     from the remote-URL probe); `zero_shot_classify` ×3, `retrieve` ×3 and `embed_image` with every sanity check
     `True` and the per-grid `evaluation_report` verdict `sample-sanity`;
   - Section 6: the majority floor (accuracy 0.167), the colour nearest neighbour (accuracy ≈ 0.26 in the build
     record) and the frozen model's test score (accuracy ≈ 0.76, macro F1 ≈ 0.76, text-to-image mAP ≈ 0.72 on the
     RTX 5070 Ti build record; scientific-name prompts: accuracy ≈ 0.48) with the per-species breakdown, and the
     cell's assertion that the frozen model is above both baselines;
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 21,264,384 trainable of 375,187,970 parameters,
     and a six-epoch history with the validation mAP selecting the epoch (`best_epoch` 4 in the build record);
   - Section 8: `pipe.evaluate` on the validation and test splits with the four-way comparison on the three
     metrics, the scientific-name prompts, the per-species breakdown and `outputs/…_evaluation_report.json` written
     (the cell asserts the adapted test mAP exceeds the frozen one — 0.871 versus 0.723 in the build record, with
     accuracy 0.760 → 0.792 and the White-throated Sparrow recall 0.44 → 0.75);
   - Section 9: the three shapes re-scored by the adapted model with the `sample-sanity` report,
     `outputs/…_shapes.json` written; `pipe.save_artifact` writing
     `outputs/…_adapter/{adapter.safetensors,manifest.json}` (45 tensors, 85,062,712 bytes) and
     `Siglip2Pipeline.from_artifact` reloading it with 8/8 identical image embeddings on eight test photographs (the
     cell asserts it); `outputs/provenance.json` and `outputs/…_result.json` written with `NOTEBOOK_SOURCE`, the
     model identity and licence, the snapshot block (`weight_file`, `weight_format`, `weight_sha256`), the `corpus`
     block, the inference-contract reports, the comparison, the artifact digest, the reload parity, the runtime
     versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the model cache, the weights directory and the photograph cache were
   clean, outcome, produced outputs, the observed metrics (as observations, not a benchmark) and any warning or
   applicable `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `siglip2_vision_language_colab.ipynb` (`E2E`) | `f3dd43e` / `15f946ab` | 2026-09-20 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-siglip2-vision-language` v4; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda`) | **PASSED** — 14/14 code cells ok (1 restart after install cell); 378 files, 1579 MB staged from the Hub into a clean cache; comparison {accuracy: {majority: 0.167, neighbour: 0.26, frozen: 0.76, adapted: 0.792}, macro_f1: {majority: 0.048, neighbour: 0.261, frozen: 0.758, adapted: 0.793}, t2i_map: {majority: 0.203, neighbour: 0.212, frozen: 0.723, adapted: 0.871}, delta_vs_frozen: {accuracy: 0.031, macro_f1: 0.034, t2i_map: 0.148}, scientific_name_prompts: {accuracy: {frozen: 0.479, adapted: 0.51}, macro_f1: {frozen: 0.459, adapted: 0.453}, t2i_map: {frozen: 0.467, adapted: 0.492}}, by_species: {american_goldfinch: {n: 16, frozen_recall: 0.88, adapted_recall: 0.81, frozen_ap: 0.93, adapted_ap: 0.97}, chipping_sparrow: {n: 16, frozen_recall: 0.75, adapted_recall: 0.62, frozen_ap: 0.71, adapted_ap: 0.8}, dark_eyed_junco: {n: 16, frozen_recall: 0.94, adapted_recall: 0.88, frozen_ap: 0.97, adapted_ap: 0.94}, house_finch: {n: 16, frozen_recall: 0.75, adapted_recall: 0.88, frozen_ap: 0.86, adapted_ap: 0.86}, song_sparrow: {n: 16, frozen_recall: 0.81, adapted_recall: 0.81, frozen_ap: 0.51, adapted_ap: 0.78}, white_throated_sparrow: {n: 16, frozen_recall: 0.44, adapted_recall: 0.75, frozen_ap: 0.35, adapted_ap: 0.89}}}; reload parity {max_abs_difference: 0, identical_rows: 8, of: 8}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-siglip2-vision-language/v4/evidence/` in the workspace |
| `siglip2_vision_language_colab.ipynb` (`MULTI-CAPABILITY`, superseded) | `0266f18` / `d8c19e7f4b8e` | 2026-09-14 | Kaggle CPU (`kurtvalcorza/dimer-nb2-siglip2-vision-language` v3) | PASS — 14/14 code cells (1 restart after install cell), 268.5 s, 18 files, 1539 MB staged; evidence for the earlier inference-only notebook, not for the `E2E` blob |
| repository-installing notebook (superseded) | `23bae80` / notebook SHA-256 `6ccf86e9…` | 2026-09-11 | Kaggle CPU (`kurtvalcorza/tut-siglip2-verify` v6) and GitHub Actions run 34485799429 (`main` integration) | PASS — 6/6 code cells; evidence for the pre-standalone carrier only |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/siglip2_vision_language_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/siglip2_vision_language_colab.ipynb`). Wall times, when recorded, are the sum of
per-cell times reported by the executor and include installs and the model download; they are measurements for the
stated runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-20 | `f3dd43e` / `15f946ab` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-siglip2-vision-language` v4; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) | 423.7 s | **PASSED** — 14/14 code cells ok (1 restart after install cell); 378 files, 1579 MB staged from the Hub into a clean cache; comparison {accuracy: {majority: 0.167, neighbour: 0.26, frozen: 0.76, adapted: 0.792}, macro_f1: {majority: 0.048, neighbour: 0.261, frozen: 0.758, adapted: 0.793}, t2i_map: {majority: 0.203, neighbour: 0.212, frozen: 0.723, adapted: 0.871}, delta_vs_frozen: {accuracy: 0.031, macro_f1: 0.034, t2i_map: 0.148}, scientific_name_prompts: {accuracy: {frozen: 0.479, adapted: 0.51}, macro_f1: {frozen: 0.459, adapted: 0.453}, t2i_map: {frozen: 0.467, adapted: 0.492}}, by_species: {american_goldfinch: {n: 16, frozen_recall: 0.88, adapted_recall: 0.81, frozen_ap: 0.93, adapted_ap: 0.97}, chipping_sparrow: {n: 16, frozen_recall: 0.75, adapted_recall: 0.62, frozen_ap: 0.71, adapted_ap: 0.8}, dark_eyed_junco: {n: 16, frozen_recall: 0.94, adapted_recall: 0.88, frozen_ap: 0.97, adapted_ap: 0.94}, house_finch: {n: 16, frozen_recall: 0.75, adapted_recall: 0.88, frozen_ap: 0.86, adapted_ap: 0.86}, song_sparrow: {n: 16, frozen_recall: 0.81, adapted_recall: 0.81, frozen_ap: 0.51, adapted_ap: 0.78}, white_throated_sparrow: {n: 16, frozen_recall: 0.44, adapted_recall: 0.75, frozen_ap: 0.35, adapted_ap: 0.89}}}; reload parity {max_abs_difference: 0, identical_rows: 8, of: 8}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-siglip2-vision-language/v4/evidence/` in the workspace |
| 2026-09-20 | generated at `4f46f58` / blob `4ff4f85135a9` | Local Windows-venv harness (`run_nb_local.py`: nbclient, fresh `python3` kernel, `CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1`, `DIMER_NOTEBOOK_CI_PREINSTALLED=1`), Python 3.12.10, torch 2.14.0+cu130, transformers 4.57.6, snapshot and the 360 photographs pre-staged | Default sample path, all 14 code cells: pinned install skipped (pre-installed), `stage_missing_files` reported nothing to fetch, `verify_snapshot` PASS (8 files), 360 photographs re-hashed from the pre-staged cache, split 216 / 48 / 96, four refusal probes raised, the three shapes classified correctly (top-1 1.0, recall@1 1.0), baselines accuracy 0.167 / 0.26, frozen test accuracy 0.760 / macro F1 0.758 / mAP 0.723 in 5.0 s (scientific-name prompts 0.479 / 0.459 / 0.467), six epochs 110.1 s (validation mAP 0.680 → 0.874 → 0.854 → 0.863 → 0.876 → 0.872 → 0.854, epoch 4 kept), adapted test accuracy 0.792 / macro F1 0.793 / mAP 0.871 (scientific-name prompts 0.510 / 0.453 / 0.492; White-throated Sparrow recall 0.44 → 0.75, House Finch 0.75 → 0.88, Chipping Sparrow 0.75 → 0.62, Dark-eyed Junco 0.94 → 0.88), adapter 85,062,712 B / 45 tensors, reload parity 8/8, 7 outputs written; the committed blob differs from the executed one in markdown prose only (timing figures filled in after this run) | 159.5 s | PASS — pre-flight only; not promotion evidence |
| 2026-09-20 | generated at `4f46f58` / blob `4ff4f85135a9` | Local WSL harness (same `run_nb_local.py`, `CUDA_VISIBLE_DEVICES=0`), Python 3.12.3, torch 2.14.0+cu130, transformers 4.57.6, RTX 5070 Ti (`cuda`), snapshot and photographs pre-staged | Default sample path, all 14 code cells; identical metrics to the CPU row (frozen 0.760 / 0.758 / 0.723 in 0.6 s, six epochs 12.2 s, epoch 4 kept, adapted 0.792 / 0.793 / 0.871, reload parity 8/8) | 92.4 s | PASS — pre-flight only; not promotion evidence |

## Current status

**Release-grade.** The `E2E` notebook blob `15f946ab` (committed at `f3dd43e`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-20 (14/14 ok (1 restart after install cell), 423.7 s, 378 files, 1579 MB fetched from the Hub and digest-verified inside the notebook) with no repository checkout — the REL1/REL10 supported-runtime evidence this file gates on. The local pre-flight rows above are what preceded it and remain history. Any later change to the carried modules or to the notebook produces a new blob, and the registry returns to **Candidate** until a clean run of that blob is recorded here.

Facts a reviewer should still weigh: the frozen model is already a usable zero-shot classifier on the six species
(accuracy 0.76 in the build record), so the adaptation gain is concentrated on the retrieval view (text-to-image mAP
0.72 → 0.87) and on the worst-separated species (White-throated Sparrow recall 0.44 → 0.75) while top-1 accuracy moves
by three photographs of 96 (0.760 → 0.792); the 48-photograph validation split moves accuracy in 2 % steps, which is
why the epoch is selected on mAP; on an earlier 48-photograph test split the same recipe moved accuracy by anything
from +2 to +12 points across epochs, which is what "no dispersion estimate" means here; the scientific-name prompt set
is scored so a reviewer can see how much of the number is the prompt's; and the drawn shapes re-scored after
adaptation are three images of evidence about behaviour outside the corpus, not a measurement.


## Shared retrieval supplemental remediation evidence — 2026-09-26

Applies only to `DIMER_MultiModel_Vision_Language_Retrieval_Workshop.ipynb`, identical executable cells in both repository entry points. Candidate remains; no real-model or hosted execution was performed here. Baseline BLIP:6 workshop+5 primary parity tests and validator passed. Baseline SigLIP2:5 primary parity tests passed; validator rejected the extra named supplemental notebook. The SigLIP2 validator now explicitly checks the named supplemental profile, standalone/candidate metadata, source syntax, saved errors and carried runtime policy while preserving strict primary parity and refusing unrelated extra notebooks.

Confirmed gaps repaired: BYOD was loader-only; archive extraction replaced a shared directory; caption inputs coerced arbitrary values and lacked bounds; ITM output zip truncation hid missing results; index parity covered only a subset without prior digest verification; FULL reload reported finite outputs without comparison to live adapted weights. The shared notebook now runs bounded optional retrieval/reranking/index exports and FULL train→validation→freeze→fresh-reload→test, with live-to-fresh validation embedding checks. NumPy retains a loaded2.x ABI rather than replacing observed Colab2.1.3 with2.5.3; fallback is2.1.3. Incompatible loaded packages retain an explicit restart guard.

Lightweight tests execute the actual loader, evaluator, reranker, index verifier, adapter export/reload and STANDARD/FULL orchestration with tiny fixtures/model doubles. They establish control flow and failure boundaries, not GPU fit, model quality or real safetensors serialization. A fake adapter that loses learned values is rejected by the actual parity comparison. Model identities and data pins are unchanged.

Pending: fresh STANDARD and FULL T4 runs with exact commit/blob, complete saved outputs, versions, restart count and wall/VRAM; valid STANDARD BYOD and FULL split BYOD through real models and adapter/index exports; invalid-caption/duplicate-image runs rejected before optional model loading. Preserve each `outputs/byod/run-*` result independently. These are open hosted/REL12 evidence gates, not presumed complete from local tests.

Local remediation checks:17 focused tests passed, including STANDARD/FULL orchestration, adapter-loss rejection and deterministic cross-image batch completion for small FULL datasets. BLIP combined suite28 passed (17 focused+6 workshop+5 primary parity). Safetensors IO and model weights are doubled in focused tests; no real model execution is claimed.


### Maintainer-supplied successful Colab run — 2026-09-26

The maintainer supplied the [executed notebook](execution-evidence/2026-09-26/DIMER_MultiModel_Vision_Language_Retrieval_Workshop.ipynb) and authorized merging PR #13 (merge commit `6fd2603`). The file is archived byte-for-byte, SHA-256 `b76acfad5db8d02119b412160164e61bdbb0722c6d09fb053f3039cd0dfccd42`. All 33 code cells have execution counts, 59 saved outputs and zero saved errors. Code-cell sources match commit `166fa991ffda3a2711b2ba20f047aa55ffda6068`, tutorial blob `3cdeee63722219eb9a06da8e81cf2985acb03850`, apart from Colab-inserted `# @title` lines. One code cell differs from the committed blob only by trailing whitespace that Colab strips. Later commits on `main` that touch the notebook (`9327e2b` (AI Use Disclosure)) change only markdown cells; its code cells are identical to the executed revision. This evidence commit does not change tutorial code.

Scope: STANDARD tier: pinned VizWiz-derived parquet shard, test gallery of 391 images and 1,737 captions; SigLIP v1, SigLIP 2 and BLIP ITC embeddings with BLIP ITM reranking of the top 5. FULL adaptation and BYOD were not exercised. The same executed notebook (blob `3cdeee63`) also covers `blip-itm-pipeline` PR #7.

Saved runtime: Python 3.13.15, torch 2.14.0+cu130, Transformers 4.57.6, pyarrow 25.0.1, CUDA Tesla T4. Execution reaches the final completion summary. The separate exported files were not supplied, so their bytes/digests were not independently inspected. Saved counts run sequentially from 1 to 33; runtime freshness and absence of manual restarts/reruns are not independently established by the artifact.

Results (sample-sanity measures on the built-in data, not general model rankings): Test recall@1 image→text / text→image: SigLIP 2 0.790 / 0.721, SigLIP v1 0.783 / 0.716, BLIP ITC 0.742 / 0.652; BLIP ITM reranking of the top 5 raises these to 0.839 / 0.731 (SigLIP 2 candidates), 0.834 / 0.734 (SigLIP v1) and 0.803 / 0.708 (BLIP ITC).

Status remains **Candidate**. Merge approval and this successful default-path run do not close the optional-path (FULL/BYOD) or REL12 qualification gates, and `metadata.dimer.clean_runtime_evidence` in the notebook stays `pending` as authored (editing it would change the verified blob).

## Notebook review fixes, revision 0.2.0-candidate — 2026-10-02

Applies only to `DIMER_MultiModel_Vision_Language_Retrieval_Workshop.ipynb`. A Notebook Review Framework v1 review of `80500e1` (`docs/reviews/2026-10-02-notebook-review/`, prefix SVR) found 3 Major and 10 Minor issues. SVR-M1..m9 are the same notebook defects as the `blip-itm-pipeline` review (BVR-M1..m9; same blob `e5651bb`); SVR-m10 is this repository's missing registry row, Colab link and README pointer. The notebook here is replaced with the `blip-itm-pipeline` `fix/bvr-workshop-review` revision byte for byte (blob `571cb2f`, sha256 `fe3a0c5d…`), so the two entry points stay identical; its edits are anchor-asserting (there is no in-repo generator) and recorded in `metadata.dimer.review_revisions`, whose `base_commit` names the `blip-itm-pipeline` commit where the edit was authored. Cell ids and order are unchanged.

- **Reading the 2026-09-26 result above.** Its text→image comparison is not like-for-like: the coarse values (0.721 / 0.716 / 0.652) are over all 1,737 captions, the reranked values (0.731 / 0.734 / 0.708) over the 391 first captions. The image→text comparison uses the same 391 queries on both sides. Revision 0.2.0 reports coarse Recall@1 on exactly the reranked queries (`coarse_i2t_r1`, `coarse_t2i_r1_canonical`) and the deltas.
- Other changes: category diagnostics add full-gallery columns; the BYOD branch displays its retrieval and reranking tables and names the failing `records.jsonl` line/id/field; caption perturbation (validation only) is implemented; the report bundle holds only files written since the current Run all started; licence and pretraining-overlap statements, sample answers, glossary entries and a results summary are added; `tutorials/README.md` gains a registry row and Colab link and `README.md` a pointer.
- Verification so far: CPU-only unit tests that execute the notebook's own cells with NumPy data and stand-in models (`tests/test_svr_review_fixes.py`, ported from `blip-itm-pipeline`, plus one registry test). This is **not** clean-runtime evidence. The 2026-09-26 run covers the previous code cells only.
- Required before release: a fresh Colab T4 `STANDARD` Run all of this revision; a `FULL` Run all (adaptation, fresh reload, adapted reranking); BYOD with a valid archive and one rejected archive (REL12). One hosted run counts for both repositories only while the two notebook blobs are identical. Status remains **Candidate**.

### Colab CLI execution of revision 0.2.0-candidate (`13040f4`, blob `571cb2f2`) — 2026-10-03

- **File:** [`execution-evidence/2026-10-03/DIMER_MultiModel_Vision_Language_Retrieval_Workshop_13040f4_colab-cli-t4.ipynb`](execution-evidence/2026-10-03/DIMER_MultiModel_Vision_Language_Retrieval_Workshop_13040f4_colab-cli-t4.ipynb), SHA-256 `88230408683946ed2b4cbac68f8c8713547feb777b07ddabb4253ece5131ac89`, a byte-for-byte copy of the CLI's output notebook. The same notebook bytes are the PR head of both `blip-itm-pipeline` #9 (`04664ca`) and `siglip2-vision-language-pipeline` #15 (`13040f4`), so this one run covers both.
- **Executor:** Google Colab CLI 0.7.4 on a fresh Colab Tesla T4 session via the workspace `colab-cli-serial-test-suite` (`colab new --gpu T4`, `colab exec -f`, `colab stop`). The notebook was downloaded from GitHub at the PR head and its git blob `571cb2f2d3dc` checked before the session was created. Code cells ran in order in one kernel; this is not a browser Run all, and the CLI records no execution counts, so order is evidenced by its `Executing cell k/33` log.
- **Result:** **PASSED** (STANDARD, default controls): 33/33 code cells, no error output, 589.1 s wall; the completion summary reports `full_adaptation: not run (STANDARD)`, `byod: not run`, and "Vision-language retrieval workshop complete".
- **Results:** coarse test recall@1 image→text / text→image over all captions equal the 2026-09-26 run exactly: SigLIP 2 0.790 / 0.721, SigLIP v1 0.783 / 0.716, BLIP ITC 0.742 / 0.652. BLIP ITM reranking of the top 5 on the 391 canonical queries also equals it (same model order: SigLIP 2 / SigLIP v1 / BLIP ITC candidates): image→text 0.839 / 0.834 / 0.803 and text→image 0.731 / 0.734 / 0.708. New in 0.2.0, the like-for-like coarse text→image recall@1 on the same 391 queries is 0.688 / 0.662 / 0.627, so reranking gains +0.044 / +0.072 / +0.082.
- **Observation:** cell "Fetch and verify the full pinned parquet" printed `Error while fetching HF_TOKEN secret value from your vault ... timed out`. Under the CLI the Colab secrets prompt cannot be answered; `huggingface_hub` then downloaded anonymously and the pinned parquet verified. This is cosmetic, CLI-specific, and not a failure.
- **Boundary:** saved outputs were inspected. FULL adaptation, BYOD and the rejected-archive path (REL12) were not exercised and remain required before release. Status remains **Candidate**.


## 2026-10-03 uv isolated environment (revision 0.3.0-candidate)

Applies only to `DIMER_MultiModel_Vision_Language_Retrieval_Workshop.ipynb`: notebook blob `571cb2f2d3dcd58d04afc1dd273742a68264c9be` (revision 0.2.0-candidate, the blob the 2026-10-03 Colab CLI run above covers) → `954cf7bc208ce10e5f41cf2d1beeec0ff5761bd1` (revision 0.3.0-candidate). This mirrors `blip-itm-pipeline` PR #9 commit `3d7dbc3`; the notebook stays byte-identical in both repositories, and the carried `tools/retrieval_workshop.py`, `tools/retrieval-workshop-requirements.in`/`.lock` and `tools/build_retrieval_workshop_carrier.py` are the same files (the builder is line-wrapped for this repository's 100-column ruff limit; its output is unchanged).

- **Change.** The in-kernel `pip install` and its "Restart session, then Run all" guard are removed. The notebook carries `tools/retrieval_workshop.py` (every model stage; the former cell functions moved unchanged) and `tools/retrieval-workshop-requirements.lock` (`uv pip compile --generate-hashes` for CPython 3.12 on `x86_64-manylinux_2_28`) in carrier cell `uvcarrier`, written by `tools/build_retrieval_workshop_carrier.py` in pieces of at most 1,000 characters. The runtime cell verifies a pinned uv 0.12.15 wheel by size and SHA-256, builds a CPython 3.12.12 venv (`--managed-python`) and installs with `--require-hashes --only-binary :all:`; every model stage runs as a separate process with that venv's interpreter, `MPLBACKEND=Agg`, and `PYTHONPATH`/`PYTHONHOME`/`PYTHONSTARTUP`/HF tokens dropped. Display cells read the stage JSON/CSV/PNG outputs.
- **Pins.** Same direct versions as before (torch 2.14.0, torchvision 0.29.0, torchaudio 2.11.0, transformers 4.57.6, huggingface-hub 0.36.2, safetensors 0.8.0, numpy 2.1.3, pillow 11.3.0, pyarrow 25.0.1); the former ranges resolve to pandas 2.2.3 and matplotlib 3.10.8; sentencepiece 0.2.1 and protobuf 6.33.5 are added (the hosted kernel used to supply them for the SigLIP v1 tokenizer).
- **User-visible.** Linux x86_64 only (Google Colab, Kaggle, Linux Jupyter); Windows/macOS kernels are refused. The experiment manifest gains an `environment` record. No metric, split, seed, model or output file changes.
- **Local verification (not clean-runtime evidence).** The local CPU harness run of this exact blob is recorded in `blip-itm-pipeline` `docs/release-verification.md` (STANDARD reproduced the 2026-10-03 Colab values exactly; FULL with one epoch, BYOD and the rejected archive passed; stand-ins: platform check patched, host interpreter instead of the uv venv). Here: `pytest -m 'not integration'` 119 passed / 1 skipped before, 133 passed / 1 skipped after; ruff, the validator, `build_notebook.py --check` and the carrier `--check` pass.
- **Pending.** A hosted Colab T4 re-run of this blob (the uv bootstrap and the hash-locked install have not run on a hosted runtime yet), then FULL, BYOD and REL12 as before. Status remains **Candidate**.
