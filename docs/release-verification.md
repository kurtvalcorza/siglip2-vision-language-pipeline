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
