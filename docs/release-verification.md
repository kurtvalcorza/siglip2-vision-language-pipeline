# Release verification

`tutorials/siglip2_vision_language_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.2 (REL8). This file is the durable release-gate
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
  `tools/build_notebook.py` output for its recorded revision; the isolated-environment bootstrap cell (generator
  /2.2: hash-locked `uv` environment, nothing installed into the kernel, no restart); `NOTEBOOK_SOURCE` recorded in exports;
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
   `pillow==11.3.0`, `huggingface-hub==0.36.2` (installed into the isolated hash-locked environment of Section 1, so the
   runtime's preinstalled torch or numpy do not matter and no restart is needed);
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
| `siglip2_vision_language_colab.ipynb` (`E2E`) | `6df20c7` / `1f274277` | 2026-10-18 | Colab CLI 0.7.4, fresh Colab Tesla T4 (session `suite-siglip2-6df20c7-c27e`) | **PASSED** — REL13: **pre-registered evaluation of record** (first hosted run of the pre-registered settings; first fresh-test scoring by any run); 17/17 code cells, 0 errors; fresh test (96) accuracy frozen 0.552 → adapted 0.677, macro F1 0.543 → 0.68, t2i mAP 0.628 → 0.771 (majority 0.167 / 0.048 / 0.312, colour neighbour 0.208 / 0.21 / 0.245); evidence in `docs/execution-evidence/2026-10-18-6df20c7/` |
| `siglip2_vision_language_colab.ipynb` (`E2E`, earlier blob, history) | `f3dd43e` / `15f946ab` | 2026-09-20 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-siglip2-vision-language` v4; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda`) | **PASSED** — 14/14 code cells ok (1 restart after install cell); 378 files, 1579 MB staged from the Hub into a clean cache; comparison {accuracy: {majority: 0.167, neighbour: 0.26, frozen: 0.76, adapted: 0.792}, macro_f1: {majority: 0.048, neighbour: 0.261, frozen: 0.758, adapted: 0.793}, t2i_map: {majority: 0.203, neighbour: 0.212, frozen: 0.723, adapted: 0.871}, delta_vs_frozen: {accuracy: 0.031, macro_f1: 0.034, t2i_map: 0.148}, scientific_name_prompts: {accuracy: {frozen: 0.479, adapted: 0.51}, macro_f1: {frozen: 0.459, adapted: 0.453}, t2i_map: {frozen: 0.467, adapted: 0.492}}, by_species: {american_goldfinch: {n: 16, frozen_recall: 0.88, adapted_recall: 0.81, frozen_ap: 0.93, adapted_ap: 0.97}, chipping_sparrow: {n: 16, frozen_recall: 0.75, adapted_recall: 0.62, frozen_ap: 0.71, adapted_ap: 0.8}, dark_eyed_junco: {n: 16, frozen_recall: 0.94, adapted_recall: 0.88, frozen_ap: 0.97, adapted_ap: 0.94}, house_finch: {n: 16, frozen_recall: 0.75, adapted_recall: 0.88, frozen_ap: 0.86, adapted_ap: 0.86}, song_sparrow: {n: 16, frozen_recall: 0.81, adapted_recall: 0.81, frozen_ap: 0.51, adapted_ap: 0.78}, white_throated_sparrow: {n: 16, frozen_recall: 0.44, adapted_recall: 0.75, frozen_ap: 0.35, adapted_ap: 0.89}}}; reload parity {max_abs_difference: 0, identical_rows: 8, of: 8}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-siglip2-vision-language/v4/evidence/` in the workspace |
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
| 2026-10-09 (14:20:50 UTC start) | commit `20ffef4` / blob `b3ae91fdb393` (`metadata.dimer.generated_from.revision` `0979f22`, the generation-time HEAD; `0979f22..20ffef4` is the ModuleSpec/spec-2.2 commit itself; generator `build_notebook.py/2.2`, `notebook_spec` 2.2) | Colab CLI 0.7.4 sequential execution (`colab exec -f`), fresh Colab Tesla T4, session `suite-siglip2-20ffef4-c27e`; isolated hash-locked uv environment, nothing installed into the kernel | Default sample path, one pass, no restart: 15/15 code cells, 0 errors (6 carried-module cells print nothing); snapshot 8 files / 1,539,456,760 B digest-verified; 360 photographs, split 216 / 48 / 96, observer overlap 53 of 222; shapes top-1 1.0 / recall@1 1.0 (`sample-sanity`); majority 0.167, colour neighbour 0.26, frozen accuracy 0.76 / macro F1 0.758 / mAP 0.723 (scientific-name prompts 0.479 / 0.459 / 0.467); validation mAP 0.68 → 0.874 → 0.854 → 0.863 → 0.876 → 0.872 …, epoch 4 kept; delta vs frozen accuracy +0.031, macro F1 +0.034, mAP +0.148 (scientific-name 0.51 / 0.453 / 0.492); White-throated Sparrow recall 0.44 → 0.75, Chipping Sparrow 0.75 → 0.62; adapter 45 tensors / 85,062,712 B; reload parity 8/8, max abs difference 0.0 | 377.5 s | **PASSED** — one pass; executed notebook SHA-256 `057015ead8bf67e4…`; optional journeys (BYOD, REL12 refusals in a hosted run) not exercised; status stays **Candidate** (REL14); earlier blob `b3ae91fd`, history since the 2026-10-18 evaluation of record |
| 2026-10-15 (00:43:58 UTC start) | `a3ec1e4` / blob `641d29c139cc` (supplemental `DIMER_MultiModel_Vision_Language_Retrieval_Workshop.ipynb`, revision 0.4.0-candidate) | Colab CLI 0.7.4, fresh Colab Tesla T4, session `suite-siglip2-a3ec1e4-4627`; CLI sequential execution, not a browser Run all | Default `STANDARD` path; forms/uploads not exercised; 37/37 code cells, 0 errors; test gallery 389 / 1,729; coarse R@1 I2T / T2I SigLIP 2 0.789 / 0.721, SigLIP v1 0.781 / 0.716, BLIP ITC 0.740 / 0.652, Flickr30k sensitivity 0.671 / 0.606; index reload parity PASS | 563.0 s | **PASSED** — executed notebook SHA-256 `6501fb697d3e…` in `docs/execution-evidence/2026-10-15/DIMER_MultiModel_Vision_Language_Retrieval_Workshop_a3ec1e4_colab-cli-t4.ipynb`; status stays **Candidate** |
| 2026-10-18 (00:06:32 UTC start) | `6df20c7` / blob `1f274277e528` (`metadata.dimer.generated_from.revision` `188c43c`; generator `build_notebook.py/2.2`, `notebook_spec` 2.5, `module_sha256` `897cfd4e32c7`) | Colab CLI 0.7.4 sequential execution (`colab exec -f`), fresh Colab Tesla T4, session `suite-siglip2-6df20c7-c27e`; isolated hash-locked uv environment (56 s) | Default pre-registered path (grid on, `CONFIG = 'grid selection'`, fresh test fetched and not masked); forms/uploads not exercised; 17/17 code cells, 0 errors; grid C0 0.8758 / C1 0.8767 / C2 0.8667, frozen 0.6798, C0 kept (103.1 s); 7b epoch 4 (31.7 s); exposed development test unchanged (0.76 → 0.792 / 0.758 → 0.793 / 0.723 → 0.871); fresh test (96) accuracy frozen 0.552 → adapted 0.677, macro F1 0.543 → 0.68, t2i mAP 0.628 → 0.771 (majority 0.167 / 0.048 / 0.312, colour neighbour 0.208 / 0.21 / 0.245); reload parity 8/8 | 895.7 s | **PASSED** — REL13: **pre-registered evaluation of record** (first hosted run of the pre-registered settings; first fresh-test scoring by any run); executed notebook SHA-256 `5774248174fd…` in `docs/execution-evidence/2026-10-18-6df20c7/`; status stays **Candidate** |
| 2026-10-18 (00:49:35 UTC start) | `e9ff7f9` / blob `3f173c9c4456` | Colab CLI 0.7.4 `colab exec -f`, fresh Colab Tesla T4, session `suite-siglip2-e9ff7f9-c27e` | Default sample path, one pass: 17/17 code cells, 0 errors; `test_look` = `reproduction of the pre-registered evaluation`, `settings_differing` = []; grid C0 0.8758 / C1 0.8767 / C2 0.8667, C0 kept, epoch 4; fresh test identical to the record (accuracy 0.552 → 0.677, macro F1 0.543 → 0.68, mAP 0.628 → 0.771; scientific-name 0.396 → 0.458 / 0.35 → 0.415 / 0.4 → 0.399); exposed development test identical; reload parity 8/8 | 393.3 s | **PASSED** — reproduction of the pre-registered evaluation (SCORE_ONCE_RULE item 3); executed notebook SHA-256 `7c5807697d0d2cd1…`; adapter bytes differ from the record (`ed193b78…` vs `ec27138c…`, GPU non-determinism) while every printed metric is equal |
| 2026-10-18 (06:41:13 UTC start) | `6176ddb` / blob `ef9048c42298` (`metadata.dimer.generated_from.revision` `b348ba1`; generator `build_notebook.py/2.2`, `notebook_spec` 2.5, `module_sha256` `d01783455fc1`) | Colab CLI 0.7.4 `colab exec -f`, fresh Colab Tesla T4, session `suite-siglip2-6176ddb-c27e`; CLI sequential execution, not a browser Run all | Default pre-registered path; forms/uploads not exercised; 17/17 code cells, 0 errors; `test_look` = `reproduction of the pre-registered evaluation`, `preregistered` True, `settings_differing` = []; grid C0 0.8758 / C1 0.8767 / C2 0.8667, frozen 0.6798, C0 kept, epoch 4; fresh test identical to the record (accuracy 0.552 → 0.677, macro F1 0.543 → 0.68, mAP 0.628 → 0.771; colour neighbour 0.208 / 0.21 / 0.245; scientific-name 0.396 → 0.458 / 0.35 → 0.415 / 0.4 → 0.399; per-species identical); `reproduction_vs_record` = `matches record`, `differing` = []; majority floor mAP printed **0.203** (tie-aware AP; the record's 0.312 is the tie-order artefact, excluded from the match); exposed development test identical; reload parity 8/8 | 470.5 s | **PASSED** — reproduction of the pre-registered evaluation (SCORE_ONCE_RULE item 3); executed notebook SHA-256 `759752937110ac6f…` in `docs/execution-evidence/2026-10-18-6176ddb/`; adapter bytes differ (`907fbe39…`, GPU non-determinism) while every printed model metric is equal; also prints a post-hoc analysis, not pre-registered (display-only; no verdict depends on it): of the 96 fresh-test photographs 18 were gained and 6 lost by the adapted model (53 → 65), exact McNemar p = 0.0227, paired bootstrap 95 % CI on the accuracy difference [0.031, 0.219]; status stays **Candidate** |

## Current status

**Candidate** (2026-10-18). **Pre-registered evaluation of record (2026-10-18).** The row 82 revision, commit `6df20c7` / blob `1f274277` (generator `build_notebook.py/2.2`, `notebook_spec` 2.5, embedded revision `188c43c`) executed on a fresh Colab Tesla T4 (Colab CLI 0.7.4 sequential execution, not a browser `Run all`; 17/17 code cells, 0 errors, 895.7 s) under the pre-registered settings (`preregistered: True`, `settings_differing: []`); the grid kept C0, epoch 4; fresh test (96) accuracy frozen 0.552 → adapted 0.677, macro F1 0.543 → 0.68, t2i mAP 0.628 → 0.771 (majority 0.167 / 0.048 / 0.312, colour neighbour 0.208 / 0.21 / 0.245). Evidence in `docs/execution-evidence/2026-10-18-6df20c7/`; see the row 82 record section below. Reproduced exactly on 2026-10-18 by `e9ff7f9` and by the round-2 head `6176ddb` / blob `ef9048c4` (Colab CLI T4, 17/17 code cells, 0 errors, `reproduction of the pre-registered evaluation`, `matches record`; it prints the corrected majority-floor mAP 0.203 in place of the recorded 0.312 tie-order artefact, and a post-hoc analysis, not pre-registered (display-only; no verdict depends on it): of the 96 fresh-test photographs 18 were gained and 6 lost by the adapted model (53 → 65), exact McNemar p = 0.0227, paired bootstrap 95 % CI on the accuracy difference [0.031, 0.219]). Forms, BYOD and uploads were not exercised, so the status stays **Candidate**. Earlier, history: **Candidate** (2026-10-09). **Colab T4 one-pass run of the isolated-environment revision (2026-10-09).** Commit `20ffef4` / blob `b3ae91fdb393` (generator `build_notebook.py/2.2`, `notebook_spec` 2.2, worker `google.colab` stubs with a `ModuleSpec`) executed in one pass with no restart and 0 errors on a fresh Colab Tesla T4 (Colab CLI 0.7.4 sequential execution, not a browser `Run all`; 15/15 code cells, 377.5 s, `cuda`); evidence in `docs/execution-evidence/2026-10-09-20ffef4/`. The 2026-09-20 Kaggle T4 run of blob `15f946ab` (14/14 ok, one restart after the install cell) is history: the notebook now builds a hash-locked isolated uv environment (generator /2.2, no restart) and declares NOTEBOOK_SPEC 2.2, so its blob changed and the status is **Candidate** (REL14) until a clean run of the new blob is recorded in `docs/release-verification.md`. Previously: the `E2E` notebook blob `15f946ab` (committed at `f3dd43e`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-20 (14/14 ok (1 restart after install cell), 423.7 s, 378 files, 1579 MB fetched from the Hub and digest-verified inside the notebook) with no repository checkout — the REL1/REL10 supported-runtime evidence this file gates on. The local pre-flight rows above are what preceded it and remain history. Any later change to the carried modules or to the notebook produces a new blob, and the registry returns to **Candidate** until a clean run of that blob is recorded here.

The current blob `5c2f4038` (code commit `09f124e`) shows a refused BYOD input as exactly one `Refused: <file>: <reason>` line (previously a differently prefixed line), with no traceback, and stops Run all at that cell; genuine failures keep their traceback (NOTEBOOK_SPEC DAT19/DAT20). Its hosted Colab T4 runs (the default path and a scripted refusal, `USE_BYOD=True;BYOD_PATH='/etc/hostname'`) are pending, and the hosted runs of earlier blobs, including `ef9048c4`, are history for it (REL14).

**Reading the record (peer review, 2026-10-18).** On the fresh test the gain is concentrated in one species: the White-throated Sparrow, which the frozen common-name prompt got wrong, went from 2 to 12 of 16 correct, while the other five species netted +2 of 80 (American Goldfinch 0, Chipping Sparrow −2, Dark-eyed Junco +4, House Finch +1, Song Sparrow −1), changes of one to four photographs that 16 per species cannot tell from noise. No primary endpoint or success criterion was pre-registered: accuracy moved from 53 to 65 of 96, and the unpaired 95 % Clopper–Pearson intervals overlap (frozen [0.447, 0.654], adapted [0.574, 0.769]), so the record shows a point improvement that n = 96 cannot establish. The recorded majority-floor mAP 0.312 is a tie-order artefact — the fresh records are grouped by species and the old metric ranked tied scores in record order — and is about 0.203 order-free, so it does not beat the colour neighbour (0.245); later runs print the corrected floor. The frozen model scored 0.552 on the fresh test against 0.76 on the exposed development test, a gap larger than the adaptation effect; two plausible, untested contributors are different photographers (excluding the 360's 222 observers removed 65–88 % of every species' pool) and the date cutoff against pretraining exposure (74 of the 96 exposed photographs predate SigLIP 2's release); the record cannot separate them.

Requote (2026-10-18, after the record): code commits `449edfc`, `29479b1` and `469f55e`, regeneration commit
`8994f49`, blob `3f173c9c` (stamp `469f55e`). It changes the worked answers and run times, which now quote the record
above with its commit and blob (REL13), and sets `RECORDED_EVALUATION`, so a later identical run labels itself
`reproduction of the pre-registered evaluation`. No model was run and nothing was scored for it. Blob `3f173c9c` has no
execution evidence of its own yet (REL14).

Facts a reviewer should still weigh (rewritten 2026-10-18 around the evaluation of record, peer review m6): On the fresh test the gain is concentrated in one species: the White-throated Sparrow, which the frozen common-name prompt got wrong, went from 2 to 12 of 16 correct, while the other five species netted +2 of 80 (American Goldfinch 0, Chipping Sparrow −2, Dark-eyed Junco +4, House Finch +1, Song Sparrow −1), changes of one to four photographs that 16 per species cannot tell from noise. No primary endpoint or success criterion was pre-registered: accuracy moved from 53 to 65 of 96, and the unpaired 95 % Clopper–Pearson intervals overlap (frozen [0.447, 0.654], adapted [0.574, 0.769]), so the record shows a point improvement that n = 96 cannot establish. The recorded majority-floor mAP 0.312 is a tie-order artefact — the fresh records are grouped by species and the old metric ranked tied scores in record order — and is about 0.203 order-free, so it does not beat the colour neighbour (0.245); later runs print the corrected floor. The frozen model scored 0.552 on the fresh test against 0.76 on the exposed development test, a gap larger than the adaptation effect; two plausible, untested contributors are different photographers (excluding the 360's 222 observers removed 65–88 % of every species' pool) and the date cutoff against pretraining exposure (74 of the 96 exposed photographs predate SigLIP 2's release); the record cannot separate them. The configuration and the epoch were chosen on 48 validation photographs, so validation numbers are biased upward; the keep-frozen rule compares a best-of-seven-epochs validation score with the single epoch-0 value (only the 0.02 margin guards against that); the scientific-name prompts did not carry the retrieval gain (mAP 0.4 → 0.399); and the drawn shapes re-scored after adaptation are three images of evidence about behaviour outside the corpus, not a measurement.


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
- **Pending.** The hosted Colab T4 re-run of this blob passed on 2026-10-03 (see "Colab CLI execution of revision 0.3.0-candidate" below, the first hosted run of the uv bootstrap and the hash-locked install); FULL, BYOD and REL12 remain pending as before. Status remains **Candidate**.

### Colab CLI execution of revision 0.3.0-candidate (`4542d3a`, blob `954cf7bc`) — 2026-10-03

- **File:** [`execution-evidence/2026-10-03/DIMER_MultiModel_Vision_Language_Retrieval_Workshop_4542d3a_colab-cli-t4.ipynb`](execution-evidence/2026-10-03/DIMER_MultiModel_Vision_Language_Retrieval_Workshop_4542d3a_colab-cli-t4.ipynb), SHA-256 `29db8b37b2dde817fc471d14bfac1978baa332db9fae650a5759107dfd3ff466`, a byte-for-byte copy of the CLI's output notebook. The same notebook bytes are the PR head of both `siglip2-vision-language-pipeline` #15 (`4542d3a`) and `blip-itm-pipeline` #9 (`3d7dbc3`), so this one run covers both PRs; the file is byte-identical to `blip-itm-pipeline`'s `docs/execution-evidence/2026-10-03/DIMER_MultiModel_Vision_Language_Retrieval_Workshop_3d7dbc3_colab-cli-t4.ipynb` (same SHA-256), named here after this repository's head.
- **Executor:** Google Colab CLI 0.7.4 on a fresh Colab Tesla T4 session via the workspace `colab-cli-serial-test-suite` (`colab new --gpu T4`, `colab exec -f`, `colab stop`). The notebook was downloaded from GitHub at the PR head (`blip-itm-pipeline` `3d7dbc3`; `git rev-parse 4542d3a:tutorials/DIMER_MultiModel_Vision_Language_Retrieval_Workshop.ipynb` gives the same blob) and its git blob `954cf7bc208c` checked before the session was created. Code cells ran in order in one kernel; this is not a browser Run all, and the CLI records no execution counts, so order is evidenced by its `Executing cell k/34` log.
- **Result:** **PASSED** (STANDARD, default controls): 34/34 code cells, no error output, 536.8 s wall for the whole session (589.1 s for the 0.2.0 run above); the completion summary reports `full_adaptation: not run (STANDARD)`, `byod: not run`, `release_status: candidate`, and "Vision-language retrieval workshop complete".
- **uv bootstrap (first hosted run).** The runtime cell built the isolated environment and installed the hash lock without error, then reported `{'python': '3.12.12', 'torch': '2.14.0+cu130', 'device': 'cuda:0', 'gpu': 'Tesla T4', 'lock_pins_checked': 61}`; the carrier cell verified `requirements.lock.txt` and `retrieval_workshop.py`. No restart was needed. The CLI records no per-cell times, so setup time on its own is not measured; it is contained in the 536.8 s total, which is 52.3 s shorter than the 0.2.0 run with its in-kernel `pip install`.
- **Results vs the 0.2.0 run above (same 391-image / 1,737-caption test gallery):** every metric is equal. Parquet SHA-256 `4492465a…` and role digest `3cb7cbd2…` are unchanged. Coarse test recall@1 image→text / text→image over all captions: SigLIP 2 0.790 / 0.721, SigLIP v1 0.783 / 0.716, BLIP ITC 0.742 / 0.652. BLIP ITM reranking of the top 5 on the 391 canonical queries (SigLIP 2 / SigLIP v1 / BLIP ITC candidates): image→text 0.839 / 0.834 / 0.803, like-for-like coarse text→image 0.688 / 0.662 / 0.627 → reranked 0.731 / 0.734 / 0.708 (+0.044 / +0.072 / +0.082), ITM pair accuracy 0.780 / 0.767 / 0.683, 4,692 pair evaluations each. The validation reranking table (40 queries), the nested-gallery table, the caption-perturbation table and the no-match nearest-neighbour scores are also unchanged. Timing columns differ, as expected (reranking 35.6–36.8 s per source vs 42.0–43.6 s). The 27-file report bundle has a different SHA-256 (`b6500aac…` vs `eef23d5e…`); its contents include the new `environment` record and run-specific timings, so a different digest is expected and is not a result difference.
- **Observation:** unlike the 0.2.0 run, no `HF_TOKEN secret ... timed out` line appears (inferred cause: the parquet fetch and model stages now run as separate processes in the uv venv with HF tokens dropped, so the kernel never queries the Colab secrets vault).
- **Boundary:** saved outputs were inspected. FULL adaptation, BYOD and the rejected-archive path (REL12) were not exercised and remain required before release. Status remains **Candidate**; `metadata.dimer.clean_runtime_evidence` is unchanged.

### Colab CLI execution of revision 0.4.0-candidate (`a3ec1e4`, blob `641d29c1`) — 2026-10-15

- **File:** [`execution-evidence/2026-10-15/DIMER_MultiModel_Vision_Language_Retrieval_Workshop_a3ec1e4_colab-cli-t4.ipynb`](execution-evidence/2026-10-15/DIMER_MultiModel_Vision_Language_Retrieval_Workshop_a3ec1e4_colab-cli-t4.ipynb), SHA-256 `6501fb697d3eb6269f138b5563eaaefbbfa98531d2f532897ec80f325a0df1d5`, a byte-for-byte copy of the CLI's output notebook. The same notebook bytes (blob `641d29c139ccd5d55cb84bf10c6420d95aa45baf`) are the PR head of both `siglip2-vision-language-pipeline` #18 (`a3ec1e4`) and `blip-itm-pipeline` #12 (`c1a1b34`), so this one run covers both PRs; the file is byte-identical to `blip-itm-pipeline`'s `docs/execution-evidence/2026-10-15/DIMER_MultiModel_Vision_Language_Retrieval_Workshop_c1a1b34_colab-cli-t4.ipynb` (same SHA-256), named here after this repository's head.
- **Executor:** Google Colab CLI 0.7.4 on a fresh Colab Tesla T4 session (`suite-siglip2-a3ec1e4-4627`, 2026-10-15 00:43:58–00:53:21 UTC) via the workspace `colab-cli-serial-test-suite` (`colab new --gpu T4`, `colab exec -f`, `colab stop`; session released). The notebook was downloaded from GitHub at `siglip2-vision-language-pipeline` `a3ec1e4` and its git blob `641d29c139cc` checked before the session was created. Code cells ran in order in one kernel; this is CLI sequential execution, not a browser Run all, and the CLI records no execution counts, so order is evidenced by its `Executing cell k/37` log.
- **Result:** **PASSED** (STANDARD, default controls): 37/37 code cells, no error output, 563.0 s wall for the whole session; the completion summary reports `notebook_spec 2.5`, `tier STANDARD`, `test_gallery_images 389`, `test_gallery_captions 1729`, `full_adaptation: not run (STANDARD)`, `byod: not run`, `release_status: candidate`, and "Vision-language retrieval workshop complete". Full index digest/shape/reload ranking parity: PASS. Report bundle: 31 files, SHA-256 `73dc30c4…`.
- **Test-set exclusions (SVR2-m5 owner decision).** The split cell printed `test: [389, 1729]` and completed, so the pinned `TEST_EXCLUSIONS` checks passed (both ids present in the gallery-only row group, neither left in the test records, exactly 389 test photographs); photographs 30231 and 30181 and their captions are out of the test gallery. Train / validation stay 208 / 888 and 40 / 179; the role digest is now `6b806c7c…` (was `3cb7cbd2…`); parquet SHA-256 `4492465a…` unchanged. The dHash audit printed 58 cross-role pairs at Hamming ≤ 10 (train–test 48, validation–test 5, train–validation 5).
- **Flickr30k sensitivity row (SVR2-M1 = B).** `Salesforce/blip-itm-base-flickr` at `a492fd8e` was staged through the verified-snapshot path, loaded and scored on the same 389 photographs / 1,729 captions (not skipped); the principal rows are unchanged by it.
- **Test retrieval (389 photographs / 1,729 captions; image→text / text→image):**

  | Model | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 | rsum |
  |---|---|---|---|---|---|---|---|
  | SigLIP 2 | 0.789 | 0.928 | 0.961 | 0.721 | 0.903 | 0.939 | 5.242 |
  | SigLIP v1 | 0.781 | 0.923 | 0.956 | 0.716 | 0.885 | 0.927 | 5.188 |
  | BLIP ITC (COCO) | 0.740 | 0.905 | 0.943 | 0.652 | 0.863 | 0.918 | 5.022 |
  | BLIP ITC (Flickr30k, sensitivity) | 0.671 | 0.866 | 0.933 | 0.606 | 0.813 | 0.875 | 4.764 |

  Chance R@1 0.003 / 0.003; colour keyword 0.008 / 0.009. BLIP ITM reranking of the top 5 (SigLIP 2 / SigLIP v1 / BLIP ITC candidates): image→text 0.841 / 0.833 / 0.802; like-for-like coarse text→image 0.686 / 0.663 / 0.625 → reranked 0.730 / 0.733 / 0.707; ITM pair accuracy 0.779 / 0.766 / 0.681; 4,668 pair evaluations each. The exploratory paired table (24 comparisons, photograph bootstrap 2,000 resamples, seed 0) and the fixed-query gallery table (70 core photographs) printed.
- **Results vs the 0.3.0 run above:** not comparable number for number, because the test gallery lost two photographs (391 / 1,737 → 389 / 1,729). Qualitatively the same: SigLIP 2 ≥ SigLIP v1 > BLIP ITC on coarse R@1 (image→text 0.790 / 0.783 / 0.742 → 0.789 / 0.781 / 0.740; text→image 0.721 / 0.716 / 0.652 at three decimals in both), ITM reranking still lifts every candidate source, and the validation table (40 photographs) is identical. The Flickr30k row, the bootstrap intervals and the fixed-query table are new in this revision.
- **Boundary:** saved outputs were inspected. Forms and uploads were not exercised; FULL adaptation, BYOD and the rejected-archive path (REL12) remain required before release. Status remains **Candidate**; `metadata.dimer.clean_runtime_evidence` is unchanged.

## Row 82 review fixes — pre-registered grid and fresh test (2026-10-18)

- **Pre-registration:** `docs/siglip2-preregistration.md`, committed alone at `1b5f790` before any grid run, any
  fresh-test download and any code change.
- **Fresh test:** 96 photographs (16 per species) drawn by `tools/draw_fresh_test.py` on 2026-10-18 under section 4 of
  the pre-registration.
  - Pool: iNaturalist API v1, research-grade CC0, created 2025-03-01..2026-10-17; created dates of the drawn photographs
    run 2025-03-09..2026-10-10.
  - Screens: 0 observers shared with the 360; 24 candidates skipped as repeat observers; 0 same-place-same-day
    exclusions; 0 exact copies.
  - Near-duplicates: pHash threshold T = 10, set on 1,440 synthetic copies of the 360 (99th percentile; 0 of 64,620
    distinct pairs of the 360 within T). 0 hash hits against the 360 or within the fresh set, so no content check was
    needed.
  - Pins: manifest SHA-256 `019d38f4026011ee88fb6606ffdbecec23cd81f1e9eb7ac2452eb93edc20c862`, 15,015,075 bytes; log in
    `docs/siglip2-fresh-test-draw.json`.
- **Code correction:** `adapt()` kept the epoch with the best validation **accuracy** although it was documented as
  text-to-image mAP; it now selects on mAP. The recorded runs of the previous blob kept epoch 4, which is also their
  mAP-best epoch.
- **Evaluation of record:** the 2026-10-18 Colab CLI T4 run of `6df20c7` / blob `1f274277` (section below). Later
  identical hosted runs are reproductions. No fresh-test number was computed before it: the CPU build record below
  ran with the fresh test **masked**, and CI's integration job did not run on this PR (it sets
  `DIMER_FRESH_TEST_MASK=1` when it does). `RECORDED_EVALUATION` in `protocol.py` names this record, so a later
  identical run labels itself `reproduction of the pre-registered evaluation`.

### CPU build record, fresh test masked (2026-10-18; not a hosted run; REL13: exploratory, no development look)

- **What ran:** `tools/run_notebook.py` on the notebook generated at `b4a02ac` (the code commit; carried modules
  identical to the regeneration stamp). Build workstation CPU, float32, torch 2.14.0+cpu, transformers 4.57.6,
  `DIMER_NOTEBOOK_CI_PREINSTALLED=1`, `HF_HUB_OFFLINE=1`, **`DIMER_FRESH_TEST_MASK=1`**. 748 s end to end, exit 0.
- **Score-once check:** the run printed `preregistered: False`, `test_look: not scored (exploratory)`,
  `settings_differing: ['fresh_manifest_sha256', 'fresh_test_masked']`. Every file it wrote carries `_exploratory_1`.
  The fresh test was neither fetched nor scored, so there is **no development look** at it.
- **Grid (validation t2i mAP, best epoch):** C0 0.8758 (epoch 4), C1 0.8763 (epoch 5), C2 0.8667 (epoch 4); frozen
  0.6798. The rule kept **C0** ("no challenger beat C0 by 0.02"). Grid 503.3 s on CPU.
- **7b (C0):** validation mAP 0.680 → 0.874 → 0.854 → 0.863 → 0.876 → 0.872 → 0.854, epoch 4 kept by mAP; 164.7 s.
- **Exposed development test (96; continuity only):**

  | Metric | Majority | Neighbour | Frozen | Adapted (old → new) |
  |---|---|---|---|---|
  | accuracy | 0.167 | 0.26 | 0.76 | 0.792 → 0.792 |
  | macro F1 | 0.048 | 0.261 | 0.758 | 0.793 → 0.793 |
  | t2i mAP | 0.203 | 0.212 | 0.723 | 0.871 → 0.871 |
  | scientific-name mAP | | | 0.467 | 0.492 → 0.492 |

  "Old" is the 2026-10-09 Colab CLI T4 run of blob `b3ae91fd`. The configuration and epoch are unchanged (C0,
  epoch 4), so the numbers are unchanged.
- **Other outputs:** shapes top-1 1.0 / recall@1 1.0 before and after; adapter 45 tensors / 85,062,712 B; reload
  parity 8/8, max abs difference 0.0.
- **Fresh test:** not scored. Its first scoring is the hosted pre-registered run (next section).

### Pre-registered evaluation of record — Colab CLI T4, 2026-10-18 (`6df20c7`, blob `1f274277`)

- **REL13 label:** pre-registered evaluation of record. It is the first hosted run of the pre-registered settings and the
  first scoring of the fresh test by any run. The notebook printed `preregistered: True`,
  `test_look: pre-registered evaluation (this runtime)`, `settings_differing: []`, `adaptation_kept: True`.
- **What ran:** Colab CLI 0.7.4 `colab exec -f` on a fresh Colab Tesla T4, session `suite-siglip2-6df20c7-c27e`, start
  2026-10-18 00:06:32 UTC, 895.7 s, exit 0, 17/17 code cells, 0 error outputs. Source identity checked with
  `inspect_run.py`: 45/45 cells, same ids and order, sources identical to blob `1f274277`. Evidence boundary: CLI
  sequential execution, not a browser Run all; forms, BYOD and uploads were not exercised.
- **Evidence:** `docs/execution-evidence/2026-10-18-6df20c7/`, byte-for-byte copies; SHA-256 checked in the run VM
  directory, the copy and the commit.
  - `siglip2_vision_language_colab_6df20c7_colab-cli-t4_output.ipynb`: `5774248174fd0c109a0130f8dc51524912ea5fc77121dd00345f45a4a9f10230`
  - `exec.log`: `449378407f9baa7d5abece28cb3e7a49e29da0a9df0032662c4c6c303d086233`
  - `run_summary.json`: `6b6a6e07d71ca849fba04efee3d75104ff3bfe4a0dee6737f5a43aaff22704d3`
- **Grid (validation t2i mAP, best epoch):** C0 0.8758 (epoch 4), C1 0.8767 (epoch 5), C2 0.8667 (epoch 4); frozen
  0.6798. No challenger beat C0 by 0.02 (C1 +0.0009, C2 −0.0091), and C0 beat the frozen model by 0.196, so **C0** was
  kept with adaptation. Grid 103.1 s (34.4 / 36.6 / 32.0 s).
- **7b (C0):** validation mAP 0.680 → 0.874 → 0.854 → 0.863 → 0.876 → 0.872 → 0.854, epoch 4 kept; 31.7 s.
- **Fresh test (96, 16 per species):**

  | Metric | Majority | Neighbour | Frozen | Adapted | Δ adapted − frozen |
  |---|---|---|---|---|---|
  | accuracy | 0.167 | 0.208 | 0.552 | 0.677 | +0.125 |
  | macro F1 | 0.048 | 0.21 | 0.543 | 0.68 | +0.137 |
  | t2i mAP | 0.312 | 0.245 | 0.628 | 0.771 | +0.143 |
  | scientific-name accuracy | | | 0.396 | 0.458 | |
  | scientific-name macro F1 | | | 0.35 | 0.415 | |
  | scientific-name t2i mAP | | | 0.4 | 0.399 | |

  - Verdicts: frozen above the baselines; adapted vs frozen `improved` on mAP and on accuracy, as printed. Note (2026-10-18, peer review M2): `improved` was a zero-threshold label; no success criterion was pre-registered, and later runs print `point gain (no pre-registered criterion)`. `ap_fell` = [];
    `recall_fell` = [chipping_sparrow (0.62 → 0.5), song_sparrow (0.75 → 0.69)].
  - Per species, recall frozen → adapted (AP): American Goldfinch 0.69 → 0.69 (0.79 → 0.89), Chipping Sparrow
    0.62 → 0.5 (0.49 → 0.58), Dark-eyed Junco 0.56 → 0.81 (0.78 → 0.8), House Finch 0.56 → 0.62 (0.65 → 0.71), Song
    Sparrow 0.75 → 0.69 (0.56 → 0.75), White-throated Sparrow 0.12 → 0.75 (0.49 → 0.89).
  - No interval: the pre-registration fixes none, and the notebook states that one seeded run on 96 photographs gives
    no dispersion estimate.
  - The scientific-name prompts did not carry the retrieval gain (mAP 0.4 → 0.399).
  - The frozen model scores lower here than on the exposed development test (accuracy 0.552 vs 0.76). This run does not
    say why.
- **Re-derived from the printed per-species rows (all 16 per species):**
  - Accuracy: frozen 11 + 10 + 9 + 9 + 12 + 2 = 53 / 96 = 0.552; adapted 11 + 8 + 13 + 10 + 11 + 12 = 65 / 96 = 0.677;
    Δ 12 / 96 = 0.125.
  - mAP as the mean of the six rounded APs: 0.627 (printed 0.628) and 0.770 (printed 0.771); the gaps are rounding.
  - Majority macro F1: 2·(1/6)/(1 + 1/6) / 6 = 0.048.
  - The majority mAP differs between the two test sets (0.312 vs 0.203) because its constant scores are ranked by a
    stable sort (`metrics.average_precision`), so ties follow record order. The fresh records are grouped by species, so the
    recorded 0.312 is a tie-order artefact, not a baseline that beats the colour neighbour; order-free it is about 0.203
    (peer review M3). The recorded number stays as printed; `83b46c1` makes AP tie-aware and later runs print the
    corrected floor. The frozen and adapted scores are continuous logits, so the correction leaves them unchanged (checked
    in the round-2 CPU record below).
  - Exposed frozen: 73 / 96 = 0.760; mAP 0.722 from rounded APs (printed 0.723).
  - Grid rule and keep-frozen rule re-applied by hand, as above.
- **Exposed development test (96; continuity only):** majority 0.167 / 0.048 / 0.203, neighbour 0.26 / 0.261 / 0.212,
  frozen 0.76 / 0.758 / 0.723, adapted 0.792 / 0.793 / 0.871 (accuracy / macro F1 / mAP); validation 0.854 / 0.854 /
  0.876 (n 48, biased upward).
- **Other outputs:** shapes top-1 1.0 / recall@1 1.0 before and after; adapter 45 tensors / 85,062,712 B
  (`ec27138cec21aca9…`); reload parity 8/8, max abs difference 0.0, reloaded best epoch 4.
- **Against the CPU build record (fresh test masked):**
  - Equal: C0 0.8758 (epoch 4) and its whole curve; C2 0.8667 (epoch 4) and its whole curve; frozen 0.6798;
    the selection (C0, adaptation kept); the 7b curve and epoch 4; every exposed development test number; the frozen
    scientific-name row 0.479 / 0.459 / 0.467; shapes; parity; adapter size.
  - Different: C1 best 0.8767 vs 0.8763, and its epoch-5 value 0.877 vs 0.876 (GPU vs CPU numerics; best epoch 5 in
    both). The selection is unchanged, since C1 needed 0.8958.
  - Different, timings: grid 103.1 s vs 503.3 s; 7b 31.7 s vs 164.7 s; whole run 895.7 s (CLI) vs 748 s (CPU harness,
    no install or downloads).
  - The adapter SHA-256 differs (`ec27138c…` vs `3c5b4099…`, GPU vs CPU tensors). The CPU run's
    `delta_vs_frozen`, scientific-name, `ap_fell` and `recall_fell` rows were exposed-test rows, because the fresh test
    was masked; here they are fresh-test rows, so they are not compared.
- **CI:** the only CI run at `6df20c7` (39081093158) ran `unit` (success); `integration` was skipped. Its log has no
  fresh-test line. Earlier CI runs predate the fresh-test pins.

## Row 82 round 2 — peer review of PR #19 (2026-10-18)

Nothing below changes a pre-registered setting or re-draws the fresh test. The evaluation of record stays the
`6df20c7` run; every Major of the review is a disclosure.

- **Reproduction recorded (m12).** `e9ff7f9` (blob `3f173c9c`) ran on a fresh Colab T4 on 2026-10-18 and labelled itself
  `reproduction of the pre-registered evaluation`. Its fresh-test and exposed-test numbers equal the record. Evidence in
  `docs/execution-evidence/2026-10-18-e9ff7f9/`:
  - `siglip2_vision_language_colab_e9ff7f9_colab-cli-t4_output.ipynb`: `7c5807697d0d2cd121846a0649a7d671d1d63ab84d6a45d5006d6568cbaa4993`
  - `exec.log`: `a90e49529694396374e0676d155a40686bcbfcec6af3aadf7908e37dd875193d`
  - `run_summary.json`: `e913cd98a74a0e34bf444e9788d099d429e2a40314a62974fb9dc0cf8433fb70`
- **Disclosures (M1, M2, M4, m9, m10).** On the fresh test the gain is concentrated in one species: the White-throated Sparrow, which the frozen common-name prompt got wrong, went from 2 to 12 of 16 correct, while the other five species netted +2 of 80 (American Goldfinch 0, Chipping Sparrow −2, Dark-eyed Junco +4, House Finch +1, Song Sparrow −1), changes of one to four photographs that 16 per species cannot tell from noise. No primary endpoint or success criterion was pre-registered: accuracy moved from 53 to 65 of 96, and the unpaired 95 % Clopper–Pearson intervals overlap (frozen [0.447, 0.654], adapted [0.574, 0.769]), so the record shows a point improvement that n = 96 cannot establish. The recorded majority-floor mAP 0.312 is a tie-order artefact — the fresh records are grouped by species and the old metric ranked tied scores in record order — and is about 0.203 order-free, so it does not beat the colour neighbour (0.245); later runs print the corrected floor. The frozen model scored 0.552 on the fresh test against 0.76 on the exposed development test, a gap larger than the adaptation effect; two plausible, untested contributors are different photographers (excluding the 360's 222 observers removed 65–88 % of every species' pool) and the date cutoff against pretraining exposure (74 of the 96 exposed photographs predate SigLIP 2's release); the record cannot separate them.
- **Tie-order artefact (M3).** See the record's majority-mAP note above. Code from `83b46c1` gives tied scores the
  expected AP over every tie order.
- **Code (m1–m4, m13, s1, s3).** All of these apply from `83b46c1` and do not alter the record:
  - **m1:** the report is written atomically, and an empty or corrupt earlier report no longer breaks a run.
  - **m2:** under keep-frozen the adapter is neither restored nor exported.
  - **m3:** a reproduction prints `matches record` / `differs from record` against the numbers carried in
    `RECORDED_EVALUATION`.
  - **m4:** the pre-registered check now also covers the model id and revision (the values the record used). Device,
    dtype and library versions are recorded but not checked.
  - **m13:** the grid and training files are run-numbered.
  - **s1:** the mask accepts only `1`, `0` or unset.
  - **s3:** the per-image fresh-test predictions are saved, with an exact McNemar test and a paired bootstrap CI labelled
    "post-hoc analysis, not pre-registered; added 2026-10-18 after the record", display-only. The record saved no
    per-image rows, so the next identical hosted run (a reproduction, rule item 3) supplies the paired data.
- **Pre-registration edited in place (m5).** `449edfc` edited `docs/siglip2-preregistration.md` §5 in place: it replaced
  "Until then `RECORDED_EVALUATION = None`" with a pointer to the `6df20c7` record and the reproduction label. This was
  a status pointer, not a setting. The file is restored to its `1b5f790` text, and the pointer now lives in
  `docs/siglip2-preregistration-addendum-2026-10-18.md`.
- **Draw deviations (m7, m8).** None of these changed the drawn set (0 hash hits, 0 place/day exclusions):
  - The visual check of confirmed near-duplicate pairs is not implemented. There were no hits to check.
  - Photograph downloads were paced 0.2 s apart; API calls were paced at 1.1 s, against the pre-registered 1 s minimum
    for both.
  - The same-place screen uses public coordinates, which obscured observations blur.
  - The log has no draw timestamp or API snapshot id.
  - `fresh-song_sparrow-09` was observed on 2018-05-28 and uploaded on 2026-06-08. 35 of the 96 have `observed_on`
    different from their upload date, and 1 was taken before the SigLIP 2 release.
  - The observer exclusion removed 65–88 % of every species' pool (1496/1862, 557/772, 883/1007, 1425/1894, 1624/2081,
    667/1020).
- **Spent for decisions (s2).** The fresh test is now spent for decisions. Any change to the recipe, grid or prompts
  needs a new draw that also excludes these 96 photographs' observers.
- **Not done:**
  - **s4:** a digest check of the tensors that `restore_base` puts back. `restore_base` copies the remembered base
    exactly, and the round-2 CPU record re-reads the frozen numbers unchanged after the grid.
  - **s5:** the failing `integration` job on main pushes is out of scope for this PR.

### Round-2 CPU check, fresh test masked (2026-10-18; not a hosted run)

- **What ran:** `tools/run_notebook.py` on the notebook generated at `83b46c1` (the round-2 code commit), workstation
  CPU, float32, `DIMER_NOTEBOOK_CI_PREINSTALLED=1`, `HF_HUB_OFFLINE=1`, **`DIMER_FRESH_TEST_MASK=1`**. 1,219 s, exit 0.
  The workstation was shared with another job, so wall times are not comparable.
- **Score-once:** `preregistered: False`, `test_look: not scored (exploratory)`, `settings_differing:
  ['fresh_manifest_sha256', 'fresh_test_masked']`. The fresh test was neither fetched nor scored, so there was no look.
  Files carry `_exploratory_1`.
- **Corrected chance floor (tie-aware AP):** majority mAP on the exposed development test **0.203**. The old metric also
  gave 0.203 on this randomly ordered split.
- **Unchanged by the tie-aware AP:** frozen 0.76 / 0.758 / 0.723 and adapted 0.792 / 0.793 / 0.871, so the model scores
  have no ties. Colour neighbour 0.26 / 0.261 / 0.212.
- **Grid:** C0 0.8758, C1 0.8763, C2 0.8667, frozen 0.6798; C0 kept; 7b epoch 4.
- **New verdict labels:** `point gain (no pre-registered criterion)`, `criterion: none pre-registered`.
- **Other outputs:** reload parity 8/8 (maximum absolute difference 0.0).

### Reproduction at the round-2 head — Colab CLI T4, 2026-10-18 (`6176ddb`, blob `ef9048c4`)

- **What ran:** blob `ef9048c42298eeb31a407a498ba9762e19fff34a` (byte-identical to `git show 6176ddb:tutorials/siglip2_vision_language_colab.ipynb`;
  `inspect_run.py`: 45/45 cells, same ids and order, source identity OK, 0 error outputs), Colab CLI 0.7.4 `colab exec -f`, fresh Colab
  Tesla T4 (`gpu-t4-s-kkb-usw4b0-3j4as1x6ikv2v`, session `suite-siglip2-6176ddb-c27e`), started 06:41:13 UTC, 17/17 code cells,
  exit 0, 470.5 s; session stopped. CLI sequential execution, not a browser Run all; forms, BYOD and uploads not exercised.
- **Score-once:** `preregistered: True`, `settings_differing: []`, `test_look: reproduction of the pre-registered evaluation`.
  Nothing was changed in response to the test number (SCORE_ONCE_RULE item 3). The evaluation of record stays the `6df20c7` run.
- **Match with the record (m3):** the notebook printed `reproduction_vs_record: matches record`, `differing: []`. A normalised line
  diff of every printed output against the `6df20c7` record differs only in the embedded revision, the `test_look` labels, the
  new verdict wording (`point gain (no pre-registered criterion)`, `criterion: none pre-registered`), the majority-floor mAP, the
  added post-hoc and match lines, the adapter digest (`907fbe39…` vs `ec27138c…`, GPU non-determinism) and the added predictions
  file. Grid, training curve, validation, exposed development test, fresh-test model numbers, scientific-name prompts and
  per-species rows are identical.
- **Corrected chance floor:** fresh-test majority mAP printed **0.203** (tie-aware AP). The recorded 0.312 stays as printed in
  the record and is a tie-order artefact (see the record section); `record_match` excludes it. The frozen and adapted scores
  are unchanged under the tie-aware AP (0.628 / 0.771), so the model scores have no ties. The printed `fresh_test` line does
  not itself carry a "corrected baseline" label; the label is in the notebook's prose (Section 8) and here.
- **Evidence** in `docs/execution-evidence/2026-10-18-6176ddb/` (SHA-256 equal in the Colab WSL run folder, the working tree and
  the commit):
  - `siglip2_vision_language_colab_6176ddb_colab-cli-t4_output.ipynb`: `759752937110ac6feccfa3d2257f1c543971db6d578d12074b36ae6d7f5c7eab`
  - `exec.log`: `f4dcda8aa3f9c182cad81203365538d17a256a1f1ddabd961057795c400498df`
  - `run_summary.json`: `c08caff17bf3afc32bfbdc72c1b0164740513c88016b8210a0cde2c93cfbc188`
  - The run wrote `outputs/siglip2_vision_language_fresh_predictions.json` on the Colab VM, but the CLI suite saves only the
    executed notebook, the log and the summary, and the VM was released, so the per-image file is **not** in the evidence.

### Post-hoc paired analysis of the `6176ddb` reproduction (post-hoc analysis, not pre-registered; added 2026-10-18 after the record)

Display-only. It was added after the fresh test had been scored, it decides nothing, and no verdict field reads it.

| Quantity | Printed | Re-derived by the recorder |
|---|---|---|
| frozen / adapted correct (n = 96) | 53 / 65 | 53 / 65 (per-species recall × 16, record) |
| gained / lost | 18 / 6 | 18 − 6 = 12 = 65 − 53 |
| exact McNemar p (two-sided, binomial 6 of 24) | 0.022656 | 0.022656 |
| paired bootstrap 95 % CI on the accuracy difference (10,000 resamples, seed 20261018) | [0.0312, 0.2188] | [0.0312, 0.2188] = [3/96, 21/96], the exact multinomial percentiles of the 18 / 6 / 72 paired differences; the repo routine (seed 20261018) gives the same interval for 15 of 20 random image orders and [0.0312, 0.2292] for 5, so the upper end depends on the image order by one photograph (1/96) |

The re-derivation uses the printed counts, not the per-image file (not retrieved, above). Read it as a paired check of the
direction of the accuracy change on these 96 photographs, made after the fact; it is not a pre-registered test and does not
change the record's reading (no primary endpoint was pre-registered; the gain is concentrated in the White-throated Sparrow).

## Execution-log redaction (2026-10-18)

The four committed Colab CLI `exec.log` files named the operator's home directory in their two
`[colab] ... notebook` lines. Those two lines now read `<local-run-dir>/...`; no other byte changed, and the
executed notebooks and `run_summary.json` files are untouched. SHA-256 as captured → as committed:

| Log | Captured | Committed |
|---|---|---|
| `2026-10-09-20ffef4/exec.log` | `49c2bc4786dd3457173347cd0bc65bdce008384e8bc5773dbc6a93783d76b22e` | `f3e8ef83087762792077d84b89d3af3cc879d619296de0803c8fc7e723c804ff` |
| `2026-10-18-6df20c7/exec.log` | `449378407f9baa7d5abece28cb3e7a49e29da0a9df0032662c4c6c303d086233` | `2903d64fbbf8d4a65d197c3c65a6ae66a2dbd6a429804dd49482a2c08855bf07` |
| `2026-10-18-e9ff7f9/exec.log` | `a90e49529694396374e0676d155a40686bcbfcec6af3aadf7908e37dd875193d` | `a611a870123787bca77e5af85de5baf20af1ccf76aa5da5c58086cf604f516ae` |
| `2026-10-18-6176ddb/exec.log` | `f4dcda8aa3f9c182cad81203365538d17a256a1f1ddabd961057795c400498df` | `087c0ee7e0322c9bf8b921c9889835b649d24324aa900a2bf65957e3cea39f3c` |

Hashes quoted elsewhere in this file for these logs are the captured values.
