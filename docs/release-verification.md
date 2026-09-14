# Release verification

`tutorials/siglip2_vision_language_colab.ipynb` (`MULTI-CAPABILITY`) is a **release candidate** until
the exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests,
JSON validation, code-cell compilation, and `tools/validate_release_assets.py` are necessary
checks but are **not** runtime evidence under DIMER Notebook Specification 1.1. This file is
the durable release-gate record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `MULTI-CAPABILITY`
  profile, the notebook-spec version and the standalone carrier; `metadata.dimer` declares that profile, spec `1.1`, `standalone: true` and `generated_from` (repository, module commit, the four carried modules, their combined SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on the
  primary path; one cell tagged `embedded_module` per module of `src/siglip2_pipeline/` (four, in dependency order:
  `config`, `model`, `provenance`, `pipeline`), each equal to its module after the generator's documented rewrites (the
  `DEFAULT_WEIGHTS_DIR` rule, the `resolve_weights_path` checkout-convenience line, and the removal of package-relative
  imports); the inline `MANIFEST` equal to the committed `weights/siglip2-base-patch16-224/dimer-base-manifest.json`
  (9 files); the inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical to
  `tools/build_notebook.py` output; the pinned-install cell with its restart-on-stale-import guard; `NOTEBOOK_SOURCE`
  recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cells (and repeated in the inline manifest,
  which the notebook asserts against the module before fetching), the revision is a 40-hex immutable commit, and the same
  identity string appears in `README.md` and `MODEL_CARD.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `Siglip2Pipeline.from_pretrained(device="cpu", weights_dir=...)`, `validate_inputs`, `zero_shot_classify`, `embed_image`,
  `embed_text`, `similarity`, `retrieve`, `evaluation_report`, `write_provenance`), the ceiling print (`TEXT_MAX_LENGTH`,
  `DEFAULT_PROMPT_TEMPLATE`, the 224×224 image contract), the exports, the learner-facing statements (uncalibrated sigmoid
  scores, prompt/label dependence, L2-normalized representations with no intrinsic metric, descending-score ordering, CPU-only
  reference, the unsupported tasks) and the gated-off BYOD default listed in the validator; forbidden patterns
  (credential-in-URL, any `git clone` / `github.com` / repository import on the primary path, a mutable `revision='main'`,
  direct `AutoModel` / `AutoProcessor` / `get_image_features` / `torch.sigmoid` / `snapshot_download` /
  `from huggingface_hub import` / `from transformers import` use **outside the carried module cells**, any `worker.run(` /
  `worker_cli(` / `subprocess.run([` outside the generator-owned install cell, `trust_remote_code=True`, `pickle.load`,
  `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter, single H1 (fenced code blocks excluded), required heading order, and the checkpoint
  invariants section.

CI also installs the frozen CPU reference environment (`requirements.lock.txt`), runs `ruff`, `scripts/check_lock.py`,
`tools/build_notebook.py --check`, and the offline unit suite (`tests/`, including `test_fleet_snapshot.py`,
`test_role_helpers.py`, `test_notebook_parity.py`; no weights, injected downloader). These are source/provenance and unit
checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime; the notebook loads on CPU even on a GPU host | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel | Kaggle CPU kernel, Python 3.12 image | Reproducible clean-room executor of the same class; the notebook is pushed verbatim plus one leading shim cell that provides `google.colab` and chdirs to a scratch directory (no repository checkout is needed — the notebook is standalone) |
| Repository CI integration job (`tools/run_notebook.py`) | GitHub-hosted Ubuntu runner, the frozen CPU reference environment with `DIMER_NOTEBOOK_CI_PREINSTALLED=1` | Executes the standalone notebook's code cells sequentially against the real pinned weights (fetched by the notebook's own `stage_missing_files` into `weights/siglip2-base-patch16-224/` under the workspace, where the committed small files already sit); a **pre-flight** on the locked stack, not a fresh-boundary run of the inline `PINS` and not promotion evidence on its own |
| Local WSL harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU runtime (Colab, or the Kaggle
   executor above) with **no repository checkout** and a clean model cache;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their
   defaults for the sample path: `USE_BYOD = False`, `BYOD_PATH = ""`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the module commit recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS` (= `pyproject.toml`);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the four carried module cells execute (define `Siglip2Pipeline`, `validate_inputs`, `evaluation_report`, `top1_accuracy`,
     `recall_at_1` and the rest) with no import of the repository package;
   - pinned `google/siglip2-base-patch16-224` acquisition at the immutable revision through the package:
     the inline `MANIFEST` is asserted against the module identity and written to `weights/siglip2-base-patch16-224/`,
     `stage_missing_files(WEIGHTS_DIR, allow_download=True)` reports all nine manifest entries on a clean runtime,
     `verify_snapshot` returns the manifest dict (the existing `verify_checkpoint` re-hashes every entry and asserts the pinned
     weight digest and byte count), and `Siglip2Pipeline.from_pretrained(device="cpu", weights_dir=WEIGHTS_DIR)` reports
     `checkpoint_source == 'explicit_path'` and `manifest_verified == True`;
   - the three synthetic PPM images generated in code with SHA-256 `b38ff0c9…` / `2e3e6576…` / `f0f4c38c…` (equal to the
     repository's `examples/sample-data/SHA256SUMS`) and the ceilings printed;
   - `validate_inputs` writes `outputs/siglip2_vision_language_input_manifest.json` (verdict `accepted`, one recorded
     rejection finding from the remote-URL probe);
   - all five operations execute (`zero_shot_classify` ×3, `embed_image`, `embed_text`, `similarity`, `retrieve` ×3);
   - `evaluation_report` writes `outputs/siglip2_vision_language_evaluation_report.json` with verdict `sample-sanity`,
     `top1_accuracy` and `recall_at_1` (both `1.0` on the synthetic set in every recorded run of the previous carrier);
   - the new-data `yellow_square.ppm` classification and the exports (`classification.json`, `image_embeddings.npz`,
     `text_embeddings.npz`, `similarity.csv`, `retrieval.json`, `metrics.json`, `new_data_classification.json`,
     `provenance.json`, `siglip2_vision_language_result.json`) written with `NOTEBOOK_SOURCE`, model revision, model licence,
     runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, transformers, device),
   model identifier and immutable revision, whether the model cache was clean, outcome, produced
   outputs, and any warning or applicable `SHOULD` deviation in the table below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/siglip2_vision_language_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/siglip2_vision_language_colab.ipynb`). Wall times, when recorded,
are the sum of per-cell times reported by the executor and include installs and the model download;
they are measurements for the stated runtime, not general estimates.

### Manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
| 2026-09-14 | `0266f18` / blob `d8c19e7f4b8e` | Kaggle CPU (`kurtvalcorza/dimer-nb2-siglip2-vision-language` v3) | Default sample path, all 14 code cells | 268.5 s | **PASS** — 14/14 ok (1 restart after install cell), 18 files, 1539 MB staged |

## Current status

A clean-runtime execution of the standalone notebook is recorded above (Kaggle CPU, 268.5 s, 14/14 ok, kernel `dimer-nb2-siglip2-vision-language` v3). Static validation (`tools/validate_release_assets.py`), nbformat validation, a
`compile()` sweep over every code cell, and the offline unit suite passed on the tutorial source at
the candidate revision, which is necessary but not sufficient. The registry status remains
**Candidate** until a reviewer confirms a recorded run against the notebook blob under review and
an integrator promotes it; promotion is not performed by the builder. Three facts a reviewer should
weigh: `stage_missing_files` was exercised only with an injected downloader in the unit suite (the
real `hf_hub_download` fetch of the 1.5 GB `model.safetensors` into a fresh `weights/siglip2-base-patch16-224/` has not been
executed); `Siglip2Pipeline.from_pretrained(weights_dir=...)` was exercised only with a stand-in `load_components` (the real
`AutoModel.from_pretrained` on the manifest-described directory has not been executed on this path); and the standalone
carrier itself — executing the four carried module cells in a runtime that has no repository checkout — has been validated
statically only (parity PASS, carrier probe up to the fetch), never run. The previous repository-installing notebook did
run on `main` CI and on Kaggle (recorded in `tutorials/README.md`) at revision `5ffaac51d5e2f3367f7dab0cad4be4cb07c0caa2`; that evidence predates the
standalone carrier and does not transfer to it.
