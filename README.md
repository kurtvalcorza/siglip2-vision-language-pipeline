# SigLIP 2 Vision-Language Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/siglip2-vision-language-pipeline/blob/main/tutorials/siglip2_vision_language_colab.ipynb)

DIMER-oriented inference and bounded fine-tuning wrapper for **one immutable open-weight SigLIP 2 checkpoint**:

- model: `google/siglip2-base-patch16-224`
- pinned revision: `5ffaac51d5e2f3367f7dab0cad4be4cb07c0caa2`
- weight file: `model.safetensors`
- expected SHA-256: `612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b`
- expected size: `1,500,800,904` bytes
- upstream model license: Apache-2.0

The wrapper code in this repository is MIT licensed. The model weights retain Google's Apache-2.0 license.

## Status

**Candidate.** The 2026-09-20 Kaggle T4 run of blob `15f946ab` (14/14 ok, one restart after the install cell) is history: the notebook now builds a hash-locked isolated uv environment (generator /2.2, no restart) and declares NOTEBOOK_SPEC 2.2, so its blob changed and the status is **Candidate** (REL14) until a clean run of the new blob is recorded in `docs/release-verification.md`. The inference contract, the adaptation contract and the real pinned checkpoint have been exercised locally (CPU and an RTX 5070 Ti), on CPU CI, and — for the `E2E` standalone tutorial at blob `15f946ab` — in a clean Kaggle Tesla T4 runtime on 2026-09-20 (recorded in `docs/release-verification.md`). A later notebook revision returns to Candidate until a clean-runtime execution of that exact blob is recorded. Production HTTP serving / DIMER worker packaging remains a separate serving-readiness milestone.

## Capabilities

```python
from siglip2_pipeline import load_pipeline

pipe = load_pipeline()

scores = pipe.zero_shot_classify(
    "photo.jpg",
    ["flooded street", "normal road", "fallen electrical pole"],
)

image_embeddings = pipe.embed_image(["a.jpg", "b.jpg"])
text_embeddings = pipe.embed_text(["flooded street", "normal road"])
cosine_matrix = pipe.similarity(["a.jpg", "b.jpg"], ["flooded street", "normal road"])
hits = pipe.retrieve("flooded street", ["a.jpg", "b.jpg"], top_k=2)
```

Public inference operations:

- `zero_shot_classify()`
- `embed_image()`
- `embed_text()`
- `similarity()`
- `retrieve()`

## Adaptation contract

```python
from siglip2_pipeline import (
    Siglip2Pipeline, build_sample_dataset, class_names, fetch_corpus, load_byod_dataset,
    read_corpus, split_dataset, validate_dataset,
)

splits = build_sample_dataset(read_corpus(fetch_corpus()), seed=42)   # 360 CC0 iNaturalist bird photographs, 216 / 48 / 96
# or: splits = split_dataset(load_byod_dataset("my_photos.zip"), seed=42)  # labels.csv: id, file, label
classes = class_names(splits["train"])

pipe = Siglip2Pipeline.from_pretrained(weights_dir="weights/siglip2-base-patch16-224")  # CUDA when visible
frozen = pipe.evaluate(splits["test"], classes=classes)            # accuracy, macro_f1, t2i_map, per_class
result = pipe.adapt(splits["train"], splits["validation"], epochs=6, lr=5e-5, trainable_vision_layers=2)
adapted = pipe.evaluate(splits["test"], classes=classes)
pipe.save_artifact("outputs/adapter")                             # adapter.safetensors + manifest.json
again = Siglip2Pipeline.from_artifact("outputs/adapter", weights_dir="weights/siglip2-base-patch16-224")
```

- `validate_dataset(records)` checks `{id, image, label}` records structurally (decodable image up to `MAX_IMAGE_SIDE` = 4096 px, label of at most 64 plain characters, 8..20,000 records over 2..100 labels, unique ids) and returns a manifest with a dataset digest; `split_dataset` is a seeded, stratified, pixel-digest-deduplicated split; `check_split_disjoint` asserts no image is shared.
- `evaluate(records, *, classes, prompt_template, class_names_map)` embeds every prompt once and every photograph once, scores them with the model's own `logits_per_image` (sigmoid-scaled cosine with the learned `logit_scale` / `logit_bias`) and returns accuracy, macro F1, per-class recall / precision / F1 / average precision and text-to-image mAP (`metrics.classification_metrics`), plus `predictions`, `verdict` (`measured` / `measured-small-sample`) and `adapted`.
- `adapt(train, val=None, *, epochs=6, lr=5e-5, batch_size=16, trainable_vision_layers=2, seed=0, progress=None)` trains only the last `trainable_vision_layers` blocks of the vision tower, its post-layernorm and its attention-pool head (21,264,384 of 375,187,970 parameters by default) with SigLIP's pairwise sigmoid loss against the frozen text tower's prompt features; AdamW, gradient clipping at 1.0, seeded shuffling, epoch 0 recorded as the frozen model, the epoch with the highest validation text-to-image mAP kept (the final epoch without validation). The update is transactional: an exception restores the frozen weights.
- `save_artifact(dir)` writes the trained tensors as `adapter.safetensors` plus a `manifest.json` (format `org.valcorza.siglip2-base-patch16-224.adapter.v1`: base id, revision and weight digest, classes, prompt template, tensor names, file size and SHA-256, training configuration, epoch history); `from_artifact(dir)` re-verifies the base snapshot, checks the manifest, the digest and the exact tensor set before deserialising, refuses any tensor outside the vision tower, and overlays the tensors onto a freshly loaded base.
- `majority_baseline` and `colour_neighbour_baseline` (`metrics.py`) are the two non-neural references the tutorial scores beside the frozen and adapted model.

## Live tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/siglip2-vision-language-pipeline/blob/main/tutorials/siglip2_vision_language_colab.ipynb)

`tutorials/siglip2_vision_language_colab.ipynb` is declared `E2E` (kind `TUTORIAL`) under DIMER Notebook Specification 2.5 and is **standalone** (§4): generated by `tools/build_notebook.py`, it carries the package's seven modules, the model identity (`google/siglip2-base-patch16-224` at the immutable revision `5ffaac51d5e2f3367f7dab0cad4be4cb07c0caa2`), the 8-file manifest digests and the runtime pins, so the exported notebook runs without this repository (parity enforced by `tests/test_notebook_parity.py` and `tools/validate_release_assets.py`). It stages and digest-verifies the snapshot, fetches 360 digest-pinned CC0 iNaturalist photographs of six bird species and splits them per species without leakage, classifies the three deterministic sample shapes through the inference contract with an input manifest and a rejection probe, scores the frozen model's zero-shot accuracy, macro F1 and text-to-image mAP on the 96 photographs of the exposed development test beside the majority-floor and colour-nearest-neighbour baselines and a second prompt set, chooses the fine-tuning configuration on validation with a pre-registered grid, runs a bounded fine-tuning of the vision tower's last blocks with validation-mAP epoch selection, scores a 96-photograph fresh test only under the pre-registered settings, re-scores the shapes, exports the adapter and reloads it with verified parity, and writes:

- `siglip2_vision_language_train.csv`
- `siglip2_vision_language_input_manifest.json`
- `siglip2_vision_language_evaluation_report.json`
- `siglip2_vision_language_shapes.json`
- `siglip2_vision_language_adapter/` (`adapter.safetensors`, `manifest.json`)
- `siglip2_vision_language_result.json`
- `provenance.json`

The default path runs on CPU and uses CUDA automatically when present (about three minutes on the build workstation's CPU after the downloads, longer on a 2-vCPU hosted runtime; about a minute and a half on an RTX 5070 Ti). The metrics it prints are one seeded split of one 360-photograph sample — evidence that the adaptation contract works, not an accuracy benchmark or production-fitness evidence. The `main` integration workflow executes the notebook's code cells on the frozen CPU reference environment as a pre-flight; see `tutorials/README.md` for the registry and `docs/release-verification.md` for the release gate.

### Supplemental workshop: vision-language retrieval

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/siglip2-vision-language-pipeline/blob/main/tutorials/DIMER_MultiModel_Vision_Language_Retrieval_Workshop.ipynb)

`tutorials/DIMER_MultiModel_Vision_Language_Retrieval_Workshop.ipynb` (`MULTI-CAPABILITY` / `WORKSHOP`, Tesla T4) compares SigLIP 2, SigLIP v1 and BLIP ITC as first-stage retrievers on a pinned VizWiz-Captions gallery and reranks their shortlists with BLIP ITM; an optional `FULL` tier adds bounded BLIP adaptation and an optional BYOD archive runs the same contracts on your own photographs. It is the same notebook as in `blip-itm-pipeline`, hand-maintained rather than generated, and its status is **Candidate** (see `tutorials/README.md`). The release status below applies to the `E2E` notebook only.

## Release status

**Candidate** — **Pre-registered evaluation of record (2026-10-18).** The row 82 revision, commit `6df20c7` / blob `1f274277` (generator `build_notebook.py/2.2`, `notebook_spec` 2.5, embedded revision `188c43c`) executed on a fresh Colab Tesla T4 (Colab CLI 0.7.4 sequential execution, not a browser `Run all`; 17/17 code cells, 0 errors, 895.7 s) under the pre-registered settings (`preregistered: True`, `settings_differing: []`); the grid kept C0, epoch 4; fresh test (96) accuracy frozen 0.552 → adapted 0.677, macro F1 0.543 → 0.68, t2i mAP 0.628 → 0.771 (majority 0.167 / 0.048 / 0.312, colour neighbour 0.208 / 0.21 / 0.245). Evidence in `docs/execution-evidence/2026-10-18-6df20c7/`, record in `docs/release-verification.md`. Forms, BYOD and uploads were not exercised. History: **Colab T4 one-pass run of the isolated-environment revision (2026-10-09).** Commit `20ffef4` / blob `b3ae91fdb393` (generator `build_notebook.py/2.2`, `notebook_spec` 2.2, worker `google.colab` stubs with a `ModuleSpec`) executed in one pass with no restart and 0 errors on a fresh Colab Tesla T4 (Colab CLI 0.7.4 sequential execution, not a browser `Run all`; 15/15 code cells, 377.5 s, `cuda`); evidence in `docs/execution-evidence/2026-10-09-20ffef4/`. The 2026-09-20 Kaggle T4 run of blob `15f946ab` (14/14 ok, one restart after the install cell) is history: the notebook now builds a hash-locked isolated uv environment (generator /2.2, no restart) and declares NOTEBOOK_SPEC 2.2, so its blob changed and the status is **Candidate** (REL14) until a clean run of the new blob is recorded in `docs/release-verification.md`. Previously, the `E2E` notebook blob `15f946ab` (committed at `f3dd43e`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-20 (14/14 ok (1 restart after install cell), 423.7 s); the record is in `docs/release-verification.md` and `STATUS.md`. Static and unit checks — including the standalone generator parity checks — are necessary but were never the evidence; the hosted run is. A later change to the carried modules or the notebook returns the status to Candidate until re-verified.

**Reading the record (peer review, 2026-10-18).** On the fresh test the gain is concentrated in one species: the White-throated Sparrow, which the frozen common-name prompt got wrong, went from 2 to 12 of 16 correct, while the other five species netted +2 of 80 (American Goldfinch 0, Chipping Sparrow −2, Dark-eyed Junco +4, House Finch +1, Song Sparrow −1), changes of one to four photographs that 16 per species cannot tell from noise. No primary endpoint or success criterion was pre-registered: accuracy moved from 53 to 65 of 96, and the unpaired 95 % Clopper–Pearson intervals overlap (frozen [0.447, 0.654], adapted [0.574, 0.769]), so the record shows a point improvement that n = 96 cannot establish. The recorded majority-floor mAP 0.312 is a tie-order artefact — the fresh records are grouped by species and the old metric ranked tied scores in record order — and is about 0.203 order-free, so it does not beat the colour neighbour (0.245); later runs print the corrected floor. The frozen model scored 0.552 on the fresh test against 0.76 on the exposed development test, a gap larger than the adaptation effect; two plausible, untested contributors are different photographers (excluding the 360's 222 observers removed 65–88 % of every species' pool) and the date cutoff against pretraining exposure (74 of the 96 exposed photographs predate SigLIP 2's release); the record cannot separate them.

## Score semantics

SigLIP uses independent sigmoid scores for image-text pairs. `zero_shot_classify()` therefore returns one sigmoid score per candidate label and **does not softmax-normalize across labels**. Scores do not need to sum to 1. They should be interpreted comparatively and validated for the deployment domain rather than as calibrated class probabilities.

The default prompt template is:

```text
This is a photo of {label}.
```

## Text preprocessing compatibility

SigLIP 2 was trained with text lowercased before tokenization and a maximum text length of 64. The pinned checkpoint currently identifies a plain Gemma tokenizer in its tokenizer metadata, so v1 explicitly lowercases model-bound text before calling the processor. This compatibility shim applies to zero-shot prompts, text embeddings, similarity, and retrieval queries. Caller-facing labels are preserved exactly as supplied.

## Embeddings and retrieval

`embed_image()` and `embed_text()` return L2-normalized vectors. `similarity()` and `retrieve()` use cosine similarity through the dot product of those normalized vectors.

Retrieval is text-to-image retrieval over an in-memory image list. It returns the source index and score for each ranked hit.

## Machine-readable provenance

```python
from siglip2_pipeline import build_provenance, load_pipeline, write_provenance

pipe = load_pipeline()
record = build_provenance(pipeline=pipe)
write_provenance("outputs/provenance.json", pipeline=pipe)
```

The record includes model ID, immutable revision, weight filename/SHA-256/size, verified checkpoint source and path, processor contract, prompt template, score/embedding semantics, Python version, platform, and runtime package versions.

## Input safety

Image inputs may be:

- a local filesystem path;
- `bytes` containing an image;
- a `PIL.Image.Image`.

Remote `http://` and `https://` image strings are rejected intentionally. The pipeline does not act as a network fetcher.

## Supply-chain controls

`load_pipeline()`:

1. resolves offline weights through `weights_path`, `SIGLIP2_WEIGHTS_DIR`, dev repo `weights/siglip2-base-patch16-224`, or pinned Hugging Face revision fallback;
2. verifies snapshot files against `dimer-base-manifest.json` when present (checking hashes and sizes of configurations and weights);
3. rejects unsafe serialized formats (`.bin`, `.pt`, `.pth`, `.ckpt`, `.pkl`, `.pickle`, `.h5`, `.msgpack`);
4. verifies the exact safetensors byte size and SHA-256 before model load;
5. loads the verified local snapshot with `trust_remote_code=False`, `use_safetensors=True`, and `local_files_only=True`.

## Reproducible reference environment

Python 3.12 is the supported runtime. The repository keeps exact direct pins in `pyproject.toml` and a fully version-pinned Linux/CPU reference graph in `requirements.lock.txt`.

```bash
python -m pip install -r requirements.lock.txt
python -m pip install --no-deps --no-build-isolation -e .
python scripts/check_lock.py
```

`requirements.lock.txt` records the exact dependency versions proven by the real-checkpoint `main` CI path, including the official CPU PyTorch wheel. It is a version lock, not a cryptographic hash lock.

## Tests

```bash
ruff check .
pytest -m "not integration"
```

`tests/test_model_backed.py` (evaluate / adapt / artifact round trip / loader scope / transactional restore, and the same path on CUDA where visible) runs only when the snapshot is staged under `weights/siglip2-base-patch16-224/`; the photograph cache `weights/inat-birds/` is git-ignored and filled by `fetch_corpus()`.

Real-checkpoint integration:

```bash
RUN_INTEGRATION=1 pytest -m integration -q
python tools/run_notebook.py tutorials/siglip2_vision_language_colab.ipynb
```

## Scope boundaries

This repository does **not** claim to provide:

- object detection;
- semantic segmentation;
- image caption generation;
- OCR;
- calibrated zero-shot probabilities;
- universal classification thresholds;
- fine-tuning of the text tower, the embeddings, `logit_scale` or `logit_bias`, or any adaptation beyond the vision tower's last blocks;
- production HTTP serving or DIMER worker packaging.

Those require separate downstream heads, models, calibration, or serving work.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.

**Evaluation design (row 82 review, 2026-10-18).** The 96 test photographs of the seeded split were scored in every earlier run, and the fine-tune's trainable scope (vision tower, text tower frozen) was kept at build time because it gained on an earlier 48-photograph test split whose record is lost. Those 96 are now the **exposed development test** (continuity only). The configuration is chosen on validation by the grid in `docs/siglip2-preregistration.md` (C0 2 blocks @ 5e-5, C1 4 blocks @ 5e-5, C2 2 blocks @ 1e-5; margin 0.02; keep C0; keep frozen unless beaten), and a **fresh test** of 96 new iNaturalist CC0 photographs (16 per species, created on or after 2025-03-01, one per observer, none of the 360's 222 observers; `docs/siglip2-fresh-test-draw.json`) is scored only under the pre-registered settings; see `docs/release-verification.md` for the recorded evaluation and its reproductions. The frozen text tower is kept by design and was not re-tested. **Pretraining overlap (DAT9):** SigLIP 2's WebLI training set has no published instance manifest, so overlap with these public photographs cannot be ruled out; 255 of the 360 (74 of the 96 exposed) were created before SigLIP 2's release on 2025-02-20, and the fresh test's 2025-03-01 cutoff narrows but does not remove the risk.
