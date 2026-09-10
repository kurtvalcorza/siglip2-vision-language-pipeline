# SigLIP 2 Vision-Language Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/siglip2-vision-language-pipeline/blob/main/tutorials/siglip2_vision_language_colab.ipynb)

DIMER-oriented inference wrapper for **one immutable open-weight SigLIP 2 checkpoint**:

- model: `google/siglip2-base-patch16-224`
- pinned revision: `5ffaac51d5e2f3367f7dab0cad4be4cb07c0caa2`
- weight file: `model.safetensors`
- expected SHA-256: `612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b`
- expected size: `1,500,800,904` bytes
- upstream model license: Apache-2.0

The wrapper code in this repository is MIT licensed. The model weights retain Google's Apache-2.0 license.

## Status

**v1 release candidate.** The core inference contract and the real pinned checkpoint have been exercised on CPU CI. The tutorial source conforms to the DIMER Notebook Specification v1.0 `MULTI-CAPABILITY` profile; each merged notebook revision becomes release evidence only after its post-merge `main` clean-runtime execution gate passes. Production HTTP serving / DIMER worker packaging remains a separate serving-readiness milestone.

## v1 capabilities

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

Public operations:

- `zero_shot_classify()`
- `embed_image()`
- `embed_text()`
- `similarity()`
- `retrieve()`

## Live tutorial

The Colab notebook is registered as `MULTI-CAPABILITY` in [`tutorials/README.md`](tutorials/README.md). It exercises all five public operations against deterministic generated sample images, provides a gated BYOD/new-image path, reports synthetic tutorial sanity metrics, and writes:

- `classification.json`
- `image_embeddings.npz`
- `text_embeddings.npz`
- `similarity.csv`
- `retrieval.json`
- `metrics.json`
- `new_data_classification.json`
- `provenance.json`

Optional BYOD execution additionally writes `byod_classification.json` and `byod_image_embedding.npz`.

The `main` workflow executes the notebook top-to-bottom against the real pinned checkpoint and uploads its outputs as an artifact. The default sample images are synthetic smoke assets and are not an accuracy benchmark or production-fitness evidence.

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

Retrieval v1 is text-to-image retrieval over an in-memory image list. It returns the source index and score for each ranked hit.

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

Python 3.12 is the supported v1 runtime. The repository keeps exact direct pins in `pyproject.toml` and a fully version-pinned Linux/CPU reference graph in `requirements.lock.txt`.

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

Real-checkpoint integration:

```bash
RUN_INTEGRATION=1 pytest -m integration -q
python tools/run_notebook.py tutorials/siglip2_vision_language_colab.ipynb
```

## v1 scope boundaries

This repository does **not** claim to provide:

- object detection;
- semantic segmentation;
- image caption generation;
- OCR;
- calibrated zero-shot probabilities;
- universal classification thresholds;
- production HTTP serving or DIMER worker packaging.

Those require separate downstream heads, models, calibration, or serving work.
