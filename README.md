# SigLIP 2 Vision-Language Pipeline

DIMER-oriented inference wrapper for **one immutable open-weight SigLIP 2 checkpoint**:

- model: `google/siglip2-base-patch16-224`
- pinned revision: `5ffaac51d5e2f3367f7dab0cad4be4cb07c0caa2`
- weight file: `model.safetensors`
- expected SHA-256: `612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b`
- expected size: `1,500,800,904` bytes
- upstream model license: Apache-2.0

The wrapper code in this repository is MIT licensed. The model weights retain Google's Apache-2.0 license.

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

## Score semantics

SigLIP uses independent sigmoid scores for image-text pairs. `zero_shot_classify()` therefore returns one sigmoid score per candidate label and **does not softmax-normalize across labels**. Scores do not need to sum to 1. They should be interpreted comparatively and validated for the deployment domain rather than as calibrated class probabilities.

The default prompt template follows the upstream Transformers example:

```text
This is a photo of {label}.
```

## Embeddings and retrieval

`embed_image()` and `embed_text()` return L2-normalized vectors. `similarity()` and `retrieve()` use cosine similarity through the dot product of those normalized vectors.

Retrieval v1 is text-to-image retrieval over an in-memory image list. It returns the source index and score for each ranked hit.

## Input safety

Image inputs may be:

- a local filesystem path;
- `bytes` containing an image;
- a `PIL.Image.Image`.

Remote `http://` and `https://` image strings are rejected intentionally. The pipeline does not act as a network fetcher, avoiding an SSRF-style interface in downstream services.

## Supply-chain controls

`load_pipeline()`:

1. downloads only an allowlisted set of checkpoint files at the pinned Hugging Face revision;
2. requires `model.safetensors` and rejects pickle `.bin` weights;
3. verifies the exact safetensors byte size and SHA-256 before model load;
4. loads the verified local snapshot with `trust_remote_code=False`, `use_safetensors=True`, and `local_files_only=True`.

## Installation

Python 3.12 is the supported v1 runtime.

```bash
python -m pip install -e '.[dev]'
```

## Tests

No-network contract tests:

```bash
pytest -m "not integration"
ruff check .
```

Real-checkpoint integration test:

```bash
RUN_INTEGRATION=1 pytest -m integration -q
```

The integration test downloads roughly 1.5 GB of weights and is intended for `main` CI or explicit workflow dispatch rather than every pull request.

## v1 scope boundaries

This repository does **not** claim to provide:

- object detection;
- semantic segmentation;
- image caption generation;
- calibrated zero-shot probabilities;
- universal classification thresholds;
- production HTTP serving or DIMER worker packaging.

Those require separate downstream heads, models, calibration, or serving work.
