# Model Profile Card

## SigLIP 2 Base Patch16 224

### Description

This repository packages `google/siglip2-base-patch16-224` as a DIMER-oriented vision-language inference pipeline. SigLIP 2 is a multilingual dual-encoder vision-language model from Google that maps images and text into a shared representation space and supports zero-shot image classification and image-text retrieval without task-specific classifier training.

### Hosted checkpoint

- **Model family:** SigLIP 2
- **Hosted checkpoint:** `google/siglip2-base-patch16-224`
- **Pinned revision:** `5ffaac51d5e2f3367f7dab0cad4be4cb07c0caa2`
- **Weight format:** safetensors
- **Weight SHA-256:** `612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b`
- **Weight size:** 1,500,800,904 bytes
- **Upstream license:** Apache-2.0
- **Pipeline code license:** MIT
- **Input resolution:** 224 x 224 through the upstream processor
- **Text max length:** 64 tokens for the v1 inference contract

The repository intentionally supports one checkpoint in v1. Other SigLIP 2 sizes, resolutions, and NaFlex variants are not silently substituted.

### Architecture and task family

SigLIP 2 uses separate vision and text Transformer towers trained into a shared embedding space. The hosted checkpoint is exposed as a vision-language encoder rather than a generative multimodal model.

Supported v1 operations:

1. zero-shot image classification against caller-supplied candidate labels;
2. image embedding extraction;
3. text embedding extraction;
4. image-text cosine similarity;
5. text-to-image retrieval over an in-memory candidate set.

### Zero-shot score semantics

SigLIP uses a sigmoid objective over image-text pairs. The pipeline returns the sigmoid of each model logit independently. Candidate-label scores are **not softmax-normalized** and are not required to sum to one.

These values are useful for ranking and comparative scoring. They are not presented as calibrated class probabilities and no universal decision threshold is provided.

The default classification prompt is:

`This is a photo of {label}.`

Changing the prompt can change scores and must be treated as part of the inference configuration.

### Text preprocessing contract

SigLIP 2 training lowercased text before tokenization and used a maximum sequence length of 64. The pinned checkpoint currently declares a plain Gemma tokenizer in its tokenizer metadata, which may not apply the SigLIP 2 lowercasing behavior automatically. This pipeline therefore explicitly lowercases all model-bound text before processor/tokenizer invocation.

The rule applies consistently to rendered zero-shot prompts, `embed_text()` input, text passed through `similarity()`, and retrieval queries. Caller-facing candidate-label strings are retained in their original form in returned classification results.

### Embedding semantics

Image and text features returned by the v1 pipeline are L2-normalized. `similarity()` and `retrieve()` operate in cosine-similarity space through the dot product of normalized embeddings.

Embedding similarity is representation similarity, not a guarantee of semantic correctness, factuality, identity, or suitability for a downstream decision.

### Reproducibility and provenance

The v1 repository provides:

- exact direct/dev/build pins in `pyproject.toml`;
- a fully version-pinned Linux/CPU reference graph in `requirements.lock.txt`;
- lock-parity CI;
- deterministic generated synthetic PPM tutorial assets with SHA-256 manifests;
- machine-readable provenance export through `build_provenance()` / `write_provenance()`;
- a live Colab notebook that covers all five public operations;
- a `main` CI gate configured to execute the notebook against the real pinned checkpoint and upload output artifacts.

The requirements file is a version lock rather than a cryptographic hash lock. The model weights themselves are independently pinned by immutable revision, expected byte size, and SHA-256.

### Intended uses

Appropriate uses include zero-shot image categorization, semantic image search, image-text retrieval, image and text feature extraction, comparative vision-language similarity analysis, and prototyping domain-specific image organization or triage workflows.

For consequential deployments, candidate labels, prompts, operating thresholds, subgroup behavior, and domain performance must be evaluated on representative local data.

### Out-of-scope claims

The hosted checkpoint is not exposed as an object detector, semantic-segmentation model, image-caption generator, OCR engine, calibrated probabilistic classifier, identity-recognition system, or universal content-safety classifier.

Downstream heads or separate models are required for detection and segmentation. Generative captioning requires a generative multimodal model rather than this encoder-only pipeline.

### Input handling and security

The pipeline accepts PIL images, image bytes, and local filesystem paths. Remote HTTP(S) image strings are rejected by design so downstream services do not inherit an implicit arbitrary-URL fetcher.

Service-layer deployments should still impose byte-size, decoded-resolution, batch-size, request-time, and concurrency limits before exposing the pipeline to untrusted callers.

### Checkpoint provenance and loading controls

The loader:

1. requests the exact pinned Hugging Face revision;
2. downloads only an allowlist of configuration, processor/tokenizer, and `model.safetensors` files;
3. rejects `.bin` pickle-style weight files;
4. verifies the exact expected safetensors size and SHA-256;
5. loads only from that verified local snapshot with `local_files_only=True`, `trust_remote_code=False`, and `use_safetensors=True`.

The byte digest and immutable revision establish the intended hosted weight artifact. A floating `main` revision is not used at runtime.

### Limitations and risks

Zero-shot performance is prompt- and domain-dependent. Similar-looking concepts, fine-grained classes, culturally specific concepts, small visual details, text-heavy images, distribution shift, and safety-critical edge cases may produce unreliable rankings. Multilingual capability does not imply uniform quality across languages or domains.

The fixed 224 x 224 v1 processor can lose fine detail or distort task-relevant spatial information relative to higher-resolution or NaFlex variants.

The synthetic tutorial samples are smoke assets only. Their behavior must not be interpreted as model accuracy or domain validation.

### Development status

**Status: v1 release candidate.**

The repository has a tested inference contract, a real-checkpoint CPU smoke path, reproducible reference environment, deterministic tutorial assets, and machine-readable provenance. Live tutorial execution on post-merge `main` remains the acceptance gate before this release candidate should be described as tutorial-ready. Production DIMER worker packaging, service-level resource limits, latency/SLO characterization, concurrency policy, observability, and deployment hardening remain separate serving-readiness work.
