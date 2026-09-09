---
license: Apache-2.0
model_card_spec: "1.0"
tags:
  - vision-language
  - zero-shot-image-classification
  - image-text-retrieval
  - embeddings
base_model: google/siglip2-base-patch16-224
---

# SigLIP 2 Base Patch16 224 (v1.0)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-google%2Fsiglip2--base--patch16--224-ffcc4d?style=flat)](https://huggingface.co/google/siglip2-base-patch16-224)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Pipeline License: MIT](https://img.shields.io/badge/Pipeline%20License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

###### Description

SigLIP 2 Base Patch16 224 is an open-weight multilingual vision-language dual-encoder foundation model developed by Google (`google/siglip2-base-patch16-224`, pinned revision `5ffaac51d5e2f3367f7dab0cad4be4cb07c0caa2`), packaged by this repository as a verified DIMER inference pipeline. Built with separate vision and text Transformer towers (~375.2M parameters in `model.safetensors`), it maps raw images and text sequences into a unified 768-dimensional representation space. At inference time, the model functions strictly as an encoder: zero-shot classification is achieved by computing pairwise sigmoid cross-entropy scores between a candidate image and prompt-expanded text strings, while retrieval and similarity operate via dot-product cosine distance across L2-normalized embedding vectors. Adaptation occurs purely through inference-time prompt conditioning without parameter updating. This repository contributes a hardened supply-chain wrapper: cryptographic SHA-256 and byte-size verification before loading, complete exclusion of unsafe pickle (`.bin`) checkpoints, local-only snapshot execution (`trust_remote_code=False`), automated text lowercasing to preserve SigLIP 2 training fidelity, strict rejection of remote HTTP(S) image fetches to prevent SSRF, and full provenance tracking.

#### Intended Use and Limitations

The sections below outline the primary machine learning tasks, targeted user cohorts, and explicit capability boundaries established for this pipeline.

###### Primary Intended Uses

The primary intended uses of this pipeline comprise five technical vision-language capabilities:
1. Zero-shot image classification (`Siglip2Pipeline.zero_shot_classify`): Ranking arbitrary candidate textual labels against an input image using independent sigmoid activations without task-specific training.
2. Image feature extraction (`Siglip2Pipeline.embed_image`): Generating dense, unit-norm 768-dimensional visual vectors for indexing, clustering, and vector search.
3. Text feature extraction (`Siglip2Pipeline.embed_text`): Generating dense, unit-norm 768-dimensional linguistic vectors from natural language descriptions up to 64 tokens.
4. Multimodal cosine similarity (`Siglip2Pipeline.similarity`): Computing pairwise similarity matrices across image and text batches.
5. In-memory semantic image retrieval (`Siglip2Pipeline.retrieve`): Finding top-k matching images from a local candidate corpus for a given text query.
Target application domains include disaster triage image filtering, digital asset cataloging, content moderation triage, wildlife photo categorization, and semantic image retrieval within the DIMER platform.

###### Primary Intended Users

Primary intended users are computer vision engineers, machine learning researchers, data scientists, and backend service developers building multimodal indexing, image classification, or search applications. Users are expected to understand dual-encoder vision-language principles—specifically that SigLIP produces independent sigmoid activations rather than calibrated softmax class probabilities across candidate labels, and that scores do not sum to one. Users must also be familiar with vector space geometry (cosine similarity over L2-normalized embeddings), realize that visual representations reflect statistical pretraining associations rather than semantic truth, and understand that candidate prompt design directly modulates classification behavior.

###### Out-of-scope use cases

1. **Capability boundaries:** This model is an encoder-only vision-language dual tower. It cannot generate text captions, explain images, read dense OCR documents, or output bounding boxes or pixel masks. It must not be marketed or deployed as an object detector (use dedicated detection pipelines like `swin-detection-pipeline`), semantic segmenter (use `swin-segmentation-pipeline`), or visual conversational assistant.
2. **Input boundaries:** Accepts local image paths, raw image bytes, and PIL Image instances. Resolution is fixed to 224x224 pixels through the upstream processor; images with drastic aspect ratios or microscopic details may suffer downsampling distortion. Texts exceeding 64 tokens are truncated. Remote URLs (`http://`, `https://`) are strictly rejected at the API boundary.
3. **Decision boundaries:** Autonomous, unreviewed deployment in safety-critical, legal, or punitive workflows—such as automated law enforcement identification, forensic analysis, medical radiology diagnosis, or automated content blocking without human review—is strictly prohibited.

---

#### Factors

This section describes factors influencing model representation and behavior, including demographic categories, capturing instruments, and operational runtime environments.

###### Groups

SigLIP 2 was pretrained on massive multilingual, multimodal web crawls (e.g., WebLI and derivative web datasets). While not exclusively human-centric, the model frequently encounters human subjects in imagery. The upstream pretraining corpus was not demographically balanced or exhaustively audited for demographic parity. As a result, representation quality, zero-shot label alignment, and semantic associations can exhibit systematic disparities across phenotypic groups, perceived gender, age brackets, skin tones, geographic regions, and cultural dress. The pipeline itself does not introduce demographic filters; operators deploying this model on human imagery bear the direct responsibility of conducting independent demographic fairness audits, false-positive disparity analyses, and bias evaluations on their target domain datasets.

###### Instrumentation

Training and evaluation images for SigLIP 2 originate from diverse consumer, commercial, and professional cameras, mobile phone sensors, web uploads, digital artwork, and synthetic graphics. Key instrumentation factors that affect data representation include camera optical resolution, lens distortion, sensor noise characteristics, compression artifacts (e.g., heavy JPEG quantization), illumination spectra, and dynamic range. Because the v1 processor resizes and center-crops inputs to 224x224 pixels, severe optical distortion, extreme low-light sensor grain, or sensor artifacts can be magnified or lost during downsampling. The pipeline validates image decoding integrity and enforces RGB color space, but cannot detect underlying physical camera miscalibration or sensor degradation.

###### Environment

1. **Operating environment:** Designed to run on Python 3.12 with PyTorch >=2.4 and `transformers` >=4.54.1. Supported hardware includes x86_64 CPUs and NVIDIA GPUs supporting CUDA 12.x. A single inference instance requires approximately 1.5 GB of memory for model weights and minimal RAM for batch activations. Float32 precision is default and fully qualified on CPU; half-precision formats require compatible GPU accelerators.
2. **Data environment:** Assumes clear, photographic or photographic-like images where visual concepts correspond meaningfully to natural language descriptions. Model performance degrades substantially in out-of-distribution visual environments: severe visual occlusions, heavy atmospheric haze, extreme weather, non-standard medical imaging (microscopy, ultrasound), or synthetic radar imagery where visual semantics diverge from web-scale photographic conventions.

---

#### Metrics

This section details performance metrics, decision thresholds, and uncertainty management applied across pipeline operations.

###### Performance Measures

The pipeline reports raw sigmoid classification scores (`scores = torch.sigmoid(logits)`) bounded in `[0.0, 1.0]` for zero-shot classification, and dot-product cosine similarity bounded in `[-1.0, 1.0]` for embedding retrieval. In technical benchmarks, model quality is quantified via top-1 and top-5 zero-shot classification accuracy across standard benchmarks (e.g., ImageNet-1k, ImageNet-A/R), Mean Average Precision (mAP), and Recall@k (R@1, R@5, R@10) on cross-modal retrieval benchmarks (e.g., MS-COCO, Flickr30k). Classification scores are chosen over softmax probabilities because SigLIP's pretraining formulation optimizes independent sigmoid loss over positive and negative pairs, allowing multi-label co-occurrence and avoiding competitive normalization artifacts.

###### Decision thresholds

The pipeline deliberately applies **no default binary classification threshold** and enforces **no argmax decision rule**. Because sigmoid logits depend on prompt phrasing, candidate label cardinality, and image domain, a universal threshold (such as 0.5) is mathematically ungrounded and operationally misleading. The API emits raw unthresholded floating-point values and ranks labels comparatively. Downstream deployment operators own threshold selection, which must be calibrated empirically against task-specific validation sets by balancing the asymmetric operational costs of false positives (erroneous categorization) against false negatives (omitted detections).

###### Approaches to uncertainty and variability

Inference across all pipeline methods is strictly deterministic on CPU: no random sampling, temperature perturbation, or stochastic dropout is active during inference (`model.eval()`). Variations across runs can arise solely from floating-point kernel differences across distinct hardware architectures or non-deterministic GPU BLAS routines. Output sigmoid scores are ranking scores and do not represent calibrated posterior probabilities or confidence intervals. Deployments requiring rigorous uncertainty quantification must employ post-hoc calibration techniques, such as Platt scaling, isotonic regression, or conformal prediction frameworks evaluated on domain-specific holdout splits.

---

#### Ethical considerations and biases

This section examines data sensitivity, life-critical implications, implemented mitigations, failure risks, and prohibited uses.

###### Data

The model weights were pretrained by Google using large-scale multilingual image-text data crawled from the public web. Upstream disclosures confirm filtering for adult and sensitive content, but do not provide an exhaustive instance-level manifest of all pretraining URLs or images. Consequently, pretraining exposure to copyrighted artwork, public figures, or sensitive web imagery cannot be ruled out. This repository distributes only open-source Python code, tests, and configuration manifests; no proprietary datasets or model weight blobs are distributed through git. Operators supplying inference images and text prompts must verify that their input data complies with data privacy laws (e.g., GDPR, HIPAA) and does not contain unauthorized personal identifiable information or classified material.

###### Human Life

SigLIP 2 is an exploratory vision-language encoder and is **not** certified, tested, or approved for life-critical applications or high-stakes decision-making. It must never be deployed as an autonomous decision-making engine in healthcare diagnostics, patient vital monitoring, autonomous vehicle emergency navigation, industrial machinery safety trips, or criminal justice profiling. Any secondary deployment in human-adjacent safety workflows demands extensive independent domain verification, redundant physical fail-safes, and continuous human-in-the-loop oversight.

###### Mitigations

This repository enforces concrete, inspectable architectural and supply-chain mitigations:
1. **Cryptographic supply-chain locking:** Pinned to immutable commit `5ffaac51d5e2f3367f7dab0cad4be4cb07c0caa2`, verifying exact safetensors byte size (`1,500,800,904`) and SHA-256 (`612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b`) prior to instantiation.
2. **Pickle execution refusal:** Scans the snapshot tree and raises a fatal `RuntimeError` if any `*.bin` weight file is detected.
3. **SSRF protection:** Rejects remote `http://` and `https://` image paths at the API boundary, accepting only validated local filesystem paths, in-memory bytes, or PIL images.
4. **Fidelity preprocessing shim:** Explicitly lowercases model-bound text prompts before tokenization to match upstream SigLIP 2 training conventions, while preserving caller label casing in returned outputs.
5. **Deterministic normalization:** Enforces explicit L2 normalization on image and text feature embeddings before cosine scoring.

###### Risks and harms

Key identified risks include:
1. **Automation bias:** Operators may treat high sigmoid scores as absolute factual confirmations of image contents rather than statistical vision-text associations, leading to unverified decisions.
2. **Cultural and linguistic bias:** Although multilingual, vocabulary and cultural concept alignment is skewed toward high-resource languages and Western visual conventions, potentially misclassifying regional artifacts or cultural dress.
3. **Adversarial susceptibility:** Like all dual-encoder CLIP-style models, SigLIP 2 is susceptible to typographic attacks (e.g., text written on an object overriding visual features) and subtle adversarial perturbations.
4. **Search and retrieval bias:** Semantic search over large uncurated image databases can surface stereotypical or offensive image associations for sensitive search queries.

###### Use cases

The following use cases are strictly prohibited by policy and developer intent:
1. Mass biometric surveillance, unauthorized facial identification, or social credit tracking in public spaces.
2. Automated demographic profiling or discriminatory filtering in housing, lending, employment, insurance, or public benefits access.
3. Deceptive or predatory systems, such as automated visual disinformation generation, deceptive deepfake indexing, or non-consensual tracking.
4. Autonomous lethal systems or automated targeting applications.
5. Any application that violates Google's upstream Apache-2.0 license terms or applicable national and international privacy regulations.

---

## Technical Specifications and Architecture

### Architecture Overview

SigLIP 2 decomposes visual and textual understanding into two parallel Transformer backbones:
- **Vision Tower:** Vision Transformer (ViT) with patch size 16x16 pixels and default resolution 224x224 pixels.
- **Text Tower:** Transformer text encoder accepting sequence lengths up to 64 tokens.
- **Embedding Projection:** Both modalities project into a shared 768-dimensional latent space.
- **Loss Formulation:** Pinned checkpoint uses Google's sigmoid contrastive loss ($L_{sig}$), evaluating each image-text pair independently via binary cross-entropy:
  $$\mathcal{L} = - \sum_{i,j} \log \sigma(z_{ij} (t_i \cdot v_j + b))$$
  where $z_{ij} = 1$ if image $j$ matches text $i$, and $-1$ otherwise.

### Checkpoint Invariants and Loading Controls

The snapshot loader (`siglip2_pipeline.model.load_components`) enforces strict supply-chain controls:
1. Pinned Hugging Face repository: `google/siglip2-base-patch16-224`
2. Pinned commit revision: `5ffaac51d5e2f3367f7dab0cad4be4cb07c0caa2`
3. Primary weight file: `model.safetensors`
4. Expected weight byte size: `1,500,800,904` bytes
5. Expected weight SHA-256: `612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b`
6. Upstream parameters: `375,187,970` F32 parameters
7. Execution policy: `trust_remote_code=False`, `use_safetensors=True`, `local_files_only=True`

### Public Inference API

```python
from siglip2_pipeline import load_pipeline

pipe = load_pipeline(device="cpu")

# 1. Zero-shot classification (returns independent sigmoid scores)
scores = pipe.zero_shot_classify(
    image="scene.jpg",
    labels=["flooded highway", "clear road", "fallen powerline"],
    prompt_template="This is a photo of {label}."
)

# 2. Dense feature embeddings (L2-normalized)
img_emb = pipe.embed_image(["scene1.jpg", "scene2.jpg"])
txt_emb = pipe.embed_text(["flooded highway", "clear road"])

# 3. Cross-modal cosine similarity
sim_matrix = pipe.similarity(["scene1.jpg"], ["flooded highway", "clear road"])

# 4. In-memory semantic image retrieval
hits = pipe.retrieve(
    query="flooded highway",
    images=["scene1.jpg", "scene2.jpg", "scene3.jpg"],
    top_k=2
)
```

### Upstream References and Citations

- **SigLIP 2 Paper:** Tschannen et al., *"SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding"*, arXiv:2502.14786 (2025).
- **SigLIP Foundation:** Zhai et al., *"Sigmoid Loss for Language Image Pre-Training"*, ICCV 2023, arXiv:2303.15343.
- **Upstream Repository:** Google Research SigLIP / Hugging Face Transformers.
