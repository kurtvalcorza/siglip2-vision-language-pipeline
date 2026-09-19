---
license: Apache-2.0
model_card_spec: "1.1"
pipeline_tag: zero-shot-image-classification
task: "Others - Vision-Language Embedding"
tags:
  - vision-language
  - zero-shot-image-classification
  - image-text-retrieval
  - embeddings
  - fine-tuning
base_model: google/siglip2-base-patch16-224
date_published: "2025-02-17"
date_published_source: "Hugging Face Hub repository creation date of the exact hosted checkpoint (`createdAt`, https://huggingface.co/api/models/google/siglip2-base-patch16-224)"
---

# SigLIP 2 Base Patch16 224 — Vision-Language Encoder (Zero-Shot Classification, Embeddings, Retrieval & Bounded Fine-Tuning)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-google%2Fsiglip2--base--patch16--224-ffcc4d?style=flat)](https://huggingface.co/google/siglip2-base-patch16-224)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-google--research%2Fbig__vision-181717?style=flat&logo=github&logoColor=white)](https://github.com/google-research/big_vision)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2502.14786-b31b1b.svg)](https://arxiv.org/abs/2502.14786)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This repository ships one standalone Google Colab tutorial that exercises its public pipeline API end to end — bootstrap a fresh runtime, stage and verify the pinned upstream revision, fetch and validate a digest-pinned labelled photograph set, measure the frozen model against two non-neural baselines, run a bounded fine-tuning, evaluate on an image-disjoint split, and export and reload the adapter:

- **E2E Fine-tuning Tutorial**: \
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/siglip2-vision-language-pipeline/blob/main/tutorials/siglip2_vision_language_colab.ipynb) [`siglip2_vision_language_colab.ipynb`](https://github.com/kurtvalcorza/siglip2-vision-language-pipeline/blob/main/tutorials/siglip2_vision_language_colab.ipynb) \
  *Zero-shot classification, embeddings, similarity and retrieval with the pinned `google/siglip2-base-patch16-224` weights, then bounded supervised fine-tuning of the vision tower's last blocks on 360 CC0 iNaturalist photographs of six bird species: the frozen model's accuracy, macro F1 and text-to-image mAP beside the majority-floor and colour-nearest-neighbour baselines, SigLIP's sigmoid loss with validation-mAP epoch selection, held-out evaluation per species on two prompt sets, the drawn shapes re-scored, and a safetensors adapter that reloads with verified parity.*

> [!NOTE]
> The notebook runs on CPU and uses CUDA automatically when present. It remains a release candidate; see [release verification](docs/release-verification.md) for execution records and promotion requirements.

---

#### Description

SigLIP 2 Base Patch16 224 is an open-weight multilingual vision-language dual-encoder foundation model developed by Google (`google/siglip2-base-patch16-224`, pinned revision `5ffaac51d5e2f3367f7dab0cad4be4cb07c0caa2`), packaged by this repository as a verified DIMER inference pipeline. Built with separate vision and text Transformer towers (~375.2M parameters in `model.safetensors`), it maps raw images and text sequences into a unified 768-dimensional representation space. At inference time, the model functions as an encoder: zero-shot classification is achieved by computing pairwise sigmoid scores between a candidate image and prompt-expanded text strings, while retrieval and similarity operate via dot-product cosine distance across L2-normalized embedding vectors. This repository contributes a hardened supply-chain wrapper — cryptographic SHA-256 and byte-size verification before loading, complete exclusion of unsafe pickle (`.bin`) checkpoints, local-only snapshot execution (`trust_remote_code=False`), automated text lowercasing to preserve SigLIP 2 training fidelity, strict rejection of remote HTTP(S) image fetches to prevent SSRF, and full provenance tracking — and a bounded adaptation contract: `evaluate` scores a labelled photograph set with the model's own `logits_per_image` (accuracy, macro F1, per-class breakdown, text-to-image mAP), `adapt` fine-tunes only the last blocks of the vision tower, its post-layernorm and its attention-pool head (21,264,384 of 375,187,970 parameters by default) with SigLIP's pairwise sigmoid loss against the frozen text tower's prompt features, and `save_artifact` / `from_artifact` export the trained tensors as a digest-manifested safetensors adapter that reloads onto a freshly verified base. The tutorial demonstrates the contract on 360 CC0 iNaturalist photographs of six North American bird species, where the frozen model's zero-shot prompts separate the species only partly.

#### Intended Use and Limitations

The sections below outline the primary machine learning tasks, targeted user cohorts, and explicit capability boundaries established for this pipeline.

###### Primary Intended Uses

The primary intended uses of this pipeline comprise five technical vision-language capabilities:
1. Zero-shot image classification (`Siglip2Pipeline.zero_shot_classify`): Ranking arbitrary candidate textual labels against an input image using independent sigmoid activations without task-specific training.
2. Image feature extraction (`Siglip2Pipeline.embed_image`): Generating dense, unit-norm 768-dimensional visual vectors for indexing, clustering, and vector search.
3. Text feature extraction (`Siglip2Pipeline.embed_text`): Generating dense, unit-norm 768-dimensional linguistic vectors from natural language descriptions up to 64 tokens.
4. Multimodal cosine similarity (`Siglip2Pipeline.similarity`): Computing pairwise similarity matrices across image and text batches.
5. In-memory semantic image retrieval (`Siglip2Pipeline.retrieve`): Finding top-k matching images from a local candidate corpus for a given text query.
6. Labelled-set evaluation (`Siglip2Pipeline.evaluate`): Scoring `{id, image, label}` records with one prompt per class and reporting accuracy, macro F1, per-class recall / precision / F1 / average precision and text-to-image mAP beside the predictions.
7. Bounded supervised fine-tuning (`Siglip2Pipeline.adapt`, `save_artifact`, `from_artifact`): Adapting the vision tower's last blocks and head to a small labelled photograph set with validation-based epoch selection, exporting the adapter, and reloading it with verified parity.
Target application domains include disaster triage image filtering, digital asset cataloging, content moderation triage, wildlife photo categorization, and semantic image retrieval within the DIMER platform.

###### Primary Intended Users

Primary intended users are computer vision engineers, machine learning researchers, data scientists, and backend service developers building multimodal indexing, image classification, or search applications. Users are expected to understand dual-encoder vision-language principles—specifically that SigLIP produces independent sigmoid activations rather than calibrated softmax class probabilities across candidate labels, and that scores do not sum to one—and, for the adaptation contract, why a fine-tuning gain on one seeded split of one sample is evidence that the contract works rather than a benchmark, why a split must be image-disjoint (and source-disjoint when photographs come from few photographers or sessions), and why the two non-neural baselines are read before the adapted number. Users must also be familiar with vector space geometry (cosine similarity over L2-normalized embeddings), realize that visual representations reflect statistical pretraining associations rather than semantic truth, and understand that candidate prompt design directly modulates classification behavior.

###### Out-of-scope use cases

1. **Capability boundaries:** This model is an encoder-only vision-language dual tower. It cannot generate text captions, explain images, read dense OCR documents, or output bounding boxes or pixel masks. It must not be marketed or deployed as an object detector (use dedicated detection pipelines like `swin-detection-pipeline`), semantic segmenter (use `swin-segmentation-pipeline`), or visual conversational assistant.
2. **Input boundaries:** Accepts local image paths, raw image bytes, and PIL Image instances. Resolution is fixed to 224x224 pixels through the upstream processor; images with drastic aspect ratios or microscopic details may suffer downsampling distortion. Texts exceeding 64 tokens are truncated. Remote URLs (`http://`, `https://`) are strictly rejected at the API boundary.
3. **Adaptation boundaries:** `adapt` trains the vision tower's last `trainable_vision_layers` blocks, post-layernorm and attention-pool head only; the text tower, the embeddings, `logit_scale` and `logit_bias` stay frozen, so a label the prompt does not describe cannot be learned through the text side, and fine-tuning on a narrow set can erode the model outside that set (the tutorial re-scores three drawn shapes as a small look at this, not a measurement). Datasets are validated structurally, never semantically: a mislabelled set is fine-tuned on without complaint. Mixed prompt sets, calibration and per-class thresholds are not provided.
4. **Decision boundaries:** Autonomous, unreviewed deployment in safety-critical, legal, or punitive workflows—such as automated law enforcement identification, forensic analysis, medical radiology diagnosis, or automated content blocking without human review—is strictly prohibited.

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

The pipeline reports raw sigmoid classification scores (`scores = torch.sigmoid(logits)`) bounded in `[0.0, 1.0]` for zero-shot classification, and dot-product cosine similarity bounded in `[-1.0, 1.0]` for embedding retrieval. In technical benchmarks, model quality is quantified via top-1 and top-5 zero-shot classification accuracy across standard benchmarks (e.g., ImageNet-1k, ImageNet-A/R), Mean Average Precision (mAP), and Recall@k (R@1, R@5, R@10) on cross-modal retrieval benchmarks (e.g., MS-COCO, Flickr30k). Classification scores are chosen over softmax probabilities because SigLIP's pretraining formulation optimizes independent sigmoid loss over positive and negative pairs, allowing multi-label co-occurrence and avoiding competitive normalization artifacts. The public `evaluation_report` helper packages the tutorial's per-grid `top1_accuracy` (against a fixed-class baseline) and `recall_at_1` sanity metrics into a machine-readable report whose verdict is `sample-sanity` on the synthetic set and `not-measurable` when no expected labels exist. The corpus-level measures are computed by `evaluate` from the model's own `logits_per_image` (sigmoid-scaled cosine with the learned `logit_scale` and `logit_bias`) and implemented in numpy in `metrics.py`: `accuracy` (top-prompt equals the gold label), `macro_f1` (unweighted mean of per-class F1), the per-class `recall` / `precision` / `f1` / `ap`, and `t2i_map` (each class prompt as a query ranking every scored photograph, the average precision of its own class, averaged over the classes); `majority_baseline` answers every record with the most frequent training label and `colour_neighbour_baseline` with the label of the training photograph whose 3×3 mean-colour grid is nearest. On the tutorial's 96-photograph test split (RTX 5070 Ti build record, seed 42): majority accuracy 0.167 / mAP 0.17; colour neighbour accuracy 0.26; frozen model accuracy 0.760 / macro F1 0.758 / mAP 0.723 (scientific-name prompts: accuracy 0.479); adapted (two vision blocks + head, lr 5e-5, six epochs, epoch 4 kept by validation mAP) accuracy 0.792 / macro F1 0.793 / mAP 0.871, with the White-throated Sparrow recall 0.44 → 0.75 and the House Finch 0.75 → 0.88 while the Chipping Sparrow fell 0.75 → 0.62 and the Dark-eyed Junco 0.94 → 0.88. These are observations on one seeded split with no dispersion estimate, not a benchmark.

###### Decision thresholds

The pipeline deliberately applies **no default binary classification threshold** and enforces **no argmax decision rule**. Because sigmoid logits depend on prompt phrasing, candidate label cardinality, and image domain, a universal threshold (such as 0.5) is mathematically ungrounded and operationally misleading. The API emits raw unthresholded floating-point values and ranks labels comparatively. Downstream deployment operators own threshold selection, which must be calibrated empirically against task-specific validation sets by balancing the asymmetric operational costs of false positives (erroneous categorization) against false negatives (omitted detections).

###### Approaches to uncertainty and variability

Inference across all pipeline methods is strictly deterministic on CPU: no random sampling, temperature perturbation, or stochastic dropout is active during inference (`model.eval()`). Variations across runs can arise solely from floating-point kernel differences across distinct hardware architectures or non-deterministic GPU BLAS routines. Adaptation is seeded (`seed=0`: shuffling order) but not bit-reproducible across devices; every corpus metric the tutorial reports is one value on one seeded split (`build_sample_dataset(seed=42)`) of one 360-photograph sample, with a 48-photograph validation split that moves accuracy in 2 % steps — the build record's sweep on an earlier 48-photograph test split moved accuracy by +2 to +12 points depending on the epoch, which is the size of the uncertainty a reader should attach to any single number here. Output sigmoid scores are ranking scores and do not represent calibrated posterior probabilities or confidence intervals. Deployments requiring rigorous uncertainty quantification must employ post-hoc calibration techniques, such as Platt scaling, isotonic regression, or conformal prediction frameworks evaluated on domain-specific holdout splits.

---

#### Ethical considerations and biases

This section examines data sensitivity, life-critical implications, implemented mitigations, failure risks, and prohibited uses.

###### Data

The model weights were pretrained by Google using large-scale multilingual image-text data crawled from the public web. Upstream disclosures confirm filtering for adult and sensitive content, but do not provide an exhaustive instance-level manifest of all pretraining URLs or images. Consequently, pretraining exposure to copyrighted artwork, public figures, or sensitive web imagery cannot be ruled out. This repository distributes only open-source Python code, tests, and configuration manifests; no proprietary datasets or model weight blobs are distributed through git. The tutorial's adaptation corpus is 360 research-grade iNaturalist photographs of six common North American birds (White-throated Sparrow, Song Sparrow, Chipping Sparrow, White-crowned Sparrow, Dark-eyed Junco, House Finch; 60 per species, one per observer), each published by its observer under CC0 1.0 and fetched at run time from the iNaturalist open-data bucket by photo id with a byte-size and SHA-256 pin recorded in `samples.py`; nothing is redistributed, every record keeps its observation URL and observer login, and the photographs contain wildlife, not people. Operators supplying inference images and text prompts must verify that their input data complies with data privacy laws (e.g., GDPR, HIPAA) and does not contain unauthorized personal identifiable information or classified material.

###### Human Life

SigLIP 2 is an exploratory vision-language encoder and is **not** certified, tested, or approved for life-critical applications or high-stakes decision-making. It must never be deployed as an autonomous decision-making engine in healthcare diagnostics, patient vital monitoring, autonomous vehicle emergency navigation, industrial machinery safety trips, or criminal justice profiling. Any secondary deployment in human-adjacent safety workflows demands extensive independent domain verification, redundant physical fail-safes, and continuous human-in-the-loop oversight.

###### Mitigations

This repository enforces concrete, inspectable architectural and supply-chain mitigations:
1. **Cryptographic supply-chain locking:** Pinned to immutable commit `5ffaac51d5e2f3367f7dab0cad4be4cb07c0caa2`, verifying exact safetensors byte size (`1,500,800,904`) and SHA-256 (`612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b`) prior to instantiation.
2. **Pickle execution refusal:** Scans the snapshot tree and raises a fatal `RuntimeError` if any `*.bin` weight file is detected.
3. **SSRF protection:** Rejects remote `http://` and `https://` image paths at the API boundary, accepting only validated local filesystem paths, in-memory bytes, or PIL images. The public `validate_inputs` helper applies exactly these input checks and returns an input manifest of the schema, ceilings, per-image observations and verdict before the model runs.
4. **Fidelity preprocessing shim:** Explicitly lowercases model-bound text prompts before tokenization to match upstream SigLIP 2 training conventions, while preserving caller label casing in returned outputs.
5. **Deterministic normalization:** Enforces explicit L2 normalization on image and text feature embeddings before cosine scoring.
6. **Adaptation integrity:** `adapt` validates the dataset before any tensor is built, trains only the named vision-tower tensors with every other parameter's `requires_grad` false, restores the frozen weights on any exception, and records the configuration and epoch history in the artifact; `from_artifact` re-verifies the base snapshot and checks the manifest's format, base identity and weight digest, the file size and SHA-256 and the exact tensor set **before** deserialising, refuses any tensor outside the vision tower, and overlays onto a freshly loaded base.

###### Risks and harms

Key identified risks include:
1. **Automation bias:** Operators may treat high sigmoid scores as absolute factual confirmations of image contents rather than statistical vision-text associations, leading to unverified decisions.
2. **Cultural and linguistic bias:** Although multilingual, vocabulary and cultural concept alignment is skewed toward high-resource languages and Western visual conventions, potentially misclassifying regional artifacts or cultural dress.
3. **Adversarial susceptibility:** Like all dual-encoder CLIP-style models, SigLIP 2 is susceptible to typographic attacks (e.g., text written on an object overriding visual features) and subtle adversarial perturbations.
4. **Search and retrieval bias:** Semantic search over large uncurated image databases can surface stereotypical or offensive image associations for sensitive search queries.
5. **Adaptation risks:** fine-tuning on a small labelled set learns that set's labelling, including its errors and its photographers' habits (background, season, camera); a gain measured on an image-disjoint but observer-overlapping split can overstate transfer; and a narrow adaptation can erode zero-shot behaviour on classes and image families it never saw.

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
8. Adapter artifact format: `org.valcorza.siglip2-base-patch16-224.adapter.v1` — `adapter.safetensors` (the trained tensors only; 45 tensors, 85,062,712 bytes for the default two blocks) plus `manifest.json` naming the base id, revision and `model.safetensors` digest, the classes and prompt template, the tensor names, the file size and SHA-256, the training configuration and the epoch history
9. Tutorial corpus: 360 iNaturalist photographs (CC0 1.0; six species, 60 each, one per observer), `CORPUS_BYTES = 39,223,447`, each pinned by photo id, byte size and SHA-256 in `siglip2_pipeline/samples.py` and fetched from `https://inaturalist-open-data.s3.amazonaws.com/photos/<id>/medium.<ext>`; split 216 / 48 / 96 by `build_sample_dataset(seed=42)`
10. Executed 2026-09-20: the **committed notebook blob** (`f3dd43e` / `15f946ab`) run top-to-bottom on a clean Kaggle Tesla T4 kernel (`kurtvalcorza/dimer-nb2-siglip2-vision-language` v4, `torch 2.14.0+cu130`, Python 3.12.13, `cuda`, empty Hugging Face cache, no repository checkout, blob SHA-1 verified against GitHub before execution): 14/14 ok (1 restart after install cell), 423.7 s, 378 files, 1579 MB fetched from the Hub and digest-verified inside the notebook; comparison {accuracy: {majority: 0.167, neighbour: 0.26, frozen: 0.76, adapted: 0.792}, macro_f1: {majority: 0.048, neighbour: 0.261, frozen: 0.758, adapted: 0.793}, t2i_map: {majority: 0.203, neighbour: 0.212, frozen: 0.723, adapted: 0.871}, delta_vs_frozen: {accuracy: 0.031, macro_f1: 0.034, t2i_map: 0.148}, scientific_name_prompts: {accuracy: {frozen: 0.479, adapted: 0.51}, macro_f1: {frozen: 0.459, adapted: 0.453}, t2i_map: {frozen: 0.467, adapted: 0.492}}, by_species: {american_goldfinch: {n: 16, frozen_recall: 0.88, adapted_recall: 0.81, frozen_ap: 0.93, adapted_ap: 0.97}, chipping_sparrow: {n: 16, frozen_recall: 0.75, adapted_recall: 0.62, frozen_ap: 0.71, adapted_ap: 0.8}, dark_eyed_junco: {n: 16, frozen_recall: 0.94, adapted_recall: 0.88, frozen_ap: 0.97, adapted_ap: 0.94}, house_finch: {n: 16, frozen_recall: 0.75, adapted_recall: 0.88, frozen_ap: 0.86, adapted_ap: 0.86}, song_sparrow: {n: 16, frozen_recall: 0.81, adapted_recall: 0.81, frozen_ap: 0.51, adapted_ap: 0.78}, white_throated_sparrow: {n: 16, frozen_recall: 0.44, adapted_recall: 0.75, frozen_ap: 0.35, adapted_ap: 0.89}}}; reload parity {max_abs_difference: 0, identical_rows: 8, of: 8}. Recorded in `docs/release-verification.md`. Not executed: any corpus other than the one 360-photograph iNaturalist sample, repeated seeds or splits (no dispersion), BYOD, and the adapted model on any photographs but that test split.

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

### Public Adaptation API

```python
from siglip2_pipeline import (
    Siglip2Pipeline, build_sample_dataset, class_names, fetch_corpus, read_corpus,
)

splits = build_sample_dataset(read_corpus(fetch_corpus()), seed=42)  # 216 / 48 / 96 photographs
classes = class_names(splits["train"])
pipe = Siglip2Pipeline.from_pretrained(weights_dir="weights/siglip2-base-patch16-224")

frozen = pipe.evaluate(splits["test"], classes=classes)      # accuracy, macro_f1, t2i_map, per_class, predictions
result = pipe.adapt(splits["train"], splits["validation"],
                    epochs=6, lr=5e-5, batch_size=16, trainable_vision_layers=2)
adapted = pipe.evaluate(splits["test"], classes=classes)
pipe.save_artifact("outputs/adapter")                       # adapter.safetensors + manifest.json
again = Siglip2Pipeline.from_artifact("outputs/adapter", weights_dir="weights/siglip2-base-patch16-224")
```

Dataset contract (`samples.py`): records `{id, image, label}` (`id` matching `[A-Za-z0-9_.:-]{1,64}` and unique; a PIL image or a decodable file with sides up to `MAX_IMAGE_SIDE = 4096`; a label of at most 64 plain characters); `validate_dataset(records, *, min_records=8, max_records=20000)` (2..100 labels); `split_dataset(records, *, val_fraction=0.15, test_fraction=0.2, seed=0)` (stratified, pixel-digest de-duplicated); `check_split_disjoint(splits)`; `observer_overlap(splits)`; `load_byod_dataset(path)` (directory or zip with `labels.csv`: `id`, `file`, `label`); `write_dataset_csv(records, path)`; `fetch_corpus(cache_dir=None)`, `read_corpus(files)`, `build_sample_dataset(records, *, seed=42, sizes=SAMPLE_SPLIT)`. Metrics (`metrics.py`): `classification_metrics(scores, gold, classes)`, `average_precision`, `majority_baseline(train, records, classes)`, `colour_signature(image)`, `colour_neighbour_baseline(train, records, classes)`.

### Upstream References and Citations

- **SigLIP 2 Paper:** Tschannen et al., *"SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding"*, arXiv:2502.14786 (2025).
- **SigLIP Foundation:** Zhai et al., *"Sigmoid Loss for Language Image Pre-Training"*, ICCV 2023, arXiv:2303.15343.
- **Upstream Repository:** Google Research SigLIP / Hugging Face Transformers.
- **Tutorial corpus:** iNaturalist open data (CC0 photographs under each observer's own licence), https://www.inaturalist.org/pages/developers — bucket https://inaturalist-open-data.s3.amazonaws.com/
