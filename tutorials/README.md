# Tutorials

Notebook specification: **DIMER Notebook Specification v1.0**

| Notebook | Profile | Capability | Default runtime | BYOD | Release status |
|---|---|---|---|---|---|
| `siglip2_vision_language_colab.ipynb` | `MULTI-CAPABILITY` | zero-shot classification, image/text embeddings, similarity, text-to-image retrieval | Python 3.12; CPU-only reference (GPU out of scope) | Optional, gated | release candidate |

## Learning contract

The release-grade tutorial exercises the repository's public `siglip2_pipeline`
API against the one immutable `google/siglip2-base-patch16-224` checkpoint.
It performs pretrained inference only: no gradient training, fine-tuning,
in-context conditioning, or fitted preprocessing state occurs.

The frozen release-reference environment installs the official CPU-only PyTorch
wheel, so the notebook intentionally executes on CPU even when a GPU is present.
GPU execution requires a separately pinned and tested environment and is outside
this tutorial's validated release contract.

The default path uses deterministic synthetic images so the notebook remains
non-interactive and reproducible. BYOD is available as an explicitly gated
Colab upload or local Jupyter path.

## Release verification

The repository CI performs static notebook JSON/Python-cell validation on pull
requests. On `main`, the integration job installs the frozen Python 3.12 CPU
reference environment, exercises the real pinned checkpoint, executes the
tutorial top-to-bottom, and uploads the tutorial outputs.

A notebook revision is release-ready only after the corresponding `main`
integration execution succeeds. Pull-request static checks alone are not
execution evidence.

## Supplemental Kaggle verification

On 2026-09-11 PHT, a private CPU Kaggle kernel independently executed the
tutorial at the immutable post-merge commit below. This is supplemental
clean-runtime evidence; the public `main` integration workflow remains the
canonical release gate.

| Evidence | Verified value |
|---|---|
| Repository commit | [`23bae80c427a1f01058ec4884db061335a2cafc0`](https://github.com/kurtvalcorza/siglip2-vision-language-pipeline/commit/23bae80c427a1f01058ec4884db061335a2cafc0) |
| Public `main` integration | [GitHub Actions run 34485799429](https://github.com/kurtvalcorza/siglip2-vision-language-pipeline/actions/runs/34485799429), `success` |
| Kaggle execution | [`kurtvalcorza/tut-siglip2-verify`, version 6](https://www.kaggle.com/code/kurtvalcorza/tut-siglip2-verify), `COMPLETE` (private; owner access required) |
| Runtime | Python 3.12.13; PyTorch 2.14.0+cpu; Transformers 4.57.6; 4-vCPU Kaggle worker |
| Notebook SHA-256 | `6ccf86e91c686d727c27eb1e9d4970abdd5bde608dbf4614ee4d8ed41f7600a1` |
| Checkpoint verification | pinned revision `5ffaac51d5e2f3367f7dab0cad4be4cb07c0caa2`; weight SHA-256 `612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b`; manifest verified |
| Notebook result | 6 of 6 code cells passed; exit code 0; all 8 required artifacts written |
| Synthetic sanity checks | top-1 accuracy `1.0`; retrieval recall@1 `1.0`; classification matches `3/3` |

Kaggle version 5 also completed all notebook cells, but its external wrapper
expected the previous four-item classification output and produced a false
terminal error. Version 6 updated that assertion to the current three-record
schema. [Issue #7](https://github.com/kurtvalcorza/siglip2-vision-language-pipeline/issues/7)
tracks moving the verifier into the repository so its contract cannot drift
silently from the tutorial.

## Applicable SHOULD deviations

- **§20.8 real-image verification:** the default automated path intentionally
  uses deterministic synthetic images to avoid adding a network- and
  license-sensitive external image dependency to CI. The notebook provides a
  real BYOD path for user-supplied images, but CI does not claim real-domain
  qualitative or quantitative validation. Real-image evaluation remains a
  deployment-specific follow-up and the notebook explicitly limits its claims.
- **DAT3 sample license:** the generated PPM assets are repository-generated
  synthetic tutorial data; no external dataset license applies.

## Outputs

The default notebook path writes:

- `classification.json`
- `image_embeddings.npz`
- `text_embeddings.npz`
- `similarity.csv`
- `retrieval.json`
- `metrics.json`
- `new_data_classification.json`
- `provenance.json`

Optional BYOD execution additionally writes `byod_classification.json` and
`byod_image_embedding.npz`.
