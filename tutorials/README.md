# Tutorials

Notebook specification: **DIMER Notebook Specification v1.0**

| Notebook | Profile | Capability | Default runtime | BYOD | Release status |
|---|---|---|---|---|---|
| `siglip2_vision_language_colab.ipynb` | `MULTI-CAPABILITY` | zero-shot classification, image/text embeddings, similarity, text-to-image retrieval | Python 3.12; CPU supported, CUDA optional | Optional, gated | release candidate |

## Learning contract

The release-grade tutorial exercises the repository's public `siglip2_pipeline`
API against the one immutable `google/siglip2-base-patch16-224` checkpoint.
It performs pretrained inference only: no gradient training, fine-tuning,
in-context conditioning, or fitted preprocessing state occurs.

The default path uses deterministic synthetic images so the notebook remains
non-interactive and reproducible. BYOD is available as an explicitly gated
Colab upload or local Jupyter path.

## Release verification

The repository CI performs static notebook JSON/Python-cell validation on pull
requests. On `main`, the integration job installs the frozen Python 3.12
reference environment, exercises the real pinned checkpoint, executes the
tutorial top-to-bottom, and uploads the tutorial outputs.

A notebook revision is release-ready only after the corresponding `main`
integration execution succeeds. Pull-request static checks alone are not
execution evidence.

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
