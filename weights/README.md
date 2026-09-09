# Base Model Weights Cache

This directory holds offline base-model weights, configurations, and cryptographic manifests for the SigLIP 2 vision-language pipeline.

The model is organized in its own isolated subfolder corresponding to its canonical identifier:

```
weights/
└── siglip2-base-patch16-224/
    ├── config.json
    ├── preprocessor_config.json
    ├── special_tokens_map.json
    ├── tokenizer.json
    ├── tokenizer.model
    ├── tokenizer_config.json
    ├── dimer-base-manifest.json
    ├── README.md (Model Card / Specification)
    ├── LICENSE
    └── model.safetensors (Excluded from Git; acquired via scripts/fetch_weights.py or DIMER upload)
```

## Available Base Model Snapshots

- [**`siglip2-base-patch16-224`**](siglip2-base-patch16-224/): Dedicated snapshot for Google's SigLIP 2 vision-language foundation model (`google/siglip2-base-patch16-224`, ~221M parameters).
  - [**Model Card & Specification**](siglip2-base-patch16-224/README.md): Full technical architecture, zero-shot classification guidelines, embedding characteristics, and Apache-2.0 license terms.
  - [**Manifest**](siglip2-base-patch16-224/dimer-base-manifest.json): Cryptographic record of byte counts and SHA-256 hashes for all snapshot files.

## DIMER Architecture & Git Tracking Strategy

In the DIMER workbench ecosystem:
1. **Large Binary Weights (`model.safetensors`):** The ~1.50 GB (1,500,800,904 bytes) weight payload is excluded from Git via `.gitignore` (`weights/**/*.safetensors`) and uploaded directly to DIMER as a model asset or downloaded using `scripts/fetch_weights.py`.
2. **Configuration & Tokenizers:** All accompanying configuration files (`config.json`, `preprocessor_config.json`, `special_tokens_map.json`), SentencePiece/tokenizer files (`tokenizer.json`, `tokenizer.model`, `tokenizer_config.json`), and cryptographic manifests are version-controlled in the repository so the pipeline and offline Docker containers can initialize vision processors and tokenizers without network dependencies.

## Management & Verification Tooling

Manage, download, and cryptographically verify model snapshots using [`scripts/fetch_weights.py`](../scripts/fetch_weights.py):

```bash
# Verify the existing snapshot in weights/siglip2-base-patch16-224:
python scripts/fetch_weights.py --verify-only

# Download and verify default model into its dedicated subfolder:
python scripts/fetch_weights.py --dest weights/siglip2-base-patch16-224

# Verify an explicit destination directory:
python scripts/fetch_weights.py --verify-only --dest weights/siglip2-base-patch16-224
```
