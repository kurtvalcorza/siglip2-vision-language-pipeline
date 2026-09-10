from __future__ import annotations

MODEL_ID = "google/siglip2-base-patch16-224"
MODEL_REVISION = "5ffaac51d5e2f3367f7dab0cad4be4cb07c0caa2"
MODEL_FILENAME = "model.safetensors"
MODEL_SHA256 = "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b"
MODEL_SIZE_BYTES = 1_500_800_904
MODEL_LICENSE = "Apache-2.0"

DEFAULT_MODEL_KEY = "siglip2-base-patch16-224"
UNSAFE_WEIGHT_EXTENSIONS = (
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".pkl",
    ".pickle",
    ".h5",
    ".msgpack",
)

ALLOWED_CHECKPOINT_FILES = (
    "config.json",
    MODEL_FILENAME,
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)

DEFAULT_PROMPT_TEMPLATE = "This is a photo of {label}."
TEXT_MAX_LENGTH = 64
