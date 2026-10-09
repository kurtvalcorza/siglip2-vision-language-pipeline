"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.2 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded package (six
modules, carried verbatim in dependency order), and the model pin/stage/verify cells are produced by
the generator from repository sources so they cannot drift from the package.

Generator /2 keys in use: ``modules`` lists every module of ``src/siglip2_pipeline/`` except
``__init__.py``; ``entry_module`` is ``config.py`` (it holds ``MODEL_ID``/``MODEL_REVISION``/
``MODEL_LICENSE`` and the model key under the package's own spelling ``DEFAULT_MODEL_KEY``, mapped by
``identity_names``); ``rewrites`` carries two rules — the fleet ``DEFAULT_WEIGHTS_DIR`` rule and the
``__file__`` use inside ``model.resolve_weights_path`` (a repository-checkout convenience that a
standalone notebook has no checkout for); ``model_load`` lets the pipeline pick CUDA when it is visible
(the fine-tuning stage is where that matters; CPU is the documented fallback).

This template configures an E2E zero-shot-classification fine-tuning workflow: the pinned
google/siglip2-base-patch16-224 snapshot is digest-verified and loaded, 360 CC0 iNaturalist photographs of
six bird species are fetched with per-file digests, validated and split by photograph, the drawn synthetic
shapes are classified through the inference contract, the frozen model's zero-shot accuracy / macro F1 /
text-to-image mAP over the held-out photographs is measured beside two non-neural baselines, a bounded
fine-tuning of the vision tower's last blocks runs in the kernel with SigLIP's sigmoid loss, the held-out
split is scored again per species, the adapted model re-scores the shapes, and the adapter is exported and
reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "siglip2_pipeline",
    "repo_name": "siglip2-vision-language-pipeline",
    "stem": "siglip2_vision_language",
    "notebook_name": "siglip2_vision_language_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "isolated_runtime": True,
    "infrastructure_labels": True,
    # The fleet's uv isolated-environment mechanism (generator /2.2): managed CPython, a
    # size- and SHA-256-verified uv wheel, and a lock compiled from the pyproject pins with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "pipeline_class": "Siglip2Pipeline",
    "weights_key": "siglip2-base-patch16-224",
    "modules": ["config.py", "model.py", "metrics.py", "samples.py", "pipeline.py", "provenance.py"],
    "entry_module": "config.py",
    "identity_names": {"MODEL_KEY": "DEFAULT_MODEL_KEY"},
    "rewrites": [
        ["^DEFAULT_WEIGHTS_DIR = Path\\(__file__\\)[^\\n]*$", 'DEFAULT_WEIGHTS_DIR = Path.cwd() / "weights" / DEFAULT_MODEL_KEY  # standalone rewrite (build_notebook.py): working-directory snapshot, no repository checkout'],
        ["^    repo_root = Path\\(__file__\\)\\.resolve\\(\\)\\.parents\\[2\\]$", "    repo_root = Path.cwd()  # standalone rewrite (build_notebook.py): no repository checkout to resolve"],
    ],
    "model_load": "Siglip2Pipeline.from_pretrained(weights_dir=WEIGHTS_DIR)",
    "runtime_imports": ["torch", "transformers", "numpy"],
    "title": "SigLIP 2 Vision-Language Pipeline — DIMER E2E zero-shot classification fine-tuning tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/siglip2-vision-language-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/siglip2-vision-language-pipeline/blob/main/tutorials/siglip2_vision_language_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-google%2Fsiglip2--base--patch16--224-ffcc4d?style=flat",
            "https://huggingface.co/google/siglip2-base-patch16-224",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-huggingface%2Ftransformers-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/huggingface/transformers",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2502.14786-b31b1b.svg", "https://arxiv.org/abs/2502.14786"),
    ],
    "capability": "zero-shot image classification, image/text embeddings, cosine similarity, text-to-image retrieval and bounded supervised fine-tuning of the vision tower's last blocks on a labelled-photograph dataset, using the pinned `google/siglip2-base-patch16-224` weights",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated, hash-locked environment with the pinned dependencies "
        "(nothing is installed into the notebook kernel, so no restart is needed), stages and digest-verifies the "
        "pinned `google/siglip2-base-patch16-224` snapshot (a 1.5 GB `model.safetensors`; no pickle is opened anywhere), fetches "
        "the 360 pinned iNaturalist photographs from the project's open-data bucket (about 39 MB, each refused on any byte-size "
        "or SHA-256 mismatch), cuts them per species into 216 / 48 / 96 training, validation and test photographs, classifies "
        "three drawn shapes through the inference contract with an input manifest and a rejection probe, measures the frozen "
        "model's zero-shot accuracy, macro F1 and text-to-image mAP over the 96 test photographs beside the majority-floor and "
        "colour-nearest-neighbour baselines, runs a bounded fine-tuning of the vision tower's last two blocks and attention-pool "
        "head with SigLIP's sigmoid loss and validation-mAP epoch selection, scores the held-out photographs again per species, "
        "re-scores the drawn shapes with the adapted model, exports the adapter as safetensors with a manifest, and reloads that "
        "artifact into a fresh pipeline to verify embedding parity. The default path needs no repository clone, no DIMER worker "
        "or service, no credential, no upload dialog and no configuration edit (NOTEBOOK_SPEC 2.2 §5). On CPU the whole path "
        "took about three minutes on the build workstation's CPU after the downloads (expect longer on a 2-vCPU hosted runtime); a CUDA runtime is used automatically when present and finishes in "
        "a few minutes."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 (and `BYOD_PATH` to the zip's path on Kaggle or "
        "Jupyter; on Colab an empty path opens the upload dialog) and re-run from that cell to supply one zip "
        "holding a `labels.csv` (columns `id`, `file`, `label`) beside the image files — at least eight photographs over at "
        "least two labels, the label text being what the prompt names. They pass through the same validation, seeded stratified "
        "split, baselines, fine-tuning, held-out evaluation, artifact export and reload-parity cells as the iNaturalist sample. "
        "The expected schema and the ceilings are stated in the Prerequisites and in Section 4, and uploaded files stay inside "
        "this runtime. BYOD is optional and never part of the default path."
    ),
    "intro": (
        "`google/siglip2-base-patch16-224` is the SigLIP 2 model of Tschannen et al. (2025) — a ViT-B/16 image tower at 224×224 "
        "with an attention-pool head and a 12-layer text tower with a 256k multilingual vocabulary; 375,187,970 parameters, "
        "published under the **Apache-2.0** licence. A photograph and a prompt are scored by the cosine of their projected "
        "embeddings, scaled and shifted by the model's learned `logit_scale` and `logit_bias` and read through a sigmoid: the "
        "scores are independent per pair, **not calibrated probabilities**, **prompt/label dependent**, and the model never "
        "abstains — the highest-scoring label is returned whatever the image shows.\n\n"
        "What this notebook adds to inference is **adaptation with labelled photographs**. The dataset is real and where the "
        "frozen model has room to improve: 360 CC0-licensed, research-grade iNaturalist photographs of six common North "
        "American birds (**CC0 1.0**; four small sparrows, a junco and two finches, 60 per species, one per observer), pinned "
        "by photo id, byte size and SHA-256 and fetched from the project's open-data bucket at run time. Zero-shot prompts from "
        "the common names separate these species only partly (the build record measured accuracy 0.76 frozen on the 96 test "
        "photographs, with the White-throated Sparrow at 0.44 recall), so the honest question is narrow: does a bounded "
        "fine-tuning of the vision tower's last blocks on 216 photographs move zero-shot accuracy and the retrieval view "
        "(**text-to-image mAP**) on an image-disjoint test split, per species, against two **non-neural baselines** (the "
        "**majority floor** and a **colour nearest neighbour**)? Nothing here is a quality claim about your photographs: it is "
        "one seeded split of one sample.\n\n"
        "**Snapshot note:** the pinned revision ships `model.safetensors` (an 8-file manifest) — no pickle is opened anywhere "
        "in this notebook. Section 3 stages and digest-verifies those files before the processor or the model is constructed."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried package guarantees; stage and digest-verify the immutable "
        "upstream snapshot; fetch a digest-pinned labelled photograph set, validate it and split it per species without "
        "leakage; classify drawn shapes through the public API and read sigmoid scores correctly (independent, uncalibrated, no "
        "abstention); measure the frozen model's zero-shot accuracy, macro F1 and text-to-image mAP beside two non-neural "
        "baselines and read the per-species breakdown; run a bounded fine-tuning with SigLIP's sigmoid loss, explicit "
        "hyperparameters and validation-based epoch selection; evaluate on an image-disjoint test split with two prompt sets; "
        "re-score drawings from a different image family with the adapted model; and export a safetensors adapter that "
        "reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "object detection, semantic segmentation, OCR, caption generation, calibrated probabilities or universal thresholds, "
        "fine-tuning of the text tower, the embeddings, `logit_scale` or `logit_bias`, training on photographs that are not the "
        "pinned sample or your own uploads, evaluation on iNaturalist or any benchmark proper (only one seeded 360-photograph "
        "sample is scored here), and any claim that six bird species stand in for your classes. The repository exposes none of "
        "these."
    ),
    "guided": {"opening": [(
        "**Who this notebook is for.** A learner who knows basic Python and PIL, has used Colab or Jupyter and has met accuracy and a cosine similarity, and wants to see what a zero-shot image–text model does with a prompt, why its sigmoid scores are not probabilities, and what a bounded fine-tuning of its vision tower changes on a real labelled set. The audience is students and practitioners deciding whether SigLIP 2 can be adapted to their own labels; no prior experience with SigLIP, transformers or fine-tuning is assumed — each term is explained where it first matters and again in the **Glossary**. CPU is adequate (minutes); a GPU is faster.\n\n**Input → Model → Output.**\n\n| | |\n|---|---|\n| Input | `{{id, image, label}}` records and a prompt template; the default is 360 pinned CC0 iNaturalist photographs of six bird species, split 216 / 48 / 96 per species, plus three 32 × 32 drawn shapes |\n| Model | `google/siglip2-base-patch16-224` (375 M parameters): a ViT-B/16 image tower with an attention-pool head and a 12-layer multilingual text tower; the fine-tune trains the last two vision blocks, the post-layernorm and the head (21.3 M parameters) |\n| Output | zero-shot accuracy, macro F1 and text-to-image mAP of the frozen and the adapted model against a majority floor and a colour nearest neighbour, per species and on two prompt sets; the shapes re-scored; an 85 MB safetensors adapter that reloads with verified embedding parity; `result.json` |\n\n**How to use this notebook.** Choose any runtime, then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed (the recorded hosted run of the previous version needed one; this version removes it). Sections 1–3 are **infrastructure** — the isolated environment, the carried modules and the verified snapshot — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited, and the defaults reproduce the recorded run. Before each principal result the notebook asks you to **Predict**; after it comes a collapsible **Check your reasoning** with a worked answer from the recorded Kaggle T4 run of 20 September 2026. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Budget about ten minutes on a T4 (424 s in the recorded run, downloads included); longer on CPU.\n\n**Roadmap:** 1–3 infrastructure → 4 the photographs, the split and four refusals *(core concept: the data contract and leakage)* → 5 the inference contract on drawn shapes *(core concept: independent sigmoid scores, no abstention)* → 6 two non-neural baselines and the frozen model, per species and on two prompt sets *(evaluation practice: baselines before model numbers)* → 7 bounded fine-tuning of the last vision blocks with mAP-based epoch selection *(core concept: what 21 M of 375 M parameters can learn)* → 8 held-out evaluation and recorded verdicts *(evaluation practice)* → 9 the shapes re-scored, export and reload parity *(engineering)* → conclude."
    )]},
    "prerequisites": [
        "- **Learner:** basic Python, PIL and Colab or Jupyter familiarity; no prior experience with SigLIP or fine-tuning. Sigmoid scores, the three metrics, the two baselines, the vision tower, the loss, epoch selection and reload parity are explained where they are first used and again in the Glossary.",
        "- **Runtime:** a fresh supported **Linux x86_64** runtime (Google Colab, Kaggle or a Linux Jupyter server). Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels, so the Python version of the kernel itself does not matter and nothing is installed into it; a Windows or macOS kernel is not supported. The default path runs on CPU (float32) and uses CUDA automatically when available. CPU is slow but adequate: the build record measured about 5 s to embed and score the 96 test photographs and 110 s for the six epochs of fine-tuning (216 photographs per epoch through the full vision tower, the last two blocks and the head training), including the per-epoch validation scoring; the whole default path took 160 s on the build workstation's CPU with the snapshot and photographs already cached (a 2-vCPU hosted runtime will be several times slower), and 92 s on an RTX 5070 Ti. The pinned `torch==2.14.0` install and the 1.5 GB checkpoint are the large downloads of the run; the photographs add about 39 MB.",
        "- **Knowledge:** basic Python and PIL; what a sigmoid score and a cosine similarity are; what accuracy, macro F1 and average precision measure and why none is a human judgement; why a high score is not a correct label.",
        "- **Data contract:** records are `{id, image, label}` — a PIL image or a file decodable by Pillow with sides up to `MAX_IMAGE_SIDE` (4096) px and a label of at most 64 plain characters (the prompt is `DEFAULT_PROMPT_TEMPLATE` with the label, or its display name, filled in). Ids match `[A-Za-z0-9_.:-]{1,64}` and are unique; a dataset needs 8..20,000 records over 2..100 labels; splitting is stratified per label after pixel-digest de-duplication so no photograph lands in two splits. BYOD accepts one zip of images plus a `labels.csv` in that shape.",
        "- **Validation is structural, not semantic:** every image is opened and decoded and every label checked, but nothing checks that a label is right — a mislabelled set is fine-tuned on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there. The default path uploads nothing.",
        "- **External access (data):** besides the model snapshot, the default path fetches 360 JPEG/PNG files from `https://inaturalist-open-data.s3.amazonaws.com/photos/<id>/medium.<ext>` (about 39 MB in total), each pinned by byte size and SHA-256 in the carried `samples.py` and refused on any mismatch; every photograph's iNaturalist observation page and observer login are kept in its record. Each photograph carries the CC0 1.0 licence its observer chose (Gurari-style redistribution is not needed: nothing is committed to the repository).",
    ],
    "cells": [
        {
            "md": (
                "## 4. iNaturalist photographs and split\n\n"
                "`fetch_corpus` returns the 360 pinned photographs from the cache under `weights/inat-birds/` or the "
                "iNaturalist open-data bucket — every cached file is re-hashed and every fetched file refused on any byte-size or "
                "SHA-256 mismatch — and `read_corpus` decodes them into `{{id, image, label}}` records with their observation "
                "page, observer and species names. `build_sample_dataset` draws a seeded stratified split per species (36 / 8 / "
                "16 → 216 / 48 / 96). `validate_dataset` then checks every record against the contract, `check_split_disjoint` "
                "asserts no photograph (by decoded-pixel digest) is shared, `observer_overlap` reports how many observers "
                "contributed to more than one split (an observation about the draw, not an assertion), and the training split's "
                "labels table is written to `outputs/{stem}_train.csv` in the shape BYOD expects.\n\n"
                "Look for: 360 photographs, the six species with 36 / 8 / 16 each, three digests, and four refusal probes — a "
                "duplicate id, an image over the side ceiling, a dataset with one label and one too small to split — each "
                "rejected before the model does anything.\n\n"
                "**Predict:** six species, 60 photographs each, one per observer. Will the split be exactly 36 / 8 / 16 per species, will any observer appear in more than one split, and will any of the four probes be accepted?"
            ),
            "code": (
                "import hashlib\n"
                "import json\n"
                "import time\n"
                "from dataclasses import asdict\n\n"
                "import numpy as np\n"
                "from PIL import Image\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                'def byod_file(path, kind, suffixes=()):\n'
                '    """BYOD path first (works on Colab, Kaggle and Jupyter); on Colab an empty path opens the upload dialog."""\n'
                '    if str(path).strip():\n'
                '        source = Path(str(path).strip()).expanduser()\n'
                '        if not source.is_file():\n'
                "            raise FileNotFoundError(f'BYOD path {{str(source)!r}} does not exist or is not a file (relative paths start at {{os.getcwd()}}); give the path of one {{kind}}.')\n"
                '    else:\n'
                '        try:\n'
                '            from google.colab import files\n'
                '        except ImportError:\n'
                "            raise RuntimeError(f'BYOD is on but BYOD_PATH is empty, and the upload dialog exists only in Google Colab: copy the {{kind}} into this runtime (or attach it as a Kaggle dataset) and set BYOD_PATH.') from None\n"
                '        uploaded = files.upload()\n'
                '        if len(uploaded) != 1:\n'
                "            raise ValueError(f'Upload exactly one {{kind}} (received {{len(uploaded)}} files; a cancelled dialog sends none). Run this cell again.')\n"
                '        name, payload = next(iter(uploaded.items()))\n'
                "        source = Path('work') / Path(name).name\n"
                '        source.parent.mkdir(parents=True, exist_ok=True)\n'
                '        source.write_bytes(payload)\n'
                '    if suffixes and not source.name.lower().endswith(tuple(suffixes)):\n'
                '        raise ValueError(f\'{{source.name}}: expected a {{kind}} ending in {{" or ".join(suffixes)}}.\')\n'
                '    return source\n'
                '\n'
                'if USE_BYOD:\n'
                "    byod_path = byod_file(BYOD_PATH, 'zip of images with a labels.csv', ('.zip',))\n"
                '    records = load_byod_dataset(byod_path)\n'
                '    splits = split_dataset(records, seed=SPLIT_SEED)\n'
                "    data_source = 'BYOD (' + byod_path.name + ')'\n"
                "    display_names = {{}}\n"
                "    raw_rows = {{'byod': len(records)}}\n"
                "else:\n"
                "    corpus_files = fetch_corpus(cache_dir='weights/inat-birds')\n"
                "    corpus = read_corpus(corpus_files)\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}}: {{CORPUS_RELEASE}} ({{CORPUS_LICENSE}})'\n"
                "    display_names = {{key: common for key, (_scientific, common) in SPECIES.items()}}\n"
                "    raw_rows = {{'photographs': len(corpus), 'bytes': sum(len(v) for v in corpus_files.values()), 'observers': len({{r['observer'] for r in corpus}})}}\n"
                "dataset_manifests = {{name: validate_dataset(part) for name, part in splits.items()}}\n"
                "splits = {{name: manifest['records'] for name, manifest in dataset_manifests.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "classes = class_names(train_records)\n"
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'raw_rows': raw_rows, 'splits': disjoint, 'observer_overlap': observer_overlap(splits), 'classes': classes}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'label_counts': manifest['label_counts'], 'image_side': manifest['image_side'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "example = train_records[0]\n"
                "print({{'example': {{'id': example['id'], 'label': example['label'], 'display_name': display_names.get(example['label'], example['label']), 'size': list(example['image'].size), 'observation': example.get('inat_observation_url')}}}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in train_records[:8]],\n"
                "    'image over the side ceiling': [{{**train_records[0], 'image': Image.new('RGB', (MAX_IMAGE_SIDE + 1, 8))}}, *train_records[1:8]],\n"
                "    'one label only': [{{**r, 'label': 'bird'}} for r in train_records[:8]],\n"
                "    'too small': train_records[:3],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>Yes, no and no. The split is stratified per species by seed, so 36 / 8 / 16 each (216 / 48 / 96) by construction; one photograph per observer per species means an observer can still appear in two splits through two species, and `observer_overlap` prints how many did — an observation about the draw, not an assertion. All four probes are refused before any model work, each naming its rule: a duplicate id, a side over `MAX_IMAGE_SIDE`, a single label, a set too small to split.</details>'
            ),
        },
        {
            "md": (
                "## 5. Classify through the inference contract\n\n"
                "The inference contract is exercised as the inference-only tutorial exercised it: three 32×32 synthetic shapes "
                "— a red square, a green circle and a blue triangle — rendered in code as ASCII PPM files exactly as the "
                "repository's `examples/sample-data/generate_samples.py` renders them and digest-asserted against "
                "`SHA256SUMS`, a different image family from the photographs, and images the model will be asked to classify "
                "again after adaptation. `validate_inputs` applies exactly the checks the public operations apply and returns an "
                "input manifest; a remote URL is validated too and its rejection recorded as a finding. `zero_shot_classify` "
                "returns one sigmoid score per candidate label, **ordered by descending score**; `retrieve` ranks the gallery by "
                "cosine to a query. The per-grid `evaluation_report` on three drawn shapes is `sample-sanity` — plumbing "
                "evidence, not a measurement; whether the classifier is *right* is what Section 6 measures on 96 photographs.\n\n"
                "**Predict:** three flat drawn shapes and four candidate labels, one of them a distractor. Will every shape get its own label on top, and will the four sigmoid scores of a shape sum to one?"
            ),
            "code": (
                "SAMPLE_DIGESTS = {{  # examples/sample-data/SHA256SUMS\n"
                "    'red_square.ppm': 'b38ff0c9131677ed6cf09832eff40a22841e1a4725d426fad3b1bf6a1dbdb096',\n"
                "    'green_circle.ppm': '2e3e657686a0f6a6df3f3621d21a410d20d9a47b109f06f9405faa4f747a663f',\n"
                "    'blue_triangle.ppm': 'f0f4c38c7af92b3a6b1d55a25272156edd87b41e029e116cfa059003dae029a3',\n"
                "}}\n"
                "WIDTH, HEIGHT, BACKGROUND = 32, 32, (245, 245, 245)\n\n\n"
                "def shape_mask(shape, x, y):\n"
                "    if shape == 'square':\n"
                "        return 8 <= x < 24 and 8 <= y < 24\n"
                "    if shape == 'circle':\n"
                "        return (x - 16) ** 2 + (y - 16) ** 2 <= 9**2\n"
                "    if not 7 <= y < 26:\n"
                "        return False\n"
                "    half = (y - 7) // 2\n"
                "    return 16 - half <= x <= 16 + half\n\n\n"
                "def render_ppm(foreground, shape):\n"
                "    # The repository's generate_samples.py rendering: ASCII P3, 24 values per line.\n"
                "    lines = ['P3', f'{{WIDTH}} {{HEIGHT}}', '255']\n"
                "    for y in range(HEIGHT):\n"
                "        row = []\n"
                "        for x in range(WIDTH):\n"
                "            pixel = foreground if shape_mask(shape, x, y) else BACKGROUND\n"
                "            row.extend(str(v) for v in pixel)\n"
                "        for start in range(0, len(row), 24):\n"
                "            lines.append(' '.join(row[start : start + 24]))\n"
                "    return '\\n'.join(lines) + '\\n'\n\n\n"
                "Path('outputs/sample-data').mkdir(parents=True, exist_ok=True)\n"
                "specs = {{'red_square.ppm': ((220, 40, 40), 'square'), 'green_circle.ppm': ((40, 170, 75), 'circle'), 'blue_triangle.ppm': ((40, 90, 220), 'triangle')}}\n"
                "shape_images = []\n"
                "for name, (foreground, shape) in specs.items():\n"
                "    path = Path('outputs/sample-data') / name\n"
                "    path.write_bytes(render_ppm(foreground, shape).encode('ascii'))\n"
                "    digest = hashlib.sha256(path.read_bytes()).hexdigest()\n"
                "    if digest != SAMPLE_DIGESTS[name]:\n"
                "        raise ValueError(f'Synthetic sample digest mismatch for {{name}}: {{digest}} != {{SAMPLE_DIGESTS[name]}}')\n"
                "    shape_images.append(path)\n"
                "shape_labels = ['red square', 'green circle', 'blue triangle']\n"
                "candidate_labels = [*shape_labels, 'abstract geometric shape']\n"
                "shape_sha256 = {{p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in shape_images}}\n"
                "print({{'ceilings': {{'TEXT_MAX_LENGTH': TEXT_MAX_LENGTH, 'DEFAULT_PROMPT_TEMPLATE': DEFAULT_PROMPT_TEMPLATE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MIN_RECORDS': MIN_RECORDS, 'MAX_RECORDS': MAX_RECORDS, 'MIN_CLASSES': MIN_CLASSES, 'image_contract': '224x224 RGB after processor resize', 'device': str(pipe.device)}}}})\n"
                "input_manifest = validate_inputs(shape_images, candidate_labels, top_k=len(shape_images), names=[p.name for p in shape_images])\n"
                "try:\n"
                "    validate_inputs('https://example.invalid/not-allowed.png', candidate_labels)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'remote-url-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print({{'shapes': [p.name for p in shape_images], 'sha256': {{k: v[:16] + '...' for k, v in shape_sha256.items()}}, 'manifest_verdict': input_manifest['verdict'], 'findings': len(input_manifest['findings'])}})\n"
                "classifications, retrievals, classification_rows = [], [], []\n"
                "t0 = time.perf_counter()\n"
                "for image_path, expected in zip(shape_images, shape_labels, strict=True):\n"
                "    scores = pipe.zero_shot_classify(image_path, candidate_labels)\n"
                "    classifications.append(scores)\n"
                "    classification_rows.append({{'image': image_path.name, 'expected_label': expected, 'predicted_label': scores[0].label, 'scores': [asdict(x) for x in scores]}})\n"
                "    print(image_path.name, '->', [(s.label, round(s.score, 4)) for s in scores])\n"
                "for query in shape_labels:\n"
                "    retrievals.append(pipe.retrieve(query, shape_images, top_k=len(shape_images)))\n"
                "shape_embeddings = pipe.embed_image(shape_images)\n"
                "checks = {{\n"
                "    'one_ranking_per_image': len(classifications) == len(shape_images),\n"
                "    'scores_in_unit_interval': all(0.0 <= s.score <= 1.0 for ranking in classifications for s in ranking),\n"
                "    'rankings_descending': all(ranking[i].score >= ranking[i + 1].score for ranking in classifications for i in range(len(ranking) - 1)),\n"
                "    'embeddings_unit_norm': bool(np.allclose(np.linalg.norm(shape_embeddings, axis=1), 1.0, atol=1e-4)),\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'inference output failed a sanity check: {{checks}}')\n"
                "result = {{'classifications': classifications, 'retrievals': retrievals, 'gallery_ids': [p.name for p in shape_images], 'embedding_shapes': {{'image': list(shape_embeddings.shape)}}}}\n"
                "targets = {{'labels': shape_labels, 'retrieval_indices': list(range(len(shape_images)))}}\n"
                "frozen_scene = evaluation_report(result, targets, sample_kind='synthetic')\n"
                "print({{'checks': checks, 'seconds': round(time.perf_counter() - t0, 2), 'frozen_scene': {{m['id']: round(m['value'], 3) for m in frozen_scene['metrics']}}, 'verdict': frozen_scene['verdict']}})"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>Each shape got its own label on top — the recorded pre-flight scored the three shapes with top-1 accuracy 1.0 and recall@1 1.0 — and no, the scores do not sum to one: each image–prompt pair goes through its own sigmoid, so a distractor like `abstract geometric shape` can score high beside the right label. The verdict is `sample-sanity`: three drawings are plumbing evidence, not a measurement.</details>'
            ),
        },
        {
            "md": (
                "## 6. Baselines and the frozen model's zero-shot score on the test photographs\n\n"
                "Three systems frame the adaptation, each read three ways. The **majority floor** answers every photograph with "
                "the most frequent training label (accuracy 1/6 on a balanced split, chance-level macro F1). The **colour "
                "nearest neighbour** answers with the label of the training photograph whose 3×3 mean-colour grid is closest — "
                "a classifier that knows the image through 27 numbers. The **frozen model** is scored by `pipe.evaluate`: one "
                "prompt per species (`DEFAULT_PROMPT_TEMPLATE` with the common name), the model's own sigmoid-scaled logits as "
                "the score grid, **accuracy** and **macro F1** of the top prompt, the per-species recall, and **text-to-image "
                "mAP** — each prompt as a query ranking all 96 photographs, the average precision of its own species, averaged "
                "over the six — the retrieval view of the same scores and the smoother of the three on a small set. A second "
                "prompt set built from the scientific names is scored too, to show how much the number is the prompt's. Expect "
                "the frozen model far above both baselines — it is a trained zero-shot classifier — and read the per-species "
                "breakdown: the build record measured accuracy 0.76 / macro F1 0.76 / mAP 0.72 frozen, with the White-throated "
                "Sparrow at 0.44 recall and the Dark-eyed Junco at 0.94. The cell records whether the frozen model is above both baselines as a verdict "
                "instead of stopping, so a BYOD set that behaves differently still reaches the export.\n\n"
                "**Predict:** rank the majority floor, the colour nearest neighbour and the frozen model by accuracy, then guess the frozen mAP. Will the scientific-name prompts score higher or lower than the common names?"
            ),
            "code": (
                '# Sections 6 and 7 describe the frozen model and what one adaptation adds to it. `pipe.adapt` trains the vision tower\n'
                '# in place, so when the pipeline already carries an adaptation both sections reload the pretrained model from the\n'
                '# verified snapshot first; a re-run never scores or trains an earlier adaptation as if it were frozen.\n'
                'def frozen_pipeline():\n'
                '    """Make `pipe` the pretrained model again: reload it from the verified snapshot when it carries an adaptation."""\n'
                '    global pipe\n'
                '    if pipe.adapter is not None:\n'
                '        pipe = Siglip2Pipeline.from_pretrained(weights_dir=WEIGHTS_DIR)\n'
                "        print({{'frozen_model': 'reloaded from the verified snapshot', 'adapted': pipe.adapter is not None}})\n"
                '    else:\n'
                "        print({{'frozen_model': 'already frozen', 'adapted': False}})\n"
                '\n'
                '\n'
                'frozen_pipeline()\n'
                'baseline_majority = majority_baseline(train_records, test_records, classes)\n'
                'baseline_neighbour = colour_neighbour_baseline(train_records, test_records, classes)\n'
                "METRICS = ('accuracy', 'macro_f1', 't2i_map')\n"
                "print({{'majority_baseline': {{k: round(baseline_majority[k], 3) for k in METRICS}}, 'n': baseline_majority['n'], 'note': baseline_majority['baseline']}})\n"
                "print({{'colour_neighbour_baseline': {{k: round(baseline_neighbour[k], 3) for k in METRICS}}, 'note': baseline_neighbour['baseline']}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records, classes=classes, class_names_map=display_names)\n"
                "print({{'frozen_model_test': {{k: round(frozen_test[k], 3) for k in METRICS}}, 'n': frozen_test['n'], 'verdict': frozen_test['verdict'], 'prompt_template': frozen_test['prompt_template'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "frozen_fields = {{c: {{'n': v['n'], 'recall': round(v['recall'], 2), 'ap': round(v['ap'], 2)}} for c, v in frozen_test['per_class'].items()}}\n"
                "print({{'by_species_frozen': frozen_fields}})\n"
                "scientific_names = {{key: scientific for key, (scientific, _common) in SPECIES.items()}} if not USE_BYOD else {{}}\n"
                "frozen_scientific = pipe.evaluate(test_records, classes=classes, class_names_map=scientific_names, prompt_template='This is a photo of {{label}}.')\n"
                "print({{'frozen_model_test_scientific_name_prompts': {{k: round(frozen_scientific[k], 3) for k in METRICS}}}})\n"
                "# A recorded verdict, not an assert: a BYOD set where the prompts do not beat the baselines still reaches the export.\n"
                "frozen_verdict = 'above the baselines' if frozen_test['t2i_map'] > baseline_majority['t2i_map'] and frozen_test['accuracy'] > baseline_neighbour['accuracy'] else 'not above the baselines'\n"
                "print({{'frozen_vs_baselines': frozen_verdict, 'frozen': {{'t2i_map': round(frozen_test['t2i_map'], 3), 'accuracy': round(frozen_test['accuracy'], 3)}}, 'majority_t2i_map': round(baseline_majority['t2i_map'], 3), 'neighbour_accuracy': round(baseline_neighbour['accuracy'], 3)}})"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>In the recorded Kaggle T4 run: majority 0.167, colour neighbour 0.26, frozen model 0.76 accuracy (macro F1 0.758, mAP 0.723), so the verdict is `above the baselines`. The scientific-name prompts scored lower — accuracy 0.479, macro F1 0.459, mAP 0.467 — because the classifier is the prompt as much as the tower, and Latin binomials are rarer in its training captions than common names. Per species, the White-throated Sparrow had recall 0.44 and the Dark-eyed Junco 0.94.</details>'
            ),
        },
        {
            "md": (
                "## 7. Bounded fine-tuning of the vision tower's last blocks\n\n"
                "`pipe.adapt` trains only the last `TRAINABLE_VISION_LAYERS` blocks of the vision tower, its post-layernorm and "
                "its attention-pool head — two blocks by default, 21,264,384 of 375,187,970 parameters; the text tower, the "
                "embeddings, `logit_scale` and `logit_bias` stay frozen. The six class prompts are embedded once by the frozen "
                "text tower; every batch of photographs is run through the vision tower, scored against the prompts with the "
                "model's own sigmoid-scaled logits, and trained with SigLIP's pairwise sigmoid loss (+1 for the gold species, "
                "−1 for the other five). AdamW at a fixed learning rate, gradient clipping at 1.0, seeded shuffling and no "
                "scheduler. Epoch 0 records the frozen model's validation metrics; every epoch is scored on the 48 validation "
                "photographs, and the epoch with the highest validation text-to-image mAP is kept — accuracy on 48 photographs "
                "moves in steps of 2 %, which is why the retrieval view selects and the 96-photograph test split is what the "
                "numbers are read from.\n\n"
                "Watch the training loss fall from about 2.2 to below 0.1 within six epochs while the validation mAP peaks "
                "early: 216 photographs are few, the last blocks memorise them, and the selector's job is to stop before that "
                "hurts. The build record's counter-examples — adapting the text tower instead, or both towers — are in the "
                "model card; the default is the configuration that gained on the held-out split. The cell first makes sure `pipe` is the "
                "frozen model (reloading it from the snapshot if an earlier run adapted it), so a second run after changing a setting trains the "
                "pretrained tower again, not the previous adaptation.\n\n"
                "**Predict:** six epochs over 216 photographs. Will the validation mAP rise at every epoch, and will the kept epoch be the last one?"
            ),
            "code": (
                "EPOCHS = 6  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 5e-5  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 16  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_VISION_LAYERS = 2  # @param {{type:\"integer\"}}\n\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row.update({{'val_' + k: round(entry['val'][k], 3) for k in METRICS}})\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n\n"
                "frozen_pipeline()  # every run of this cell trains the frozen model, never an earlier adaptation\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_vision_layers=TRAINABLE_VISION_LAYERS, class_names_map=display_names, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'classes': adapt_result['classes'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>Neither. The recorded workstation pre-flight logged validation mAP 0.680 → 0.874 → 0.854 → 0.863 → 0.876 → 0.872 → 0.854 and kept epoch 4: the retrieval view peaks early and wobbles while the training loss keeps falling, because 216 photographs are few and the last blocks memorise them. Read the `val_t2i_map` column to see which epoch is kept, and expect your own kept epoch to differ by one or two.</details>'
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test photographs were never used for training or epoch selection, and no photograph appears in two splits. "
                "The adapted model is scored exactly as the frozen model was in Section 6 — the same six prompts — the four "
                "systems are put side by side on the three metrics, the per-species breakdown is repeated, and the "
                "scientific-name prompt set is scored again (the vision tower was adapted, not the prompts, so a gain that "
                "carries to a prompt set it never saw is the more general one). Read it in this order: **text-to-image mAP** "
                "first (the metric the epoch was selected on — the build record measured 0.72 → 0.87), then accuracy and macro "
                "F1 (0.76 → 0.79 and 0.76 → 0.79, three photographs of 96), then the per-species recall, where the "
                "White-throated Sparrow moved from 0.44 to 0.75 while the Chipping Sparrow lost ground. The cell records verdicts instead of "
                "asserting — `improved` / `no gain` / `worse` for the adapted mAP and accuracy against the frozen model — so a BYOD run that does "
                "not gain still exports and reloads; the verdicts go into the evaluation report and `result.json`. Ninety-six photographs from one seeded split give **no dispersion "
                "estimate**; the deltas are sample-sanity evidence that the adaptation contract works, not a benchmark, and a "
                "gain on six birds says nothing about your classes until you measure them.\n\n"
                "**Predict:** write down a direction and a size for the mAP delta and for the accuracy delta. Will the scientific-name prompts, which the tower never saw during training, move too — and which species will gain most?"
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records, classes=classes, class_names_map=display_names)\n"
                "adapted_val = pipe.evaluate(val_records, classes=classes, class_names_map=display_names)\n"
                "adapted_fields = {{c: {{'n': v['n'], 'recall': round(v['recall'], 2), 'ap': round(v['ap'], 2)}} for c, v in adapted_test['per_class'].items()}}\n"
                "adapted_scientific = pipe.evaluate(test_records, classes=classes, class_names_map=scientific_names, prompt_template='This is a photo of {{label}}.')\n"
                "comparison = {{metric: {{'majority': round(baseline_majority[metric], 3), 'neighbour': round(baseline_neighbour[metric], 3), 'frozen': round(frozen_test[metric], 3), 'adapted': round(adapted_test[metric], 3)}} for metric in METRICS}}\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 3) for metric in METRICS}}\n"
                "comparison['scientific_name_prompts'] = {{metric: {{'frozen': round(frozen_scientific[metric], 3), 'adapted': round(adapted_scientific[metric], 3)}} for metric in METRICS}}\n"
                "comparison['by_species'] = {{c: {{'n': frozen_fields[c]['n'], 'frozen_recall': frozen_fields[c]['recall'], 'adapted_recall': adapted_fields[c]['recall'], 'frozen_ap': frozen_fields[c]['ap'], 'adapted_ap': adapted_fields[c]['ap']}} for c in classes}}\n"
                "# Recorded verdicts, not asserts: a BYOD run whose mAP does not gain still exports, reloads and writes result.json.\n"
                "delta_map = adapted_test['t2i_map'] - frozen_test['t2i_map']\n"
                "adaptation_verdict = 'improved' if delta_map > 0 else ('no gain' if delta_map == 0 else 'worse')\n"
                "comparison['verdicts'] = {{'frozen_vs_baselines': frozen_verdict, 'adapted_vs_frozen_t2i_map': adaptation_verdict, 'adapted_vs_frozen_accuracy': 'improved' if adapted_test['accuracy'] > frozen_test['accuracy'] else ('no gain' if adapted_test['accuracy'] == frozen_test['accuracy'] else 'worse')}}\n"
                "for key, row in comparison.items():\n"
                "    print({{key: row}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': DEFAULT_MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'classes': classes,\n"
                "    'display_names': display_names,\n"
                "    'baselines': {{'majority': baseline_majority, 'colour_neighbour': baseline_neighbour}},\n"
                "    'frozen_test': frozen_test,\n"
                "    'frozen_test_scientific_names': frozen_scientific,\n"
                "    'validation_metrics': adapted_val,\n"
                "    'test_metrics': adapted_test,\n"
                "    'test_metrics_scientific_names': adapted_scientific,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json', 'verdicts': comparison['verdicts']}})"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>In the recorded Kaggle T4 run mAP rose 0.723 → 0.871 (+0.148), accuracy 0.76 → 0.792 (+0.031, three photographs of 96) and macro F1 0.758 → 0.793 — verdicts `improved` and `improved`. The scientific-name prompts, never seen in training, moved with it: accuracy 0.479 → 0.51 and mAP 0.467 → 0.492, while their macro F1 slipped 0.459 → 0.453. The White-throated Sparrow gained most (recall 0.44 → 0.75, AP 0.35 → 0.89) and the Chipping Sparrow lost ground (recall 0.75 → 0.62): a gain concentrated where the prompt separated worst, and not uniform.</details>'
            ),
        },
        {
            "md": (
                "## 9. Re-score the drawn shapes, export the adapter and reload it\n\n"
                "The three shapes from Section 5 are classified again by the adapted model — drawings, a different image family "
                "from the photographs it was tuned on, so this is a small look at what the adaptation did *outside* its corpus "
                "(the build record's rankings are in the model card; a changed ranking here is a finding to record, not a "
                "failure) — and reported with the per-grid `evaluation_report` (`sample-sanity`). Both score sets are written "
                "as JSON.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the vision tower's last two blocks, post-layernorm and "
                "attention-pool head, about 85 MB — as `adapter.safetensors`, with a `manifest.json` recording the artifact "
                "format, the base model id and revision, the digest of the base `model.safetensors`, the classes and prompt "
                "template, the tensor names, the file size and SHA-256, the training configuration and the epoch history "
                "(OUT8). `Siglip2Pipeline.from_artifact` re-verifies the base snapshot, checks the artifact manifest, its digest "
                "and its exact tensor set **before** deserialising, refuses any tensor outside the vision tower, and overlays "
                "the tensors onto a freshly loaded base — a new object from files, not the in-memory model (VER2). The cell "
                "asserts identical image embeddings on eight test photographs (VER4).\n\n"
                "**Predict:** the reloaded pipeline is a new object built from files. Will all eight embeddings match to within 1e-5, and will the adapted tower still put the right label on top for each drawn shape?"
            ),
            "code": (
                "import shutil\n\n"
                "adapted_classifications = [pipe.zero_shot_classify(image_path, candidate_labels) for image_path in shape_images]\n"
                "adapted_retrievals = [pipe.retrieve(query, shape_images, top_k=len(shape_images)) for query in shape_labels]\n"
                "adapted_scene = evaluation_report({{'classifications': adapted_classifications, 'retrievals': adapted_retrievals, 'gallery_ids': [p.name for p in shape_images]}}, targets, sample_kind='synthetic')\n"
                "for image_path, before, after in zip(shape_images, classifications, adapted_classifications, strict=True):\n"
                "    print({{'image': image_path.name, 'frozen': [(s.label, round(s.score, 3)) for s in before[:2]], 'adapted': [(s.label, round(s.score, 3)) for s in after[:2]]}})\n"
                "print({{'scene_after_adaptation': {{m['id']: round(m['value'], 3) for m in adapted_scene['metrics']}}, 'verdict': adapted_scene['verdict']}})\n"
                "with open('outputs/{stem}_shapes.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump({{'frozen': classification_rows, 'adapted': [{{'image': p.name, 'scores': [asdict(x) for x in ranking]}} for p, ranking in zip(shape_images, adapted_classifications, strict=True)]}}, handle, indent=2)\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = Siglip2Pipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "before = pipe.embed_image([r['image'] for r in test_records[:8]])\n"
                "after = reloaded.embed_image([r['image'] for r in test_records[:8]])\n"
                "parity = {{'max_abs_difference': float(np.abs(before - after).max()), 'identical_rows': int((np.abs(before - after).max(axis=1) < 1e-5).sum()), 'of': int(before.shape[0])}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['identical_rows'] == parity['of']\n\n"
                "write_provenance('outputs/provenance.json', pipeline=pipe)\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': snapshot['files'], 'fetched_this_run': fetched, 'weight_file': MODEL_FILENAME, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': MODEL_SHA256}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'release': CORPUS_RELEASE, 'license': CORPUS_LICENSE, 'base_url': CORPUS_BASE_URL, 'bytes': CORPUS_BYTES, 'pinned_photographs': len(SAMPLE_RECORDS), 'species': {{k: list(v) for k, v in SPECIES.items()}}}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'sanity_checks': checks, 'shapes': {{'names': [p.name for p in shape_images], 'sha256': shape_sha256, 'labels': shape_labels, 'candidate_labels': candidate_labels}}, 'frozen_report': frozen_scene, 'adapted_report': adapted_scene}},\n"
                "    'comparison': comparison,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'numpy': numpy.__version__, 'device': str(pipe.device), 'dtype': 'float32', 'checkpoint_source': pipe.checkpoint_source}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>All eight embeddings matched — the recorded run reported a maximum absolute difference of 0 and `identical_rows: 8 of 8` — because the reloaded pipeline runs the same tensors on the same device. The shapes stayed correctly labelled after adaptation in the recorded pre-flight (a changed ranking would be a finding to record, not a failure); the adapter was 85,062,712 bytes in 45 tensors.</details>'
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The frozen model is already a usable zero-shot classifier on six bird species it was never told about — far above "
        "the two non-neural baselines — and a bounded fine-tuning of the vision tower's last two blocks and head on 216 "
        "photographs moves the retrieval view clearly (text-to-image mAP 0.72 → 0.87 in the build record) and the top-1 "
        "accuracy by a few photographs (0.76 → 0.79), with the gain concentrated on the species the prompt separated worst, "
        "and an 85 MB adapter that reloads to identical embeddings. That is the claim: the adaptation contract works end to "
        "end on a real labelled photograph set, and the numbers it produces are read on three metrics, per species, on two "
        "prompt sets, against two non-neural baselines and the frozen model rather than in isolation.\n\n"
        "The test split is 96 photographs from one seeded draw of one sample, the validation split that picks the epoch is 48, "
        "the metrics are three reference-based scores (own numpy implementations; none a human judgement), accuracy moves in "
        "steps of one photograph, and the build record's own sweep shows the estimate's fragility: on a 48-photograph test "
        "split the same recipe moved accuracy by anything from +2 to +12 points depending on the epoch. So a gain here says the "
        "contract works, not that the adapted model is better on your photographs, that its scores are calibrated, or that a "
        "sigmoid score is a probability of being right — it still returns a best label for every image, and it can be wrong "
        "confidently. Fine-tuning on a narrow set can also erode the model elsewhere; the drawn shapes re-scored in Section 9 "
        "are three images of evidence about that, not a measurement.\n\n"
        "Three things to carry to real data. **Baselines first:** the majority floor, the colour neighbour and the frozen "
        "model's score on *your* labels are the numbers to read before any adapted one, per class and on the retrieval view. "
        "**Leakage:** keep every photograph in one split (the contract de-duplicates by decoded pixels) and split by "
        "photographer or session when your images come from few sources — the sample's observer overlap is printed for exactly "
        "that reason. **Prompts:** the classifier is the prompt as much as the tower; a second prompt set is scored here so the "
        "difference is visible, and a label an annotator wrote is not a prompt the model understands.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this standalone notebook, can "
        "acquire and digest-verify the pinned model snapshot, fetch and digest-verify a real labelled photograph set, validate "
        "the demonstrated dataset contract without leakage, execute the inference contract and a bounded fine-tuning, evaluate "
        "against two trivial baselines and the frozen model on an image-disjoint split, and emit the shown machine-readable "
        "artifacts — without the repository being reachable. It does **not** establish benchmark superiority, zero-shot "
        "accuracy on any other population or camera, calibration, or production fitness.\n\n"
        '## Troubleshooting\n'
        '\n'
        '- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — use Google Colab, Kaggle or a Linux x86_64 Jupyter server.\n'
        '- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 again; a complete environment is reused and an incomplete one is finished. If it repeats, `files.pythonhosted.org` or `pypi.org` is blocked or altered.\n'
        '- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable. After a session restart, run from the top.\n'
        '- **"The isolated environment\'s Python process exited"** — usually out of memory; restart the session and choose **Run all**.\n'
        '- **Section 3 reports a size or SHA-256 mismatch, or cannot reach the Hub** — the message names the file. Delete the folder Section 3 prints as `weights_dir` and run Section 3 again (1.5 GB).\n'
        '- **Section 4 refuses a photograph** — its byte size or SHA-256 does not match the pin: the cached file under `weights/inat-birds/` is damaged (delete it) or the bucket served something else; the message names the photo id.\n'
        '- **The frozen or adapted numbers differ from the recorded ones by a photograph or two** — expected: accuracy on 96 photographs moves in steps of about 1 %, and the recorded T4 and workstation runs themselves differ in the second decimal. A verdict of `not above the baselines` or `no gain` / `worse` is a finding to read, not an error.\n'
        '- **You re-ran Section 6 or 7 after training** — both reload the frozen model first when the pipeline carries an adaptation, so the frozen numbers stay frozen and a second fine-tune starts from the pretrained tower, never from the previous adaptation.\n'
        "- **Section 9's parity check fails** — the export or reload is broken; run Sections 7–9 again. Do not use the artifact.\n"
        '- **BYOD: "BYOD path … does not exist" / "the upload dialog exists only in Google Colab" / "Upload exactly one"** — set `BYOD_PATH` to a zip in the runtime (it works on Kaggle and Jupyter); on Colab an empty path opens the dialog, and a cancelled dialog stops with that message.\n'
        '- **A `ValueError` from `load_byod_dataset` or `validate_dataset`** — it names the file and the rule: a missing `labels.csv`, an image side over 4096 px, a label over 64 characters, fewer than eight records, one label only, or a duplicate id.\n'
        '\n'
        '## Change one thing (next experiments)\n'
        '\n'
        'Each of these changes one default and keeps the rest of the path; Sections 6 and 7 reload the frozen model before they run, so the frozen numbers are the fixed reference. Set `TRAINABLE_VISION_LAYERS = 4` and compare the artifact size and the held-out mAP; raise `EPOCHS` and watch the validation mAP pick the epoch while the training loss keeps falling; change `LEARNING_RATE` to `1e-5` and read a smaller, steadier gain; change `SPLIT_SEED` for another draw of the 360 photographs; or bring your own photographs through BYOD and read the two baselines before the adapted number.\n'
        '\n'
        '## Glossary\n'
        '\n'
        '- **Zero-shot classification** — scoring an image against a prompt per label (`DEFAULT_PROMPT_TEMPLATE` with the label filled in) and returning the highest; no label-specific training.\n'
        '- **Sigmoid score** — each image–prompt pair is scored on its own through `logit_scale`, `logit_bias` and a sigmoid; the scores do not sum to one, are not calibrated, and the model never abstains.\n'
        '- **Accuracy / macro F1** — the fraction of test photographs whose top prompt is right; the mean per-species F1, which weighs each species equally.\n'
        '- **Text-to-image mAP** — each prompt as a query ranking all 96 photographs; the average precision of its own species, averaged over the six — the retrieval view of the same scores, and the smoother metric on a small set.\n'
        '- **Majority floor / colour nearest neighbour** — answering every photograph with the most frequent training label; answering with the label of the training photograph whose 3 × 3 mean-colour grid is closest (27 numbers, no neural network).\n'
        '- **Vision tower / attention-pool head** — the ViT-B/16 image encoder and the head that pools its patch tokens into one embedding; the fine-tune trains the last `TRAINABLE_VISION_LAYERS` blocks, the post-layernorm and the head.\n'
        '- **SigLIP sigmoid loss** — the pairwise training objective: +1 for the gold prompt, −1 for every other, each pair scored independently.\n'
        '- **Epoch selection** — the kept epoch has the highest validation text-to-image mAP; accuracy on 48 photographs moves in steps of 2 %, so the retrieval view selects.\n'
        '- **Frozen model reload** — when the pipeline carries an adaptation, Sections 6 and 7 load the pretrained model again from the verified snapshot before scoring or training.\n'
        '- **Reload parity** — the adapter written to disk, loaded onto a fresh base, reproduces the in-memory image embeddings of eight test photographs.\n'
        '- **Isolated environment** — the separate Python 3.12.12 environment Section 1 builds from the hash lock; every later cell runs there.\n'
        '- **BYOD** — bring your own data: a zip with `labels.csv` beside the image files, read from `BYOD_PATH` or the Colab upload dialog.\n'
        '\n'
        '## Conclusion (your notes)\n'
        '\n'
        "Before you leave, write three lines in this cell: (1) the frozen model's accuracy and mAP beside the two baselines, and which species the common-name prompt separated worst; (2) the verdicts Section 8 recorded — did the mAP gain, and did the scientific-name prompt set, which the tower never saw, move with it? (3) one reason the same recipe could land differently on your own labels (prompts, split, photographers) and the baseline you would read first.\n"
        '\n'
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/siglip2-vision-language-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/siglip2-vision-language-pipeline/blob/main/MODEL_CARD.md\n"
        "- Sample dataset card (synthetic shapes): https://github.com/kurtvalcorza/siglip2-vision-language-pipeline/blob/main/examples/sample-data/DATASET_CARD.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream library: https://github.com/huggingface/transformers\n"
        "- SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features (Tschannen et al., 2025): https://arxiv.org/abs/2502.14786\n"
        "- Sigmoid Loss for Language Image Pre-Training (Zhai et al., 2023): https://arxiv.org/abs/2303.15343\n"
        "- iNaturalist open data (CC0 photographs, each observer's own licence): https://www.inaturalist.org/pages/developers — bucket https://inaturalist-open-data.s3.amazonaws.com/\n"
        "- DIMER Notebook Specification 2.2 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
