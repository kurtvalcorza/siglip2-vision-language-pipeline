"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded package (four
modules, carried verbatim in dependency order), and the model pin/stage/verify cells are produced by
the generator from repository sources so they cannot drift from the package.

Generator /2 keys in use: ``modules`` lists every module of ``src/siglip2_pipeline/`` except
``__init__.py``; ``entry_module`` is ``config.py`` (it holds ``MODEL_ID``/``MODEL_REVISION``/
``MODEL_LICENSE`` and the model key under the package's own spelling ``DEFAULT_MODEL_KEY``, mapped by
``identity_names``); ``rewrites`` carries two rules — the fleet ``DEFAULT_WEIGHTS_DIR`` rule and the
``__file__`` use inside ``model.resolve_weights_path`` (a repository-checkout convenience that a
standalone notebook has no checkout for); ``model_load`` keeps the tutorial on CPU
(``device="cpu"``) because the release lock is the CPU-only reference graph.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    'package': 'siglip2_pipeline',
    'repo_name': 'siglip2-vision-language-pipeline',
    'stem': 'siglip2_vision_language',
    'notebook_name': 'siglip2_vision_language_colab.ipynb',
    'profile': 'MULTI-CAPABILITY',
    'pipeline_class': 'Siglip2Pipeline',
    'weights_key': 'siglip2-base-patch16-224',
    'modules': ['config.py', 'model.py', 'pipeline.py', 'provenance.py'],
    'entry_module': 'config.py',
    'identity_names': {'MODEL_KEY': 'DEFAULT_MODEL_KEY'},
    'rewrites': [['^DEFAULT_WEIGHTS_DIR = Path\\(__file__\\)[^\\n]*$', 'DEFAULT_WEIGHTS_DIR = Path.cwd() / "weights" / DEFAULT_MODEL_KEY  # standalone rewrite (build_notebook.py): working-directory-relative'], ['^    repo_root = Path\\(__file__\\)\\.resolve\\(\\)\\.parents\\[2\\]$', '    repo_root = Path.cwd()  # standalone rewrite (build_notebook.py): no repository checkout exists; the notebook passes weights_dir explicitly']],
    'model_load': 'Siglip2Pipeline.from_pretrained(device="cpu", weights_dir=WEIGHTS_DIR)',
    'runtime_imports': ['torch', 'transformers', 'numpy'],
    'title': 'SigLIP 2 Vision-Language Pipeline — DIMER `MULTI-CAPABILITY` tutorial (standalone)',
    'badges': [
        (
            'GitHub',
            'https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white',
            'https://github.com/kurtvalcorza/siglip2-vision-language-pipeline',
        ),
        (
            'Open In Colab',
            'https://colab.research.google.com/assets/colab-badge.svg',
            'https://colab.research.google.com/github/kurtvalcorza/siglip2-vision-language-pipeline/blob/main/tutorials/siglip2_vision_language_colab.ipynb',
        ),
        (
            'Hugging Face',
            'https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-google%2Fsiglip2--base--patch16--224-ffcc4d?style=flat',
            'https://huggingface.co/google/siglip2-base-patch16-224',
        ),
        (
            'Upstream',
            'https://img.shields.io/badge/Upstream-huggingface%2Ftransformers-181717?style=flat&logo=github&logoColor=white',
            'https://github.com/huggingface/transformers',
        ),
        (
            'arXiv',
            'https://img.shields.io/badge/arXiv-2502.14786-b31b1b.svg',
            'https://arxiv.org/abs/2502.14786',
        ),
    ],
    'capability': (
        'zero-shot image classification, image/text embeddings, cosine similarity, and text-to-image retrieval with the pinned `google/siglip2-base-patch16-224` checkpoint'
    ),
    'intro': (
        "This notebook uses the pinned `google/siglip2-base-patch16-224` checkpoint through the repository's public API — carried in this notebook — for zero-shot classification, image/text embeddings, cosine similarity, and text-to-image retrieval. **No gradient training, fine-tuning, in-context conditioning, or fitted preprocessing state occurs** — **no adaptation occurs.** Upstream provides the pretrained model/processor; this repository adds immutable pinning, integrity verification, safe local loading, stable inference contracts, the `validate_inputs` / `evaluation_report` role stages, exports, and provenance. Outputs are uncalibrated inference/ranking evidence. The default data are deterministic **synthetic tutorial/smoke assets** generated in code, not benchmark data."
    ),
    'learning_objectives': (
        "install the pinned runtime, read what the carried package guarantees, resolve and digest-verify the immutable model revision, generate the three synthetic sample images and validate them into an input manifest, run all five public operations (classification, image embeddings, text embeddings, similarity, retrieval) and read each capability's input/output contract, produce an evaluation report whose `top1_accuracy` and `recall_at_1` are `sample-sanity` evidence on the synthetic set, exercise an optional BYOD path and a default new-data path, and export machine-readable outputs plus provenance."
    ),
    'exclusions': (
        'object detection, semantic segmentation, OCR, caption generation, calibrated probabilities, universal thresholds, or production serving. GPU execution is outside this release contract: the tutorial runs on CPU even on a GPU host.'
    ),
    'prerequisites': [
        '- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The release reference is **CPU-only**: the pinned `torch==2.14.0` install is the largest download of the run and the model is loaded with `device="cpu"` even on a GPU host; GPU execution requires a separately pinned/tested environment. The verified weight file is 1,500,800,904 bytes.',
        '- **Knowledge:** basic Python and PIL image handling; what a sigmoid score and a cosine similarity are.',
        '- **Data:** the default sample is three deterministic 32×32 synthetic images (red square, green circle, blue triangle) generated in code and digest-asserted, so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one image file decodable by Pillow (any colour mode; the pipeline converts to RGB and applies the pinned 224×224 image contract), plus your own candidate labels; model-bound text is lowercased and truncated to a 64-token maximum; candidate labels/queries must be non-empty; HTTP(S) image URLs are rejected. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send image contents to a hosted inference API.',
    ],
    "cells": [
        {
            "md": (
                '## 4. Generate the synthetic sample or optional BYOD\n'
                '\n'
                "The default sample is **synthetic**: three 32×32 RGB images — a red square, a green circle and a blue triangle on a light background — rendered in code as ASCII PPM files exactly as the repository's `examples/sample-data/generate_samples.py` writes them, so each file's SHA-256 is asserted against the digest the repository checks in. They provide deterministic tutorial ground truth (the expected label of each image) for the sanity metrics later; they are **synthetic tutorial/smoke assets**, not benchmark data. BYOD is optional and disabled by default; when enabled, upload one image (Colab) or set `BYOD_PATH` (other Jupyter environments), and edit `BYOD_LABELS` for the intended domain. Look for the three file names, sizes and digests."
            ),
            "code": (
                'import hashlib\n'
                'import io\n'
                'import os\n'
                'from pathlib import Path\n'
                '\n'
                'import numpy as np\n'
                'from PIL import Image\n'
                '\n'
                'USE_BYOD = False  # @param {{type:"boolean"}}\n'
                'BYOD_PATH = ""  # @param {{type:"string"}}\n'
                'BYOD_LABELS = ["flooded street", "normal road", "fallen electrical pole"]\n'
                'SAMPLE_DIGESTS = {{  # examples/sample-data/SHA256SUMS\n'
                '    "red_square.ppm": "b38ff0c9131677ed6cf09832eff40a22841e1a4725d426fad3b1bf6a1dbdb096",\n'
                '    "green_circle.ppm": "2e3e657686a0f6a6df3f3621d21a410d20d9a47b109f06f9405faa4f747a663f",\n'
                '    "blue_triangle.ppm": "f0f4c38c7af92b3a6b1d55a25272156edd87b41e029e116cfa059003dae029a3",\n'
                '}}\n'
                'WIDTH, HEIGHT, BACKGROUND = 32, 32, (245, 245, 245)\n'
                '\n'
                '\n'
                'def shape_mask(shape: str, x: int, y: int) -> bool:\n'
                '    if shape == "square":\n'
                '        return 8 <= x < 24 and 8 <= y < 24\n'
                '    if shape == "circle":\n'
                '        return (x - 16) ** 2 + (y - 16) ** 2 <= 9**2\n'
                '    if not 7 <= y < 26:\n'
                '        return False\n'
                '    half = (y - 7) // 2\n'
                '    return 16 - half <= x <= 16 + half\n'
                '\n'
                '\n'
                'def render_ppm(foreground: tuple, shape: str) -> str:\n'
                "    # The repository's generate_samples.py rendering: ASCII P3, 24 values per line.\n"
                '    lines = ["P3", f"{{WIDTH}} {{HEIGHT}}", "255"]\n'
                '    for y in range(HEIGHT):\n'
                '        row = []\n'
                '        for x in range(WIDTH):\n'
                '            pixel = foreground if shape_mask(shape, x, y) else BACKGROUND\n'
                '            row.extend(str(value) for value in pixel)\n'
                '        for offset in range(0, len(row), 24):\n'
                '            lines.append(" ".join(row[offset : offset + 24]))\n'
                '    return "\\n".join(lines) + "\\n"\n'
                '\n'
                '\n'
                'os.makedirs("outputs/sample-data", exist_ok=True)\n'
                'specs = {{"red_square.ppm": ((220, 40, 40), "square"), "green_circle.ppm": ((40, 170, 75), "circle"), "blue_triangle.ppm": ((40, 90, 220), "triangle")}}\n'
                'images = []\n'
                'for name, (foreground, shape) in specs.items():\n'
                '    path = Path("outputs/sample-data") / name\n'
                '    path.write_bytes(render_ppm(foreground, shape).encode("ascii"))\n'
                '    digest = hashlib.sha256(path.read_bytes()).hexdigest()\n'
                '    if digest != SAMPLE_DIGESTS[name]:\n'
                '        raise ValueError(f"Synthetic sample digest mismatch for {{name}}: {{digest}} != {{SAMPLE_DIGESTS[name]}}")\n'
                '    images.append(path)\n'
                'expected_labels = ["red square", "green circle", "blue triangle"]\n'
                'candidate_labels = [*expected_labels, "abstract geometric shape"]\n'
                'sample_kind = "synthetic"\n'
                'sample_sha256 = {{p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in images}}\n'
                'for path in images:\n'
                '    with Image.open(path) as im:\n'
                '        print({{"name": path.name, "mode": im.mode, "size": im.size, "sha256": sample_sha256[path.name]}})\n'
                '\n'
                'byod_image = None\n'
                'if USE_BYOD:\n'
                '    try:\n'
                '        from google.colab import files as colab_files\n'
                '        uploaded = colab_files.upload()\n'
                '        if len(uploaded) != 1:\n'
                '            raise ValueError("Upload exactly one image.")\n'
                '        name, data = next(iter(uploaded.items()))\n'
                '        target = Path("outputs/byod") / Path(name).name\n'
                '        target.parent.mkdir(parents=True, exist_ok=True)\n'
                '        target.write_bytes(data)\n'
                '    except ModuleNotFoundError:\n'
                '        if not BYOD_PATH:\n'
                '            raise RuntimeError("Outside Colab, set BYOD_PATH before enabling BYOD.") from None\n'
                '        target = Path(BYOD_PATH).expanduser()\n'
                '    if not target.is_file():\n'
                '        raise FileNotFoundError(target)\n'
                '    try:\n'
                '        with Image.open(target) as im:\n'
                '            im.verify()\n'
                '    except Exception as exc:\n'
                '        raise ValueError(f"BYOD file is not a decodable image: {{target}}") from exc\n'
                '    if not BYOD_LABELS or any(not x.strip() for x in BYOD_LABELS):\n'
                '        raise ValueError("BYOD_LABELS must contain non-empty labels.")\n'
                '    byod_image = target\n'
                '    sample_kind = "BYOD"\n'
                '    print({{"byod_image": str(byod_image), "byod_labels": BYOD_LABELS}})\n'
                'else:\n'
                '    print("BYOD disabled; default path is non-interactive.")'
            ),
        },
        {
            "md": (
                '## 5. Validate the inputs → input manifest\n'
                '\n'
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks the five public operations apply — every image coerced to RGB the way `zero_shot_classify` / `embed_image` do (local bytes or paths only; HTTP(S) URLs are rejected), non-empty labels/queries, a `prompt_template` carrying the `{{label}}` placeholder, `top_k >= 1` — and returns an **input manifest** naming the schema and ceilings. The effective runtime and model identity (Python, PyTorch, Transformers, device, model id, immutable revision, verified weight digest and byte count) are printed first (the 224×224 image contract, the 64-token text maximum, the default prompt template), each image's observed mode and size, the texts, and the verdict. The manifest is written to `outputs/{stem}_input_manifest.json`. To show what rejection looks like, the cell also validates a remote URL and records the pipeline's own error message as a finding. The ceilings are printed before the model runs anything."
            ),
            "code": (
                'import json\n'
                '\n'
                "os.makedirs('outputs', exist_ok=True)\n"
                'print("Python:", platform.python_version())\n'
                'print("PyTorch:", torch.__version__)\n'
                'print("Transformers:", transformers.__version__)\n'
                'print("Device:", pipe.device)\n'
                'print("Model ID:", MODEL_ID)\n'
                'print("Revision:", MODEL_REVISION)\n'
                'print("Checkpoint source:", pipe.checkpoint_source, "| manifest verified:", pipe.manifest_verified)\n'
                'print("Weight SHA-256:", pipe.weight_sha256)\n'
                'print("Weight bytes:", pipe.weight_size_bytes)\n'
                "print({{'ceilings': {{'TEXT_MAX_LENGTH': TEXT_MAX_LENGTH, 'DEFAULT_PROMPT_TEMPLATE': DEFAULT_PROMPT_TEMPLATE, 'image_contract': '224x224 RGB after processor resize', 'device': str(pipe.device)}}}})\n"
                'validation_images = [*images] + ([byod_image] if byod_image is not None else [])\n'
                'validation_names = [p.name for p in validation_images]\n'
                'input_manifest = validate_inputs(validation_images, candidate_labels, top_k=len(images), names=validation_names)\n'
                '# Demonstrate rejection on an input the pipeline refuses; the finding is recorded, not swallowed.\n'
                'try:\n'
                '    validate_inputs("https://example.invalid/not-allowed.png", candidate_labels)\n'
                'except ValueError as exc:\n'
                '    input_manifest["findings"].append({{"input": "remote-url-probe", "verdict": "rejected", "message": str(exc)}})\n'
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                '    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n'
                'print(json.dumps(input_manifest, indent=2))'
            ),
        },
        {
            "md": (
                '## 6. Capability A — zero-shot classification\n'
                '\n'
                "`zero_shot_classify(image, labels)` scores each candidate label's prompt (`DEFAULT_PROMPT_TEMPLATE`, model-bound text lowercased) against the image and returns the labels **ranked by descending score**. Zero-shot scores are independent SigLIP sigmoids: **not calibrated probabilities** and not required to sum to one; they are **prompt/label dependent** — rewording a label changes its score. The tutorial uses argmax only for its sanity check; any deployment threshold/abstention rule belongs to the **downstream application** and must be calibrated on representative labelled data. Look for one ranked list per sample image and the BYOD ranking when enabled."
            ),
            "code": (
                'from dataclasses import asdict\n'
                '\n'
                'classifications = []\n'
                'classification_rows = []\n'
                'for image_path, expected in zip(images, expected_labels, strict=True):\n'
                '    scores = pipe.zero_shot_classify(image_path, candidate_labels)\n'
                '    classifications.append(scores)\n'
                '    classification_rows.append({{"image": image_path.name, "expected_label": expected, "predicted_label": scores[0].label, "scores": [asdict(x) for x in scores]}})\n'
                '    print(image_path.name, "->", [(s.label, round(s.score, 4)) for s in scores])\n'
                '\n'
                'byod_scores = None\n'
                'if byod_image is not None:\n'
                '    byod_scores = pipe.zero_shot_classify(byod_image, BYOD_LABELS)\n'
                '    print("BYOD top label:", byod_scores[0].label, [(s.label, round(s.score, 4)) for s in byod_scores])'
            ),
        },
        {
            "md": (
                '## 7. Capability B — image and text embeddings, similarity\n'
                '\n'
                '`embed_image` and `embed_text` return **L2-normalized** vectors, one per input, in a shared space: embeddings are representations, not predictions, and have **no intrinsic accuracy metric**. `similarity(images, texts)` is their cosine similarity matrix (rows = images, columns = texts); a similarity is a raw score, not a calibrated probability. Look for the two embedding shapes and the 3×3 matrix whose diagonal should dominate on the synthetic set.'
            ),
            "code": (
                'image_embeddings = pipe.embed_image(images)\n'
                'text_embeddings = pipe.embed_text(expected_labels)\n'
                'similarity = pipe.similarity(images, expected_labels)\n'
                'print("Image embeddings:", image_embeddings.shape, "| L2 norms:", np.round(np.linalg.norm(image_embeddings, axis=1), 4).tolist())\n'
                'print("Text embeddings:", text_embeddings.shape)\n'
                'print("Similarity rows:", [p.name for p in images])\n'
                'print("Similarity columns:", expected_labels)\n'
                'print(np.array2string(similarity, precision=4))\n'
                'byod_embedding = pipe.embed_image([byod_image]) if byod_image is not None else None'
            ),
        },
        {
            "md": (
                '## 8. Capability C — text-to-image retrieval\n'
                '\n'
                '`retrieve(query, images, top_k)` ranks the gallery images by cosine similarity to one text query and returns `RetrievalHit(index, score)` entries **ordered by descending score** (ties broken by stable index order). Retrieval usefulness on real data requires a downstream labelled evaluation; here each expected label is used as the query over the three-image gallery so the ranking is falsifiable.'
            ),
            "code": (
                'retrievals = []\n'
                'retrieval_rows = []\n'
                'for query, expected_image in zip(expected_labels, images, strict=True):\n'
                '    hits = pipe.retrieve(query, images, top_k=len(images))\n'
                '    retrievals.append(hits)\n'
                '    retrieval_rows.append({{"query": query, "expected_image": expected_image.name, "hits": [{{"rank": r, "index": h.index, "filename": images[h.index].name, "score": h.score}} for r, h in enumerate(hits, 1)]}})\n'
                '    print(query, "->", [(images[h.index].name, round(h.score, 4)) for h in hits])'
            ),
        },
        {
            "md": (
                '## 9. Evaluate → evaluation report\n'
                '\n'
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. On the synthetic set it carries `top1_accuracy` (the repository's helper: the fraction of images whose top-scoring label equals the expected label, compared with the fixed-class baseline of always predicting one candidate, 1/4 here) and `recall_at_1` (the fraction of queries whose top hit is the expected image) with the verdict `sample-sanity` — tiny synthetic sanity metrics, not estimates of generalization. Embeddings and similarity have no intrinsic metric and are reported as representations. Without expected labels or expected retrieval indices (the BYOD case) the verdict is `not-measurable` and the report states what labelled data would make the task measurable. The report is written to `outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                'result = {{"classifications": classifications, "retrievals": retrievals, "gallery_ids": [p.name for p in images], "embedding_shapes": {{"image": list(image_embeddings.shape), "text": list(text_embeddings.shape)}}, "similarity_shape": list(similarity.shape)}}\n'
                'targets = {{"labels": expected_labels, "retrieval_indices": list(range(len(images)))}}\n'
                'report = evaluation_report(result, targets, sample_kind=sample_kind)\n'
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                '    json.dump(report, handle, indent=2, ensure_ascii=False)\n'
                'print(json.dumps(report, indent=2))\n'
                'if byod_scores is not None:\n'
                '    byod_report = evaluation_report({{"classifications": [byod_scores]}}, sample_kind="BYOD")\n'
                '    print("BYOD report verdict:", byod_report["verdict"], "-", byod_report["reason"])\n'
                'top1_sanity_accuracy = next(m["value"] for m in report["metrics"] if m["id"] == "top1_accuracy")\n'
                'retrieval_recall_at_1 = next(m["value"] for m in report["metrics"] if m["id"] == "recall_at_1")\n'
                'print("Top-1 sanity accuracy:", top1_sanity_accuracy, "| fixed-class baseline:", report["baselines"][0]["value"], "| recall@1:", retrieval_recall_at_1)'
            ),
        },
        {
            "md": (
                '## 10. Default new-data inference\n'
                '\n'
                'A deterministic `yellow_square.ppm`, distinct from the three-image evaluation set, exercises the new-input path without interaction. It remains synthetic evidence and does not establish deployment accuracy.'
            ),
            "code": (
                'os.makedirs("outputs/new-data", exist_ok=True)\n'
                'new_image_path = Path("outputs/new-data") / "yellow_square.ppm"\n'
                'im = Image.new("RGB", (32, 32), "white")\n'
                'px = im.load()\n'
                'for y in range(8, 24):\n'
                '    for x in range(8, 24):\n'
                '        px[x, y] = (255, 255, 0)\n'
                'im.save(new_image_path)\n'
                'new_labels = ["yellow square", "blue circle", "red triangle", "abstract geometric shape"]\n'
                'new_data_scores = pipe.zero_shot_classify(new_image_path, new_labels)\n'
                'print("New-data top label:", new_data_scores[0].label, [(s.label, round(s.score, 4)) for s in new_data_scores])'
            ),
        },
        {
            "md": (
                '## 11. Export outputs and provenance\n'
                '\n'
                "Every demonstrated capability gets a stable export: `outputs/classification.json`, `outputs/similarity.csv` (rows = images, columns = expected labels), `outputs/retrieval.json`, `outputs/image_embeddings.npz` and `outputs/text_embeddings.npz` (input identifiers stored beside the vectors), `outputs/new_data_classification.json`, `outputs/metrics.json`, `outputs/provenance.json` (the package's own provenance: effective model, immutable revision, verified checkpoint, runtime packages, CPU device, inference semantics), and `outputs/{stem}_result.json` with the input manifest, the evaluation report, the sample identity and digests, the notebook's source (repository, revision, embedded module digests, generator), the model identifier, the immutable model revision and licence, and the runtime identity. No trained/adapted model artifact is produced because this is pretrained inference. No credentials are recorded."
            ),
            "code": (
                'import csv\n'
                '\n'
                'with open("outputs/classification.json", "w", encoding="utf-8") as handle:\n'
                '    json.dump(classification_rows, handle, indent=2)\n'
                'with open("outputs/retrieval.json", "w", encoding="utf-8") as handle:\n'
                '    json.dump(retrieval_rows, handle, indent=2)\n'
                'with open("outputs/new_data_classification.json", "w", encoding="utf-8") as handle:\n'
                '    json.dump({{"image": new_image_path.name, "scores": [asdict(x) for x in new_data_scores]}}, handle, indent=2)\n'
                'with open("outputs/similarity.csv", "w", newline="", encoding="utf-8") as handle:\n'
                '    writer = csv.writer(handle)\n'
                '    writer.writerow(["image", *expected_labels])\n'
                '    for path, row in zip(images, similarity, strict=True):\n'
                '        writer.writerow([path.name, *map(float, row)])\n'
                'np.savez_compressed("outputs/image_embeddings.npz", image_ids=np.asarray([p.name for p in images]), vectors=image_embeddings)\n'
                'np.savez_compressed("outputs/text_embeddings.npz", text_ids=np.asarray(expected_labels), vectors=text_embeddings)\n'
                'metrics = {{"evidence_type": "synthetic_tutorial_sanity_only", "sample_sha256": sample_sha256, "classification": {{"metric": "top1_accuracy", "value": top1_sanity_accuracy, "fixed_class_baseline": report["baselines"][0]["value"]}}, "retrieval": {{"metric": "recall_at_1", "value": retrieval_recall_at_1}}}}\n'
                'with open("outputs/metrics.json", "w", encoding="utf-8") as handle:\n'
                '    json.dump(metrics, handle, indent=2)\n'
                'if byod_image is not None:\n'
                '    with open("outputs/byod_classification.json", "w", encoding="utf-8") as handle:\n'
                '        json.dump({{"image": byod_image.name, "scores": [asdict(x) for x in byod_scores]}}, handle, indent=2)\n'
                '    np.savez_compressed("outputs/byod_image_embedding.npz", image_ids=np.asarray([byod_image.name]), vectors=byod_embedding)\n'
                'write_provenance("outputs/provenance.json", pipeline=pipe)\n'
                '\n'
                'payload = {{\n'
                "    'classification': classification_rows,\n"
                "    'retrieval': retrieval_rows,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'sample': {{'kind': sample_kind, 'names': [p.name for p in images], 'sha256': sample_sha256, 'expected_labels': expected_labels, 'candidate_labels': candidate_labels}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'numpy': numpy.__version__, 'device': str(pipe.device), 'checkpoint_source': pipe.checkpoint_source, 'manifest_verified': pipe.manifest_verified}},\n"
                '}}\n'
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                '    json.dump(payload, handle, indent=2, ensure_ascii=False)\n'
                'expected_outputs = ["classification.json", "image_embeddings.npz", "text_embeddings.npz", "similarity.csv", "retrieval.json", "metrics.json", "new_data_classification.json", "provenance.json", "{stem}_input_manifest.json", "{stem}_evaluation_report.json", "{stem}_result.json"]\n'
                'missing = [n for n in expected_outputs if not (Path("outputs") / n).is_file()]\n'
                'if missing:\n'
                '    raise RuntimeError(f"Missing expected tutorial outputs: {{missing}}")\n'
                'print(sorted(os.listdir("outputs")))'
            ),
        },
    ],
    "closing": (
        '## Interpretation and limits\n'
        '\n'
        "A successful run proves that the pinned runtime installs, the pinned checkpoint passes integrity checks, all five public operations execute on the validated inputs, and machine-readable outputs/provenance are produced. Successful execution proves that the recorded repository revision's package, carried in this notebook, can do exactly that — without the repository being reachable — and no more.\n"
        '\n'
        'It **does not** prove production fitness, domain accuracy, fairness, robustness, calibration, latency, GPU compatibility, or universal thresholds; the evaluation report says `sample-sanity` on the synthetic set for that reason and `not-measurable` on BYOD. It does **not** establish benchmark superiority, deployment calibration, safety for high-consequence decisions, or production fitness on an unseen domain. SigLIP scores are prompt-dependent and uncalibrated: unexpected scores should first trigger review of prompt/label wording. Integrity/download failures must be fixed rather than bypassing pin/hash checks; for memory pressure, reduce images per call; a GPU host is expected to remain on CPU because the release lock is CPU-only.\n'
        '\n'
        '**Next experiments:** enable `USE_BYOD` with non-sensitive data and your own labels; compare prompt templates for the same labels and watch the scores move; evaluate representative labelled real images with `top1_accuracy` against the fixed-class baseline and labelled real-image retrieval with `recall_at_1`; run a downstream evaluation of the exported embeddings; for deployment, define the production prompt/label policy, inspect failure modes/subgroups, calibrate thresholds or abstention rules, and benchmark the intended serving hardware.\n'
        '\n'
        '## References\n'
        '\n'
        '- Repository README: https://github.com/kurtvalcorza/siglip2-vision-language-pipeline/blob/main/README.md\n'
        '- Repository model card: https://github.com/kurtvalcorza/siglip2-vision-language-pipeline/blob/main/MODEL_CARD.md\n'
        '- Sample dataset card: https://github.com/kurtvalcorza/siglip2-vision-language-pipeline/blob/main/examples/sample-data/DATASET_CARD.md\n'
        '- Upstream model: https://huggingface.co/{MODEL_ID}\n'
        '- Upstream library: https://github.com/huggingface/transformers\n'
        '- SigLIP 2: Multilingual Vision-Language Encoders: https://arxiv.org/abs/2502.14786'
    ),
}
