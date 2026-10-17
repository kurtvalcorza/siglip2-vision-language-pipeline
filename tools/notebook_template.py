"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.5 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded package (seven
modules, carried verbatim in dependency order), and the model pin/stage/verify cells are produced by
the generator from repository sources so they cannot drift from the package.

Generator keys in use: ``modules`` lists every module of ``src/siglip2_pipeline/`` except
``__init__.py``; ``entry_module`` is ``config.py`` (it holds ``MODEL_ID``/``MODEL_REVISION``/
``MODEL_LICENSE`` and the model key under the package's own spelling ``DEFAULT_MODEL_KEY``, mapped by
``identity_names``); ``rewrites`` carries two rules — the fleet ``DEFAULT_WEIGHTS_DIR`` rule and the
``__file__`` use inside ``model.resolve_weights_path`` (a repository-checkout convenience that a
standalone notebook has no checkout for); ``model_load`` lets the pipeline pick CUDA when it is visible;
``notebook_kind`` and ``use_notice`` are the NOTEBOOK_SPEC 2.5 KND1 and DOC17 declarations.

This template configures an E2E zero-shot-classification fine-tuning workflow under the pre-registration in
docs/siglip2-preregistration.md: the pinned google/siglip2-base-patch16-224 snapshot is digest-verified and loaded,
360 CC0 iNaturalist photographs of six bird species and a 96-photograph fresh test are fetched with per-file digests,
validated and split by photograph, the drawn synthetic shapes are classified through the inference contract, the
frozen model is measured on the exposed development test beside two non-neural baselines, a three-configuration grid
selects the fine-tuning configuration on validation, a bounded fine-tuning of the vision tower's last blocks runs with
SigLIP's sigmoid loss, the fresh test is scored only when every pre-registered setting holds (SCORE_ONCE_RULE),
the adapted model re-scores the shapes, and the adapter is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    'package': 'siglip2_pipeline',
    'repo_name': 'siglip2-vision-language-pipeline',
    'stem': 'siglip2_vision_language',
    'notebook_name': 'siglip2_vision_language_colab.ipynb',
    'profile': 'E2E',
    'mode': 'GUIDED',
    'notebook_kind': 'TUTORIAL',
    'use_notice': (
        '> **⚠️ Notice.** For research, training and evaluation only; not validated for production or safety-critical use.\n'
        '> Running this notebook downloads third-party model weights and photographs under their own licences (see the\n'
        '> model card) and uses compute on your own Colab, Kaggle or Jupyter runtime. Data you bring stays in your runtime,\n'
        '> and you are responsible for having the right to use it. Results here are tutorial evidence, not benchmarks.\n'
        '> Everything is provided "as is", without warranty of any kind. To the maximum extent permitted by law, the\n'
        '> maintainers and the DIMER platform accept no liability for its use.'
    ),
    'isolated_runtime': True,
    'infrastructure_labels': True,
    'managed_python': '3.12.12',
    'uv': {
        'version': '0.12.15',
        'url': 'https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl',
        'bytes': 20081404,
        'sha256': 'aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60',
    },
    'lock': 'tutorials/requirements-colab.lock.txt',
    'pipeline_class': 'Siglip2Pipeline',
    'weights_key': 'siglip2-base-patch16-224',
    'modules': [
        'config.py',
        'model.py',
        'metrics.py',
        'samples.py',
        'protocol.py',
        'pipeline.py',
        'provenance.py',
    ],
    'entry_module': 'config.py',
    'identity_names': {
        'MODEL_KEY': 'DEFAULT_MODEL_KEY',
    },
    'rewrites': [
        [
            '^DEFAULT_WEIGHTS_DIR = Path\\(__file__\\)[^\\n]*$',
            'DEFAULT_WEIGHTS_DIR = Path.cwd() / "weights" / DEFAULT_MODEL_KEY  # standalone rewrite (build_notebook.py): working-directory snapshot, no repository checkout',
        ],
        [
            '^    repo_root = Path\\(__file__\\)\\.resolve\\(\\)\\.parents\\[2\\]$',
            '    repo_root = Path.cwd()  # standalone rewrite (build_notebook.py): no repository checkout to resolve',
        ],
    ],
    'model_load': 'Siglip2Pipeline.from_pretrained(weights_dir=WEIGHTS_DIR)',
    'runtime_imports': [
        'torch',
        'transformers',
        'numpy',
    ],
    'title': 'SigLIP 2 Vision-Language Pipeline — DIMER E2E zero-shot classification fine-tuning tutorial (standalone)',
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
    'capability': "zero-shot image classification, image/text embeddings, cosine similarity, text-to-image retrieval and bounded supervised fine-tuning of the vision tower's last blocks on a labelled-photograph dataset, using the pinned `google/siglip2-base-patch16-224` weights",
    'run_all': "Selecting **Run all** in a fresh supported runtime builds an isolated, hash-locked environment with the pinned dependencies (nothing is installed into the notebook kernel, so no restart is needed), stages and digest-verifies the pinned `google/siglip2-base-patch16-224` snapshot (a 1.5 GB `model.safetensors`; no pickle is opened anywhere), fetches the 360 pinned iNaturalist photographs and the 96 photographs of the fresh test from the project's open-data bucket (about 54 MB, each refused on any byte-size or SHA-256 mismatch), cuts the 360 per species into 216 / 48 / 96 training, validation and **exposed development test** photographs, classifies three drawn shapes through the inference contract with an input manifest and a rejection probe, measures the frozen model's zero-shot accuracy, macro F1 and text-to-image mAP on the exposed development test beside the majority-floor and colour-nearest-neighbour baselines, chooses the fine-tuning configuration on **validation** with a pre-registered three-configuration grid, runs a bounded fine-tuning of the vision tower's last blocks and attention-pool head with SigLIP's sigmoid loss and validation-mAP epoch selection, scores the **fresh test** only when every pre-registered setting holds, re-scores the drawn shapes with the adapted model, exports the adapter as safetensors with a manifest, and reloads that artifact into a fresh pipeline to verify embedding parity. The default path needs no repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration edit (NOTEBOOK_SPEC 2.5 §5). A CUDA runtime is used automatically when present; on a T4 the whole path takes about ten minutes, on CPU much longer (see the Prerequisites).",
    'byod': "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 (and `BYOD_PATH` to the zip's path on Kaggle or Jupyter; on Colab an empty path opens the upload dialog) and re-run from that cell to supply one zip holding a `labels.csv` (columns `id`, `file`, `label`) beside the image files — at least eight photographs over at least two labels, the label text being what the prompt names. They pass through the same validation, seeded stratified split, baselines, fine-tuning, held-out evaluation, artifact export and reload-parity cells as the iNaturalist sample. The expected schema and the ceilings are stated in the Prerequisites and in Section 4, and uploaded files stay inside this runtime. BYOD is optional and never part of the default path.",
    'intro': (
        "`google/siglip2-base-patch16-224` is the SigLIP 2 model of Tschannen et al. (2025) [1] — a ViT-B/16 image tower at 224×224 with an attention-pool head and a 12-layer text tower with a 256k multilingual vocabulary; 375,187,970 parameters, published under the **Apache-2.0** licence. A photograph and a prompt are scored by the cosine of their projected embeddings, scaled and shifted by the model's learned `logit_scale` and `logit_bias` and read through a sigmoid [2]: the scores are independent per pair, **not calibrated probabilities**, **prompt/label dependent**, and the model never abstains — the highest-scoring label is returned whatever the image shows.\n"
        '\n'
        "What this notebook adds to inference is **adaptation with labelled photographs**. The dataset is real and where the frozen model has room to improve: 360 CC0-licensed, research-grade iNaturalist photographs of six common North American birds (**CC0 1.0**; four small sparrows, a junco and two finches, 60 per species, one per observer), pinned by photo id, byte size and SHA-256 and fetched from the project's open-data bucket at run time. Zero-shot prompts from the common names separate these species only partly (the Colab CLI T4 run of 9 October 2026 of the previous version, blob `b3ae91fd`, measured accuracy 0.76 frozen on the 96 photographs now called the exposed development test, with the White-throated Sparrow at 0.44 recall), so the question is narrow: does a bounded fine-tuning of the vision tower's last blocks on 216 photographs move zero-shot accuracy and the retrieval view (**text-to-image mAP**), per species, against two **non-neural baselines** (the **majority floor** and a **colour nearest neighbour**)? Nothing here is a quality claim about your photographs: it is one seeded split of one sample.\n"
        '\n'
        "**Which numbers are evidence.** The 96 test photographs of the seeded split were scored in every earlier run, and the fine-tune's trainable scope — the vision tower's last blocks with the text tower frozen — was kept at build time because it gained on an earlier 48-photograph *test* split; no record of that comparison survives. Those 96 are therefore the **exposed development test**: reported for continuity, informing no decision. From this version the configuration is chosen on **validation** by a grid written down before any run (`docs/siglip2-preregistration.md` [7]), and a **fresh test** of 96 new photographs (16 per species, CC0, research grade, created on iNaturalist on or after 2025-03-01, one per observer and none from the 360's 222 observers) is scored **only under the pre-registered settings**; see `docs/release-verification.md` for the recorded evaluation and its reproductions. The frozen text tower is kept by design and not re-tested: the scientific-name prompt lesson in Sections 6 and 8 depends on it.\n"
        '\n'
        "**Pretraining overlap (DAT9).** The photographs are public web images, and SigLIP 2 was trained on WebLI, a web crawl with no published instance manifest, so overlap with its training data **cannot be ruled out**. 255 of the 360 pinned photographs (74 of the 96 exposed) were created on iNaturalist before SigLIP 2's release on 2025-02-20. The fresh test's cutoff of 2025-03-01 narrows this risk but does not remove it: a photograph can be older than its upload, and it can have been posted elsewhere first. The frozen score is therefore not evidence of zero-shot transfer to unseen photographs.\n"
        '\n'
        '**Snapshot note:** the pinned revision ships `model.safetensors` (an 8-file manifest) — no pickle is opened anywhere in this notebook. Section 3 stages and digest-verifies those files before the processor or the model is constructed.'
    ),
    'learning_objectives': "install the pinned runtime; read what the carried package guarantees; stage and digest-verify the immutable upstream snapshot; fetch a digest-pinned labelled photograph set, validate it and split it per species without leakage; classify drawn shapes through the public API and read sigmoid scores correctly (independent, uncalibrated, no abstention); measure the frozen model's zero-shot accuracy, macro F1 and text-to-image mAP beside two non-neural baselines and read the per-species breakdown; choose a fine-tuning configuration on validation with a pre-registered grid; run a bounded fine-tuning with SigLIP's sigmoid loss, explicit hyperparameters and validation-based epoch selection; score a fresh test only under pre-registered settings, with two prompt sets, and tell a pre-registered run from an exploratory one; re-score drawings from a different image family with the adapted model; and export a safetensors adapter that reloads against the pinned base with verified parity.",
    'exclusions': 'object detection, semantic segmentation, OCR, caption generation, calibrated probabilities or universal thresholds, fine-tuning of the text tower, the embeddings, `logit_scale` or `logit_bias`, training on photographs that are not the pinned sample or your own uploads, evaluation on iNaturalist or any benchmark proper (one seeded 360-photograph sample and a 96-photograph fresh test are scored here), and any claim that six bird species stand in for your classes. The repository exposes none of these.',
    'guided': {
        'opening': [
            (
                '**Who this notebook is for.** A learner who knows basic Python and PIL, has used Colab or Jupyter and has met accuracy and a cosine similarity, and wants to see what a zero-shot image–text model does with a prompt, why its sigmoid scores are not probabilities, and what a bounded fine-tuning of its vision tower changes on a real labelled set. The audience is students and practitioners deciding whether SigLIP 2 can be adapted to their own labels; no prior experience with SigLIP, transformers or fine-tuning is assumed — each term is explained where it first matters and again in the **Glossary**. A T4 takes about ten minutes; CPU works but is slow.\n'
                '\n'
                '**Input → Model → Output.**\n'
                '\n'
                '| | |\n'
                '|---|---|\n'
                '| Input | `{{id, image, label}}` records and a prompt template; the default is 360 pinned CC0 iNaturalist photographs of six bird species, split 216 / 48 / 96 per species (training / validation / exposed development test), a 96-photograph fresh test, and three 32 × 32 drawn shapes |\n'
                '| Model | `google/siglip2-base-patch16-224` (375 M parameters): a ViT-B/16 image tower with an attention-pool head and a 12-layer multilingual text tower; the fine-tune trains the last vision blocks the validation grid selects, the post-layernorm and the head (21.3 M parameters with two blocks) |\n'
                '| Output | the validation grid and its selection; zero-shot accuracy, macro F1 and text-to-image mAP of the frozen and the adapted model against a majority floor and a colour nearest neighbour, per species and on two prompt sets, on the exposed development test and — under the pre-registered settings only — the fresh test; `test_look` and `settings_differing`; the shapes re-scored; a safetensors adapter that reloads with verified embedding parity; `result.json` |\n'
                '\n'
                "**How to use this notebook.** Choose any runtime, then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed. Sections 1–3 are **infrastructure** — the isolated environment, the carried modules and the verified snapshot — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited; the defaults are the **pre-registered settings**, and changing any of them makes the run **exploratory** (it is compared on validation and the exposed development test, and the fresh test is not scored). Before each principal result the notebook asks you to **Predict**; after it comes a collapsible **Check your reasoning** whose worked answer names the run its numbers come from — the Colab CLI T4 run of 9 October 2026 of the previous version (blob `b3ae91fd`, revision `20ffef4`), or the CPU build record of 18 October 2026 of this version, run with the fresh test masked. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Budget about ten minutes on a T4: the previous version took 377.5 s in its Colab CLI T4 run of 9 October 2026, downloads included, and the validation grid adds about two minutes.\n"
                '\n'
                '**Roadmap:** 1–3 infrastructure → 4 the photographs, the splits and four refusals *(core concept: the data contract and leakage)* → 5 the inference contract on drawn shapes *(core concept: independent sigmoid scores, no abstention)* → 6 two non-neural baselines and the frozen model on the exposed development test *(evaluation practice: baselines before model numbers)* → 7 a pre-registered grid on validation *(evaluation practice: choose on validation)* → 7b bounded fine-tuning with mAP-based epoch selection *(core concept: what 21 M of 375 M parameters can learn)* → 8 the fresh test, scored only under the pre-registered settings *(evaluation practice)* → 9 the shapes re-scored, export and reload parity *(engineering)* → conclude.'
            ),
        ],
    },
    'prerequisites': [
        '- **Learner:** basic Python, PIL and Colab or Jupyter familiarity; no prior experience with SigLIP or fine-tuning. Sigmoid scores, the three metrics, the two baselines, the vision tower, the loss, epoch selection and reload parity are explained where they are first used and again in the Glossary.',
        '- **Runtime:** a fresh supported **Linux x86_64** runtime (Google Colab, Kaggle or a Linux Jupyter server). Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels into `dimer_isolated_env_<lock digest>/` (set `DIMER_ISOLATED_ENV` to a path to put it elsewhere), so the Python version of the kernel itself does not matter and nothing is installed into it; a Windows or macOS kernel is not supported. The default path runs on CPU (float32) and uses CUDA automatically when available. On a T4 the six-epoch fine-tune took 32.9 s in the Colab CLI T4 run of 9 October 2026 of the previous version; the validation grid runs three such fine-tunes. On CPU the grid took 503 s on the build workstation in the CPU build record of 18 October 2026 (a 2-vCPU hosted runtime will be several times slower). The pinned `torch==2.14.0` install and the 1.5 GB checkpoint are the large downloads of the run; the photographs add about 54 MB.',
        '- **Knowledge:** basic Python and PIL; what a sigmoid score and a cosine similarity are; what accuracy, macro F1 and average precision measure and why none is a human judgement; why a high score is not a correct label.',
        '- **Data contract:** records are `{id, image, label}` — a PIL image or a file decodable by Pillow with sides up to `MAX_IMAGE_SIDE` (4096) px and a label of at most 64 plain characters (the prompt is `DEFAULT_PROMPT_TEMPLATE` with the label, or its display name, filled in). Ids match `[A-Za-z0-9_.:-]{1,64}` and are unique; a dataset needs 8..20,000 records over 2..100 labels; splitting is stratified per label after pixel-digest de-duplication so no photograph lands in two splits. BYOD accepts one zip of images plus a `labels.csv` in that shape.',
        '- **Validation is structural, not semantic:** every image is opened and decoded and every label checked, but nothing checks that a label is right — a mislabelled set is fine-tuned on without complaint.',
        '- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there. The default path uploads nothing.',
        "- **External access (data):** besides the model snapshot, the default path fetches 360 + 96 JPEG/PNG files from `https://inaturalist-open-data.s3.amazonaws.com/photos/<id>/medium.<ext>` (about 54 MB in total), each pinned by byte size and SHA-256 in the carried `samples.py` and refused on any mismatch; every photograph's iNaturalist observation page and observer login are kept in its record. Each photograph carries the CC0 1.0 licence its observer chose; nothing is committed to the repository.",
    ],
    'cells': [
        {
            'md': (
                '## 4. iNaturalist photographs, splits and the fresh test\n'
                '\n'
                "`fetch_corpus` returns the 360 pinned photographs from the cache under `weights/inat-birds/` or the iNaturalist open-data bucket — every cached file is re-hashed and every fetched file refused on any byte-size or SHA-256 mismatch — and `read_corpus` decodes them into `{{id, image, label}}` records with their observation page, observer and species names. `build_sample_dataset` draws a seeded stratified split per species (36 / 8 / 16 → 216 training / 48 validation / 96 **exposed development test**). `fetch_fresh_test` and `read_fresh_test` do the same for the 96 photographs of the **fresh test** (16 per species, created on iNaturalist on or after 2025-03-01, one per observer, none from the 360's observers; `FRESH_TEST_RECORDS`). Loading them is not scoring them: Section 8 scores the fresh test only under the pre-registered settings. When the environment variable `DIMER_FRESH_TEST_MASK=1` is set (the repository's CI and CPU development checks), the fresh test is not even fetched and the run is exploratory. `validate_dataset` then checks every record against the contract, `check_split_disjoint` asserts no photograph (by decoded-pixel digest) is shared by two sets, `observer_overlap` reports how many observers contributed to more than one split (an observation about the draw, not an assertion), and the training split's labels table is written to `outputs/{stem}_train.csv` in the shape BYOD expects.\n"
                '\n'
                'Look for: 360 photographs, the six species with 36 / 8 / 16 each, the fresh test with 16 each, the digests, and four refusal probes — a duplicate id, an image over the side ceiling, a dataset with one label and one too small to split — each rejected before the model does anything.\n'
                '\n'
                '**Predict:** six species, 60 photographs each, one per observer. Will the split be exactly 36 / 8 / 16 per species, will any observer appear in more than one split, and will any of the four probes be accepted?'
            ),
            'code': (
                'import hashlib\n'
                'import json\n'
                'import time\n'
                'from dataclasses import asdict\n'
                '\n'
                'import numpy as np\n'
                'from PIL import Image\n'
                '\n'
                'USE_BYOD = False  # @param {{type:"boolean"}}\n'
                'BYOD_PATH = \'\'  # @param {{type:"string"}}\n'
                'SPLIT_SEED = 42  # @param {{type:"integer"}}\n'
                '\n'
                "os.makedirs('outputs', exist_ok=True)\n"
                '# A grid selection or a Section 7b result from an earlier data load never carries over to this one.\n'
                'grid_selection = adapt_result = None\n'
                "FRESH_TEST_MASKED = os.environ.get(FRESH_TEST_MASK_ENV) == '1'  # CI / CPU development checks (SCORE_ONCE_RULE item 8)\n"
                '\n'
                '\n'
                'class BYODRefusal(ValueError):\n'
                '    """A refused BYOD input: shown as this one message, without a traceback (it names the file and the rule)."""\n'
                '\n'
                '    dimer_refusal = True  # the isolated-environment router prints the message alone\n'
                '\n'
                '    def _render_traceback_(self):  # the same when the cell runs in the kernel itself\n'
                '        return [str(self)]\n'
                '\n'
                '\n'
                'def refuse_byod(reason):\n'
                '    """DAT20: forget the previous data so no later cell reuses it, mark earlier output files as not from this run, and\n'
                '    return one actionable refusal. Earlier files are kept; the marker says they belong to an earlier run."""\n'
                "    for stale in ('splits', 'dataset_manifests', 'train_records', 'val_records', 'dev_records', 'fresh_records', 'classes'):\n"
                '        globals().pop(stale, None)\n'
                "    earlier = sorted(name for name in os.listdir('outputs') if name.endswith('.json') and not name.endswith('run_invalidated.json'))\n"
                "    with open('outputs/{stem}_run_invalidated.json', 'w', encoding='utf-8') as handle:\n"
                "        json.dump({{'reason': str(reason), 'files_from_an_earlier_run': earlier}}, handle, indent=2)\n"
                "    return BYODRefusal(f'BYOD input refused: {{reason}} Fix the upload or BYOD_PATH and run this cell again. Files already in outputs/ are from an earlier run (listed in outputs/{stem}_run_invalidated.json).')\n"
                '\n'
                '\n'
                "if os.path.exists('outputs/{stem}_run_invalidated.json'):\n"
                "    os.remove('outputs/{stem}_run_invalidated.json')\n"
                '\n'
                '\n'
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
                '        uploaded = files.upload() or {{}}\n'
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
                '\n'
                'if USE_BYOD:\n'
                '    # DAT20: a refused input (missing path, wrong file, bad labels.csv, too few images) prints one message; other errors keep their traceback.\n'
                '    try:\n'
                "        byod_path = byod_file(BYOD_PATH, 'zip of images with a labels.csv', ('.zip',))\n"
                '    except (FileNotFoundError, RuntimeError, ValueError) as exc:\n'
                '        raise refuse_byod(exc) from None\n'
                '    try:\n'
                '        records = load_byod_dataset(byod_path)\n'
                '        splits = split_dataset(records, seed=SPLIT_SEED)\n'
                '        for part in splits.values():\n'
                '            validate_dataset(part, min_records=1)\n'
                '    except (KeyError, OSError, ValueError) as exc:\n'
                "        raise refuse_byod(f'{{byod_path.name}}: {{exc}}.') from None\n"
                "    data_source = 'BYOD (' + byod_path.name + ')'\n"
                '    display_names = {{}}\n'
                "    raw_rows = {{'byod': len(records)}}\n"
                "    fresh_records = None  # BYOD runs are exploratory: the sample's fresh test is never read\n"
                'else:\n'
                "    corpus_files = fetch_corpus(cache_dir='weights/inat-birds')\n"
                '    corpus = read_corpus(corpus_files)\n'
                '    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n'
                "    data_source = f'{{CORPUS_NAME}}: {{CORPUS_RELEASE}} ({{CORPUS_LICENSE}})'\n"
                '    display_names = {{key: common for key, (_scientific, common) in SPECIES.items()}}\n'
                "    raw_rows = {{'photographs': len(corpus), 'bytes': sum(len(v) for v in corpus_files.values()), 'observers': len({{r['observer'] for r in corpus}})}}\n"
                '    if FRESH_TEST_MASKED:\n'
                '        fresh_records = None\n'
                "        print({{'fresh_test': 'masked (' + FRESH_TEST_MASK_ENV + '=1): not fetched; this run is exploratory'}})\n"
                '    else:\n'
                "        fresh_files = fetch_fresh_test(cache_dir='weights/inat-birds/fresh')\n"
                '        fresh_records = read_fresh_test(fresh_files)\n'
                "        raw_rows['fresh_test'] = {{'photographs': len(fresh_records), 'bytes': sum(len(v) for v in fresh_files.values()), 'observers': len({{r['observer'] for r in fresh_records}}), 'observers_shared_with_the_360': len({{r['observer'] for r in fresh_records}} & {{r['observer'] for r in corpus}}), 'manifest_sha256': fresh_manifest_digest()[:16] + '...'}}\n"
                "dataset_manifests = {{name: validate_dataset(part, min_records=MIN_RECORDS if name == 'train' else 1) for name, part in splits.items()}}\n"
                "splits = {{name: manifest['records'] for name, manifest in dataset_manifests.items()}}\n"
                'fresh_manifest = validate_dataset(fresh_records) if fresh_records else None\n'
                "disjoint = check_split_disjoint({{**splits, **({{'fresh_test': fresh_manifest['records']}} if fresh_manifest else {{}})}})\n"
                "train_records, val_records, dev_records = splits['train'], splits['validation'], splits['test']\n"
                "fresh_records = fresh_manifest['records'] if fresh_manifest else None\n"
                "dev_label = 'exposed development test' if not USE_BYOD else 'your own test split (BYOD, exploratory)'\n"
                'classes = class_names(train_records)\n'
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'raw_rows': raw_rows, 'sets': disjoint, 'test_role': dev_label, 'observer_overlap': observer_overlap(splits), 'classes': classes}})\n"
                "for name, manifest in {{**dataset_manifests, **({{'fresh_test': fresh_manifest}} if fresh_manifest else {{}})}}.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'label_counts': manifest['label_counts'], 'image_side': manifest['image_side'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                'example = train_records[0]\n'
                "print({{'example': {{'id': example['id'], 'label': example['label'], 'display_name': display_names.get(example['label'], example['label']), 'size': list(example['image'].size), 'observation': example.get('inat_observation_url')}}}})\n"
                '\n'
                'probes = {{\n'
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in train_records[:8]],\n"
                "    'image over the side ceiling': [{{**train_records[0], 'image': Image.new('RGB', (MAX_IMAGE_SIDE + 1, 8))}}, *train_records[1:8]],\n"
                "    'one label only': [{{**r, 'label': 'bird'}} for r in train_records[:8]],\n"
                "    'too small': train_records[:3],\n"
                '}}\n'
                'for name, probe in probes.items():\n'
                '    try:\n'
                '        validate_dataset(probe)\n'
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                '    except (TypeError, ValueError) as exc:\n'
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            'md': '<details><summary>Check your reasoning</summary>Yes, no and no. The split is stratified per species by seed, so 36 / 8 / 16 each (216 / 48 / 96) by construction; one photograph per observer per species means an observer can still appear in two splits through two species, and `observer_overlap` prints how many did — an observation about the draw, not an assertion. The fresh test shares no observer with the 360 by construction (`observers_shared_with_the_360` is 0). All four probes are refused before any model work, each naming its rule: a duplicate id, a side over `MAX_IMAGE_SIDE`, a single label, a set too small to split.</details>',
        },
        {
            'md': (
                '## 5. Classify through the inference contract\n'
                '\n'
                "The inference contract is exercised as the inference-only tutorial exercised it: three 32×32 synthetic shapes — a red square, a green circle and a blue triangle — rendered in code as ASCII PPM files exactly as the repository's `examples/sample-data/generate_samples.py` renders them and digest-asserted against `SHA256SUMS`, a different image family from the photographs, and images the model will be asked to classify again after adaptation. `validate_inputs` applies exactly the checks the public operations apply and returns an input manifest; a remote URL is validated too and its rejection recorded as a finding. `zero_shot_classify` returns one sigmoid score per candidate label, **ordered by descending score**; `retrieve` ranks the gallery by cosine to a query. The per-grid `evaluation_report` on three drawn shapes is `sample-sanity` — plumbing evidence, not a measurement; whether the classifier is *right* is what Section 6 measures on 96 photographs.\n"
                '\n'
                '**Predict:** three flat drawn shapes and four candidate labels, one of them a distractor. Will every shape get its own label on top, and will the four sigmoid scores of a shape sum to one?'
            ),
            'code': (
                'SAMPLE_DIGESTS = {{  # examples/sample-data/SHA256SUMS\n'
                "    'red_square.ppm': 'b38ff0c9131677ed6cf09832eff40a22841e1a4725d426fad3b1bf6a1dbdb096',\n"
                "    'green_circle.ppm': '2e3e657686a0f6a6df3f3621d21a410d20d9a47b109f06f9405faa4f747a663f',\n"
                "    'blue_triangle.ppm': 'f0f4c38c7af92b3a6b1d55a25272156edd87b41e029e116cfa059003dae029a3',\n"
                '}}\n'
                'WIDTH, HEIGHT, BACKGROUND = 32, 32, (245, 245, 245)\n'
                '\n'
                '\n'
                'def shape_mask(shape, x, y):\n'
                "    if shape == 'square':\n"
                '        return 8 <= x < 24 and 8 <= y < 24\n'
                "    if shape == 'circle':\n"
                '        return (x - 16) ** 2 + (y - 16) ** 2 <= 9**2\n'
                '    if not 7 <= y < 26:\n'
                '        return False\n'
                '    half = (y - 7) // 2\n'
                '    return 16 - half <= x <= 16 + half\n'
                '\n'
                '\n'
                'def render_ppm(foreground, shape):\n'
                "    # The repository's generate_samples.py rendering: ASCII P3, 24 values per line.\n"
                "    lines = ['P3', f'{{WIDTH}} {{HEIGHT}}', '255']\n"
                '    for y in range(HEIGHT):\n'
                '        row = []\n'
                '        for x in range(WIDTH):\n'
                '            pixel = foreground if shape_mask(shape, x, y) else BACKGROUND\n'
                '            row.extend(str(v) for v in pixel)\n'
                '        for start in range(0, len(row), 24):\n'
                "            lines.append(' '.join(row[start : start + 24]))\n"
                "    return '\\n'.join(lines) + '\\n'\n"
                '\n'
                '\n'
                "Path('outputs/sample-data').mkdir(parents=True, exist_ok=True)\n"
                "specs = {{'red_square.ppm': ((220, 40, 40), 'square'), 'green_circle.ppm': ((40, 170, 75), 'circle'), 'blue_triangle.ppm': ((40, 90, 220), 'triangle')}}\n"
                'shape_images = []\n'
                'for name, (foreground, shape) in specs.items():\n'
                "    path = Path('outputs/sample-data') / name\n"
                "    path.write_bytes(render_ppm(foreground, shape).encode('ascii'))\n"
                '    digest = hashlib.sha256(path.read_bytes()).hexdigest()\n'
                '    if digest != SAMPLE_DIGESTS[name]:\n'
                "        raise ValueError(f'Synthetic sample digest mismatch for {{name}}: {{digest}} != {{SAMPLE_DIGESTS[name]}}')\n"
                '    shape_images.append(path)\n'
                "shape_labels = ['red square', 'green circle', 'blue triangle']\n"
                "candidate_labels = [*shape_labels, 'abstract geometric shape']\n"
                'shape_sha256 = {{p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in shape_images}}\n'
                "print({{'ceilings': {{'TEXT_MAX_LENGTH': TEXT_MAX_LENGTH, 'DEFAULT_PROMPT_TEMPLATE': DEFAULT_PROMPT_TEMPLATE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MIN_RECORDS': MIN_RECORDS, 'MAX_RECORDS': MAX_RECORDS, 'MIN_CLASSES': MIN_CLASSES, 'image_contract': '224x224 RGB after processor resize', 'device': str(pipe.device)}}}})\n"
                'input_manifest = validate_inputs(shape_images, candidate_labels, top_k=len(shape_images), names=[p.name for p in shape_images])\n'
                'try:\n'
                "    validate_inputs('https://example.invalid/not-allowed.png', candidate_labels)\n"
                'except ValueError as exc:\n'
                "    input_manifest['findings'].append({{'input': 'remote-url-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                '    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n'
                "print({{'shapes': [p.name for p in shape_images], 'sha256': {{k: v[:16] + '...' for k, v in shape_sha256.items()}}, 'manifest_verdict': input_manifest['verdict'], 'findings': len(input_manifest['findings'])}})\n"
                'classifications, retrievals, classification_rows = [], [], []\n'
                't0 = time.perf_counter()\n'
                'for image_path, expected in zip(shape_images, shape_labels, strict=True):\n'
                '    scores = pipe.zero_shot_classify(image_path, candidate_labels)\n'
                '    classifications.append(scores)\n'
                "    classification_rows.append({{'image': image_path.name, 'expected_label': expected, 'predicted_label': scores[0].label, 'scores': [asdict(x) for x in scores]}})\n"
                "    print(image_path.name, '->', [(s.label, round(s.score, 4)) for s in scores])\n"
                'for query in shape_labels:\n'
                '    retrievals.append(pipe.retrieve(query, shape_images, top_k=len(shape_images)))\n'
                'shape_embeddings = pipe.embed_image(shape_images)\n'
                'checks = {{\n'
                "    'one_ranking_per_image': len(classifications) == len(shape_images),\n"
                "    'scores_in_unit_interval': all(0.0 <= s.score <= 1.0 for ranking in classifications for s in ranking),\n"
                "    'rankings_descending': all(ranking[i].score >= ranking[i + 1].score for ranking in classifications for i in range(len(ranking) - 1)),\n"
                "    'embeddings_unit_norm': bool(np.allclose(np.linalg.norm(shape_embeddings, axis=1), 1.0, atol=1e-4)),\n"
                '}}\n'
                'if not all(checks.values()):\n'
                "    raise RuntimeError(f'inference output failed a sanity check: {{checks}}')\n"
                "result = {{'classifications': classifications, 'retrievals': retrievals, 'gallery_ids': [p.name for p in shape_images], 'embedding_shapes': {{'image': list(shape_embeddings.shape)}}}}\n"
                "targets = {{'labels': shape_labels, 'retrieval_indices': list(range(len(shape_images)))}}\n"
                "frozen_scene = evaluation_report(result, targets, sample_kind='synthetic')\n"
                "print({{'checks': checks, 'seconds': round(time.perf_counter() - t0, 2), 'frozen_scene': {{m['id']: round(m['value'], 3) for m in frozen_scene['metrics']}}, 'verdict': frozen_scene['verdict']}})"
            ),
        },
        {
            'md': '<details><summary>Check your reasoning</summary>Each shape got its own label on top — top-1 accuracy 1.0 and recall@1 1.0 in the CPU build record of 18 October 2026 — and no, the scores do not sum to one: each image–prompt pair goes through its own sigmoid, so a distractor like `abstract geometric shape` can score high beside the right label. The verdict is `sample-sanity`: three drawings are plumbing evidence, not a measurement.</details>',
        },
        {
            'md': (
                '## 6. Baselines and the frozen model on the exposed development test\n'
                '\n'
                "Three systems frame the adaptation, each read three ways. The **majority floor** answers every photograph with the most frequent training label (accuracy 1/6 on a balanced split, chance-level macro F1). The **colour nearest neighbour** answers with the label of the training photograph whose 3×3 mean-colour grid is closest — a classifier that knows the image through 27 numbers. The **frozen model** is scored by `pipe.evaluate`: one prompt per species (`DEFAULT_PROMPT_TEMPLATE` with the common name), the model's own sigmoid-scaled logits as the score grid, **accuracy** and **macro F1** of the top prompt, the per-species recall, and **text-to-image mAP** — each prompt as a query ranking all 96 photographs, the average precision of its own species, averaged over the six — the retrieval view of the same scores and the smoother of the three on a small set. A second prompt set built from the scientific names is scored too, to show how much the number is the prompt's.\n"
                '\n'
                "These numbers are read on the **exposed development test** — the 96 photographs every earlier run scored — so they describe the frozen model for continuity and inform no decision; the fresh test is scored in Section 8, and only under the pre-registered settings. Expect the frozen model far above both baselines and read the per-species breakdown: the Colab CLI T4 run of 9 October 2026 (previous blob `b3ae91fd`; the frozen numbers do not depend on the fine-tune) measured accuracy 0.76 / macro F1 0.758 / mAP 0.723 frozen, with the White-throated Sparrow at 0.44 recall and the Dark-eyed Junco at 0.94. Because the photographs are public and SigLIP 2's training data is not published, the frozen score may partly be recall of photographs it saw in pretraining (DAT9; see the introduction). The cell records whether the frozen model is above both baselines as a verdict instead of stopping, so a BYOD set that behaves differently still reaches the export.\n"
                '\n'
                '**Predict:** rank the majority floor, the colour nearest neighbour and the frozen model by accuracy, then guess the frozen mAP. Will the scientific-name prompts score higher or lower than the common names?'
            ),
            'code': (
                '# Sections 6, 7 and 7b describe the frozen model and what one adaptation adds to it. `pipe.adapt` trains the vision\n'
                '# tower in place, so when the pipeline carries an adaptation they put the pinned base tensors back first\n'
                '# (`restore_base`, exact and in memory); a re-run never scores or trains an earlier adaptation as if it were frozen.\n'
                'def frozen_pipeline():\n'
                '    """Make `pipe` the pretrained model again when it carries an adaptation."""\n'
                '    if pipe.adapter is not None:\n'
                '        pipe.restore_base()\n'
                "        print({{'frozen_model': 'base tensors restored', 'adapted': pipe.adapter is not None}})\n"
                '\n'
                '\n'
                'frozen_pipeline()\n'
                'baseline_majority = majority_baseline(train_records, dev_records, classes)\n'
                'baseline_neighbour = colour_neighbour_baseline(train_records, dev_records, classes)\n'
                "METRICS = ('accuracy', 'macro_f1', 't2i_map')\n"
                "print({{'set': dev_label, 'majority_baseline': {{k: round(baseline_majority[k], 3) for k in METRICS}}, 'n': baseline_majority['n'], 'note': baseline_majority['baseline']}})\n"
                "print({{'colour_neighbour_baseline': {{k: round(baseline_neighbour[k], 3) for k in METRICS}}, 'note': baseline_neighbour['baseline']}})\n"
                't0 = time.perf_counter()\n'
                'frozen_dev = pipe.evaluate(dev_records, classes=classes, class_names_map=display_names)\n'
                "print({{'frozen_model': {{k: round(frozen_dev[k], 3) for k in METRICS}}, 'n': frozen_dev['n'], 'verdict': frozen_dev['verdict'], 'prompt_template': frozen_dev['prompt_template'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'definitions': frozen_dev['definitions']}})\n"
                "frozen_fields = {{c: {{'n': v['n'], 'recall': round(v['recall'], 2), 'ap': round(v['ap'], 2)}} for c, v in frozen_dev['per_class'].items()}}\n"
                "print({{'by_species_frozen': frozen_fields}})\n"
                'scientific_names = {{key: scientific for key, (scientific, _common) in SPECIES.items()}} if not USE_BYOD else {{}}\n'
                "frozen_dev_scientific = pipe.evaluate(dev_records, classes=classes, class_names_map=scientific_names, prompt_template='This is a photo of {{label}}.')\n"
                "print({{'frozen_model_scientific_name_prompts': {{k: round(frozen_dev_scientific[k], 3) for k in METRICS}}}})\n"
                '# A recorded verdict, not an assert: a BYOD set where the prompts do not beat the baselines still reaches the export.\n'
                "frozen_verdict = 'above the baselines' if frozen_dev['t2i_map'] > baseline_majority['t2i_map'] and frozen_dev['accuracy'] > baseline_neighbour['accuracy'] else 'not above the baselines'\n"
                "print({{'frozen_vs_baselines': frozen_verdict, 'frozen': {{'t2i_map': round(frozen_dev['t2i_map'], 3), 'accuracy': round(frozen_dev['accuracy'], 3)}}, 'majority_t2i_map': round(baseline_majority['t2i_map'], 3), 'neighbour_accuracy': round(baseline_neighbour['accuracy'], 3)}})"
            ),
        },
        {
            'md': "<details><summary>Check your reasoning</summary>In the Colab CLI T4 run of 9 October 2026 (previous blob `b3ae91fd`) on these 96 photographs: majority 0.167, colour neighbour 0.26, frozen model 0.76 accuracy (macro F1 0.758, mAP 0.723), so the verdict is `above the baselines`. The scientific-name prompts scored lower — accuracy 0.479, macro F1 0.459, mAP 0.467. A plausible reading, not a measured one (SigLIP 2's training captions are not published), is that common names are far more frequent than Latin binomials in web captions: the classifier is the prompt as much as the tower. Per species, the White-throated Sparrow had recall 0.44 and the Dark-eyed Junco 0.94.</details>",
        },
        {
            'md': (
                '## 7. A pre-registered configuration grid on validation\n'
                '\n'
                "Before the tutorial adapts the model it **chooses the configuration on validation**, by a rule written down before any run (`docs/siglip2-preregistration.md`). Three configurations differ in one knob each — **C0** the last two vision blocks at learning rate 5e-5 (the default), **C1** the last four blocks at 5e-5, **C2** the last two blocks at 1e-5 — and everything else is fixed: six epochs, batch 16, seed 0, the common-name prompts, AdamW, clipping at 1.0, the text tower frozen. Each is scored by its best-epoch **validation** text-to-image mAP on the 48 validation photographs. A challenger replaces C0 only if it beats C0 by at least **0.02** (a practical tolerance, not a significance test — on 48 photographs a difference that small is noise); C1, the larger scope, must also beat the best other configuration by 0.02; and the selected configuration must beat the frozen model's validation mAP by 0.02, or the **frozen model** is kept. The grid never reads either test set; it writes `outputs/{stem}_grid.json`. On a T4 it takes about two minutes; turning `RUN_GRID` off makes the run exploratory (Section 8 then does not score the fresh test).\n"
                '\n'
                '**What was not re-tested, and why.** At build time the trainable scope was compared with adapting the text tower, or both towers, on an earlier 48-photograph *test* split, and the vision-only scope was kept because it gained there; no record of that comparison survives. The grid varies only the vision side. The frozen text tower is kept **by design**: the scientific-name prompts in Sections 6 and 8 are a transfer check only because the text tower that embeds them is never trained.\n'
                '\n'
                '*Evaluation practice.* **Predict:** which configuration will win, and by more or less than 0.02?'
            ),
            'code': (
                'RUN_GRID = True  # @param {{type:"boolean"}}\n'
                "GRID_EPOCHS, GRID_BATCH_SIZE, GRID_SEED = PREREGISTERED_SETTINGS['epochs'], PREREGISTERED_SETTINGS['batch_size'], PREREGISTERED_SETTINGS['adapt_seed']\n"
                'grid_results, grid_selection, grid_seconds = {{}}, None, 0.0\n'
                'if RUN_GRID:\n'
                '    frozen_pipeline()\n'
                '    t_grid = time.perf_counter()\n'
                '    for grid_name, grid_config in GRID_CONFIGS.items():\n'
                '        t0 = time.perf_counter()\n'
                "        run = pipe.adapt(train_records, val_records, epochs=GRID_EPOCHS, lr=grid_config['lr'], batch_size=GRID_BATCH_SIZE, trainable_vision_layers=grid_config['trainable_vision_layers'], class_names_map=display_names, seed=GRID_SEED)\n"
                "        by_epoch = [round(float(h['val']['t2i_map']), 3) for h in run['history']]\n"
                "        grid_results[grid_name] = {{'config': grid_config, 'best_epoch': run['best_epoch'], 'best_val_t2i_map': round(float(run['history'][run['best_epoch']]['val']['t2i_map']), 4), 'val_t2i_map_by_epoch': by_epoch, 'seconds': round(time.perf_counter() - t0, 1)}}\n"
                '        print({{grid_name: grid_results[grid_name]}})\n'
                '        pipe.restore_base()  # every configuration starts from the pinned base\n'
                "    frozen_val_map = float(run['history'][0]['val']['t2i_map'])  # epoch 0 = the frozen model, the same in every config\n"
                "    grid_fingerprint = run_fingerprint(data_source='pinned sample' if not USE_BYOD else 'BYOD', pin_manifest_sha256=pin_manifest_digest() if not USE_BYOD else None, split_seed=SPLIT_SEED, validation_digest=dataset_manifests['validation']['digest'], epochs=GRID_EPOCHS, batch_size=GRID_BATCH_SIZE, seed=GRID_SEED, prompt_template=DEFAULT_PROMPT_TEMPLATE)\n"
                "    grid_selection = select_config({{n: r['best_val_t2i_map'] for n, r in grid_results.items()}}, frozen_val_map, fingerprint=grid_fingerprint)\n"
                '    grid_seconds = round(time.perf_counter() - t_grid, 1)\n'
                "    print({{'grid_selection': {{k: grid_selection[k] for k in ('selected', 'config', 'adaptation_kept', 'reason', 'scores', 'frozen_val_map', 'margin')}}, 'seconds': grid_seconds}})\n"
                "    with open('outputs/{stem}_grid.json', 'w', encoding='utf-8') as handle:\n"
                "        json.dump({{'purpose': 'selection only (validation); no test set is read', 'preregistration': PREREGISTRATION, 'data_source': data_source, 'n_validation': len(val_records), 'fixed': {{'epochs': GRID_EPOCHS, 'batch_size': GRID_BATCH_SIZE, 'seed': GRID_SEED, 'prompt_template': DEFAULT_PROMPT_TEMPLATE}}, 'results': grid_results, 'selection': grid_selection, 'seconds': grid_seconds}}, handle, indent=2)\n"
                'else:\n'
                "    print('RUN_GRID is off: no configuration is selected in this run, so the run is exploratory and Section 8 will not score the fresh test.')"
            ),
        },
        {
            'md': (
                "**What to notice:** each configuration's validation mAP by epoch, the three `best_val_t2i_map` values against the 0.02 margin, `frozen_val_map`, and `reason` in `grid_selection`.\n"
                '\n'
                '<details><summary>Check your reasoning</summary>In the CPU build record of 18 October 2026 (this version on the build workstation, fresh test masked; pending the hosted run) the best-epoch validation mAP was C0 0.876 (epoch 4), C1 0.876 (epoch 5) and C2 0.867 (epoch 4) against a frozen 0.680, and the rule selected **C0** (no challenger beat C0 by 0.02 — C1 was 0.0005 higher and C2 0.009 lower — and C0 beat the frozen model by far more than 0.02). A difference under 0.02 on 48 photographs is treated as no difference, which is why a challenger has to clear the margin rather than merely come out ahead.</details>'
            ),
        },
        {
            'md': (
                "## 7b. Bounded fine-tuning of the vision tower's last blocks\n"
                '\n'
                "`pipe.adapt` trains only the last `TRAINABLE_VISION_LAYERS` blocks of the vision tower, its post-layernorm and its attention-pool head — with two blocks, 21,264,384 of 375,187,970 parameters; the text tower, the embeddings, `logit_scale` and `logit_bias` stay frozen. The six class prompts are embedded once by the frozen text tower; every batch of photographs is run through the vision tower, scored against the prompts with the model's own sigmoid-scaled logits, and trained with SigLIP's pairwise sigmoid loss (+1 for the gold species, −1 for the other five). AdamW at a fixed learning rate, gradient clipping at 1.0, seeded shuffling and no scheduler. Epoch 0 records the frozen model's validation metrics; every epoch is scored on the 48 validation photographs, and the epoch with the highest validation text-to-image mAP is kept — accuracy on 48 photographs moves in steps of 2 %, which is why the retrieval view selects. With `CONFIG = 'grid selection'` (the default) the configuration is the one Section 7 selected; any other choice, and any change to `EPOCHS` or `BATCH_SIZE`, makes the run **exploratory** — Section 8 then reads validation and the exposed development test, not the fresh test.\n"
                '\n'
                "Watch the training loss fall within six epochs while the validation mAP peaks early: 216 photographs are few, the last blocks memorise them, and the selector's job is to stop before that hurts. The reported validation numbers are biased upward, because the configuration and the epoch were both chosen on them. The cell first makes sure `pipe` is the frozen model, so a second run after changing a setting trains the pretrained tower again, not the previous adaptation.\n"
                '\n'
                '**Predict:** six epochs over 216 photographs. Will the validation mAP rise at every epoch, and will the kept epoch be the last one?'
            ),
            'code': (
                'CONFIG = \'grid selection\'  # @param ["grid selection", "C0", "C1", "C2", "manual"]\n'
                'EPOCHS = 6  # @param {{type:"integer"}}\n'
                'BATCH_SIZE = 16  # @param {{type:"integer"}}\n'
                "# Used only when CONFIG is 'manual' (an exploratory run):\n"
                'LEARNING_RATE = 5e-5  # @param {{type:"number"}}\n'
                'TRAINABLE_VISION_LAYERS = 2  # @param {{type:"integer"}}\n'
                'ADAPT_SEED = 0  # pre-registered; not a form field\n'
                "if CONFIG == 'grid selection':\n"
                "    chosen = (grid_selection or {{}}).get('config') or GRID_CONFIGS['C0']\n"
                'elif CONFIG in GRID_CONFIGS:\n'
                '    chosen = GRID_CONFIGS[CONFIG]\n'
                'else:\n'
                "    chosen = {{'trainable_vision_layers': TRAINABLE_VISION_LAYERS, 'lr': LEARNING_RATE}}\n"
                "lr_used, layers_used = chosen['lr'], chosen['trainable_vision_layers']\n"
                "print({{'config': CONFIG, 'lr': lr_used, 'trainable_vision_layers': layers_used, 'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'seed': ADAPT_SEED}})\n"
                '\n'
                '\n'
                'def report(entry):\n'
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row.update({{'val_' + k: round(entry['val'][k], 3) for k in METRICS}})\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                '    print(row)\n'
                '\n'
                '\n'
                'frozen_pipeline()  # every run of this cell trains the frozen model, never an earlier adaptation\n'
                't0 = time.perf_counter()\n'
                'adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=lr_used, batch_size=BATCH_SIZE, trainable_vision_layers=layers_used, class_names_map=display_names, seed=ADAPT_SEED, progress=report)\n'
                'adapt_seconds = round(time.perf_counter() - t0, 1)\n'
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'classes': adapt_result['classes'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            'md': '<details><summary>Check your reasoning</summary>Neither, as a rule. In the CPU build record of 18 October 2026 the selected configuration logged validation mAP 0.680 → 0.874 → 0.854 → 0.863 → 0.876 → 0.872 → 0.854 and kept epoch 4; the Colab CLI T4 run of 9 October 2026 of the previous version (C0) logged 0.680 → 0.874 → 0.854 → 0.863 → 0.876 → 0.872 → 0.854 and kept epoch 4. The retrieval view peaks early and wobbles while the training loss keeps falling, because 216 photographs are few and the last blocks memorise them. Read the `val_t2i_map` column to see which epoch is kept; differences of about 0.01 between epochs are noise on 48 photographs.</details>',
        },
        {
            'md': (
                '## 8. The fresh test, scored only under the pre-registered settings\n'
                '\n'
                "The cell first decides whether this run is **pre-registered**: every setting the pre-registration fixes (the pinned sample and its pin manifest, the fresh test's pin manifest and that it is not masked, `SPLIT_SEED` and the 36 / 8 / 16 split, six epochs, batch 16, seed 0, the common-name prompts, a grid run) and the configuration the grid selected. `settings_differing` lists any that differ; `test_look` is then one of `pre-registered evaluation (this runtime)`, `reproduction (this runtime)` (an earlier report in this runtime already holds a pre-registered scoring; it is written to `*_reproduction_<n>.json`, never over the first) or `not scored (exploratory)`. A runtime cannot see other runtimes: the evaluation of record is the first hosted pre-registered run, named in `docs/release-verification.md`, and a run of yours with identical settings is a reproduction of it. The check also requires that the grid ran on this run's data and settings and that the model in memory is the one Section 7b trained (re-running Section 6 or 7 after 7b puts the pinned base back, and the run becomes exploratory).\n"
                '\n'
                '- **Pre-registered:** the majority floor, the colour neighbour, the frozen model and — if the grid kept adaptation — the adapted model are scored on the **96-photograph fresh test**, with both prompt sets. Its photographs were drawn after the configuration rule was fixed and were never used for training, configuration or epoch selection.\n'
                '- **Exploratory** (any form change, BYOD, `RUN_GRID` off, a smaller batch for memory, the optional experiments, or `DIMER_FRESH_TEST_MASK=1`): the fresh test is **not** scored. The cell reads validation and the development set (the exposed development test, or your own test split on BYOD) and writes `outputs/{stem}_evaluation_report_exploratory_<n>.json`, never the pre-registered file.\n'
                '\n'
                'Either way the adapted model is also read on the exposed development test, for continuity with the earlier records (it informs no decision). Read **text-to-image mAP** first (the metric the configuration and the epoch were selected on), then accuracy and macro F1, then the per-species rows: `ap_fell` and `recall_fell` name the species whose AP or recall went down, so read those rather than trusting a mean. The scientific-name prompts are the transfer check: the vision tower was adapted, not the prompts, so a gain that carries to a prompt set it never saw is the more general one. Outcomes are recorded **verdicts** (`improved`, `no change`, `worse`), never asserts, so a run that does not improve still exports and reloads. Ninety-six photographs from one seeded run give **no dispersion estimate**; a gain on six birds says nothing about your classes until you measure them.\n'
                '\n'
                '*Evaluation practice.* **Predict:** write down a direction and a size for the mAP delta and the accuracy delta. Will the scientific-name prompts, which the tower never saw during training, move too — and which species will gain most?'
            ),
            'code': (
                'import shutil\n'
                '\n'
                "run_settings = {{'data_source': 'pinned sample' if not USE_BYOD else 'BYOD', 'pin_manifest_sha256': pin_manifest_digest() if not USE_BYOD else None, 'fresh_manifest_sha256': fresh_manifest_digest() if fresh_records else None, 'fresh_test_masked': FRESH_TEST_MASKED, 'split_seed': SPLIT_SEED, 'split_sizes': dict(SAMPLE_SPLIT) if not USE_BYOD else 'BYOD fractions', 'adapt_seed': (adapt_result or {{}}).get('seed'), 'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'prompt_template': DEFAULT_PROMPT_TEMPLATE, 'class_names': 'common names' if not USE_BYOD else 'BYOD labels', 'grid_run': grid_selection is not None, 'trainable_vision_layers': layers_used, 'lr': lr_used}}\n"
                "run_fingerprint_now = run_fingerprint(data_source=run_settings['data_source'], pin_manifest_sha256=run_settings['pin_manifest_sha256'], split_seed=SPLIT_SEED, validation_digest=dataset_manifests['validation']['digest'], epochs=PREREGISTERED_SETTINGS['epochs'], batch_size=PREREGISTERED_SETTINGS['batch_size'], seed=PREREGISTERED_SETTINGS['adapt_seed'], prompt_template=DEFAULT_PROMPT_TEMPLATE)\n"
                '# The fresh test needs the model Section 7b trained, not a base restored by a later re-run of Section 6 or 7.\n'
                'model_state = adapter_mismatch(pipe.adapter, adapt_result)\n'
                'differing = settings_differing(run_settings, grid_selection, run_fingerprint_now, model_state)\n'
                'preregistered = not differing  # SCORE_ONCE_RULE: computed from ALL pre-registered settings, not one field\n'
                "report_path = Path('outputs/{stem}_evaluation_report.json')\n"
                "prior_report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else None\n"
                'test_look = look_at_test(differing, prior_report)\n'
                "adaptation_kept = bool(grid_selection['adaptation_kept']) if grid_selection else True\n"
                "print({{'preregistered': preregistered, 'test_look': test_look, 'settings_differing': differing, 'adaptation_kept': adaptation_kept}})\n"
                'if not preregistered:\n'
                "    print('Exploratory run: the fresh test is NOT scored. Compare on validation; the development view below is the ' + dev_label + '.')\n"
                '\n'
                '\n'
                'def fields(result):\n'
                "    return {{c: {{'n': v['n'], 'recall': round(float(v['recall']), 2), 'ap': round(float(v['ap']), 2)}} for c, v in result['per_class'].items()}}\n"
                '\n'
                '\n'
                'def compare(majority, neighbour, frozen, adapted, frozen_sci, adapted_sci):\n'
                "    systems = {{'majority': majority, 'neighbour': neighbour, 'frozen': frozen, **({{'adapted': adapted}} if adapted else {{}})}}\n"
                '    table = {{metric: {{name: round(float(r[metric]), 3) for name, r in systems.items()}} for metric in METRICS}}\n'
                "    table['scientific_name_prompts'] = {{metric: {{'frozen': round(float(frozen_sci[metric]), 3), **({{'adapted': round(float(adapted_sci[metric]), 3)}} if adapted_sci else {{}})}} for metric in METRICS}}\n"
                '    before = fields(frozen)\n'
                "    table['verdict'] = {{'frozen_vs_baselines': 'above the baselines' if frozen['t2i_map'] > majority['t2i_map'] and frozen['accuracy'] > neighbour['accuracy'] else 'not above the baselines'}}\n"
                '    if adapted:\n'
                '        after = fields(adapted)\n'
                "        table['delta_vs_frozen'] = {{metric: round(float(adapted[metric]) - float(frozen[metric]), 3) for metric in METRICS}}\n"
                "        for metric in ('t2i_map', 'accuracy'):\n"
                '            delta = float(adapted[metric]) - float(frozen[metric])\n'
                "            table['verdict']['adapted_vs_frozen_' + metric] = 'improved' if delta > 0 else ('no change' if delta == 0 else 'worse')\n"
                "        table['ap_fell'] = sorted(c for c in classes if after[c]['ap'] < before[c]['ap'])\n"
                "        table['recall_fell'] = sorted(c for c in classes if after[c]['recall'] < before[c]['recall'])\n"
                "        table['by_species'] = {{c: {{'n': before[c]['n'], 'frozen_recall': before[c]['recall'], 'adapted_recall': after[c]['recall'], 'frozen_ap': before[c]['ap'], 'adapted_ap': after[c]['ap']}} for c in classes}}\n"
                '    else:\n'
                "        table['verdict']['adapted_vs_frozen_t2i_map'] = 'adaptation not kept (did not beat frozen on validation by the margin)'\n"
                "        table['by_species'] = before\n"
                '    return table\n'
                '\n'
                '\n'
                'adapted_val = pipe.evaluate(val_records, classes=classes, class_names_map=display_names)\n'
                'adapted_dev = pipe.evaluate(dev_records, classes=classes, class_names_map=display_names)\n'
                "adapted_dev_scientific = pipe.evaluate(dev_records, classes=classes, class_names_map=scientific_names, prompt_template='This is a photo of {{label}}.')\n"
                "development_view = {{'label': dev_label, 'n': len(dev_records), 'note': 'reported in every run for continuity with earlier records; informs no decision' if not USE_BYOD else 'your own test split (exploratory)', **compare(baseline_majority, baseline_neighbour, frozen_dev, adapted_dev, frozen_dev_scientific, adapted_dev_scientific)}}\n"
                "print({{'validation': {{k: round(float(adapted_val[k]), 3) for k in METRICS}}, 'n': adapted_val['n'], 'note': 'the configuration and the epoch were chosen on these photographs, so these numbers are biased upward'}})\n"
                "print({{'development_view': {{k: development_view[k] for k in ('label', 'n', 'note', *METRICS, 'verdict')}}}})\n"
                'fresh_test = None\n'
                'if preregistered:\n'
                '    # THE fresh test: scored only here, only under the pre-registered settings. The adapted state is parked in work/ so\n'
                '    # the frozen model can be scored on the same photographs, then put back.\n'
                "    parked = Path('work/{stem}_adapter_parked')\n"
                '    shutil.rmtree(parked, ignore_errors=True)\n'
                '    adapted_fresh = pipe.evaluate(fresh_records, classes=classes, class_names_map=display_names) if adaptation_kept else None\n'
                "    adapted_fresh_sci = pipe.evaluate(fresh_records, classes=classes, class_names_map=scientific_names, prompt_template='This is a photo of {{label}}.') if adaptation_kept else None\n"
                '    pipe.save_artifact(parked)\n'
                '    pipe.restore_base()\n'
                '    frozen_fresh = pipe.evaluate(fresh_records, classes=classes, class_names_map=display_names)\n'
                "    frozen_fresh_sci = pipe.evaluate(fresh_records, classes=classes, class_names_map=scientific_names, prompt_template='This is a photo of {{label}}.')\n"
                '    pipe.load_artifact(parked)\n'
                "    fresh_test = {{'n': len(fresh_records), **compare(majority_baseline(train_records, fresh_records, classes), colour_neighbour_baseline(train_records, fresh_records, classes), frozen_fresh, adapted_fresh, frozen_fresh_sci, adapted_fresh_sci)}}\n"
                "    print({{'fresh_test': {{k: fresh_test[k] for k in ('n', *METRICS, 'verdict')}}}})\n"
                'comparison = fresh_test if fresh_test is not None else development_view\n'
                "for key in ('scientific_name_prompts', 'delta_vs_frozen', 'ap_fell', 'recall_fell', 'by_species'):\n"
                '    if key in comparison:\n'
                '        print({{key: comparison[key]}})\n'
                'evaluation_report_payload = {{\n'
                "    'test_look': test_look,\n"
                "    'evaluation_of_record': RECORDED_EVALUATION or 'see docs/release-verification.md (the first hosted pre-registered run); this runtime cannot know about others',\n"
                "    'preregistered': preregistered,\n"
                "    'settings_differing': differing,\n"
                "    'preregistration': PREREGISTRATION,\n"
                "    'run_settings': run_settings,\n"
                "    'grid_selection': grid_selection,\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': DEFAULT_MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'sets': disjoint,\n"
                "    'classes': classes,\n"
                "    'display_names': display_names,\n"
                "    'fresh_test': fresh_test,\n"
                "    'development_view': development_view,\n"
                "    'validation_metrics': adapted_val,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                '}}\n'
                "run_suffix = output_suffix(report_path, test_look)  # '' / _reproduction_<n> / _exploratory_<n>, once for every file this run writes\n"
                '\n'
                '\n'
                'def suffixed(path):\n'
                '    path = Path(path)\n'
                '    return path.with_name(path.stem + run_suffix + path.suffix)\n'
                '\n'
                '\n'
                'written = suffixed(report_path)  # never overwrites the first pre-registered report (this runtime)\n'
                "with open(written, 'w', encoding='utf-8') as f:\n"
                '    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n'
                "print({{'report': str(written), 'test_look': test_look}})"
            ),
        },
        {
            'md': (
                '**What to notice:** `preregistered`, `test_look` and `settings_differing` first; then the four systems on the three metrics, `delta_vs_frozen`, the scientific-name rows, `ap_fell` / `recall_fell`, and the `verdict`.\n'
                '\n'
                '<details><summary>Check your reasoning</summary>No fresh-test number exists yet: the evaluation of record will be the first hosted pre-registered run, recorded in `docs/release-verification.md`. On the exposed development test (continuity only) the CPU build record of 18 October 2026 (fresh test masked, so `not scored (exploratory)`) read majority 0.167, colour neighbour 0.26, accuracy 0.76 frozen → 0.792 adapted, macro F1 0.758 → 0.793 and mAP 0.723 → 0.871 (scientific-name prompts: mAP 0.467 → 0.492), with `ap_fell` = [dark_eyed_junco] and `recall_fell` = [american_goldfinch, chipping_sparrow, dark_eyed_junco]. The Colab CLI T4 run of 9 October 2026 of the previous version read mAP 0.723 → 0.871, accuracy 0.76 → 0.792 and macro F1 0.758 → 0.793 there, and the scientific-name prompts moved with it (accuracy 0.479 → 0.51, mAP 0.467 → 0.492, macro F1 0.459 → 0.453), the White-throated Sparrow gaining most (recall 0.44 → 0.75) while the Chipping Sparrow lost ground (0.75 → 0.62). A higher mean can hide a regression in one species, and one seeded run on 96 photographs has no dispersion estimate.</details>'
            ),
        },
        {
            'md': (
                '## 9. Re-score the drawn shapes, export the adapter and reload it\n'
                '\n'
                'The three shapes from Section 5 are classified again by the adapted model — drawings, a different image family from the photographs it was tuned on, so this is a small look at what the adaptation did *outside* its corpus (a changed ranking here is a finding to record, not a failure) — and reported with the per-grid `evaluation_report` (`sample-sanity`). Both score sets are written as JSON.\n'
                '\n'
                "`pipe.save_artifact` writes the trained tensors — the vision tower's selected last blocks, post-layernorm and attention-pool head (about 85 MB with two blocks) — as `adapter.safetensors`, with a `manifest.json` recording the artifact format, the base model id and revision, the digest of the base `model.safetensors`, the classes and prompt template, the tensor names, the file size and SHA-256, the training configuration and the epoch history (OUT8). `Siglip2Pipeline.from_artifact` re-verifies the base snapshot, checks the artifact manifest, its digest and its exact tensor set **before** deserialising, refuses any tensor outside the vision tower, and overlays the tensors onto a freshly loaded base — a new object from files, not the in-memory model (VER2). The cell asserts matching image embeddings (within 1e-5) on eight photographs of the development set (VER4; images only, no labels or metrics). Every file this run writes carries the same suffix as its evaluation report, and `result.json` repeats `test_look` and `settings_differing`.\n"
                '\n'
                '**Predict:** the reloaded pipeline is a new object built from files. Will all eight embeddings match to within 1e-5, and will the adapted tower still put the right label on top for each drawn shape?'
            ),
            'code': (
                'adapted_classifications = [pipe.zero_shot_classify(image_path, candidate_labels) for image_path in shape_images]\n'
                'adapted_retrievals = [pipe.retrieve(query, shape_images, top_k=len(shape_images)) for query in shape_labels]\n'
                "adapted_scene = evaluation_report({{'classifications': adapted_classifications, 'retrievals': adapted_retrievals, 'gallery_ids': [p.name for p in shape_images]}}, targets, sample_kind='synthetic')\n"
                'for image_path, before, after in zip(shape_images, classifications, adapted_classifications, strict=True):\n'
                "    print({{'image': image_path.name, 'frozen': [(s.label, round(s.score, 3)) for s in before[:2]], 'adapted': [(s.label, round(s.score, 3)) for s in after[:2]]}})\n"
                "print({{'scene_after_adaptation': {{m['id']: round(m['value'], 3) for m in adapted_scene['metrics']}}, 'verdict': adapted_scene['verdict']}})\n"
                "with open(suffixed('outputs/{stem}_shapes.json'), 'w', encoding='utf-8') as handle:\n"
                "    json.dump({{'frozen': classification_rows, 'adapted': [{{'image': p.name, 'scores': [asdict(x) for x in ranking]}} for p, ranking in zip(shape_images, adapted_classifications, strict=True)]}}, handle, indent=2)\n"
                '\n'
                "artifact_dir = Path('outputs/{stem}_adapter' + run_suffix)\n"
                'shutil.rmtree(artifact_dir, ignore_errors=True)\n'
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source, 'test_look': test_look, 'settings_differing': differing}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n"
                '\n'
                'reloaded = Siglip2Pipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n'
                "before = pipe.embed_image([r['image'] for r in dev_records[:8]])  # images only: no labels, no metrics\n"
                "after = reloaded.embed_image([r['image'] for r in dev_records[:8]])\n"
                "parity = {{'max_abs_difference': float(np.abs(before - after).max()), 'identical_rows': int((np.abs(before - after).max(axis=1) < 1e-5).sum()), 'of': int(before.shape[0])}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['identical_rows'] == parity['of']\n"
                '\n'
                "write_provenance(suffixed('outputs/provenance.json'), pipeline=pipe)\n"
                'result_payload = {{\n'
                "    'test_look': test_look,\n"
                "    'preregistered': preregistered,\n"
                "    'settings_differing': differing,\n"
                "    'evaluation_of_record': RECORDED_EVALUATION or 'see docs/release-verification.md (the first hosted pre-registered run)',\n"
                "    'grid_selection': grid_selection,\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': snapshot['files'], 'fetched_this_run': fetched, 'weight_file': MODEL_FILENAME, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': MODEL_SHA256}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'release': CORPUS_RELEASE, 'license': CORPUS_LICENSE, 'base_url': CORPUS_BASE_URL, 'bytes': CORPUS_BYTES, 'pinned_photographs': len(SAMPLE_RECORDS), 'pin_manifest_sha256': pin_manifest_digest(), 'species': {{k: list(v) for k, v in SPECIES.items()}}}},\n"
                "    'fresh_test_pins': {{'source': FRESH_TEST_SOURCE, 'photographs': len(FRESH_TEST_RECORDS), 'manifest_sha256': fresh_manifest_digest(), 'masked': FRESH_TEST_MASKED}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'sanity_checks': checks, 'shapes': {{'names': [p.name for p in shape_images], 'sha256': shape_sha256, 'labels': shape_labels, 'candidate_labels': candidate_labels}}, 'frozen_report': frozen_scene, 'adapted_report': adapted_scene}},\n"
                "    'comparison': comparison,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'numpy': numpy.__version__, 'device': str(pipe.device), 'dtype': 'float32', 'checkpoint_source': pipe.checkpoint_source}},\n"
                '}}\n'
                "with open(suffixed('outputs/{stem}_result.json'), 'w', encoding='utf-8') as handle:\n"
                '    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n'
                "print(sorted(os.listdir('outputs')))"
            ),
        },
        {
            'md': '<details><summary>Check your reasoning</summary>All eight embeddings matched within 1e-5 — the CPU build record of 18 October 2026 reported a maximum absolute difference of 0.0 and `identical_rows: 8 of 8` — because the reloaded pipeline runs the same tensors on the same device. The shapes kept their labels after adaptation in that record (top-1 1.0, recall@1 1.0; a changed ranking would be a finding to record, not a failure); the adapter was 85,062,712 bytes in 45 tensors.</details>',
        },
    ],
    'closing': (
        '## Interpretation and limits\n'
        '\n'
        "The frozen model is already a usable zero-shot classifier on six bird species it was not fine-tuned on — far above the two non-neural baselines — and a bounded fine-tuning of the vision tower's last blocks and head on 216 photographs moved the retrieval view on the exposed development test in every recorded run (text-to-image mAP 0.723 → 0.871 in the Colab CLI T4 run of 9 October 2026 of the previous version). That is the claim this notebook can support: the adaptation contract works end to end on a real labelled photograph set, and its numbers are read on three metrics, per species, on two prompt sets, against two non-neural baselines and the frozen model rather than in isolation. Whether the gain holds on photographs nobody looked at is what the fresh test answers, once, under the pre-registered settings; see `docs/release-verification.md` for the recorded evaluation and its reproductions.\n"
        '\n'
        "Four limits shape every number. **Exposure:** the 96-photograph development test was scored in every earlier run, and the trainable scope (vision tower, text tower frozen) was kept at build time because it gained on an earlier test split whose record is lost; the grid in Section 7 re-chooses the vision-side configuration on validation, but the frozen text tower was not re-tested. **Pretraining overlap (DAT9):** the photographs are public and SigLIP 2's WebLI training set is not published, so the frozen model may have seen some of them; the fresh test's 2025-03-01 cutoff narrows that risk but does not remove it. **Size:** 96 photographs per test set and 48 for validation give no dispersion estimate; accuracy moves in steps of one photograph, and the configuration and the epoch were chosen on the 48 validation photographs, so validation numbers are biased upward. **Scope:** a gain says the contract works, not that the adapted model is better on your photographs, that its scores are calibrated, or that a sigmoid score is a probability of being right — it still returns a best label for every image, and it can be wrong confidently. Fine-tuning on a narrow set can also erode the model elsewhere; the drawn shapes re-scored in Section 9 are three images of evidence about that, not a measurement.\n"
        '\n'
        "Three things to carry to real data. **Baselines first:** the majority floor, the colour neighbour and the frozen model's score on *your* labels are the numbers to read before any adapted one, per class and on the retrieval view. **Leakage and exposure:** keep every photograph in one split (the contract de-duplicates by decoded pixels), split by photographer or session when your images come from few sources, and choose settings on validation — a test set you have looked at while choosing is a development set. **Prompts:** the classifier is the prompt as much as the tower; a second prompt set is scored here so the difference is visible, and a label an annotator wrote is not a prompt the model understands.\n"
        '\n'
        "Successful execution proves that the recorded repository revision's package, carried in this standalone notebook, can acquire and digest-verify the pinned model snapshot, fetch and digest-verify a real labelled photograph set and a pinned fresh test, validate the demonstrated dataset contract without leakage, choose a configuration on validation by a pre-registered rule, execute the inference contract and a bounded fine-tuning, evaluate against two trivial baselines and the frozen model, and emit the shown machine-readable artifacts — without the repository being reachable. It does **not** establish benchmark superiority, zero-shot accuracy on any other population or camera, calibration, or production fitness.\n"
        '\n'
        '## Troubleshooting\n'
        '\n'
        '- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — use Google Colab, Kaggle or a Linux x86_64 Jupyter server.\n'
        '- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 again; a complete environment is reused and an incomplete one is finished. If it repeats, `files.pythonhosted.org` or `pypi.org` is blocked or altered.\n'
        '- **Section 1 runs out of disk, or you want the environment elsewhere** — set the environment variable `DIMER_ISOLATED_ENV` to a writable path before running Section 1; by default it is `dimer_isolated_env_<lock digest>/` in the working directory.\n'
        '- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable. After a session restart, run from the top.\n'
        '- **"The isolated environment\'s Python process exited"** — usually out of memory; restart the session and choose **Run all**. A smaller `BATCH_SIZE` also helps, but it makes the run exploratory (the fresh test is then not scored).\n'
        '- **Section 3 reports a size or SHA-256 mismatch, or cannot reach the Hub** — the message names the file. Delete the folder Section 3 prints as `weights_dir` and run Section 3 again (1.5 GB).\n'
        '- **Section 4 refuses a photograph** — its byte size or SHA-256 does not match the pin: the cached file under `weights/inat-birds/` (or `weights/inat-birds/fresh/`) is damaged (delete it) or the bucket served something else; the message names the photo id.\n'
        "- **Section 8 says `not scored (exploratory)` although you changed nothing** — read `settings_differing`: a re-run of Section 6 or 7 after Section 7b restores the base model, and a grid that ran on other data does not count. Run Sections 4–8 in order. In the repository's CI the fresh test is masked on purpose (`DIMER_FRESH_TEST_MASK=1`).\n"
        '- **The frozen or adapted numbers differ from the quoted ones by a photograph or two** — expected across hardware and library builds: accuracy on 96 photographs moves in steps of about 1 %. A verdict of `not above the baselines`, `no change` or `worse` is a finding to read, not an error.\n'
        "- **Section 9's parity check fails** — the export or reload is broken; run Sections 7b–9 again. Do not use the artifact.\n"
        '- **BYOD: "BYOD input refused: …"** — the message names the file and the rule (a missing path, no upload, not a .zip, a missing `labels.csv`, an unreadable image, an image side over 4096 px, a label over 64 characters, too few records, one label only, or a duplicate id). Fix it and run Section 4 again; files already in `outputs/` are from an earlier run and are listed in `outputs/{stem}_run_invalidated.json`.\n'
        '\n'
        '## Change one thing (next experiments)\n'
        '\n'
        'Each of these makes the run **exploratory**: compare on the **validation** numbers Section 8 prints (and, for continuity, the exposed development test), never on the fresh test, which is scored only under the pre-registered settings. Set `CONFIG` to `C1` and compare the artifact size and the validation mAP; raise `EPOCHS` and watch the validation mAP pick the epoch while the training loss keeps falling; set `CONFIG` to `C2` (learning rate 1e-5) and read a smaller, steadier gain; change `SPLIT_SEED` for another draw of the 360 photographs; or bring your own photographs through BYOD and read the two baselines before the adapted number.\n'
        '\n'
        '## Glossary\n'
        '\n'
        '- **Zero-shot classification** — scoring an image against a prompt per label (`DEFAULT_PROMPT_TEMPLATE` with the label filled in) and returning the highest; no label-specific training.\n'
        '- **Sigmoid score** — each image–prompt pair is scored on its own through `logit_scale`, `logit_bias` and a sigmoid; the scores do not sum to one, are not calibrated, and the model never abstains.\n'
        '- **Accuracy / macro F1** — the fraction of photographs whose top prompt is right; the mean per-species F1, which weighs each species equally.\n'
        '- **Text-to-image mAP** — each prompt as a query ranking all the photographs of a set; the average precision of its own species, averaged over the six — the retrieval view of the same scores, and the smoother metric on a small set.\n'
        '- **Majority floor / colour nearest neighbour** — answering every photograph with the most frequent training label; answering with the label of the training photograph whose 3 × 3 mean-colour grid is closest (27 numbers, no neural network).\n'
        '- **Exposed development test** — the 96 test photographs of the seeded split; every earlier run scored them, so they are reported for continuity and inform no decision.\n'
        '- **Fresh test** — 96 new photographs pinned by id, size and SHA-256, scored only when every pre-registered setting holds.\n'
        '- **Pre-registered / exploratory** — a run whose settings all equal the pre-registration (it scores the fresh test) / any other run (validation and the exposed development test only, written to `*_exploratory_<n>.json`).\n'
        '- **`test_look`** — what this run did with the fresh test: `pre-registered evaluation (this runtime)`, `reproduction (this runtime)` or `not scored (exploratory)`.\n'
        '- **Vision tower / attention-pool head** — the ViT-B/16 image encoder and the head that pools its patch tokens into one embedding; the fine-tune trains the last `TRAINABLE_VISION_LAYERS` blocks, the post-layernorm and the head.\n'
        '- **SigLIP sigmoid loss** — the pairwise training objective: +1 for the gold prompt, −1 for every other, each pair scored independently.\n'
        '- **Epoch selection** — the kept epoch has the highest validation text-to-image mAP; accuracy on 48 photographs moves in steps of 2 %, so the retrieval view selects.\n'
        '- **Restore base** — when the pipeline carries an adaptation, Sections 6, 7 and 7b put the pinned base tensors back (in memory, exactly) before scoring or training.\n'
        '- **Reload parity** — the adapter written to disk, loaded onto a fresh base, reproduces the in-memory image embeddings of eight photographs within 1e-5.\n'
        '- **Isolated environment** — the separate Python 3.12.12 environment Section 1 builds from the hash lock (`DIMER_ISOLATED_ENV` moves it); every later cell runs there.\n'
        '- **BYOD** — bring your own data: a zip with `labels.csv` beside the image files, read from `BYOD_PATH` or the Colab upload dialog.\n'
        '\n'
        '## Conclusion (your notes)\n'
        '\n'
        "Before you leave, write three lines in this cell: (1) the frozen model's accuracy and mAP beside the two baselines, and which species the common-name prompt separated worst; (2) which configuration the grid selected and why, and the `test_look` Section 8 recorded — did the mAP gain, and did the scientific-name prompt set, which the tower never saw, move with it? (3) one reason the same recipe could land differently on your own labels (prompts, split, photographers, pretraining overlap) and the baseline you would read first.\n"
        '\n'
        '## References\n'
        '\n'
        '[1] M. Tschannen *et al.*, “SigLIP 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features,” arXiv:2502.14786, Feb. 2025. [Online]. Available: https://arxiv.org/abs/2502.14786\n'
        '\n'
        '[2] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, “Sigmoid loss for language image pre-training,” in *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, 2023. [Online]. Available: https://arxiv.org/abs/2303.15343\n'
        '\n'
        '[3] Google, “{MODEL_ID},” Hugging Face model card, revision `{MODEL_REVISION}`. [Online]. Available: https://huggingface.co/{MODEL_ID}\n'
        '\n'
        '[4] T. Wolf *et al.*, “Transformers: State-of-the-art natural language processing,” in *Proc. 2020 Conf. Empirical Methods in Natural Language Processing: System Demonstrations*, 2020, pp. 38–45. [Online]. Available: https://github.com/huggingface/transformers\n'
        '\n'
        '[5] iNaturalist, “iNaturalist open data and API v1,” [Online]. Available: https://www.inaturalist.org/pages/developers — bucket https://inaturalist-open-data.s3.amazonaws.com/\n'
        '\n'
        '[6] K. Valcorza, “siglip2-vision-language-pipeline: README, model card and sample dataset card,” GitHub repository. [Online]. Available: https://github.com/kurtvalcorza/siglip2-vision-language-pipeline\n'
        '\n'
        '[7] K. Valcorza, “Pre-registration: SigLIP 2 tutorial configuration grid and fresh test,” `docs/siglip2-preregistration.md`, 2026-10-18; draw log `docs/siglip2-fresh-test-draw.json`.\n'
        '\n'
        '[8] DIMER, “DIMER Notebook Specification 2.5” and “Model Card Specification 1.1,” `integrations/dimer/fleet-specs/`, ml-worker repository, revision `bfe4150`.'
    ),
}
