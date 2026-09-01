<div align="center">

# latex-ocr

**Standalone LaTeX formula OCR — image in, LaTeX out.**

A 67M-parameter [CoCa](https://arxiv.org/abs/2205.01917)-based model that runs comfortably on a **laptop CPU**,
trained on a new dense, style-aware LaTeX dataset.

[Model (release 1.0.0)](https://huggingface.co/PadishahIIIXXX/latex-ocr) ·
[Dataset](https://huggingface.co/datasets/PadishahIIIXXX/latex-ocr-dataset) ·
[Code](https://github.com/PadishahIII/latex-ocr)

</div>

## Introduction

`latex-ocr` is a standalone LaTeX formula OCR system: feed it an image of a rendered
formula, get the LaTeX source back. This repository contains:

- **A pretrained model** — [release 1.0.0](https://huggingface.co/PadishahIIIXXX/latex-ocr),
  a 67M-parameter CoCa model that is small enough to run on a laptop CPU with no GPU required.
- **A fresh training dataset** —
  [PadishahIIIXXX/latex-ocr-dataset](https://huggingface.co/datasets/PadishahIIIXXX/latex-ocr-dataset),
  built from [UniMER-1M](https://huggingface.co/datasets/wanderkid/UniMER_Dataset) and the
  [LaTeX-OCR dataset](https://huggingface.co/datasets/lukbl/LaTeX-OCR-dataset) with denser LaTeX
  text and fully re-rendered images.
- **Training recipes** (`recipe/`, `latex_ocr/trainers/`) — epoch loop, AMP, gradient
  accumulation, DDP (torchrun), early stopping, LR schedules, checkpointing, and
  MLflow/MinIO experiment tracking.
- **An inference API server** (`latex_ocr/serve.py`) — a small FastAPI service: image in, LaTeX out.

## Motivation

Two gaps motivated this project.

### 1. An enriched dataset with style-aware LaTeX syntax

Real documents do not render every symbol in the same font: you meet
`\mathbb{R}`, `\mathcal{F}`, `\mathfrak{g}`, `\mathbf{x}`, `\mathrm{d}x`, and friends on
every page. Existing OCR datasets mostly render plain LaTeX, so models trained on them
either misread style-macro glyphs or hallucinate the macros back.

The new dataset addresses this in two ways:

- **Denser, re-rendered text.** Formulas from UniMER-1M and LaTeX-OCR are sanitized
  (space normalization, legacy-command normalization) and re-rendered through a
  MathJax → SVG → PDF → PNG pipeline at 150 DPI, giving clean PDF-quality images of
  denser formulas (~1.17M plain + ~158k styled samples).
- **A `styled` split with LaTeX font styles.** Font macros (`\mathbf`, `\mathbb`,
  `\mathcal`, `\mathit`, `\mathrm`, `\mathsf`, `\mathtt`, `\mathfrak`, `\mathscr`) are
  injected with semantic heuristics — number sets tend to become `\mathbb`, vectors
  `\mathbf`, differentials `\mathrm` — with a per-formula consistency rule (the same
  variable keeps the same style). Training on this split makes the model _read_ the
  visual style of a glyph and emit the corresponding style macro, instead of ignoring it.

See the dataset card
([README_synth](https://huggingface.co/datasets/PadishahIIIXXX/latex-ocr-dataset/))
for the full generation pipeline and statistics.

### 2. Laptop-CPU-friendly use

Large formula-OCR models (300M+) are awkward to use locally: slow to load, slow to run,
and impractical on machines without a GPU. The release model is deliberately compact —
**67M parameters** — so that:

- inference is fine on a **laptop CPU** (`--device cpu`), no CUDA needed;
- it still beats the 107M UniMER-tiny baseline and approaches the 325M UniMER-base
  baseline on our test set (see [Benchmarks](#benchmarks)).

## Model

The release model ([release 1.0.0](https://huggingface.co/PadishahIIIXXX/latex-ocr),
67M parameters) uses the **CoCa** structure — a contrastive captioner adapted for OCR:

```
                 formula image (192 × 672 RGB)
                            │
              ┌─────────────▼──────────────┐
              │  Swin encoder (Swin-Small, │  hierarchical vision encoder
              │  ImageNet-22k pretrained)  │  → N image tokens (384-d)
              └─────────────┬──────────────┘
                            │
              ┌─────────────▼──────────────┐
              │    attentional pooling     │  256 learned image queries
              └───────┬────────────┬───────┘  + 1 CLS query
                      │            │
        image queries │            │ image CLS (384-d)
        (256 × 384)   │            │
                      │            │
  LaTeX tokens ──► ┌──▼───────┐ ┌──▼───────────────┐
  (BOS + prefix)   │ unimodal │ │    multimodal    │
                   │ decoder  │ │     decoder      │
                   │ 3 × self │ │ 3 × self-attn    │──► per-token logits
                   │  -attn   │ │ + cross-attn ────┘    (1,122-token LaTeX
                   └──┬───────┘ └──────────────────┘     vocab, weight-tied)
                      │                                ⇒ LaTeX output
                text CLS (384-d)          inference = autoregressive
                      │                   beam search in the multimodal
        ┌─────────────▼─────────────────┐   decoder
        │  contrastive head (CLIP-style)│
        │  img CLS ↔ text CLS projected │
        │  into a shared 1152-d space   │
        └───────────────────────────────┘

  training:  captioning cross-entropy (λ = 2.0, label smoothing 0.1)
           + image–text contrastive loss (λ = 1.0)
```

Key points:

- **Two text decoders, one trunk.** A _unimodal_ stack (causal self-attention only)
  embeds the token sequence into a text CLS for the contrastive loss; a _multimodal_
  stack (self-attention + cross-attention into the pooled image queries) produces the
  captioning logits used at inference time.
- Training is two-stage: **pretrain** on the `plain` split, then **finetune** on the
  plain + styled mixture (stage A: frozen encoder, stage B: end-to-end).

## Benchmarks

Test sets (both are test splits of
[the dataset](https://huggingface.co/datasets/PadishahIIIXXX/latex-ocr-dataset)):

- **plain** split — 23,868 items
- **styled** split — 2,955 items

Baselines are UniMER-base / -small / -tiny from
[UniMERNet](https://github.com/opendatalab/unimernet).

| Benchmark | Model                           | Param | BLEU ↑     | Edit distance ↓ | Exact match ratio ↑ |
| --------- | ------------------------------- | ----- | ---------- | --------------- | ------------------- |
| Plain     | UniMER-base                     | 325M  | 0.891463   | 0.138271        | -                   |
|           | UniMER-small                    | 202M  | 0.885505   | 0.141475        | -                   |
|           | UniMER-tiny                     | 107M  | 0.869181   | 0.149117        | -                   |
|           | **latex-ocr** _(release-1.0.0)_ |       | **0.8737** | **0.0695**      | **0.5220**          |
| Styled    | UniMER-base †                   | 325M  | 0.757091   | 0.253583        |                     |
|           | UniMER-small †                  | 202M  | 0.755511   | 0.255015        |                     |
|           | UniMER-tiny †                   | 107M  | 0.741603   | 0.267083        |                     |
|           | **latex-ocr** _(release-1.0.0)_ |       | **0.9049** | **0.0562**      | **0.5316**          |     |     |

> **† Limitation of the comparison:** the UniMER baseline numbers on the **styled** test
> set are **zero-shot** — UniMERNet was never trained on the new styled data, so the
> styled gap is partly by construction. The plain-split numbers, however, are a fair
> like-for-like comparison, and there the 67M release model matches or beats the
> 107M–325M baselines.

Headline: the release model (67M) **beats UniMER-tiny (107M)** and comes close to
UniMER-small (202M) / UniMER-base (325M) on plain, while strongly outperforming all
baselines on styled formulas — thanks to the style-aware training data.

## Usage

### Installation

Requires Python ≥ 3.11. With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/PadishahIII/latex-ocr.git # LFS files are only for dataset processing
cd latex-ocr
uv sync --extra server      # + FastAPI inference server
```

or plain pip:

```bash
pip install -e ".[server]"
```
By default, `torch`/`torchvision` come from PyPI, which works for CPU inference
and most CUDA GPUs. To select a specific CUDA build, add the matching extra:

| GPU | Install |
| --- | --- |
| CPU or recent CUDA GPUs | `uv sync --extra server` (default) |
| Volta/Pascal (compute capability ≤ 7.0, e.g. V100) | `uv sync --extra server --extra cu126` |
| Turing (sm_75) or newer | `uv sync --extra server --extra cu128` |

> The `cu126`/`cu128` extras repoint torch/torchvision at the matching
> `download.pytorch.org` index and are mutually exclusive — uv rejects syncing
> both. PyTorch 2.11+ cu128 builds dropped Volta (sm_70) support, which is why
> older GPUs need the cu126 flavor. The committed `uv.lock` covers all three
> variants. Plain pip always gets the PyPI build; for other CUDA flavors follow
> [pytorch.org](https://pytorch.org/get-started/locally/).
### Get the release model

Download the checkpoint from Hugging Face:

```bash
hf download PadishahIIIXXX/latex-ocr --local-dir models/checkpoints/
```

### Run inference (Python)

```python
from PIL import Image
from latex_ocr.serve import LatexOCRPredictor

predictor = LatexOCRPredictor(
    model_path="models/checkpoints/model.pth",
    device="cpu",     # runs fine on a laptop CPU
    beam_size=4,
)
print(predictor.predict(Image.open("formula.png")))
```

### Web UI (Gradio)

A browser UI — upload an image, see the LaTeX and its rendered preview:

```bash
latex-ocr webui \
  --model models/checkpoints/model.pth \
  --device cpu --port 7860
```

Then open `http://localhost:7860`. Add `--share` for a temporary public
gradio.live link.

### Facade CLI

Both servers are also reachable through a single entry point:

```bash
latex-ocr webui --model <ckpt.pth> --device cpu   # Gradio web UI (port 7860)
latex-ocr api   --model <ckpt.pth> --device cpu   # FastAPI API server (port 8000)
```

### Inference API server

```bash
latex-ocr-server \
  --model models/checkpoints/model.pth \
  --device cpu --port 8000
```

Endpoints:

```bash
# multipart upload
curl -s -F file=@formula.png http://localhost:8000/predict
# -> {"latex": "E = mc^2"}

# raw image bytes
curl -s --data-binary @formula.png http://localhost:8000/predict

# base64 JSON batch
curl -s http://localhost:8000/predict_json \
  -H 'Content-Type: application/json' \
  -d '{"images_b64": ["<base64>"]}'

curl http://localhost:8000/health
```

Interactive docs: `http://localhost:8000/docs`.

## Showcase
<img width="1345" height="329" alt="image" src="https://github.com/user-attachments/assets/086344d5-45d3-4a71-8368-583d941ece64" />

<img width="1357" height="302" alt="image" src="https://github.com/user-attachments/assets/d2fd4240-3c2c-4a0d-b936-ff67aad4b98c" />

<img width="1354" height="365" alt="image" src="https://github.com/user-attachments/assets/c4707b15-fef9-4245-b6dc-ae9b7d003503" />

<img width="1873" height="373" alt="image" src="https://github.com/user-attachments/assets/afbd7156-c219-418f-b7ad-4a4626895676" />



## Training

Training is a standalone topic: everything lives in `latex_ocr/trainers/` +
`recipe/`, driven by named recipes.

### 1. Data

All data is open and hosted on Hugging Face; it downloads and caches automatically
(respecting `HF_HOME`) on first use:

| Dataset                 | HF repo                            | Used for                                        |
| ----------------------- | ---------------------------------- | ----------------------------------------------- |
| **latex-ocr-dataset**   | `PadishahIIIXXX/latex-ocr-dataset` | main training data (`plain` + `styled` configs) |
| UniMER-1M / UniMER-Test | `wanderkid/UniMER_Dataset`         | upstream formula corpus                         |
| LaTeX-OCR               | `lukbl/LaTeX-OCR-dataset`          | upstream formula corpus                         |

Optional prefetch:

```bash
hf download PadishahIIIXXX/latex-ocr-dataset --repo-type=dataset
```

The `datasets/` directory in this repo only carries small metadata (formula corpora
JSON, tokenizer symbol list); the heavy image data comes from HF Hub.

### 2. Configure MLflow tracking (metrics + artifacts)

All training runs are recorded to [MLflow](https://mlflow.org) — metrics, params,
and model checkpoints as artifacts — with a MinIO (S3-compatible) artifact store.
The recipe reads its settings from the environment; copy the template and fill it in
(never commit real credentials):

```bash
cp .env.example .env
```

| Variable                                                | Purpose                                                     |
| ------------------------------------------------------- | ----------------------------------------------------------- |
| `MLFLOW_ENABLE`                                         | master switch (`false` trains with no MLflow connectivity)  |
| `MLFLOW_HOST` / `MLFLOW_PORT`                           | MLflow tracking server address                              |
| `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` | basic auth for the tracking server                          |
| `MLFLOW_ARTIFACT_TIMEOUT`                               | artifact upload/download timeout (seconds)                  |
| `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY`                 | artifact-store credentials                                  |
| `MINIO_PORT`, `MINIO_SECURE`, `MINIO_BUCKET`            | artifact-store endpoint / TLS / bucket (`mlflow-artifacts`) |

What gets recorded per run: training/validation metrics (loss, LR, BLEU, edit distance,
gradient norms), the full trainer config as params, and checkpointed models as MLflow
artifacts (`enable_state_ckp_to_mlflow`). Those artifacts are what the evaluation CLI
consumes by `--run-id` below. Set `MLFLOW_ENABLE=false` to train without any tracking.

### 3. Launch training

```bash
# CoCa pretrain (plain split)
PYTHONPATH=. uv run python latex_ocr/trainers/train.py train --recipe-name coca_pretrain

# CoCa finetune (plain + styled mixture; stage A frozen encoder, stage B end-to-end)
PYTHONPATH=. uv run python latex_ocr/trainers/train.py train --recipe-name coca_finetune

# Multi-GPU with DDP
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. torchrun --standalone \
  --nproc_per_node=2 -m latex_ocr.trainers.train train --recipe-name coca_finetune
```

All hyperparameters (data mixture, LR schedule, epochs, batch size) live in
`latex_ocr/trainers/train.py` as typed configs.

### 4. Evaluate

Evaluate a run's checkpoints against the plain/styled test configs:

```bash
PYTHONPATH=. uv run python latex_ocr/trainers/eval.py evaluate \
  --run-id <mlflow-run-id> \
  --experiment-name latex-ocr-coca-finetune \
  --model-name <model-artifact> \
  --config-name plain \
  --split validation
```

Repeat with `--config-name styled` for the styled test set.

## Repository layout

```
latex_ocr/
  models/        model architectures (CoCa) + packaged LaTeX tokenizer
  trainers/      configs, model factory, train CLI, eval CLI
  data/          HF Hub dataset loaders + synthetic-data rendering pipeline
  evaluation/    BLEU / edit-distance / exact-match metrics, offline evaluator
  serve.py       FastAPI inference server
  cli.py         facade CLI (webui / api)
  webui.py       Gradio web UI (image upload → LaTeX)
recipe/          training framework (runner, DDP, MLflow/MinIO tracking)
datasets/        small metadata only; images come from HF Hub
models/          release artifacts (release-1.0.0 model card)
.githooks/       large-file detector script (pre-commit)
.github/         CI workflows (gitleaks + large files)
```

## Development

### Pre-commit hooks

Two checks guard every commit — secret scanning and large-file detection:

```bash
uv tool install pre-commit   # or: pip install pre-commit
pre-commit install
```

| Hook                 | What it does                                                       |
| -------------------- | ------------------------------------------------------------------ |
| `gitleaks` (v8.30.1) | scans staged changes for leaked secrets/credentials                |
| `no-large-files`     | rejects files > 10 MB (weights/datasets belong on HF Hub, not git) |

Run on the whole repo: `pre-commit run --all-files`. The large-file threshold
is `10` MB, configurable via the first arg in `.pre-commit-config.yaml`; the
detector script is `.githooks/check-large-files`.

### CI

`.github/workflows/ci.yml` runs on every push to `main` and every PR:

- **gitleaks** — full-history secret scan (via `gitleaks/gitleaks-action@v3`)
- **large-file detector** — fails if any tracked file exceeds 10 MB, same rule as the pre-commit hook

## License

MIT for the code and model weights; see the dataset cards for dataset licenses (CC-BY-4.0).
