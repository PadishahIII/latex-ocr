<div align="center">

# latex-ocr

**Standalone LaTeX formula OCR: image in, LaTeX out.**

Training recipes + inference API server. Models are trained on open datasets
([UniMER-1M](https://huggingface.co/datasets/wanderkid/UniMER_Dataset),
[LaTeX-OCR](https://huggingface.co/datasets/lukbl/LaTeX-OCR-dataset), and a
rendered [styled synthetic set](https://huggingface.co/datasets/PadishahIIIXXX/latex-ocr-dataset)).

</div>

## Overview

This repo contains:

- **Model architectures** (`latex_ocr/models/`):
  - `CoCaSwinOCR` — Swin encoder + CoCa-style two-decoder (caption + contrastive), the flagship model.
  - `XceptionGRUCaptioner` — Xception encoder + attention GRU decoder (lightweight).
  - `SwinGRUCaptioner` / `SwinTBase` — Swin-T with GRU / Transformer decoders.
- **Training framework** (`recipe/`) — the recipe runner used for all training:
  epoch loop, AMP, grad accumulation, DDP (torchrun), early stopping,
  LR schedules (warmup+cosine, ReduceLROnPlateau, 2-stage finetune),
  checkpointing and experiment tracking to MLflow + MinIO.
- **Data pipeline** (`latex_ocr/data/`) — HF Hub dataset loaders, token-length
  filtering with caches, bucket batch sampler, and the synthetic-data
  rendering pipeline (`data/pipeline/`).
- **Evaluation** (`latex_ocr/evaluation/`, `latex_ocr/trainers/eval.py`) —
  BLEU / edit-distance / token metrics, offline evaluator, MLflow-based eval CLI.
- **Inference API server** (`latex_ocr/serve.py`) — FastAPI service:
  image in, LaTeX out.

## Installation

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/) (recommended):

```bash
git clone https://github.com/<you>/latex-ocr.git
cd latex-ocr
uv sync --extra server      # + FastAPI inference server
```

or plain pip: `pip install -e ".[server]"`.

> Torch is installed from the cu128 index by default (see `pyproject.toml`).
> For CPU-only environments, adjust `[tool.uv.sources]` accordingly.

## Datasets

All training data is open and hosted on Hugging Face; it downloads and caches
automatically (respecting `HF_HOME`) on first use:

| Dataset | HF repo | Used for |
|---|---|---|
| UniMER-1M / UniMER-Test | `wanderkid/UniMER_Dataset` | real formulas (train/test) |
| LaTeX-OCR | `lukbl/LaTeX-OCR-dataset` | real rendered formulas |
| Synthetic plain/styled | `PadishahIIIXXX/latex-ocr-dataset` | style-augmented pretraining |

The `datasets/` directory in this repo only carries small metadata (formula
corpora JSON, tokenizer symbol list); the heavy image data comes from HF Hub.

```bash
# optional: prefetch
hf download PadishahIIIXXX/latex-ocr-dataset --repo-type=dataset
hf download wanderkid/UniMER_Dataset   --repo-type=dataset
```

## Training

Configure MLflow/MinIO tracking by copying `.env.example` → `.env` and filling
in your server's credentials (never committed). Then:

```bash
# CoCa pretrain (synthetic plain)
python -m latex_ocr.trainers.train train --recipe-name coca_pretrain

# CoCa finetune (stage A: frozen encoder, stage B: end-to-end)
python -m latex_ocr.trainers.train train --recipe-name coca_finetune

# Xception + GRU (styled)
python -m latex_ocr.trainers.train train --recipe-name xception_gru

# Multi-GPU with DDP
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone \
  --nproc_per_node=2 -m latex_ocr.trainers.train train --recipe-name coca_pretrain
```

All hyperparameters live in `latex_ocr/trainers/train.py` (dataclass/pydantic
configs). Tracking can be disabled entirely with `MLFLOW_ENABLE=false`.

## Evaluation

```bash
python -m latex_ocr.trainers.eval evaluate \
  --run-id <mlflow-run-id> --model-name <model-artifact> \
  --experiment-name <experiment> --config-name plain --split validation
```

or use the offline evaluator directly (`latex_ocr/evaluation/`).

## Inference API server

```bash
latex-ocr-server \
  --model models/checkpoints/latex-ocr-coca-finetune.pth \
  --device cuda --port 8000
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

## Checkpoints

Trained checkpoints (full-model `.pth`) are published to Hugging Face; place
them under `models/checkpoints/` or point `--model` anywhere on disk:

```bash
hf download <you>/latex-ocr-coca-finetune --local-dir models/checkpoints/
```

## Repository layout

```
latex_ocr/
  models/        architectures (CoCa-Swin, Xception+GRU, Swin variants), tokenizer
  trainers/      configs, model factory, train CLI, eval CLI, decoder pretraining
  data/          HF Hub loaders + synthetic data pipeline
  evaluation/    BLEU / edit-distance / token metrics, offline evaluator
  serve.py       FastAPI inference server
recipe/          training framework (runner, DDP, MLflow/MinIO tracking)
datasets/        small metadata only; images come from HF Hub
```

## License

MIT for the code; see dataset cards for dataset licenses (CC-BY-4.0).
