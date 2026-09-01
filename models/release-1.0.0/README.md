---
license: mit
pipeline_tag: image-to-text
base_model: timm/swin_small_patch4_window7_224.ms_in22k
datasets:
  - PadishahIIIXXX/latex-ocr-dataset
tags:
  - latex
  - ocr
  - formula-recognition
  - coca
  - swin
  - math
model-index:
  - name: latex-ocr release-1.0.0
    results:
      - task:
          type: image-to-text
          name: LaTeX formula OCR
        dataset:
          name: PadishahIIIXXX/latex-ocr-dataset (plain test split)
          type: PadishahIIIXXX/latex-ocr-dataset
          config: plain
          split: test
        metrics:
          - type: bleu
            name: BLEU
            value: 0.8737
          - type: exact_match
            name: Exact match ratio
            value: 0.522
          - type: edit_distance
            name: Edit distance (lower is better)
            value: 0.0695
      - task:
          type: image-to-text
          name: LaTeX formula OCR (styled)
        dataset:
          name: PadishahIIIXXX/latex-ocr-dataset (styled test split)
          type: PadishahIIIXXX/latex-ocr-dataset
          config: styled
          split: test
        metrics:
          - type: bleu
            name: BLEU
            value: 0.9049
          - type: exact_match
            name: Exact match ratio
            value: 0.5316
          - type: edit_distance
            name: Edit distance (lower is better)
            value: 0.0562
---

<div align="center">

# latex-ocr · release 1.0.0

**CoCa-based LaTeX formula OCR — image in, LaTeX out — 67M parameters, laptop-CPU friendly.**

[Dataset](https://huggingface.co/datasets/PadishahIIIXXX/latex-ocr-dataset) ·
[Training code](https://github.com/PadishahIII/latex-ocr) ·
[Benchmarks](https://github.com/PadishahIII/latex-ocr#benchmarks)

</div>

This is the **release-1.0.0** model card for [`latex-ocr`](https://github.com/PadishahIII/latex-ocr).
The full weights ship in this repo as `model.pth` (torch full-model checkpoint,
MLflow/pytorch-logged format).

- **Architecture:** CoCa (Swin encoder + unimodal/multimodal dual text decoder)
- **Parameters:** 67M
- **LaTeX vocabulary:** 1,122 tokens (packaged SentencePiece tokenizer)
- **Input:** rendered formula image, resized to 192 × 672 RGB
- **Output:** LaTeX source string, autoregressive beam search (default beam 4,
  generation limit 354 tokens)
- **Hardware:** runs fine on a **laptop CPU** — no GPU required

## Model structure

CoCa structure adapted for OCR — a contrastive captioner with two text decoders over
one pooled visual memory:

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

- **Unimodal decoder** (causal self-attention only) embeds the token sequence into a
  text CLS for the contrastive loss.
- **Multimodal decoder** (self-attention + cross-attention into the pooled image
  queries) produces the captioning logits used at inference time.
- **Attentional pooling** compresses the variable-length Swin token grid into 256
  queries, so decode cost is independent of image size.
- **Weight tying** between input embeddings and the output projection keeps the model compact.

The Swin-Small encoder is initialized from
[`timm/swin_small_patch4_window7_224.ms_in22k`](https://huggingface.co/timm/swin_small_patch4_window7_224.ms_in22k)
(ImageNet-22k pretrained weights); all other components are trained from scratch.

## Training

Two-stage training on
[PadishahIIIXXX/latex-ocr-dataset](https://huggingface.co/datasets/PadishahIIIXXX/latex-ocr-dataset):

1. **Pretrain** on the `plain` config (~1.14M re-rendered, sanitized formulas from
   UniMER-1M + LaTeX-OCR).
2. **Finetune** on the plain + styled mixture (stage A: frozen encoder; stage B:
   end-to-end). The `styled` config injects LaTeX font macros (`\mathbf`, `\mathbb`,
   `\mathcal`, `\mathit`, `\mathrm`, `\mathsf`, `\mathtt`, `\mathfrak`, `\mathscr`)
   with semantic heuristics, so the model learns to **read style glyphs and emit the
   corresponding style macros**.

Full recipes and hyperparameters: [training docs](https://github.com/PadishahIII/latex-ocr#training).

## Results

Test sets are the test splits of
[the dataset](https://huggingface.co/datasets/PadishahIIIXXX/latex-ocr-dataset):
`plain` (23,868 items) and `styled` (2,955 items). Baselines:
[UniMERNet](https://github.com/opendatalab/unimernet) (zero-shot on this data —
never trained on the new dataset, especially the styled split).

| Test set | Model | Param | BLEU $\uparrow$ | Edit distance $\downarrow$ | Exact match ratio $\uparrow$ |
|-|-|-|-|-|-|
| plain | UniMER-base | 325M | 0.891463 | 0.138271 | |
| plain | UniMER-small | 202M | 0.885505 | 0.141475 | |
| plain | UniMER-tiny | 107M | 0.869181 | 0.149117 | |
| plain | **latex-ocr (this release)** | **67M** | **0.8737** | **0.0695** | **0.5220** |
| styled | UniMER-base † | 325M | 0.757091 | 0.253583 | |
| styled | UniMER-small † | 202M | 0.755511 | 0.255015 | |
| styled | UniMER-tiny † | 107M | 0.741603 | 0.267083 | |
| styled | **latex-ocr (this release)** | **67M** | **0.9049** | **0.0562** | **0.5316** |

> **†** Baseline numbers on the styled set are **zero-shot** (UniMERNet was not trained
> on the new styled data), so the styled gap is partly by construction. On the plain
> set the comparison is like-for-like: this 67M model beats UniMER-tiny (107M) on all
> reported metrics and approaches the 202M–325M baselines.

Full benchmark table with all development iterations:
[repo benchmarks](https://github.com/PadishahIII/latex-ocr#benchmarks).

## Usage

Download the weights:

```bash
hf download PadishahIIIXXX/latex-ocr --local-dir models/checkpoints/
```

### Python

```python
from PIL import Image
from latex_ocr.serve import LatexOCRPredictor

predictor = LatexOCRPredictor(
    model_path="models/checkpoints/model.pth",
    device="cpu",     # laptop CPU is enough
    beam_size=4,
)
print(predictor.predict(Image.open("formula.png")))
```

### Inference API server

```bash
pip install -e "git+https://github.com/PadishahIII/latex-ocr.git#egg=latex-ocr[server]"

latex-ocr-server --model models/checkpoints/model.pth --device cpu --port 8000

curl -s -F file=@formula.png http://localhost:8000/predict
# -> {"latex": "E = mc^2"}
```

Interactive docs: `http://localhost:8000/docs`. More endpoints and options:
[usage docs](https://github.com/PadishahIII/latex-ocr#usage).

## Intended use & limitations

- **In scope:** rendered (print-style) LaTeX formula images — PDFs, screenshots,
  textbook/paper crops — including formulas typeset with font-style macros.
- **Out of scope / known limits:** handwritten formulas (the upstream UniMER HWE split
  was not used), very long multi-line equations beyond the 354-token generation limit,
  and non-LaTeX math notation (e.g. MathML, UnicodeMath).
- The styled advantage is measured against zero-shot baselines; treat cross-dataset
  comparisons accordingly.

## License

MIT for the model weights. Training data derives from
[UniMER-1M](https://huggingface.co/datasets/wanderkid/UniMER_Dataset) and
[LaTeX-OCR](https://huggingface.co/datasets/lukbl/LaTeX-OCR-dataset) — see the
[dataset card](https://huggingface.co/datasets/PadishahIIIXXX/latex-ocr-dataset) for
dataset licenses (CC-BY-4.0).

## Citation

If you use this model, please cite the repository and the underlying work:

```bibtex
@software{latex_ocr_2026,
  author  = {PadishahIII},
  title   = {latex-ocr: a 67M-parameter CoCa model for LaTeX formula OCR},
  year    = {2026},
  url     = {https://github.com/PadishahIII/latex-ocr}
}

@article{yu2022coca,
  title   = {CoCa: Contrastive Captioners are Image-Text Foundation Models},
  author  = {Yu, Jiahui and Wang, Zirui and Vasudevan, Vijay and Yeung, Legg and
             Seyedhosseini, Mojtaba and Wu, Yonghui},
  journal = {Transactions on Machine Learning Research},
  year    = {2022},
  url     = {https://arxiv.org/abs/2205.01917}
}
```
