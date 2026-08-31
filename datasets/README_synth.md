---
dataset_info:
- config_name: plain
  splits:
  - name: train
  - name: validation
- config_name: styled
  splits:
  - name: train
  - name: validation
configs:
- config_name: plain
  data_files:
  - split: train
    path: plain/train-*
  - split: validation
    path: plain/validation-*
- config_name: styled
  data_files:
  - split: train
    path: styled/train-*
  - split: validation
    path: styled/validation-*
license: cc-by-4.0
task_categories:
- image-to-text
size_categories:
- 1M<n<10M
---


# Synthetic LaTeX OCR Dataset

![Dataset Type](https://img.shields.io/badge/type-synthetic-blue)
![Format](https://img.shields.io/badge/format-HuggingFace%20JSONL-green)
![License](https://img.shields.io/badge/license-MIT-yellow)

## Overview

This dataset contains synthetically generated LaTeX formula images designed to augment training data for LaTeX OCR models. The dataset applies style enrichment techniques to existing real-world LaTeX datasets, creating diverse visual representations of mathematical formulas through PDF-rendering and font styling.

## Dataset Structure

```
synth/
├── plain/
│   ├── train/
│   │   ├── images/
│   │   │   ├── train_0000000.png
│   │   │   ├── train_0000001.png
│   │   │   └── ...
│   │   └── metadata.jsonl
│   └── validation/
│       ├── images/
│       └── metadata.jsonl
├── styled/
│   ├── train/
│   │   ├── images/
│   │   └── metadata.jsonl
│   └── validation/
│       ├── images/
│       └── metadata.jsonl
└── README.md
```

### Data Format

Each `metadata.jsonl` file contains one JSON object per line:

```json
{"text": "x^2 + y^2 = z^2", "file_name": "images/train_0000001.png"}
{"text": "\\frac{a}{b} + \\frac{c}{d}", "file_name": "images/train_0000002.png"}
{"text": "\\int_{0}^{\\infty} e^{-x} dx", "file_name": "images/train_0000003.png"}
```

**Fields:**
- `text` (str): LaTeX formula string
- `file_name` (str): Relative path to the image file

## Dataset Statistics

### Plain Dataset

| Split | Samples | Avg Length | Median Length | Min Length | Max Length | Std Length | P95 Length | P99 Length |
|-------|---------|------------|---------------|------------|------------|------------|------------|------------|
| Train | 1,143,025 | 86.1 chars | 35.0 chars | 1 chars | 7,058 chars | 142.3 chars | 341.0 chars | 651.0 chars |
| Validation | 30,461 | 136.4 chars | 60.0 chars | 1 chars | 3,257 chars | 195.1 chars | 490.0 chars | 894.0 chars |
| **Total** | **1,173,486** | - | - | - | - | - | - | - |

### Styled Dataset

| Split | Samples | Avg Length | Median Length | Min Length | Max Length | Std Length | P95 Length | P99 Length |
|-------|---------|------------|---------------|------------|------------|------------|------------|------------|
| Train | 154,616 | 75.0 chars | 49.0 chars | 10 chars | 3,111 chars | 80.9 chars | 237.0 chars | 367.0 chars |
| Validation | 4,133 | 98.1 chars | 71.0 chars | 10 chars | 845 chars | 87.6 chars | 291.4 chars | 400.0 chars |
| **Total** | **158,749** | - | - | - | - | - | - | - |

### Combined Statistics

- **Total Plain Dataset**: 1,173,486 samples
- **Total Styled Dataset**: 158,749 samples
- **Grand Total**: 1,332,235 samples

## Dataset Constitution

This synthetic dataset is generated from the following sources:

### 1. Source Datasets

#### UniMER-1M
- **Description**: Large-scale mathematical expression recognition dataset
- **Source**: UniMER-1M training set
- **Formulas**: `XXX,XXX` mathematical expressions
- **Coverage**: Diverse mathematical notation including algebra, calculus, geometry, and advanced mathematics

#### pix2tex Dataset
- **Description**: HuggingFace pix2tex LaTeX OCR dataset
- **Source**: Loaded automatically from HuggingFace Hub
- **Splits**: Training + Validation splits
- **Formulas**: LaTeX expressions from academic sources
- **Coverage**: Academic papers, textbooks, and research documents

### 2. Synthetic Plain Dataset

**Generation Method**: PDF-style rendering without font styling

- **Formula Source**: Combined UniMER-1M + pix2tex datasets
- **Processing**:
  - Sanitized formulas with comprehensive space normalization
  - Normalized legacy LaTeX commands (`\bf` → `\mathbf`, etc.)
  - Rendered using MathJax + CairoSVG pipeline (LaTeX → SVG → PDF → PNG)
  - Rasterized at 150 DPI for screenshot-style images
  - RGB format with white background
- **Train/Validation Split**: Train from `unimer_train` + `pix2tex_train`; Validation from `unimer_test` + `pix2tex_val`
- **Usage in Training**: 20% random subset used in mixed training dataset

**Characteristics:**
- Clean, PDF-quality rendering
- Consistent font style (default LaTeX fonts)
- Minimal visual variation
- Serves as baseline/anchor for style robustness

### 3. Synthetic Styled Dataset

**Generation Method**: Style-enriched PDF rendering with font macros

- **Formula Source**: Combined UniMER-1M + pix2tex datasets
- **Style Injection Strategy** (Section 3.2):
  - Random injection of `\mathxx` font macros (`\mathbf`, `\mathbb`, `\mathcal`, `\mathit`, `\mathrm`, `\mathsf`, `\mathtt`, `\mathfrak`, `\mathscr`)
  - Semantic heuristics for variable types:
    - Sets (R, C, N, Z, Q): 35% `\mathbb`, 15% `\mathcal`, 50% plain
    - Vectors (x, y, z, u, v, w, A, B, M): 30% `\mathbf`, 10% `\mathit`, 60% plain
    - Operators (d, e, i): 20% `\mathrm`, 80% plain
    - Generic: 2% each specialty font, 90% plain
  - Global cap: ~40% of identifiers styled per formula
  - Consistency rule: Same variable gets same style throughout formula
- **Styled Formula Ratio**: ~10-50% of original formulas produce distinct styled variants
- **Rendering**: MathJax with full TeX package support (amsmath, amssymb, mathrsfs, amsfonts)
- **Train/Validation Split**: Train from `unimer_train` + `pix2tex_train`; Validation from `unimer_test` + `pix2tex_val`
- **Usage in Training**: 100% used in mixed training dataset

**Characteristics:**
- Rich visual diversity in mathematical typography
- Realistic font variations from academic publications
- Maintains semantic correctness
- Improves model robustness to font styling

## Generation Pipeline

### Pipeline Steps

1. **Extract Formulas**: Extract LaTeX formulas from UniMER and pix2tex source datasets
2. **Sanitize Formulas**: Apply LaTeX normalization rules for robust tokenization
3. **Inject Macros & Render**: Generate styled variants and render images via MathJax

### Technical Details

1. **Style Injection**: Probability-based identifier selection with semantic heuristics
2. **Rendering**: MathJax (Node.js) for LaTeX → SVG conversion with full TeX package support
3. **Rasterization**: CairoSVG for SVG → PDF conversion, then PDF → PNG at 150 DPI

### Rendering Pipeline

```
LaTeX Formula → MathJax (SVG) → CairoSVG (PDF) → pdf2image (PNG)
```


### Source Dataset Citations

```bibtex
@inproceedings{unimer2024,
  title={UniMER: Universal Mathematical Expression Recognition},
  author={UniMER Authors},
  booktitle={Conference},
  year={2024}
}

@dataset{latex_ocr_dataset,
  title={LaTeX-OCR Dataset},
  author={lukbl},
  year={2023},
  publisher={HuggingFace},
  howpublished={\url{https://huggingface.co/datasets/lukbl/LaTeX-OCR-dataset}}
}
```

## License

This dataset is released under the MIT License. See source dataset licenses for additional restrictions:

- UniMER-1M: [Original License]
- LaTeX-OCR Dataset: [Original License]

The synthetic generation code and methodology are provided under MIT License.

## Changelog

### Version 2.0.0 (Formula Sanitization)

- 🔧 **Added LaTeX formula sanitization** with comprehensive space normalization rules:
  - **Control word boundary protection**: Prevents token merging (e.g., `\mathcal A` stays as `\mathcal A`)
  - **Text-like command preservation**: Maintains spaces in `\text{}`, `\mathrm{}`, `\operatorname{}`, etc.
  - **Control symbol handling**: Preserves spaces after control symbols (`\,`, `\;`, `\\`, etc.)
  - **Verbatim command exemption**: `\verb` and `\lstinline` commands remain untouched
  - **Array/tabular optimization**: Removes unnecessary spaces around `&` and `\\` in structured environments
  - **General space cleanup**: Removes all other redundant whitespace
- 📈 Improved formula consistency and reduced tokenization errors

  **Example:**
  ```latex
  Old:  \mathcal  A  +  \text{ cost }  +  \begin{array}{c} x \\ y \end{array}
  New: \mathcal A+\text{ cost }+\begin{array}{c}x\\y\end{array}
  ```

### Version 1.0.0 (Initial Release)

- ✨ Initial release with plain and styled variants
- ✨ HuggingFace JSONL format
- ✨ Train/validation splits (90/10)
- ✨ Quality-controlled generation pipeline
- 📊 Total samples: 1,332,235 (Plain: 1,173,486 | Styled: 158,749)

