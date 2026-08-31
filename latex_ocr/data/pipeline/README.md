# Prerequisite
## Install Latex Tools
On Mac:
```bash
brew install --cask mactex
eval "$(/usr/libexec/path_helper)"
brew install poppler
```
On linux:
```bash
sudo apt update
sudo apt install texlive-xetex --fix-missing
sudo apt install poppler-utils

```

To verify the installation, run:
```bash
xelatex --version


```

## Install HF CLI
```bash
curl -LsSf https://hf.co/cli/install.sh | bash
# or
uv tool install hf
```

---

# Data Processing Pipeline

This section describes the complete pipeline for generating the LaTeX OCR synthetic dataset.

## Overview

```
┌─────────────────────────┐     ┌─────────────────────────┐
│  UniMER Dataset         │     │  pix2tex Dataset        │
│  (local download)       │     │  (loaded from HF)       │
└───────────┬─────────────┘     └───────────┬─────────────┘
            │                               │
            └───────────┬───────────────────┘
                        ▼
            ┌───────────────────────┐
            │ 1. Extract Formulas   │
            │    (extract_formula_  │
            │     corpus.py)        │
            └───────────┬───────────┘
                        ▼
            ┌───────────────────────┐
            │ 2. Sanitize Formulas  │
            │    (sanitize_formula. │
            │     py)               │
            └───────────┬───────────┘
                        ▼
            ┌───────────────────────┐
            │ 3. Inject Macros &    │
            │    Render Images      │
            │    (syntax_enrich.py) │
            └───────────┬───────────┘
                        ▼
            ┌───────────────────────┐
            │ 4. Validate & Upload  │
            │    (upload_synth.py)  │
            └───────────────────────┘
```

---

## Step 1: Install the Original Datasets

### UniMER Dataset (local download required)

Follow the instructions in [datasets/README.md](../../datasets/README.md):

```bash
cd latex_ocr/datasets

# Option 1: Using git-xet
brew install git-xet
git xet install
git clone https://huggingface.co/datasets/wanderkid/UniMER_Dataset

# Option 2: Using HuggingFace CLI
uv tool install hf
hf download wanderkid/UniMER_Dataset --repo-type=dataset --local-dir UniMER_Dataset

# Uncompress the dataset
cd UniMER_Dataset
unzip UniMER-1M.zip
unzip UniMER-Test.zip
```

### pix2tex Dataset (auto-loaded from HuggingFace)

No manual download required — the pipeline automatically loads it from HuggingFace Hub.

---

## Step 2: Extract Formula Corpus

Extract LaTeX formulas from the source datasets into JSON format for processing.

```bash
# Extract UniMER formulas (train + test splits)
PYTHONPATH=. uv run python latex_ocr/data/pipeline/extract_formula_corpus.py extract-unimer

# Extract pix2tex formulas (train + validation splits)
PYTHONPATH=. uv run python latex_ocr/data/pipeline/extract_formula_corpus.py extract-pix2tex

# (Optional) Generate macro statistics
PYTHONPATH=. uv run python latex_ocr/data/pipeline/extract_formula_corpus.py stats \
    --corpus latex_ocr/datasets/formula_corpus.txt
```

**Output files** (in `latex_ocr/datasets/formula_corpora/`):
- `unimer_train.json` — UniMER-1M training formulas
- `unimer_test.json` — UniMER test formulas
- `pix2tex_train.json` — pix2tex training formulas
- `pix2tex_val.json` — pix2tex validation formulas

---

## Step 3: Sanitize Formulas

Apply LaTeX normalization rules to ensure robust tokenization and prevent rendering errors.

```bash
# Sanitize UniMER training formulas
PYTHONPATH=. uv run python latex_ocr/data/pipeline/sanitize_formula.py sanitize \
    --input unimer_train.json

# Sanitize UniMER test formulas
PYTHONPATH=. uv run python latex_ocr/data/pipeline/sanitize_formula.py sanitize \
    --input unimer_test.json

# Sanitize pix2tex training formulas
PYTHONPATH=. uv run python latex_ocr/data/pipeline/sanitize_formula.py sanitize \
    --input pix2tex_train.json

# Sanitize pix2tex validation formulas
PYTHONPATH=. uv run python latex_ocr/data/pipeline/sanitize_formula.py sanitize \
    --input pix2tex_val.json

# (Optional) Run sanitization tests
PYTHONPATH=. uv run python latex_ocr/data/pipeline/sanitize_formula.py sanitize --test
```

**Output files** (in `latex_ocr/datasets/formula_corpora/`):
- `unimer_train_sanitized.json`
- `unimer_test_sanitized.json`
- `pix2tex_train_sanitized.json`
- `pix2tex_val_sanitized.json`

---

## Step 4: Inject Macros & Render Images

Generate the synthetic dataset by injecting font macros (e.g., `\mathbf`, `\mathbb`) and rendering images.

```bash
# Generate training data from UniMER
PYTHONPATH=. uv run python latex_ocr/data/pipeline/syntax_enrich.py \
    --output_root latex_ocr/datasets \
    --corpus-file unimer_train_sanitized.json \
    --num_workers 16 \
    --split train

# Generate training data from pix2tex
PYTHONPATH=. uv run python latex_ocr/data/pipeline/syntax_enrich.py \
    --output_root latex_ocr/datasets \
    --corpus-file pix2tex_train_sanitized.json \
    --num_workers 16 \
    --split train

# Generate validation data from UniMER test
PYTHONPATH=. uv run python latex_ocr/data/pipeline/syntax_enrich.py \
    --output_root latex_ocr/datasets \
    --corpus-file unimer_test_sanitized.json \
    --num_workers 16 \
    --split validation

# Generate validation data from pix2tex
PYTHONPATH=. uv run python latex_ocr/data/pipeline/syntax_enrich.py \
    --output_root latex_ocr/datasets \
    --corpus-file pix2tex_val_sanitized.json \
    --num_workers 16 \
    --split validation
```

### Running in Background

For long-running jobs, use `nohup`:

```bash
PYTHONPATH=. nohup uv run python latex_ocr/data/pipeline/syntax_enrich.py \
    --output_root latex_ocr/datasets \
    --corpus-file unimer_train_sanitized.json \
    --num_workers 16 \
    --split train &> syntax_enrich_unimer_train.log &
```

### Resuming Interrupted Jobs

If the process is interrupted, resume from a specific batch:

```bash
PYTHONPATH=. uv run python latex_ocr/data/pipeline/syntax_enrich.py \
    --output_root latex_ocr/datasets \
    --corpus-file unimer_train_sanitized.json \
    --num_workers 16 \
    --split train \
    --resume-from-batch 50  # Resume from batch 50
```

**Output structure**:
```
latex_ocr/datasets/synth/
├── plain/
│   ├── train/
│   │   ├── images/
│   │   │   ├── train_0000001.png
│   │   │   └── ...
│   │   └── metadata.jsonl
│   └── validation/
│       ├── images/
│       └── metadata.jsonl
└── styled/
    ├── train/
    │   ├── images/
    │   └── metadata.jsonl
    └── validation/
        ├── images/
        └── metadata.jsonl
```

---

## Step 5: Validate the Dataset (Optional)

Visualize randomly sampled data points to validate the generated dataset.

```bash
# detect image duplicate, which would lead to image-formula misalignment
PYTHONPATH=. uv run python latex_ocr/data/pipeline/detect_duplicates.py \
    --data_root latex_ocr/datasets/synth

# Install streamlit if needed
uv pip install streamlit pillow datasets

# Run the visualization app (loads from HuggingFace Hub after upload)
PYTHONPATH=. uv run streamlit run latex_ocr/data/pipeline/visualize_samples.py -- \
    --num_samples 100 \
    --split train

# Visualize validation split
PYTHONPATH=. uv run streamlit run latex_ocr/data/pipeline/visualize_samples.py -- \
    --num_samples 100 \
    --split validation
```

This opens a Streamlit web app where you can:
- View random samples from both plain and styled datasets
- Filter by formula length
- Compare statistics between datasets

---

## Step 6: Compute Dataset Statistics

Before uploading, compute and review the dataset statistics to update the dataset's README on HuggingFace Hub.

```bash
# Compute statistics for plain and styled datasets
PYTHONPATH=. uv run python latex_ocr/data/pipeline/compute_dataset_stats.py \
    --plain_root latex_ocr/datasets/synth/plain \
    --styled_root latex_ocr/datasets/synth/styled

# Generate histogram visualization
PYTHONPATH=. uv run python latex_ocr/data/pipeline/compute_dataset_stats.py \
    --plain_root latex_ocr/datasets/synth/plain \
    --styled_root latex_ocr/datasets/synth/styled \
    --histogram_output latex_ocr/datasets/synth/length_distribution.png
```

This outputs:
- **Sample counts** for each split (train/validation)
- **Length statistics**: avg, median, min, max, std, **P95**, **P99**
- **Top-10 longest formulas** with corresponding image paths
- **Histogram** showing the length distribution (if `--histogram_output` is specified)

Use this information to update the dataset README on HuggingFace Hub with accurate statistics.

---

## Step 7: Upload to HuggingFace Hub

Upload the generated dataset to HuggingFace Hub in parquet format.

```bash
# Upload both plain and styled datasets with custom commit message
PYTHONPATH=. uv run python latex_ocr/data/pipeline/upload_synth.py \
    --message "Add v2.0 dataset with improved rendering"

# Upload only plain dataset
PYTHONPATH=. uv run python latex_ocr/data/pipeline/upload_synth.py \
    --config plain \
    --message "Update plain dataset"

# Upload only styled dataset
PYTHONPATH=. uv run python latex_ocr/data/pipeline/upload_synth.py \
    --config styled \
    --message "Update styled dataset"

# Upload with auto-generated commit message
PYTHONPATH=. uv run python latex_ocr/data/pipeline/upload_synth.py
```

---

## Quick Reference: Full Pipeline

```bash
# 1. Download UniMER dataset
cd latex_ocr/datasets
hf download wanderkid/UniMER_Dataset --repo-type=dataset --local-dir UniMER_Dataset
cd UniMER_Dataset && unzip UniMER-1M.zip && unzip UniMER-Test.zip && cd ..

# 2. Extract formulas
cd ../..  # Back to project root
PYTHONPATH=. uv run python latex_ocr/data/pipeline/extract_formula_corpus.py extract-unimer
PYTHONPATH=. uv run python latex_ocr/data/pipeline/extract_formula_corpus.py extract-pix2tex

# 3. Sanitize formulas
PYTHONPATH=. uv run python latex_ocr/data/pipeline/sanitize_formula.py sanitize --input unimer_train.json
PYTHONPATH=. uv run python latex_ocr/data/pipeline/sanitize_formula.py sanitize --input unimer_test.json
PYTHONPATH=. uv run python latex_ocr/data/pipeline/sanitize_formula.py sanitize --input pix2tex_train.json
PYTHONPATH=. uv run python latex_ocr/data/pipeline/sanitize_formula.py sanitize --input pix2tex_val.json

# 4. Generate synthetic dataset (run in background)
PYTHONPATH=. nohup uv run python latex_ocr/data/pipeline/syntax_enrich.py \
    --output_root latex_ocr/datasets \
    --corpus-file unimer_train_sanitized.json,pix2tex_train_sanitized.json \
    --num_workers 16 --split train &> train.log &

PYTHONPATH=. nohup uv run python latex_ocr/data/pipeline/syntax_enrich.py \
    --output_root latex_ocr/datasets \
    --corpus-file unimer_test_sanitized.json,pix2tex_val_sanitized.json \
    --num_workers 16 --split validation &> val.log &

# 5. (Optional) Visualize samples
PYTHONPATH=. uv run streamlit run latex_ocr/data/pipeline/visualize_samples.py -- --num_samples 100

# 6. Compute dataset statistics (update HF README with these values)
PYTHONPATH=. uv run python latex_ocr/data/pipeline/compute_dataset_stats.py \
    --plain_root latex_ocr/datasets/synth/plain \
    --styled_root latex_ocr/datasets/synth/styled \
    --histogram_output latex_ocr/datasets/synth/length_distribution.png

# 7. Upload to HuggingFace Hub
PYTHONPATH=. uv run python latex_ocr/data/pipeline/upload_synth.py -m "Dataset v2.0"
```
