# LaTeX OCR Offline Evaluation

Comprehensive offline evaluation module for LaTeX OCR models with support for:
- Multiple evaluation metrics (BLEU, Edit Distance, Token-level accuracy)
- Separate evaluation of mixed dataset parts (real, synthetic plain, synthetic styled)
- Customizable inference functions
- Detailed result logging and visualization

## Features

### 📊 Metrics Implemented

1. **BLEU Scores** (`metrics.py`)
   - BLEU-1, BLEU-2, BLEU-3, BLEU-4
   - Overall BLEU with brevity penalty
   - LaTeX-aware tokenization

2. **Edit Distance** (`metrics.py`)
   - Levenshtein distance (character-level)
   - Normalized edit distance (error rate)
   - Exact match ratio
   - Supports fast C implementation (python-Levenshtein) with Python fallback

3. **Token-level Metrics** (`metrics.py`)
   - Token precision, recall, F1
   - LaTeX command-aware tokenization

### 🔍 Evaluation Capabilities

- **Mixed Dataset Evaluation**: Separate evaluation for real, synthetic plain, and synthetic styled parts
- **Custom Inference**: Support for custom inference functions
- **Batch Processing**: Efficient batch-wise evaluation
- **Result Persistence**: JSON export with detailed predictions
- **Progress Tracking**: tqdm progress bars

## Installation

```bash
# Required
pip install torch torchvision numpy tqdm pydantic

# Optional (for faster edit distance)
pip install python-Levenshtein
```

## Quick Start

### Basic Usage

```python
from latex_ocr.evaluation import (
    LaTeXOCREvaluator,
    EvaluationConfig,
)

# Create evaluation config
config = EvaluationConfig(
    model_path="path/to/model.pth",
    output_dir="latex_ocr/output/evaluation",
    batch_size=32,
    max_samples=1000,  # Optional: limit for quick testing
)

# Create evaluator
evaluator = LaTeXOCREvaluator(model=your_model, config=config)

# Evaluate on a dataset
result = evaluator.evaluate_dataset(
    dataset=your_dataset,
    dataset_name="test_set",
)

# Print results
evaluator.print_results()

# Save results
evaluator.save_results()
```

### Evaluate Mixed Dataset Parts Separately

```python
from latex_ocr.data.loader import MixedTrainingDataset

# Assume you have a mixed_dataset
results = evaluator.evaluate_mixed_dataset(
    mixed_dataset=mixed_dataset,
    evaluate_parts=True,  # Evaluate real, plain, styled separately
)

# Results will contain:
# - results["real"]: Real dataset performance
# - results["synthetic_plain"]: Synthetic plain (20% subset) performance
# - results["synthetic_styled"]: Synthetic styled performance  
# - results["mixed_full"]: Full mixed dataset performance
```

### Custom Inference Function

```python
def custom_inference(model, batch):
    """Custom inference with post-processing"""
    images = batch["image"].to("cuda")
    
    with torch.no_grad():
        outputs = model(images)
        predictions = model.decode(outputs)
        
        # Add custom post-processing
        predictions = [pred.strip() for pred in predictions]
    
    return predictions

evaluator = LaTeXOCREvaluator(
    model=model,
    config=config,
    inference_fn=custom_inference,
)
```

## Module Structure

```
evaluation/
├── __init__.py           # Package exports
├── metrics.py            # Metric implementations (BLEU, Edit Distance, Token-level)
├── evaluator.py          # Main evaluator class
├── example_evaluation.py # Usage examples
└── README.md            # This file
```

## Metrics Details

### BLEU Score
- Uses LaTeX-aware tokenization (treats `\command` as single token)
- Computes n-gram precision (n=1,2,3,4)
- Applies brevity penalty for short predictions
- Weighted geometric mean of precisions

### Edit Distance
- Character-level Levenshtein distance
- Normalized by reference length for error rate
- Exact match counting
- Fast C implementation via python-Levenshtein (optional)

### Token-level Metrics
- LaTeX command-aware tokenization
- Set-based precision, recall, F1
- Useful for partial credit evaluation

## Output Format

### Console Output
```
================================================================================
EVALUATION RESULTS
================================================================================

REAL:
  Samples: 1000
  Metrics:
    BLEU Scores:
      bleu-1: 0.9234
      bleu-2: 0.8567
      bleu-3: 0.7891
      bleu-4: 0.7234
      bleu: 0.8123
    Edit Distance:
      edit_distance: 3.45
      normalized_edit_distance: 0.0892
      exact_match_ratio: 0.6700
    Token-level:
      token_precision: 0.9123
      token_recall: 0.9034
      token_f1: 0.9078
```

### JSON Output
```json
{
  "timestamp": "2025-12-02T10:30:00",
  "config": {
    "model_path": "model.pth",
    "batch_size": 32,
    "device": "cuda"
  },
  "results": {
    "real": {
      "name": "real",
      "num_samples": 1000,
      "metrics": {
        "bleu": 0.8123,
        "edit_distance": 3.45,
        "exact_match_ratio": 0.67
      }
    }
  }
}
```

### Predictions Output (JSONL)
```jsonl
{"sample_id": 0, "reference": "x^2 + y^2", "prediction": "x^2 + y^2", "match": true}
{"sample_id": 1, "reference": "\\frac{a}{b}", "prediction": "\\frac{a}{c}", "match": false}
```

## Examples

See `example_evaluation.py` for complete examples:

```bash
# Example 1: Evaluate mixed dataset with separate parts
PYTHONPATH=. python latex_ocr/evaluation/example_evaluation.py --example 1

# Example 2: Evaluate single dataset
PYTHONPATH=. python latex_ocr/evaluation/example_evaluation.py --example 2

# Example 3: Custom inference function
PYTHONPATH=. python latex_ocr/evaluation/example_evaluation.py --example 3
```

## Advanced Usage

### Evaluate Multiple Datasets

```python
eval_datasets = {
    "SPE": spe_dataset,
    "CPE": cpe_dataset,
    "SCE": sce_dataset,
}

all_results = {}
for name, dataset in eval_datasets.items():
    result = evaluator.evaluate_dataset(dataset, name)
    all_results[name] = result

evaluator.print_results(all_results)
evaluator.save_results(results=all_results)
```

### Custom Metrics

```python
from latex_ocr.evaluation.metrics import tokenize_latex

# Use tokenizer for custom analysis
formula = r"\frac{a}{b} + \sum_{i=1}^{n} x_i"
tokens = tokenize_latex(formula)
print(tokens)  # ['\\frac', '{', 'a', '}', '{', 'b', '}', ...]
```

## Performance Tips

1. **Use python-Levenshtein**: Install for 10-100x faster edit distance computation
2. **Batch size**: Larger batches (64-128) for faster evaluation
3. **Num workers**: Use 4-8 workers for parallel data loading
4. **Max samples**: Use `max_samples` for quick testing/debugging

## Citation

If you use this evaluation code, please cite:

```bibtex
@software{latex_ocr_eval,
  title = {LaTeX OCR Offline Evaluation},
  year = {2025},
  author = {Your Name},
}
```

## License

MIT License
