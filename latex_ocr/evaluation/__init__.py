"""
Evaluation module for LaTeX OCR models.
"""

from latex_ocr.evaluation.metrics import (
    compute_bleu,
    compute_edit_distance,
    compute_token_level_metrics,
    compute_all_metrics,
    tokenize_latex,
)

from latex_ocr.evaluation.evaluator import (
    EvaluationConfig,
    DatasetEvaluationResult,
    LaTeXOCREvaluator,
    create_evaluator,
)

__all__ = [
    # Metrics
    "compute_bleu",
    "compute_edit_distance",
    "compute_token_level_metrics",
    "compute_all_metrics",
    "tokenize_latex",
    # Evaluator
    "EvaluationConfig",
    "DatasetEvaluationResult",
    "LaTeXOCREvaluator",
    "create_evaluator",
]
