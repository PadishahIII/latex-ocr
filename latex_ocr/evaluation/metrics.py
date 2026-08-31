"""
Evaluation Metrics for LaTeX OCR
Implements BLEU score and Edit Distance for LaTeX formula evaluation.
"""

from typing import List, Dict, Tuple
import numpy as np
from collections import Counter


def _levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein distance (edit distance) between two strings.
    Fallback implementation if python-Levenshtein is not installed.
    """
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


# Try to use fast C implementation, fallback to Python
try:
    from Levenshtein import distance as levenshtein_distance  # type: ignore
except ImportError:
    levenshtein_distance = _levenshtein_distance
    print("Warning: python-Levenshtein not installed. Using Python fallback (slower).")


def tokenize_latex(formula: str) -> List[str]:
    r"""
    Tokenize LaTeX formula into tokens.
    Treats LaTeX commands (\command) as single tokens.

    Args:
        formula: LaTeX formula string

    Returns:
        List of tokens
    """
    tokens = []
    i = 0
    while i < len(formula):
        if formula[i] == "\\":
            # Extract LaTeX command
            j = i + 1
            while j < len(formula) and (formula[j].isalpha() or formula[j] == "_"):
                j += 1
            if j > i + 1:
                tokens.append(formula[i:j])
                i = j
            else:
                tokens.append(formula[i])
                i += 1
        elif formula[i] in [
            "{",
            "}",
            "^",
            "_",
            "(",
            ")",
            "[",
            "]",
            "=",
            "+",
            "-",
            "*",
            "/",
            "|",
        ]:
            # Special characters as separate tokens
            tokens.append(formula[i])
            i += 1
        elif formula[i].isspace():
            # Skip whitespace
            i += 1
        else:
            # Regular characters
            tokens.append(formula[i])
            i += 1
    return tokens


def compute_bleu(
    references: List[str],
    predictions: List[str],
    max_n: int = 4,
    weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25),
) -> Dict[str, float]:
    """
    Compute BLEU score for LaTeX formulas.

    Args:
        references: List of reference (ground truth) formulas
        predictions: List of predicted formulas
        max_n: Maximum n-gram order (default: 4)
        weights: Weights for each n-gram (default: equal weights)

    Returns:
        Dictionary with BLEU scores (bleu-1, bleu-2, bleu-3, bleu-4, bleu)
    """
    assert len(references) == len(predictions), (
        "Must have same number of references and predictions"
    )

    if len(references) == 0:
        return {f"bleu-{i}": 0.0 for i in range(1, max_n + 1)} | {"bleu": 0.0}

    # Tokenize all formulas
    ref_tokens_list = [tokenize_latex(ref) for ref in references]
    pred_tokens_list = [tokenize_latex(pred) for pred in predictions]

    # Compute n-gram precisions
    precisions = []

    for n in range(1, max_n + 1):
        total_pred_ngrams = 0
        total_matched_ngrams = 0

        for ref_tokens, pred_tokens in zip(ref_tokens_list, pred_tokens_list):
            # Generate n-grams
            ref_ngrams = Counter(
                [tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1)]
            )
            pred_ngrams = Counter(
                [tuple(pred_tokens[i : i + n]) for i in range(len(pred_tokens) - n + 1)]
            )

            # Count matches (clipped)
            matched = sum((pred_ngrams & ref_ngrams).values())
            total_matched_ngrams += matched
            total_pred_ngrams += sum(pred_ngrams.values())

        # Compute precision for this n-gram
        precision = (
            total_matched_ngrams / total_pred_ngrams if total_pred_ngrams > 0 else 0.0
        )
        precisions.append(precision)

    # Compute brevity penalty
    total_ref_length = sum(len(ref_tokens) for ref_tokens in ref_tokens_list)
    total_pred_length = sum(len(pred_tokens) for pred_tokens in pred_tokens_list)

    if total_pred_length > total_ref_length:
        brevity_penalty = 1.0
    elif total_pred_length == 0:
        brevity_penalty = 0.0
    else:
        brevity_penalty = np.exp(1 - total_ref_length / total_pred_length)

    # Compute weighted geometric mean of precisions
    log_precisions = [np.log(p) if p > 0 else float("-inf") for p in precisions]
    weighted_log_precision = sum(w * lp for w, lp in zip(weights, log_precisions))

    if weighted_log_precision == float("-inf"):
        bleu_score = 0.0
    else:
        bleu_score = brevity_penalty * np.exp(weighted_log_precision)

    # Return individual BLEU scores and overall BLEU
    results = {f"bleu-{i + 1}": precisions[i] for i in range(max_n)}
    results["bleu"] = bleu_score
    results["brevity_penalty"] = brevity_penalty

    return results


def compute_edit_distance(
    references: List[str],
    predictions: List[str],
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Compute edit distance (Levenshtein distance) for LaTeX formulas.

    Args:
        references: List of reference (ground truth) formulas
        predictions: List of predicted formulas
        normalize: If True, normalize by reference length (returns error rate)

    Returns:
        Dictionary with edit distance metrics
    """
    assert len(references) == len(predictions), (
        "Must have same number of references and predictions"
    )

    if len(references) == 0:
        return {
            "edit_distance": 0.0,
            "normalized_edit_distance": 0.0,
            "exact_match_ratio": 0.0,
        }

    distances = []
    normalized_distances = []
    exact_matches = 0

    for ref, pred in zip(references, predictions):
        # Compute Levenshtein distance
        distance = levenshtein_distance(ref, pred)
        distances.append(distance)

        # Normalize by reference length
        ref_len = len(ref)
        normalized_dist = (
            distance / ref_len if ref_len > 0 else (1.0 if len(pred) > 0 else 0.0)
        )
        normalized_distances.append(normalized_dist)

        # Check exact match
        if distance == 0:
            exact_matches += 1

    results = {
        "edit_distance": np.mean(distances),
        "normalized_edit_distance": np.mean(normalized_distances),
        "exact_match_ratio": exact_matches / len(references),
        "total_samples": len(references),
        "exact_matches": exact_matches,
    }

    return results


def compute_token_level_metrics(
    references: List[str],
    predictions: List[str],
) -> Dict[str, float]:
    """
    Compute token-level accuracy and F1 score.

    Args:
        references: List of reference (ground truth) formulas
        predictions: List of predicted formulas

    Returns:
        Dictionary with token-level metrics
    """
    assert len(references) == len(predictions), (
        "Must have same number of references and predictions"
    )

    if len(references) == 0:
        return {
            "token_accuracy": 0.0,
            "token_precision": 0.0,
            "token_recall": 0.0,
            "token_f1": 0.0,
        }

    correct_tokens = 0
    total_pred_tokens = 0
    total_ref_tokens = 0

    for ref, pred in zip(references, predictions):
        ref_tokens = tokenize_latex(ref)
        pred_tokens = tokenize_latex(pred)

        # Use edit distance to find aligned tokens
        # For simplicity, we use character-level alignment
        ref_set = set(ref_tokens)
        pred_set = set(pred_tokens)

        intersection = len(ref_set & pred_set)

        total_ref_tokens += len(ref_tokens)
        total_pred_tokens += len(pred_tokens)
        correct_tokens += intersection

    # Compute metrics
    precision = correct_tokens / total_pred_tokens if total_pred_tokens > 0 else 0.0
    recall = correct_tokens / total_ref_tokens if total_ref_tokens > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    results = {
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
    }

    return results


def compute_all_metrics(
    references: List[str],
    predictions: List[str],
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        references: List of reference (ground truth) formulas
        predictions: List of predicted formulas

    Returns:
        Dictionary with all metrics
    """
    metrics = {}

    # BLEU scores
    bleu_scores = compute_bleu(references, predictions)
    metrics.update(bleu_scores)

    # Edit distance
    edit_scores = compute_edit_distance(references, predictions)
    metrics.update(edit_scores)

    # Token-level metrics
    token_scores = compute_token_level_metrics(references, predictions)
    metrics.update(token_scores)

    return metrics


if __name__ == "__main__":
    """Test the metrics"""

    # Test data
    references = [
        r"x^2 + y^2 = z^2",
        r"\frac{a}{b} + \frac{c}{d}",
        r"\int_{0}^{\infty} e^{-x} dx",
    ]

    predictions = [
        r"x^2 + y^2 = z^2",  # Perfect match
        r"\frac{a}{b} + \frac{d}{c}",  # Slight error
        r"\int_{0}^{1} e^{-x} dx",  # Different limits
    ]

    print("=" * 80)
    print("METRICS TEST")
    print("=" * 80)
    print()

    # Test BLEU
    print("BLEU Scores:")
    bleu_results = compute_bleu(references, predictions)
    for key, value in bleu_results.items():
        print(f"  {key}: {value:.4f}")
    print()

    # Test Edit Distance
    print("Edit Distance:")
    edit_results = compute_edit_distance(references, predictions)
    for key, value in edit_results.items():
        print(f"  {key}: {value:.4f}")
    print()

    # Test Token-level Metrics
    print("Token-level Metrics:")
    token_results = compute_token_level_metrics(references, predictions)
    for key, value in token_results.items():
        print(f"  {key}: {value:.4f}")
    print()

    # Test All Metrics
    print("All Metrics Combined:")
    all_metrics = compute_all_metrics(references, predictions)
    for key, value in all_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
