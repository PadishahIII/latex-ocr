"""
Offline Evaluation for LaTeX OCR Models
Supports evaluation on separate dataset parts (real, plain, styled) from MixedTrainingDataset.
"""

from pathlib import Path
from typing import Dict, List, Optional, Callable
import json
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

from latex_ocr.evaluation.metrics import compute_all_metrics
from latex_ocr.data.loader import MixedTrainingDataset


class EvaluationConfig(BaseModel):
    """
    Configuration for LaTeX OCR model evaluation.

    This Pydantic model defines all parameters needed to run evaluation on a dataset,
    including batch processing, device selection, and output options.

    Attributes:
        output_dir: Directory path where evaluation results will be saved. Includes
                   JSON summaries and optional per-sample predictions. Default creates
                   a directory in latex_ocr/output/evaluation.

        batch_size: Number of samples to process in each batch during evaluation.
                   Larger values are faster but require more memory. Default: 32.

        num_workers: Number of worker processes for DataLoader. More workers speed up
                    data loading but consume more memory. Use 0 for single-process.
                    Default: 4.

        device: PyTorch device for model inference ("cuda", "cpu", "mps", etc.).
               Automatically defaults to "cuda" if available, otherwise "cpu".

        max_samples: Optional limit on number of samples to evaluate per dataset.
                    Useful for quick testing or debugging. None means evaluate all.
                    Default: None (evaluate all samples).

        save_predictions: Whether to save detailed per-sample predictions to JSONL files.
                         When True, creates separate files for each dataset part with
                         predictions, references, and match status. Default: True.

    Example Construction:
        Basic usage with defaults:

        >>> config = EvaluationConfig(output_dir="results/eval_run_1")

        Custom configuration:

        >>> config = EvaluationConfig(
        ...     output_dir="experiments/model_v2/evaluation",
        ...     batch_size=64,
        ...     num_workers=8,
        ...     device="cuda:1",
        ...     max_samples=1000,
        ...     save_predictions=True
        ... )

        Quick testing configuration:

        >>> config = EvaluationConfig(
        ...     output_dir="debug/quick_test",
        ...     batch_size=8,
        ...     max_samples=10,
        ...     save_predictions=False
        ... )

        CPU-only evaluation:

        >>> config = EvaluationConfig(
        ...     output_dir="results/cpu_eval",
        ...     device="cpu",
        ...     num_workers=0  # Single process for CPU
        ... )

    Validation:
        - output_dir is automatically created if it doesn't exist
        - batch_size must be positive
        - num_workers must be non-negative
        - max_samples must be positive if specified

    Notes:
        - The output directory is created automatically when the config is instantiated
        - Device string can be any valid PyTorch device identifier
        - For large datasets, consider setting max_samples for quick validation runs
    """

    output_dir: str = Field(
        default="latex_ocr/output/evaluation",
        description="Directory to save evaluation results",
    )
    batch_size: int = Field(default=32, gt=0, description="Batch size for evaluation")
    num_workers: int = Field(
        default=4, ge=0, description="Number of DataLoader workers"
    )
    device: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu",
        description="Device for model inference",
    )
    max_samples: Optional[int] = Field(
        default=None, gt=0, description="Optional limit on samples to evaluate"
    )
    save_predictions: bool = Field(
        default=True, description="Save detailed predictions to JSONL"
    )

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        frozen=False,
    )

    @model_validator(mode="after")
    def create_output_dir(self) -> "EvaluationConfig":
        """Create output directory if it doesn't exist."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        return self


class DatasetEvaluationResult(BaseModel):
    """
    Results container for evaluation of a single dataset part.

    This Pydantic model stores comprehensive evaluation results including metrics,
    predictions, references, and sample metadata for a specific dataset evaluation run.

    Attributes:
        name: Human-readable identifier for this dataset part (e.g., "real", "synthetic_plain",
              "synthetic_styled", "mixed_full"). Used for organizing and reporting results.

        num_samples: Total number of samples evaluated in this dataset part. Should match
                    the length of predictions, references, and sample_ids lists.

        metrics: Dictionary mapping metric names to their computed values. Common metrics include:
                - "bleu_1", "bleu_2", "bleu_3", "bleu_4": BLEU scores at different n-gram levels
                - "exact_match": Proportion of predictions that exactly match references
                - "edit_distance": Average normalized edit distance between predictions and references
                - "token_accuracy": Token-level accuracy
                All metric values are floats typically in range [0, 1] or [0, 100].

        predictions: List of model-predicted LaTeX formulas, one per sample. Order must match
                    references and sample_ids. Empty strings indicate prediction failures.

        references: List of ground-truth LaTeX formulas from the dataset. These are the target
                   formulas the model should produce. Order must match predictions and sample_ids.

        sample_ids: List of integer sample identifiers, typically sequential indices from the
                   original dataset. Useful for tracking which samples were evaluated and
                   correlating results back to the source data.

    Example Construction:
        Basic usage with minimal data:

        >>> result = DatasetEvaluationResult(
        ...     name="validation_set",
        ...     num_samples=100,
        ...     metrics={"bleu_4": 0.75, "exact_match": 0.68},
        ...     predictions=["x^2", "\\frac{1}{2}", ...],  # 100 predictions
        ...     references=["x^2", "\\frac{1}{2}", ...],   # 100 references
        ...     sample_ids=[0, 1, 2, ..., 99]
        ... )

        Construction after model evaluation:

        >>> # After running inference on a dataset
        >>> predictions = model.predict(dataset)
        >>> references = [sample["formula"] for sample in dataset]
        >>> metrics = compute_all_metrics(references, predictions)
        >>>
        >>> result = DatasetEvaluationResult(
        ...     name="test_dataset",
        ...     num_samples=len(predictions),
        ...     metrics=metrics,
        ...     predictions=predictions,
        ...     references=references,
        ...     sample_ids=list(range(len(predictions)))
        ... )

        Using default factory for empty initialization:

        >>> # Create empty result and populate incrementally
        >>> result = DatasetEvaluationResult(
        ...     name="streaming_eval",
        ...     num_samples=0  # Will update as we process
        ... )
        >>> # Later, add results
        >>> result.predictions.append("x + y")
        >>> result.references.append("x + y")
        >>> result.sample_ids.append(0)
        >>> result.num_samples = len(result.predictions)

        Batch evaluation pattern:

        >>> result = DatasetEvaluationResult(name="batch_eval", num_samples=0)
        >>> for batch_idx, (images, formulas) in enumerate(dataloader):
        ...     batch_preds = model(images)
        ...     result.predictions.extend(batch_preds)
        ...     result.references.extend(formulas)
        ...     start_idx = batch_idx * batch_size
        ...     result.sample_ids.extend(range(start_idx, start_idx + len(batch_preds)))
        >>> result.num_samples = len(result.predictions)
        >>> result.metrics = compute_all_metrics(result.references, result.predictions)

    Serialization:

        >>> # Convert to dictionary (excludes large lists by default)
        >>> summary = result.to_dict()
        >>> print(summary)
        {
            "name": "validation_set",
            "num_samples": 100,
            "metrics": {"bleu_4": 0.75, "exact_match": 0.68},
            "sample_count": 100
        }

        >>> # Full serialization with Pydantic
        >>> json_str = result.model_dump_json(indent=2)
        >>>
        >>> # Deserialize from JSON
        >>> loaded_result = DatasetEvaluationResult.model_validate_json(json_str)

    Notes:
        - The to_dict() method provides a compact summary suitable for JSON export,
          excluding the potentially large predictions and references lists.
        - For full serialization including predictions, use Pydantic's model_dump().
        - All lists (predictions, references, sample_ids) should maintain consistent ordering.
        - The num_samples field should always equal len(predictions) for data consistency.
    """

    name: str
    num_samples: int
    metrics: Dict[str, float] = Field(default_factory=dict)
    predictions: List[str] = Field(default_factory=list)
    references: List[str] = Field(default_factory=list)
    sample_ids: List[int] = Field(default_factory=list)

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        frozen=False,
    )

    def to_dict(self) -> Dict:
        """
        Convert to compact dictionary for JSON serialization.

        Returns a summary dictionary suitable for saving evaluation results,
        excluding the potentially large predictions and references lists.

        Returns:
            Dictionary with keys: name, num_samples, metrics, sample_count
        """
        return {
            "name": self.name,
            "num_samples": self.num_samples,
            "metrics": self.metrics,
            "sample_count": len(self.predictions),
        }


class LaTeXOCREvaluator:
    """
    Evaluator for LaTeX OCR models with support for evaluating
    different dataset parts separately.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: EvaluationConfig,
        inference_fn: Optional[Callable] = None,
    ):
        """
        Initialize evaluator.

        Args:
            model: PyTorch model to evaluate
            config: Evaluation configuration
            inference_fn: Optional custom inference function.
                         Should take (model, batch) and return list of predicted formulas.
                         If None, uses default forward pass.
        """
        self.model = model
        self.config = config
        self.inference_fn = inference_fn or self._default_inference

        # Move model to device and set to eval mode
        self.model.to(self.config.device)
        self.model.eval()

        self.results: Dict[str, DatasetEvaluationResult] = {}

    def _default_inference(self, model: torch.nn.Module, batch: Dict) -> List[str]:
        """
        Default inference function.
        Override this or provide custom inference_fn for your model.

        Args:
            model: The model
            batch: Batch dictionary with 'image' key

        Returns:
            List of predicted LaTeX formulas
        """
        images = batch["image"].to(self.config.device)

        with torch.no_grad():
            # This is a placeholder - adjust based on your model's output format
            outputs = model(images)

            # Assuming model returns logits and you have a decode function
            # You'll need to adapt this to your model's actual interface
            if hasattr(model, "decode"):
                predictions = model.decode(outputs)  # type: ignore
            else:
                # Fallback: just return empty strings
                predictions = [""] * images.shape[0]

        return predictions

    def evaluate_dataset(
        self,
        dataset: Dataset,
        dataset_name: str,
        dataloader: Optional[DataLoader] = None,
    ) -> DatasetEvaluationResult:
        """
        Evaluate model on a single dataset.

        Args:
            dataset: PyTorch Dataset
            dataset_name: Name for this dataset part
            dataloader: Optional pre-configured DataLoader

        Returns:
            Evaluation results for this dataset
        """
        # Create dataloader if not provided
        if dataloader is None:
            # Limit samples if specified
            if self.config.max_samples and len(dataset) > self.config.max_samples:  # type: ignore
                indices = list(range(self.config.max_samples))
                dataset = Subset(dataset, indices)

            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True,
            )

        predictions = []
        references = []
        sample_ids = []

        print(f"\nEvaluating {dataset_name}...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(dataloader, desc=f"Evaluating {dataset_name}")
            ):
                # Get predictions
                batch_predictions = self.inference_fn(self.model, batch)
                predictions.extend(batch_predictions)

                # Get ground truth
                if isinstance(batch, dict) and "formula" in batch:
                    references.extend(batch["formula"])
                else:
                    # Handle tuple format (image, formula)
                    references.extend(batch[1])

                # Track sample IDs
                start_id = batch_idx * self.config.batch_size
                sample_ids.extend(range(start_id, start_id + len(batch_predictions)))

        # Compute metrics
        metrics = compute_all_metrics(references, predictions)

        # Create result object
        result = DatasetEvaluationResult(
            name=dataset_name,
            num_samples=len(predictions),
            metrics=metrics,
            predictions=predictions,
            references=references,
            sample_ids=sample_ids,
        )

        self.results[dataset_name] = result

        return result

    def evaluate_mixed_dataset(
        self,
        mixed_dataset: MixedTrainingDataset,
        evaluate_parts: bool = True,
    ) -> Dict[str, DatasetEvaluationResult]:
        """
        Evaluate on MixedTrainingDataset with separate evaluation for each part.

        Args:
            mixed_dataset: MixedTrainingDataset instance
            evaluate_parts: If True, evaluate real, plain, and styled parts separately

        Returns:
            Dictionary of evaluation results for each part
        """
        results = {}

        if evaluate_parts:
            # Evaluate real dataset
            print("\n" + "=" * 80)
            print("Evaluating Real Dataset Part")
            print("=" * 80)
            real_result = self.evaluate_dataset(
                mixed_dataset.real,
                dataset_name="real",
            )
            results["real"] = real_result

            # Evaluate plain dataset (note: this is the 20% subset)
            print("\n" + "=" * 80)
            print("Evaluating Synthetic Plain Dataset Part (20% subset)")
            print("=" * 80)
            plain_result = self.evaluate_dataset(
                mixed_dataset.plain,
                dataset_name="synthetic_plain",
            )
            results["synthetic_plain"] = plain_result

            # Evaluate styled dataset
            print("\n" + "=" * 80)
            print("Evaluating Synthetic Styled Dataset Part")
            print("=" * 80)
            styled_result = self.evaluate_dataset(
                mixed_dataset.styled,
                dataset_name="synthetic_styled",
            )
            results["synthetic_styled"] = styled_result

        # Evaluate full mixed dataset
        print("\n" + "=" * 80)
        print("Evaluating Full Mixed Dataset")
        print("=" * 80)
        full_result = self.evaluate_dataset(
            mixed_dataset.dataset,  # The ConcatDataset
            dataset_name="mixed_full",
        )
        results["mixed_full"] = full_result

        return results

    def print_results(
        self, results: Optional[Dict[str, DatasetEvaluationResult]] = None
    ):
        """
        Print evaluation results in a formatted table.

        Args:
            results: Optional dict of results. If None, uses self.results
        """
        if results is None:
            results = self.results

        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)

        for dataset_name, result in results.items():
            print(f"\n{dataset_name.upper()}:")
            print(f"  Samples: {result.num_samples}")
            print("  Metrics:")

            # Group metrics by type
            bleu_metrics = {k: v for k, v in result.metrics.items() if "bleu" in k}
            edit_metrics = {
                k: v for k, v in result.metrics.items() if "edit" in k or "exact" in k
            }
            token_metrics = {k: v for k, v in result.metrics.items() if "token" in k}

            if bleu_metrics:
                print("    BLEU Scores:")
                for key, value in bleu_metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"      {key}: {value:.4f}")

            if edit_metrics:
                print("    Edit Distance:")
                for key, value in edit_metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"      {key}: {value:.4f}")

            if token_metrics:
                print("    Token-level:")
                for key, value in token_metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"      {key}: {value:.4f}")

    def save_results(
        self,
        output_path: Optional[str] = None,
        results: Optional[Dict[str, DatasetEvaluationResult]] = None,
    ):
        """
        Save evaluation results to JSON file.

        Args:
            output_path: Optional custom output path
            results: Optional dict of results. If None, uses self.results
        """
        if results is None:
            results = self.results

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = (
                f"{self.config.output_dir}/evaluation_results_{timestamp}.json"
            )

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data for JSON serialization
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "output_dir": self.config.output_dir,
                "batch_size": self.config.batch_size,
                "device": self.config.device,
                "max_samples": self.config.max_samples,
                "num_workers": self.config.num_workers,
                "save_predictions": self.config.save_predictions,
            },
            "results": {name: result.to_dict() for name, result in results.items()},
        }

        # Save to file
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✓ Results saved to: {output_path}")

        # Save detailed predictions if requested
        if self.config.save_predictions:
            for dataset_name, result in results.items():
                pred_path = (
                    output_path_obj.parent
                    / f"predictions_{dataset_name}_{output_path_obj.stem}.jsonl"
                )
                with open(pred_path, "w") as f:
                    for idx, (ref, pred, sample_id) in enumerate(
                        zip(result.references, result.predictions, result.sample_ids)
                    ):
                        entry = {
                            "sample_id": sample_id,
                            "reference": ref,
                            "prediction": pred,
                            "match": ref == pred,
                        }
                        f.write(json.dumps(entry) + "\n")

                print(f"✓ Predictions for {dataset_name} saved to: {pred_path}")


def create_evaluator(
    model: torch.nn.Module,
    output_dir: str = "latex_ocr/output/evaluation",
    inference_fn: Optional[Callable] = None,
    **kwargs,
) -> LaTeXOCREvaluator:
    """
    Convenience function to create an evaluator.

    Args:
        model: PyTorch model to evaluate
        output_dir: Directory to save evaluation results
        inference_fn: Optional custom inference function that takes (model, batch)
                     and returns list of predicted formulas
        **kwargs: Additional arguments for EvaluationConfig (batch_size, device, etc.)

    Returns:
        LaTeXOCREvaluator instance ready for evaluation

    Example:
        >>> model = load_model("checkpoint.pth")
        >>> evaluator = create_evaluator(
        ...     model=model,
        ...     output_dir="results/eval_v1",
        ...     batch_size=64,
        ...     device="cuda"
        ... )
    """
    config = EvaluationConfig(
        output_dir=output_dir,
        **kwargs,
    )

    return LaTeXOCREvaluator(model=model, config=config, inference_fn=inference_fn)
