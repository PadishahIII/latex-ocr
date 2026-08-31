"""
Evaluation module for Swin-based models (both Transformer and GRU decoders).

This module provides a unified evaluation interface that works with both model architectures.
"""

import os
import warnings
import traceback
import torch
import click
from typing import Dict, List, Any, Union
from pathlib import Path
from tqdm import tqdm

# Disable tokenizers parallelism warning before any imports that use tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress torch.distributed redirect warning on macOS/Windows
warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed")

from latex_ocr.data.synth.data import (
    TokenizedDataset,
    ConcatTokenizedDataset,
    create_dataloader,
)
from latex_ocr.trainers.config import (
    ModelCfg,
    ModelType,
    ImageToSeqModel,
)
from recipe.utils.train_util import get_device
from latex_ocr.trainers.model import get_model
from recipe.logging.logger import get_logger_by_name
from recipe.utils.mlflow_util import (
    load_configuration,
    load_model,
    load_model_state_dict,
    setup_mlflow,
)

logger = get_logger_by_name("evaluator")


class ModelEvaluator:
    """
    Unified evaluator for Swin-based models.

    Supports both Transformer and GRU decoder architectures.
    """

    def __init__(
        self,
        model: ImageToSeqModel,
        dataset: Union[TokenizedDataset, ConcatTokenizedDataset],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the evaluator.

        Args:
            model: The model to evaluate (SwinTBase or SwinGRUCaptioner)
            dataset: Dataset for evaluation
            device: Device to run evaluation on
        """
        self.model = model.to(device)
        self.model.eval()
        self.dataset = dataset
        self.device = device
        self.tokenizer = dataset.tokenizer

    @torch.no_grad()
    def evaluate_batch(
        self,
        batch: Dict[str, torch.Tensor],
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        beam_size: int = 4,
    ) -> Dict[str, Any]:
        """
        Evaluate a single batch.

        Args:
            batch: Dictionary containing 'images' and 'labels'
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter

        Returns:
            Dictionary with predictions, ground truth, and metrics
        """
        if batch.get("images") is None:
            raise ValueError(
                "Batch has no images (got images=None). Ensure the dataset is created with formula_only=False."
            )

        images = batch["images"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Generate predictions
        # Both models should have a generate method with compatible signatures
        if hasattr(self.model, "generate"):
            # Get special token IDs
            bos_id = getattr(self.tokenizer, "bos_token_id", 1)
            eos_id = getattr(self.tokenizer, "eos_token_id", 2)

            # Check if model is GRU-based (only needs max_len) or Transformer-based
            try:
                if beam_size > 0:
                    generated = self.model.generate(
                        src=images,
                        bos_token_id=bos_id,
                        eos_token_id=eos_id,
                        max_length=max_length,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        beam_size=beam_size,
                    )
                else:
                    generated = self.model.generate(
                        src=images,
                        bos_token_id=bos_id,
                        eos_token_id=eos_id,
                        max_length=max_length,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )

            except TypeError as e:
                # Fallback for GRU model which has simpler generate signature
                logger.warning(
                    f"Discard inference arguments due to error: {e}, traceback: {traceback.format_exc()}"
                )
                generated = self.model.generate(
                    src=images,
                    max_len=max_length,
                )
        else:
            raise NotImplementedError("Model does not have a generate method")

        # Decode predictions and ground truth
        predictions = []
        ground_truths = []

        for pred_ids, label_ids in zip(generated, labels):
            # Decode prediction
            pred_text = self.tokenizer.decode(pred_ids.tolist())
            predictions.append(pred_text)

            # Decode ground truth
            # Filter out padding tokens
            label_ids_filtered = [
                tid for tid in label_ids.tolist() if tid != self.tokenizer.pad_token_id
            ]
            gt_text = self.tokenizer.decode(label_ids_filtered)
            ground_truths.append(gt_text)

        return {
            "predictions": predictions,
            "ground_truths": ground_truths,
            "generated_ids": generated,
            "label_ids": labels,
        }

    def evaluate_dataset(
        self,
        batch_size: int = 8,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        beam_size: int = 4,
        num_workers: int = 4,
    ) -> Dict[str, Any]:
        """
        Evaluate the entire dataset.

        Args:
            batch_size: Batch size for evaluation
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            num_workers: Number of workers for data loading

        Returns:
            Dictionary with all predictions, ground truths, and aggregated metrics
        """
        from torch.utils.data import DataLoader

        from functools import partial
        from latex_ocr.data.synth.data import collate_fn

        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=partial(
                collate_fn,
                padding_value=self.tokenizer.pad_token_id,
                # Generation code here doesn't consume attention_mask.
                return_attention_mask=False,
            ),
        )

        all_predictions = []
        all_ground_truths = []

        logger.info(f"Evaluating {len(dataloader)} batches...")

        for batch in tqdm(dataloader, desc="Evaluating"):
            results = self.evaluate_batch(
                batch,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                beam_size=beam_size,
            )
            all_predictions.extend(results["predictions"])
            all_ground_truths.extend(results["ground_truths"])

        # Calculate metrics
        from latex_ocr.evaluation.metrics import (
            compute_bleu,
            compute_edit_distance,
        )

        bleu_scores = compute_bleu(all_ground_truths, all_predictions)
        edit_scores = compute_edit_distance(all_ground_truths, all_predictions)

        logger.info(
            "BLEU: "
            + ", ".join(
                f"{k}={v:.4f}"
                for k, v in bleu_scores.items()
                if isinstance(v, (int, float))
            )
        )
        logger.info(
            "Edit distance: "
            + ", ".join(
                f"{k}={v:.4f}"
                for k, v in edit_scores.items()
                if isinstance(v, (int, float))
            )
        )

        return {
            "predictions": all_predictions,
            "ground_truths": all_ground_truths,
            "num_samples": len(all_predictions),
            **bleu_scores,
            **edit_scores,
        }

    def save_results(self, results: Dict[str, Any], output_path: str | Path):
        """
        Save evaluation results to a JSON file.

        Args:
            results: Results dictionary from evaluate_dataset
            output_path: Path to save the results
        """
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_path}")


@click.group()
def cli():
    """Evaluation CLI for Swin-based models"""
    pass


@cli.command()
@click.option(
    "--run-id",
    type=str,
    help="MLflow run ID (required; loads config + weights like infer-images)",
)
@click.option(
    "--model-name",
    type=str,
    help="MLflow model artifact name (required with --run-id)",
)
@click.option(
    "--experiment-name",
    type=str,
    help="MLflow experiment name (required with --run-id)",
)
@click.option("--config-file", type=str, default="train-configuration.json")
@click.option("--mlflow-host", type=str, default=None,
              help="MLflow server host; defaults to the MLFLOW_HOST env var (see .env.example).")
@click.option(
    "--config-name",
    type=str,
    default="plain",
    help="Dataset configuration name",
)
@click.option(
    "--split",
    type=str,
    default="validation",
    help="Dataset split to evaluate",
)
@click.option(
    "--batch-size",
    type=int,
    default=8,
    help="Batch size for evaluation",
)
@click.option(
    "--max-length",
    type=int,
    default=512,
    help="Maximum generation length",
)
@click.option(
    "--output",
    type=click.Path(),
    default="data/output/eval_results.json",
    help="Path to save evaluation results",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Device to run evaluation on (e.g. 'cpu', 'cuda', 'cuda:0', 'mps'). Auto-detected if not set.",
)
def evaluate(
    run_id: str | None,
    model_name: str | None,
    experiment_name: str | None,
    config_file: str,
    mlflow_host: str,
    config_name: str,
    split: str,
    batch_size: int,
    max_length: int,
    output: str,
    device: str | None,
):
    """Evaluate a trained model on a dataset."""
    # Load dataset
    logger.info(f"Loading dataset: {config_name}, split: {split}")
    dataset, _ = create_dataloader(
        config_name=config_name,
        split=split,
        batch_size=1,  # Will create our own dataloader
        shuffle=False,
    )

    if not run_id:
        raise click.UsageError("--run-id is required (evaluate loads from MLflow)")
    if not (model_name and experiment_name):
        raise click.UsageError(
            "When using --run-id, also provide --model-name and --experiment-name"
        )

    setup_mlflow(experiment_name, mlflow_host)
    logger.info(
        f"Loading configuration from MLflow for run_id={run_id}, mlflow host: {mlflow_host}"
    )
    loaded_cfg = load_configuration(run_id=run_id, file_name=config_file)
    model_cfg_payload = (
        loaded_cfg.get("model_cfg")
        if isinstance(loaded_cfg, dict)
        and isinstance(loaded_cfg.get("model_cfg"), dict)
        else loaded_cfg
    )
    model_cfg = ModelCfg.model_validate(model_cfg_payload)
    model_cfg.encoder_pretrained = False
    model_cfg.decoder_pretrained = False

    # Keep evaluation overrides consistent
    model_cfg.dropout = 0.0
    model_cfg.max_seq_length = max_length
    model_cfg.vocab_size = dataset.tokenizer.vocab_size
    model_cfg.bos_id = getattr(dataset.tokenizer, "bos_token_id", model_cfg.bos_id)
    model_cfg.eos_id = getattr(dataset.tokenizer, "eos_token_id", model_cfg.eos_id)
    model_cfg.pad_id = getattr(dataset.tokenizer, "pad_token_id", model_cfg.pad_id)

    model = get_model(model_cfg)

    logger.info(
        f"Loading model weights from MLflow (run_id={run_id}, model_name={model_name}), mlflow host: {mlflow_host}"
    )
    # state_dict = load_model_state_dict(run_id, model_name)  # type: ignore[arg-type]
    # model.load_state_dict(state_dict)
    model = load_model(run_id, model_name)
    # model = torch.load(
    #     Path(__file__).parent.parent
    #     / "models"
    #     / "pretrained"
    #     / "finetune_on_converged_pretrain"
    #     / "latex-ocr-coca-finetune_epoch00010"
    #     / "data"
    #     / "model.pth",
    #     weights_only=False,
    #     map_location=torch.device("cpu"),
    # )

    resolved_device = device if device is not None else get_device()
    logger.info(f"Using device: {resolved_device}")

    # Create evaluator and run evaluation
    evaluator = ModelEvaluator(model, dataset, device=resolved_device)
    results = evaluator.evaluate_dataset(
        batch_size=batch_size,
        max_length=max_length,
        beam_size=4,
    )

    # Save results
    evaluator.save_results(results, output)


@cli.command()
@click.option(
    "--run-id",
    type=str,
    help="MLflow run ID to load model from (use with --model-name)",
)
@click.option(
    "--model-name",
    type=str,
    help="Model artifact name in MLflow (use with --run-id)",
)
@click.option(
    "--experiment-name",
    type=str,
    help="MLflow experiment name (required when using --run-id)",
)
@click.option(
    "--image-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Directory containing images to process",
)
@click.option(
    "--batch-size",
    type=int,
    default=8,
    help="Batch size for inference",
)
@click.option(
    "--max-length",
    type=int,
    default=512,
    help="Maximum generation length",
)
@click.option(
    "--output",
    type=click.Path(),
    default="data/output/inference_results.json",
    help="Path to save inference results (JSON with image-formula pairs)",
)
@click.option("--config-file", type=str, default="train-configuration.json")
@click.option("--mlflow-host", type=str, default=None,
              help="MLflow server host; defaults to the MLFLOW_HOST env var (see .env.example).")
@click.option("--temperature", type=float, default=1.0)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Device to run inference on (e.g. 'cpu', 'cuda', 'cuda:0', 'mps'). Auto-detected if not set.",
)
def infer_images(
    run_id: str,
    model_name: str,
    experiment_name: str,
    image_dir: str,
    batch_size: int,
    max_length: int,
    output: str,
    config_file: str,
    mlflow_host: str,
    temperature: float,
    device: str | None,
):
    """Infer formulas from images in a directory"""
    import json
    from PIL import Image
    from latex_ocr.data.synth.loader import get_synthetic_transforms
    from latex_ocr.models.tokenizer.latex_tokenizer import Tokenizer

    # Validate input options
    if not (run_id and model_name and experiment_name):
        logger.error("Must specify --experiment-name --run-id and --model-name ")
        return

    # Get image files
    image_dir_path = Path(image_dir)
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    image_files = sorted(
        [f for f in image_dir_path.iterdir() if f.suffix.lower() in image_extensions]
    )

    if not image_files:
        logger.error(f"No images found in {image_dir}")
        return

    logger.info(f"Found {len(image_files)} images in {image_dir}")

    # Initialize tokenizer
    tokenizer = Tokenizer()
    tokenizer.load(None)

    # Create model config
    setup_mlflow(experiment_name, mlflow_host)

    logger.info(
        f"Loading configuration from MLflow for run_id={run_id}, mlflow host: {mlflow_host}"
    )

    # MLflow configs are typically saved as the full TrainerCfg (top-level keys like
    # `Train`, `Checkpoint`, etc.) with the actual model config nested under `model_cfg`.
    # For backwards compatibility, also accept a direct ModelCfg payload.
    loaded_cfg = load_configuration(run_id=run_id, file_name=config_file)
    model_cfg_payload = (
        loaded_cfg.get("model_cfg")
        if isinstance(loaded_cfg, dict)
        and isinstance(loaded_cfg.get("model_cfg"), dict)
        else loaded_cfg
    )
    model_cfg = ModelCfg.model_validate(model_cfg_payload)
    model_cfg.encoder_pretrained = False
    model_cfg.decoder_pretrained = False

    # Load model using factory
    model = get_model(model_cfg)

    # Load weights
    assert run_id is not None and model_name is not None and experiment_name is not None
    logger.info(
        f"Loading model from MLflow (run_id={run_id}, model_name={model_name}), mlflow host: {mlflow_host}"
    )
    model = load_model(run_id, model_name)
    # state_dict = load_model_state_dict(run_id, model_name)  # type: ignore[arg-type]
    # model.load_state_dict(state_dict)
    # model = torch.load(
    #     Path(__file__).parent.parent
    #     / "models"
    #     / "pretrained"
    #     / "coca_styled_finetune_epoch00028.pth",
    #     weights_only=False,
    #     map_location=torch.device("cpu"),
    # )

    device = device if device is not None else get_device()
    logger.info(f"Using device: {device}")
    model = model.to(device)
    model.eval()

    # Get transforms (same as training data)
    transform = get_synthetic_transforms(is_train=False)

    # Process images in batches
    results = []

    for i in tqdm(range(0, len(image_files), batch_size), desc="Processing images"):
        batch_files = image_files[i : i + batch_size]
        batch_images = []
        batch_names = []

        # Load and transform images
        for img_file in batch_files:
            try:
                img = Image.open(img_file).convert("RGB")
                img_tensor = transform(img)
                batch_images.append(img_tensor)
                batch_names.append(img_file.name)
            except Exception as e:
                logger.warning(f"Failed to load {img_file}: {e}")
                continue

        if not batch_images:
            continue

        # Stack images and move to device
        images = torch.stack(batch_images).to(device)

        # Generate predictions
        with torch.no_grad():
            bos_id = getattr(tokenizer, "bos_token_id", 1)
            eos_id = getattr(tokenizer, "eos_token_id", 2)

            try:
                generated = model.generate(
                    src=images,
                    bos_token_id=bos_id,
                    eos_token_id=eos_id,
                    max_length=max_length,
                    temperature=temperature,
                )
            except TypeError:
                # Fallback for GRU model
                generated = model.generate(
                    src=images,
                    max_len=max_length,
                )

        # Decode predictions
        for img_name, pred_ids in zip(batch_names, generated):
            pred_text = tokenizer.decode(pred_ids.tolist())
            results.append(
                {
                    "image": img_name,
                    "formula": pred_text,
                }
            )

    # Save results
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Processed {len(results)} images")
    logger.info(f"Results saved to {output_path}")


@cli.command()
@click.option(
    "--results-file",
    type=click.Path(exists=True),
    required=True,
    help="Path to inference_results.json file",
)
@click.option(
    "--image-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Directory containing the original images",
)
@click.option(
    "--port",
    type=int,
    default=8501,
    help="Port to run Streamlit app on",
)
def visualize(results_file: str, image_dir: str, port: int):
    """
    Launch a Streamlit app to visualize inference results.
    
    This command renders the predicted LaTeX formulas and displays them
    side-by-side with the original images for comparison.
    
    Example usage:
        PYTHONPATH=. uv run python latex_ocr/trainers/eval.py visualize \
            --results-file inference_results.json \
            --image-dir /path/to/images \
            --port 8501
    """
    import subprocess
    import tempfile
    import sys

    # Create a temporary Streamlit app file
    # Using triple quotes and regular string formatting to avoid f-string escaping issues
    streamlit_code = '''
import streamlit as st
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import io

# LaTeX rendering
try:
    from matplotlib import mathtext
except ImportError:
    st.error("matplotlib is required for LaTeX rendering")
    st.stop()

st.set_page_config(page_title="LaTeX OCR Results Viewer", layout="wide")

# Load results
results_file = Path("{results_file}")
image_dir = Path("{image_dir}")

@st.cache_data
def load_results():
    with open(results_file, "r") as f:
        return json.load(f)

def render_latex_to_image(latex_str: str, dpi: int = 150, fontsize: int = 20):
    """Render LaTeX formula to an image using matplotlib"""
    try:
        # Create a figure with transparent background
        fig = plt.figure(figsize=(10, 2), dpi=dpi)
        fig.patch.set_alpha(0.0)
        
        # Render the LaTeX
        # Ensure the LaTeX is in math mode
        if not latex_str.strip().startswith('$'):
            latex_str = f"${{latex_str}}$"
        
        plt.text(
            0.5, 0.5, latex_str,
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=fontsize,
            transform=fig.transFigure
        )
        plt.axis('off')
        
        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', 
                   pad_inches=0.1, transparent=True, dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        
        return Image.open(buf)
    except Exception as e:
        st.error(f"Failed to render LaTeX: {{e}}")
        return None

# UI
st.title("📊 LaTeX OCR Results Viewer")
st.markdown("Compare original images with predicted LaTeX formulas")

results = load_results()
st.sidebar.header("Navigation")
st.sidebar.markdown(f"**Total results:** {{len(results)}}")

# Filters
filter_option = st.sidebar.radio(
    "View mode:",
    ["All results", "Browse by index", "Search by filename"]
)

if filter_option == "Browse by index":
    idx = st.sidebar.number_input(
        "Select index:", 
        min_value=0, 
        max_value=len(results)-1, 
        value=0, 
        step=1
    )
    selected_results = [results[idx]]
elif filter_option == "Search by filename":
    search_query = st.sidebar.text_input("Search filename:")
    if search_query:
        selected_results = [
            r for r in results 
            if search_query.lower() in r["image"].lower()
        ]
    else:
        selected_results = results
else:
    selected_results = results

# Display options
st.sidebar.header("Display Settings")
font_size = st.sidebar.slider("LaTeX Font Size", 10, 40, 20)
dpi = st.sidebar.slider("Render DPI", 50, 300, 150)
show_raw_latex = st.sidebar.checkbox("Show Raw LaTeX", value=True)

# Pagination
items_per_page = st.sidebar.selectbox("Items per page:", [5, 10, 20, 50, 100], index=1)
total_pages = (len(selected_results) + items_per_page - 1) // items_per_page
max_pages = max(1, total_pages)
page = st.sidebar.number_input("Page:", min_value=1, max_value=max_pages, value=1)

start_idx = (page - 1) * items_per_page
end_idx = min(start_idx + items_per_page, len(selected_results))
page_results = selected_results[start_idx:end_idx]

st.sidebar.markdown(f"Showing {{start_idx + 1}}-{{end_idx}} of {{len(selected_results)}} results")

# Main content
if not page_results:
    st.warning("No results to display")
else:
    for i, result in enumerate(page_results, start=start_idx):
        st.markdown("---")
        st.subheader(f"Result #{{i + 1}}: {{result['image']}}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📷 Original Image")
            img_path = image_dir / result["image"]
            if img_path.exists():
                orig_img = Image.open(img_path)
                st.image(orig_img, use_container_width=True, caption="Original Image")
            else:
                st.error(f"Image not found: {{img_path}}")
        
        with col2:
            st.markdown("### 🔮 Predicted Formula (Rendered)")
            formula = result["formula"]
            
            if show_raw_latex:
                st.code(formula, language="latex")
            
            # Render LaTeX
            with st.spinner("Rendering LaTeX..."):
                rendered_img = render_latex_to_image(formula, dpi=dpi, fontsize=font_size)
                if rendered_img:
                    st.image(rendered_img, use_container_width=True, caption="Rendered Formula")
                else:
                    st.warning("Could not render LaTeX formula")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**LaTeX OCR Results Viewer**")
st.sidebar.markdown(f"Results file: `{{results_file.name}}`")
'''.format(results_file=results_file, image_dir=image_dir)

    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(streamlit_code)
        temp_file = f.name

    logger.info(f"Starting Streamlit app on port {port}...")
    logger.info(f"Results file: {results_file}")
    logger.info(f"Image directory: {image_dir}")

    try:
        # Run streamlit
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                temp_file,
                "--server.port",
                str(port),
            ],
            check=True,
        )
    except KeyboardInterrupt:
        logger.info("Streamlit app stopped")
    finally:
        # Clean up temp file
        import os

        try:
            os.unlink(temp_file)
        except:
            pass


if __name__ == "__main__":
    """
    Example usage:
    
    # Evaluate a model (loads config + weights from MLflow; same as infer-images)
    PYTHONPATH=. nohup uv run python latex_ocr/trainers/eval.py evaluate \
        --run-id 416b0c1f4f2c4bbc8cfad64f07aba3ea \
        --experiment-name "latex-ocr-coca-finetune" \
        --model-name "latex-ocr-coca-finetune_epoch00045" \
        --config-name plain \
        --split validation \
        --batch-size 64 \
        --device cuda:1 \
        --output data/output/eval_results_coca_plain_v2.9.json 2>&1 &> eval.log &
    PYTHONPATH=. nohup uv run python latex_ocr/trainers/eval.py evaluate \
        --run-id 416b0c1f4f2c4bbc8cfad64f07aba3ea \
        --experiment-name "latex-ocr-coca-finetune" \
        --model-name "latex-ocr-coca-finetune_epoch00045" \
        --config-name styled \
        --split validation \
        --batch-size 64 \
        --device cuda:0 \
        --output data/output/eval_results_coca_styled_v2.9.json 2>&1 &> eval2.log &
    
    # Infer formulas from images in a directory (using MLflow model)
    PYTHONPATH=. uv run python latex_ocr/trainers/eval.py infer-images \
        --run-id 1c0eb12acf40406d91a642624354fe96 \
        --experiment-name "latex-ocr-coca-pretrain" \
        --model-name "latex-ocr-coca-pretrain_epoch00021" \
        --image-dir ~/formulas \
        --batch-size 8 \
        --max-length 512 \
        --output inference_results_test.json \
        --temperature=0.5
    
    # Visualize inference results with Streamlit
    PYTHONPATH=. uv run python latex_ocr/trainers/eval.py visualize \
        --results-file inference_results_test.json \
        --image-dir ~/formulas \
        --port 8501
    """
    cli()
