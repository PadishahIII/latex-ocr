"""
Upload synthesized LaTeX OCR datasets to Hugging Face Hub.

Usage:
    PYTHONPATH=. uv run python workspace/latex_ocr/data/pipeline/upload_synth.py --message "Add v2.0 dataset"
    PYTHONPATH=. uv run python workspace/latex_ocr/data/pipeline/upload_synth.py --config plain --message "Update plain only"
    PYTHONPATH=. uv run python workspace/latex_ocr/data/pipeline/upload_synth.py  # uses auto-generated message
"""

import click
from datetime import datetime
from datasets import load_dataset
from pathlib import Path
from typing import Optional

REPO_ID = "PadishahIIIXXX/latex-ocr-dataset"
DATA_DIR = Path(__file__).parent.parent.parent / "datasets" / "synth"

DATA_FILES = {
    "plain": {"train": "plain/train/**/*", "validation": "plain/validation/**/*"},
    "styled": {"train": "styled/train/**/*", "validation": "styled/validation/**/*"},
}


def get_commit_message(config_name: str, custom_message: Optional[str] = None) -> str:
    """Generate a commit message with timestamp and config name."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if custom_message:
        return f"[{config_name}] {custom_message} - {timestamp}"
    return f"Upload {config_name} dataset - {timestamp}"


def upload_config(config_name: str, custom_message: Optional[str] = None) -> None:
    """Upload a single dataset config to the Hub."""
    click.echo(f"Loading {config_name} dataset...")
    dataset = load_dataset(
        "imagefolder",
        data_dir=DATA_DIR.absolute().__str__(),
        data_files=DATA_FILES[config_name],
    )

    commit_msg = get_commit_message(config_name, custom_message)
    click.echo(f"Uploading {config_name} with commit: {commit_msg}")

    dataset.push_to_hub(
        REPO_ID,
        max_shard_size="500MB",
        config_name=config_name,
        commit_message=commit_msg,
    )
    click.echo(f"✓ {config_name} uploaded successfully")


@click.command()
@click.option(
    "--message", "-m",
    type=str,
    default=None,
    help="Custom commit message. If not provided, uses auto-generated message.",
)
@click.option(
    "--config", "-c",
    type=click.Choice(["plain", "styled", "both"]),
    default="both",
    help="Which dataset config to upload. Default: both",
)
def main(message: Optional[str], config: str) -> None:
    """Upload synthesized LaTeX OCR datasets to Hugging Face Hub."""
    if config == "both":
        upload_config("plain", message)
        upload_config("styled", message)
    else:
        upload_config(config, message)

    click.echo("🎉 All uploads completed!")


if __name__ == "__main__":
    main()
