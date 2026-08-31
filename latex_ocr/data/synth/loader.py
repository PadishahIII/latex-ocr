"""
Synthetic Dataset Loaders for LaTeX OCR
Loads datasets from HuggingFace Hub (PadishahIIIXXX/latex-ocr-dataset).

Example usage:

    # Load synthetic datasets
    from latex_ocr.data.synth.loader import (
        SyntheticPlainDataset,
        SyntheticStyledDataset,
        get_synthetic_transforms
    )

    # Create datasets - now loads from HuggingFace Hub
    plain_ds = SyntheticPlainDataset(
        split="train",
        transform=get_synthetic_transforms(is_train=True)
    )

    styled_ds = SyntheticStyledDataset(
        split="train",
        transform=get_synthetic_transforms(is_train=True)
    )

    # Use with MixedTrainingDataset
    from latex_ocr.data.loader import MixedTrainingDataset
    from latex_ocr.data.unimer.loader import UniMERDataset

    real_ds = UniMERDataset(
        image_dir="datasets/UniMER-1M/images",
        formula_file="datasets/UniMER-1M/train.txt",
        transform=get_synthetic_transforms(is_train=True)
    )

    # Create mixed dataset with 100% real, 20% plain, 100% styled
    mixed_ds = MixedTrainingDataset(
        real_dataset=real_ds,
        synth_plain_dataset=plain_ds,
        synth_styled_dataset=styled_ds,
    )
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import io

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset


class SyntheticLaTeXDataset(Dataset):
    """
    PyTorch Dataset for synthetic LaTeX OCR data loaded from HuggingFace Hub

    Loads from PadishahIIIXXX/latex-ocr-dataset repository with configurations:
    - "plain": Plain formulas without style enrichment
    - "styled": Styled formulas with mathxx enrichment

    Each sample returns a dictionary with:
    - "image": torch.Tensor of shape (C, H, W)
    - "text": str, the LaTeX formula
    """

    REPO_ID = "PadishahIIIXXX/latex-ocr-dataset"

    def __init__(
        self,
        config_name: str,
        local_dir: Optional[str] = None,  # Kept for backward compatibility, not used
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        max_samples: Optional[int] = None,
        formula_only: bool = False,
    ):
        """
        Args:
            config_name: Configuration name ("plain" or "styled")
            local_dir: Local dir of dataset
            split: Either "train" or "validation"
            transform: Optional torchvision transforms to apply to images
            max_samples: Optional limit on number of samples to load
            formula_only: If True, only load formulas without images
        """
        self.config_name = config_name
        self.split = split
        self.transform = transform or self._default_transform()
        self.formula_only = formula_only

        # Load dataset from HuggingFace Hub
        print(f"Loading {config_name} dataset from HuggingFace Hub ({split} split)...")
        self.hf_dataset = load_dataset(
            self.REPO_ID,
            config_name,
            split=split,
            verification_mode="no_checks",
            # download_mode="force_redownload",
            # data_dir=local_dir,
        )

        if max_samples and max_samples < len(self.hf_dataset):  # type: ignore
            self.hf_dataset = self.hf_dataset.select(range(max_samples))  # type: ignore

        print(
            f"✓ Loaded {len(self.hf_dataset)} samples from {self.REPO_ID}/{config_name} ({split} split)"  # type: ignore
        )

    def _default_transform(self) -> transforms.Compose:
        """Default image transformation pipeline"""
        return transforms.Compose(
            [
                transforms.Resize((192, 672)),  # Standard LaTeX OCR size
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.hf_dataset)  # type: ignore

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample

        Returns:
            Dictionary with keys:
            - "image": torch.Tensor
            - "text": str (LaTeX formula)
        """
        sample = self.hf_dataset[idx]  # type: ignore

        # Get image from HuggingFace dataset (only "image" and "text" fields)
        # The image is already a PIL Image object
        image = None
        if not self.formula_only:
            image = sample["image"]
            if not isinstance(image, Image.Image):
                # If it's bytes, convert to PIL Image
                if isinstance(image, bytes):
                    image = Image.open(io.BytesIO(image)).convert("RGB")
                else:
                    # Handle other formats if needed
                    image = Image.open(io.BytesIO(bytes(image))).convert("RGB")  # type: ignore
            else:
                image = image.convert("RGB")

        # Get formula text
        text = sample["text"]

        # Apply transforms
        if self.transform and image is not None:
            image = self.transform(image)
        if image is None:
            return {"text": text}

        return {
            "image": image,
            "text": text,
        }


class SyntheticPlainDataset(SyntheticLaTeXDataset):
    """
    Dataset for synthetic plain formulas (without style enrichment)
    Loads from HuggingFace Hub with config="plain"
    """

    def __init__(
        self,
        local_dir: Optional[str] = None,  # Kept for backward compatibility
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        max_samples: Optional[int] = None,
    ):
        super().__init__(
            config_name="plain",
            local_dir=local_dir,
            split=split,
            transform=transform,
            max_samples=max_samples,
        )


class SyntheticStyledDataset(SyntheticLaTeXDataset):
    """
    Dataset for synthetic styled formulas (with mathxx enrichment)
    Loads from HuggingFace Hub with config="styled"
    """

    def __init__(
        self,
        local_dir: Optional[str] = None,  # Kept for backward compatibility
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        max_samples: Optional[int] = None,
    ):
        super().__init__(
            config_name="styled",
            local_dir=local_dir,
            split=split,
            transform=transform,
            max_samples=max_samples,
        )


def get_synthetic_transforms(
    image_size: Tuple[int, int] = (192, 672), is_train: bool = True
) -> transforms.Compose:
    """
    Get image transformations for synthetic datasets

    Args:
        image_size: (height, width) for resizing
        is_train: Whether for training (adds augmentation)

    Returns:
        Composed transforms
    """
    if is_train:
        return transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.3
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


def create_synthetic_dataloaders(
    config_names: list[str],
    local_dir: Optional[str] = None,
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (192, 672),
    is_train: bool = True,
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create dataloaders for synthetic datasets from HuggingFace Hub

    Args:
        plain_root: Deprecated, no longer used (kept for backward compatibility)
        styled_root: Deprecated, no longer used (kept for backward compatibility)
        split: Dataset split to load ("train" or "validation")
        batch_size: Batch size
        num_workers: Number of worker processes
        image_size: Image dimensions (H, W)
        is_train: Whether for training (affects transforms and shuffling)

    Returns:
        Dictionary of dataloaders with keys "plain" and/or "styled"
    """
    dataloaders = {}
    transform = get_synthetic_transforms(image_size, is_train=is_train)

    # Load plain dataset if requested (plain_root being not None indicates request)
    if config_names.__contains__("plain"):
        plain_dataset = SyntheticPlainDataset(
            split=split,
            local_dir=local_dir,
            transform=transform,
        )
        dataloaders["plain"] = torch.utils.data.DataLoader(
            plain_dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=True,
        )

    # Load styled dataset if requested (styled_root being not None indicates request)
    if config_names.__contains__("styled"):
        styled_dataset = SyntheticStyledDataset(
            split=split,
            local_dir=local_dir,
            transform=transform,
        )
        dataloaders["styled"] = torch.utils.data.DataLoader(
            styled_dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=True,
        )

    return dataloaders


if __name__ == "__main__":
    """
    Example usage:
    
    PYTHONPATH=. uv run python latex_ocr/data/synth/loader.py \
        --plain \
        --styled \
        --split train 
    
    Note: Now loads directly from HuggingFace Hub (PadishahIIIXXX/latex-ocr-dataset)
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Synthetic LaTeX Dataset Loader (HuggingFace Hub)"
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--styled",
        action="store_true",
        help="Load synthetic styled dataset",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Load synthetic plain dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation"],
        help="Dataset split to load",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Limit samples for testing"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("LOADING SYNTHETIC DATASETS FROM HUGGINGFACE HUB")
    print("=" * 80)

    # Create dataloaders
    # Pass dummy values to trigger loading (actual values no longer used)
    configs = []
    if args.plain:
        configs.append("plain")
    if args.styled:
        configs.append("styled")

    if not configs:
        raise ValueError("At least one of --plain or --styled must be specified.")
    dataloaders = create_synthetic_dataloaders(
        config_names=configs,
        local_dir=args.local_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Test each dataloader
    for name, loader in dataloaders.items():
        print(f"\n{name.upper()} Dataset:")
        print(f"  - Total samples: {len(loader.dataset)}")  # type: ignore
        print(f"  - Number of batches: {len(loader)}")

        # Fetch first batch
        batch = next(iter(loader))
        print(f"  - Batch image shape: {batch['image'].shape}")
        print(f"  - Sample text: {batch['text'][0][:80]}...")

    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)

    for name, loader in dataloaders.items():
        total_samples = len(loader.dataset)  # type: ignore
        num_batches = len(loader)
        print(f"{name:10s}: {total_samples:6d} samples, {num_batches:4d} batches")
