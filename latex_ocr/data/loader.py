"""
LaTeX OCR Data Loader
Unified dataloader factory for LaTeX OCR training and evaluation.

Supports:
- UniMER datasets (real data)
- LaTeXOCR datasets (HuggingFace)
- Synthetic Plain/Styled datasets (HuggingFace JSONL format)
- Mixed training dataset with configurable ratios
- Multiple evaluation datasets

Example usage:

    from latex_ocr.data.loader import (
        create_dataloaders,
        TrainConfig,
        EvalConfig,
        DataLoaderConfig,
    )

    # Create all dataloaders
    train_config = TrainConfig(
        unimer_image_dir="datasets/UniMER-1M/images",
        unimer_formula_file="datasets/UniMER-1M/train.txt",
        latex_ocr_split="train",
        plain_root="datasets/synth/plain",
        styled_root="datasets/synth/styled",
    )

    eval_configs = [
        EvalConfig(name="SPE", dataset_type="unimer", image_dir="datasets/UniMER-Test/spe", formula_file="datasets/UniMER-Test/spe.txt"),
        EvalConfig(name="CPE", dataset_type="unimer", image_dir="datasets/UniMER-Test/cpe", formula_file="datasets/UniMER-Test/cpe.txt"),
    ]

    loader_config = DataLoaderConfig(batch_size=32, num_workers=4)

    dataloaders = create_dataloaders(
        train_config=train_config,
        eval_configs=eval_configs,
        loader_config=loader_config,
    )

    # Access dataloaders
    train_loader = dataloaders["train"]
    spe_loader = dataloaders["SPE"]
"""

from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import math
import torch
from pydantic import BaseModel, Field, field_validator
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import transforms


# ==============================================================================
# Pydantic Configuration Models
# ==============================================================================


class TrainConfig(BaseModel):
    """
    Configuration for training datasets.

    Attributes:
        unimer_image_dir: Path to UniMER dataset images
        unimer_formula_file: Path to UniMER dataset formulas
        latex_ocr_split: LaTeXOCR dataset split ("train" or "validation")
        latex_ocr_cache_dir: Cache directory for LaTeXOCR dataset
        plain_root: Path to synthetic plain dataset
        styled_root: Path to synthetic styled dataset
        use_mixed_dataset: If True and synthetic data is provided, create mixed dataset
    """

    unimer_image_dir: Optional[str] = None
    unimer_formula_file: Optional[str] = None
    latex_ocr_split: Optional[Literal["train", "validation"]] = None
    latex_ocr_cache_dir: Optional[str] = None
    plain_root: Optional[str] = None
    styled_root: Optional[str] = None
    use_mixed_dataset: bool = True

    @field_validator(
        "unimer_image_dir",
        "unimer_formula_file",
        "plain_root",
        "styled_root",
        "latex_ocr_cache_dir",
    )
    @classmethod
    def validate_paths(cls, v: Optional[str]) -> Optional[str]:
        """Validate that paths exist if provided"""
        if v is not None and v.strip():
            path = Path(v)
            # Note: We don't check existence here as datasets might be downloaded later
            return str(path)
        return v

    def model_post_init(self, __context) -> None:
        """Validate that at least one real dataset is configured"""
        has_unimer = self.unimer_image_dir and self.unimer_formula_file
        has_latex_ocr = self.latex_ocr_split is not None

        if not has_unimer and not has_latex_ocr:
            raise ValueError(
                "At least one real dataset must be configured: "
                "UniMER (unimer_image_dir + unimer_formula_file) or "
                "LaTeXOCR (latex_ocr_split)"
            )


class EvalConfig(BaseModel):
    """
    Configuration for a single evaluation dataset.

    Attributes:
        name: Name/identifier for the evaluation set
        image_dir: Path to evaluation images (for UniMER-style datasets)
        formula_file: Path to evaluation formulas (for UniMER-style datasets)
        dataset_type: Type of dataset ("unimer", "latex_ocr", or "synthetic")
        latex_ocr_split: LaTeXOCR split to use (if dataset_type="latex_ocr")
        latex_ocr_cache_dir: Cache directory for LaTeXOCR (if dataset_type="latex_ocr")
        synth_root: Root directory for synthetic dataset (if dataset_type="synthetic")
        synth_split: Split for synthetic dataset (if dataset_type="synthetic")
    """

    name: str = Field(..., description="Name/identifier for the evaluation set")
    dataset_type: Literal["unimer", "latex_ocr", "synthetic"] = "unimer"

    # UniMER-style dataset fields
    image_dir: Optional[str] = None
    formula_file: Optional[str] = None

    # LaTeXOCR dataset fields (validation split is for benchmark/evaluation)
    latex_ocr_split: Optional[Literal["train", "validation"]] = None
    latex_ocr_cache_dir: Optional[str] = None

    # Synthetic dataset fields (validation split is for benchmark/evaluation)
    synth_root: Optional[str] = None
    synth_split: Literal["train", "validation"] = "validation"

    @field_validator("image_dir", "formula_file", "latex_ocr_cache_dir", "synth_root")
    @classmethod
    def validate_paths(cls, v: Optional[str]) -> Optional[str]:
        """Convert to absolute path if provided"""
        if v is not None:
            return str(Path(v))
        return v

    def model_post_init(self, __context) -> None:
        """Validate that required fields are set based on dataset_type"""
        if self.dataset_type == "unimer":
            if not self.image_dir or not self.formula_file:
                raise ValueError(
                    "image_dir and formula_file are required for dataset_type='unimer'"
                )
        elif self.dataset_type == "latex_ocr":
            if not self.latex_ocr_split:
                raise ValueError(
                    "latex_ocr_split is required for dataset_type='latex_ocr'"
                )
        elif self.dataset_type == "synthetic":
            if not self.synth_root:
                raise ValueError("synth_root is required for dataset_type='synthetic'")


class DataLoaderConfig(BaseModel):
    """
    Configuration for PyTorch DataLoader settings.

    Attributes:
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        image_height: Image height for resizing
        image_width: Image width for resizing
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last_train: Whether to drop last incomplete batch in training
    """

    batch_size: int = 32
    num_workers: int = 4
    image_height: int = 192
    image_width: int = 672
    pin_memory: bool = True
    drop_last_train: bool = True

    @property
    def image_size(self) -> Tuple[int, int]:
        """Return (height, width) tuple"""
        return (self.image_height, self.image_width)


# ==============================================================================
# Dataset Classes
# ==============================================================================


class MixedTrainingDataset(Dataset):
    """
    Implements the Data Mix per batch (Section 5):
    - Original UniMER
    - 20% Synthetic Plain
    - Synthetic Styled

    Note: Uses fixed seed for reproducible 20% plain dataset selection.
    This ensures consistent ordering across runs, which is critical for
    resume functionality in dataset generation.
    """

    def __init__(
        self,
        real_dataset: Dataset,
        synth_plain_dataset: Dataset,
        synth_styled_dataset: Dataset,
        plain_sample_seed: int = 42,
    ):
        self.real = real_dataset
        self.plain = synth_plain_dataset
        self.styled = synth_styled_dataset

        # 20% of the synthetic PDF-style plain dataset is used.
        # Use fixed seed for deterministic selection across runs.
        plain_len = len(self.plain)  # type: ignore[arg-type]
        generator = torch.Generator().manual_seed(plain_sample_seed)
        plain_indices = torch.randperm(plain_len, generator=generator)[
            : math.floor(0.2 * plain_len)
        ]
        self.plain = Subset(self.plain, plain_indices.tolist())

        self.dataset = ConcatDataset([self.real, self.plain, self.styled])

    def __len__(self):
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx):
        return self.dataset[idx]  # type: ignore[return-value]


def get_default_transforms(
    image_size: Tuple[int, int] = (192, 672), is_train: bool = True
) -> transforms.Compose:
    """
    Get default image transformations for LaTeX OCR.

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


def collate_mixed_batch(batch):
    """
    Custom collate function to handle mixed dataset formats.

    Handles:
    - tuple format from LaTeXOCRDataset: (image_tensor, formula_string)
    - dict format from UniMERDataset/SyntheticDatasets: {"image": tensor, "formula": str, ...}

    Returns:
        Dict with "image" and "formula" keys, compatible with UniMER format
    """
    images = []
    formulas = []

    for item in batch:
        if isinstance(item, dict):
            # UniMER/Synthetic format
            images.append(item["image"])
            formulas.append(item["formula"])
        elif isinstance(item, tuple) and len(item) == 2:
            # LaTeXOCR format: (image, formula)
            images.append(item[0])
            formulas.append(item[1])
        else:
            raise ValueError(f"Unexpected item format: {type(item)}")

    return {
        "image": torch.stack(images),
        "formula": formulas,
    }


def create_dataloaders(
    train_config: Optional[TrainConfig] = None,
    eval_configs: Optional[List[EvalConfig]] = None,
    loader_config: Optional[DataLoaderConfig] = None,
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for LaTeX OCR training and evaluation.

    Args:
        train_config: Training dataset configuration (Pydantic model)
        eval_configs: List of evaluation dataset configurations (Pydantic models)
        loader_config: DataLoader settings (Pydantic model)

    Returns:
        Dictionary of dataloaders with keys "train" and evaluation set names

    Example:
        from latex_ocr.data.loader import (
            create_dataloaders,
            TrainConfig,
            EvalConfig,
            DataLoaderConfig,
        )

        train_config = TrainConfig(
            unimer_image_dir="datasets/UniMER-1M/images",
            unimer_formula_file="datasets/UniMER-1M/train.txt",
            latex_ocr_split="train",
            plain_root="datasets/synth/plain",
            styled_root="datasets/synth/styled",
            use_mixed_dataset=True,
        )

        eval_configs = [
            # UniMER-Test benchmark datasets
            EvalConfig(
                name="SPE",
                dataset_type="unimer",
                image_dir="datasets/UniMER-Test/spe",
                formula_file="datasets/UniMER-Test/spe.txt"
            ),
            # LaTeXOCR validation split for benchmark
            EvalConfig(
                name="LaTeXOCR-Val",
                dataset_type="latex_ocr",
                latex_ocr_split="validation"
            ),
            # Synthetic validation splits for benchmark
            EvalConfig(
                name="Synth-Plain-Val",
                dataset_type="synthetic",
                synth_root="datasets/synth/plain",
                synth_split="validation"
            ),
        ]

        loader_config = DataLoaderConfig(batch_size=32, num_workers=4)

        dataloaders = create_dataloaders(
            train_config=train_config,
            eval_configs=eval_configs,
            loader_config=loader_config,
        )
    """
    # Lazy import to avoid circular dependencies
    from latex_ocr.data.unimer.loader import UniMERDataset
    from latex_ocr.data.pix2tex.loader import LaTeXOCRDataset
    from latex_ocr.data.synth.loader import (
        SyntheticPlainDataset,
        SyntheticStyledDataset,
    )

    # Use default loader config if not provided
    if loader_config is None:
        loader_config = DataLoaderConfig()

    dataloaders = {}

    # Create training dataloader
    if train_config:
        train_transform = get_default_transforms(
            loader_config.image_size, is_train=True
        )
        real_datasets = []

        # Load UniMER dataset if provided
        if train_config.unimer_image_dir and train_config.unimer_formula_file:
            unimer_dataset = UniMERDataset(
                image_dir=train_config.unimer_image_dir,
                formula_file=train_config.unimer_formula_file,
                transform=train_transform,
            )
            real_datasets.append(unimer_dataset)
            print(f"✓ Loaded UniMER dataset with {len(unimer_dataset)} samples")

        # Load LaTeXOCR dataset if specified
        if train_config.latex_ocr_split:
            try:
                latex_ocr_dataset = LaTeXOCRDataset(
                    split=train_config.latex_ocr_split,
                    transform=train_transform,
                    cache_dir=train_config.latex_ocr_cache_dir,
                )
                real_datasets.append(latex_ocr_dataset)
                print(
                    f"✓ Loaded LaTeXOCR dataset ({train_config.latex_ocr_split}) "
                    f"with {len(latex_ocr_dataset)} samples"
                )
            except Exception as e:
                print(f"⚠ Failed to load LaTeXOCR dataset: {e}")

        # Combine real datasets
        if len(real_datasets) > 1:
            real_dataset = ConcatDataset(real_datasets)
            print(f"✓ Combined real datasets: {len(real_dataset)} total samples")
        elif len(real_datasets) == 1:
            real_dataset = real_datasets[0]
        else:
            print("⚠ No real datasets provided for training")
            return dataloaders

        # Optionally create mixed dataset with synthetic data
        if (
            train_config.use_mixed_dataset
            and train_config.plain_root
            and train_config.styled_root
        ):
            plain_dataset = SyntheticPlainDataset(
                local_dir=train_config.plain_root,
                split="train",  # Use training split for training
                transform=train_transform,
            )
            styled_dataset = SyntheticStyledDataset(
                local_dir=train_config.styled_root,
                split="train",  # Use training split for training
                transform=train_transform,
            )

            train_dataset = MixedTrainingDataset(
                real_dataset=real_dataset,
                synth_plain_dataset=plain_dataset,
                synth_styled_dataset=styled_dataset,
            )
            print("✓ Created mixed training dataset with synthetic data")
        else:
            train_dataset = real_dataset
            print("✓ Using combined real dataset for training")

        dataloaders["train"] = DataLoader(
            train_dataset,
            batch_size=loader_config.batch_size,
            shuffle=True,
            num_workers=loader_config.num_workers,
            pin_memory=loader_config.pin_memory,
            drop_last=loader_config.drop_last_train,
            collate_fn=collate_mixed_batch,
        )

    # Create evaluation dataloaders
    if eval_configs:
        from latex_ocr.data.unimer.loader import UniMERDataset
        from latex_ocr.data.pix2tex.loader import LaTeXOCRDataset
        from latex_ocr.data.synth.loader import (
            SyntheticPlainDataset,
            SyntheticStyledDataset,
        )

        eval_transform = get_default_transforms(
            loader_config.image_size, is_train=False
        )

        for eval_cfg in eval_configs:
            if eval_cfg.dataset_type == "unimer":
                # UniMER-style dataset
                if not eval_cfg.image_dir or not eval_cfg.formula_file:
                    print(
                        f"⚠ Skipping {eval_cfg.name}: missing image_dir or formula_file"
                    )
                    continue

                eval_dataset = UniMERDataset(
                    image_dir=eval_cfg.image_dir,
                    formula_file=eval_cfg.formula_file,
                    transform=eval_transform,
                )
            elif eval_cfg.dataset_type == "latex_ocr":
                # LaTeXOCR dataset
                if not eval_cfg.latex_ocr_split:
                    print(f"⚠ Skipping {eval_cfg.name}: missing latex_ocr_split")
                    continue

                try:
                    eval_dataset = LaTeXOCRDataset(
                        split=eval_cfg.latex_ocr_split,
                        transform=eval_transform,
                        cache_dir=eval_cfg.latex_ocr_cache_dir,
                    )
                except Exception as e:
                    print(f"⚠ Failed to load LaTeXOCR dataset {eval_cfg.name}: {e}")
                    continue
            elif eval_cfg.dataset_type == "synthetic":
                # Synthetic dataset
                if not eval_cfg.synth_root:
                    print(f"⚠ Skipping {eval_cfg.name}: missing synth_root")
                    continue

                # Determine if plain or styled based on name or path
                if "styled" in eval_cfg.name.lower() or "styled" in eval_cfg.synth_root:
                    eval_dataset = SyntheticStyledDataset(
                        local_dir=eval_cfg.synth_root,
                        split=eval_cfg.synth_split,
                        transform=eval_transform,
                    )
                else:
                    eval_dataset = SyntheticPlainDataset(
                        local_dir=eval_cfg.synth_root,
                        split=eval_cfg.synth_split,
                        transform=eval_transform,
                    )
            else:
                print(
                    f"⚠ Unknown dataset type for {eval_cfg.name}: {eval_cfg.dataset_type}"
                )
                continue

            dataloaders[eval_cfg.name] = DataLoader(
                eval_dataset,
                batch_size=loader_config.batch_size,
                shuffle=False,
                num_workers=loader_config.num_workers,
                pin_memory=loader_config.pin_memory,
                collate_fn=collate_mixed_batch,  # Use collate function for mixed formats
            )
            print(
                f"✓ Created {eval_cfg.name} evaluation dataset ({eval_cfg.dataset_type}) with {len(eval_dataset)} samples"
            )

    return dataloaders


if __name__ == "__main__":
    """
    Test the dataloader creation.
    
    Example usage:
        # With UniMER only
        PYTHONPATH=. uv run python workspace/latex_ocr/data/loader.py \
            --unimer_image_dir workspace/latex_ocr/datasets/UniMER_Dataset/UniMER-1M/images \
            --unimer_formula_file workspace/latex_ocr/datasets/UniMER_Dataset/UniMER-1M/train.txt \
            --plain_root workspace/latex_ocr/datasets/synth/plain \
            --styled_root workspace/latex_ocr/datasets/synth/styled
        
        # With both UniMER and LaTeXOCR, along with synthetic datasets
        PYTHONPATH=. uv run python workspace/latex_ocr/data/loader.py \
            --unimer_image_dir workspace/latex_ocr/datasets/UniMER_Dataset/UniMER-1M/images \
            --unimer_formula_file workspace/latex_ocr/datasets/UniMER_Dataset/UniMER-1M/train.txt \
            --latex_ocr_split train \
            --use_mixed \
            --plain_root workspace/latex_ocr/datasets/synth/plain \
            --styled_root workspace/latex_ocr/datasets/synth/styled
    """
    import argparse

    parser = argparse.ArgumentParser(description="LaTeX OCR Dataloader Factory")
    parser.add_argument(
        "--unimer_image_dir",
        type=str,
        default=None,
        help="Path to UniMER dataset images",
    )
    parser.add_argument(
        "--unimer_formula_file",
        type=str,
        default=None,
        help="Path to UniMER dataset formulas",
    )
    parser.add_argument(
        "--latex_ocr_split",
        type=str,
        default=None,
        choices=["train", "validation"],
        help="LaTeXOCR dataset split to load",
    )
    parser.add_argument(
        "--latex_ocr_cache_dir",
        type=str,
        default=None,
        help="Cache directory for LaTeXOCR dataset",
    )
    parser.add_argument(
        "--plain_root",
        type=str,
        default=None,
        help="Path to synthetic plain dataset",
    )
    parser.add_argument(
        "--styled_root",
        type=str,
        default=None,
        help="Path to synthetic styled dataset",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--use_mixed",
        action="store_true",
        help="Use mixed dataset if synthetic data is available",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("CREATING DATALOADERS")
    print("=" * 80)

    # Prepare training config using Pydantic model
    train_config = None
    if args.unimer_image_dir or args.latex_ocr_split:
        config_dict = {}
        if args.unimer_image_dir:
            config_dict["unimer_image_dir"] = args.unimer_image_dir
        if args.unimer_formula_file:
            config_dict["unimer_formula_file"] = args.unimer_formula_file
        if args.latex_ocr_split:
            config_dict["latex_ocr_split"] = args.latex_ocr_split
        if args.latex_ocr_cache_dir:
            config_dict["latex_ocr_cache_dir"] = args.latex_ocr_cache_dir
        if args.plain_root:
            config_dict["plain_root"] = args.plain_root
        if args.styled_root:
            config_dict["styled_root"] = args.styled_root
        config_dict["use_mixed_dataset"] = args.use_mixed

        train_config = TrainConfig(**config_dict)

    # Create loader config
    loader_config = DataLoaderConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Create dataloaders
    dataloaders = create_dataloaders(
        train_config=train_config,
        loader_config=loader_config,
    )

    # Test dataloaders
    for name, loader in dataloaders.items():
        print(f"\n{name.upper()} Dataloader:")
        print(f"  - Total samples: {len(loader.dataset)}")  # type: ignore
        print(f"  - Number of batches: {len(loader)}")

        # Fetch first batch
        batch = next(iter(loader))
        print(f"  - Batch image shape: {batch['image'].shape}")
        print(f"  - Sample formula: {batch['formula'][0][:80]}...")
        if "image_path" in batch:
            print(f"  - Sample image path: {batch['image_path'][0]}")

    print("\n" + "=" * 80)
    print("DATALOADER STATISTICS")
    print("=" * 80)

    for name, loader in dataloaders.items():
        total_samples = len(loader.dataset)  # type: ignore
        num_batches = len(loader)
        print(f"{name:10s}: {total_samples:6d} samples, {num_batches:4d} batches")
