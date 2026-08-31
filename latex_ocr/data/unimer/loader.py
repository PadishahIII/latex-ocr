"""
UniMERNet Dataset Processor - Single File Version
Simplified dataset loader for UniMER-1M (train) and UniMER-Test (eval)
"""

import os
from typing import List, Tuple, Optional, Dict
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import argparse
from tqdm import tqdm


class UniMERDataset(Dataset):
    """
    Universal dataset class for UniMER-1M and UniMER-Test

    Dataset structure:
    - Images: 0000000.png, 0000001.png, ... in image folder
    - LaTeX formulas: line-by-line in .txt file (line i corresponds to image i)
    """

    def __init__(
        self,
        image_dir: str,
        formula_file: str,
        transform=None,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            image_dir: Directory containing images (*.png files)
            formula_file: Text file with LaTeX formulas (one per line)
            transform: Image transformations
            max_samples: Limit number of samples (useful for testing)
        """
        self.image_dir = Path(image_dir)
        self.formula_file = Path(formula_file)
        self.transform = transform

        # Load image paths and formulas
        self.image_paths, self.formulas = self._load_data()

        if max_samples:
            self.image_paths = self.image_paths[:max_samples]
            self.formulas = self.formulas[:max_samples]

        print(f"✓ Loaded {len(self.image_paths)} samples from {image_dir}")

    def _load_data(self) -> Tuple[List[str], List[str]]:
        """Load image paths and corresponding formulas"""
        # Get sorted image files (deterministic ordering for reproducibility)
        image_files = sorted(
            [f for f in os.listdir(self.image_dir) if f.endswith(".png")]
        )
        image_files_set = set(image_files)  # For O(1) lookup
        image_paths = [str(self.image_dir / f) for f in image_files]

        # Load formulas from text file
        formulas = []
        bar = tqdm(total=len(image_files), desc="Loading formulas", ncols=80)
        with open(self.formula_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=0):
                image_name = f"{i:07d}.png"
                # Only add formula if corresponding image exists
                if image_name in image_files_set:
                    formulas.append(line.strip())
                bar.update(1)

        # Validate matching counts
        if len(image_paths) != len(formulas):
            raise ValueError(
                f"Mismatch: {len(image_paths)} images vs {len(formulas)} formulas"
            )

        return image_paths, formulas

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict:
        # Load image
        image = Image.open(self.image_paths[idx]).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "text": self.formulas[idx],
        }

    @staticmethod
    def get_formulas_only(
        formula_file: str,
        image_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> List[str]:
        """
        Load only the formula text without loading images.
        This is much faster for generating synthetic datasets where we only need the LaTeX strings.

        Args:
            formula_file: Text file with LaTeX formulas (one per line)
            image_dir: Optional directory containing images (used for validation if provided)
            max_samples: Optional limit on number of formulas to load

        Returns:
            List[str]: List of LaTeX formula strings
        """
        formula_path = Path(formula_file)
        print(f"Loading formulas from {formula_path.name} (text only, no images)...")

        formulas: List[str] = []

        # If image_dir is provided, check which images exist
        if image_dir:
            image_dir_path = Path(image_dir)
            if image_dir_path.exists():
                image_files = set(
                    f for f in os.listdir(image_dir_path) if f.endswith(".png")
                )
            else:
                image_files = set()
        else:
            image_files = None

        # Load formulas from text file
        with open(formula_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(tqdm(f, desc="Loading formulas", ncols=80)):
                formula = line.strip()

                # Skip if corresponding image doesn't exist (when image_dir provided)
                if image_files is not None:
                    image_name = f"{i:07d}.png"
                    if image_name not in image_files:
                        continue

                if formula:
                    formulas.append(formula)

                if max_samples and len(formulas) >= max_samples:
                    break

        print(f"✓ Loaded {len(formulas)} formulas from {formula_path.name}")
        return formulas


def get_transforms(image_size: Tuple[int, int] = (192, 672), is_train: bool = True):
    """
    Get image transformations

    Args:
        image_size: (height, width) for resizing
        is_train: Whether for training (adds augmentation)
    """
    if is_train:
        return transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.3
                ),
                transforms.ToTensor(),
                transforms.Normalize(  # ImageNet normalization
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
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


def create_dataloaders(
    train_image_dir: Optional[str] = None,
    train_formula_file: Optional[str] = None,
    eval_configs: Optional[List[Tuple[str, str, str]]] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (192, 672),
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for training and evaluation

    Args:
        train_image_dir: Training images directory
        train_formula_file: Training formulas text file
        eval_configs: List of (name, image_dir, formula_file) for evaluation sets
        batch_size: Batch size
        num_workers: Number of worker processes
        image_size: Image dimensions (H, W)

    Returns:
        Dictionary of dataloaders
    """
    dataloaders = {}

    # Training dataloader
    if train_image_dir and train_formula_file:
        train_dataset = UniMERDataset(
            image_dir=train_image_dir,
            formula_file=train_formula_file,
            transform=get_transforms(image_size, is_train=True),
        )
        dataloaders["train"] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    # Evaluation dataloaders
    if eval_configs:
        for name, image_dir, formula_file in eval_configs:
            eval_dataset = UniMERDataset(
                image_dir=image_dir,
                formula_file=formula_file,
                transform=get_transforms(image_size, is_train=False),
            )
            dataloaders[name] = DataLoader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

    return dataloaders


def main():
    """Example usage"""
    parser = argparse.ArgumentParser(description="UniMERNet Dataset Processor")
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Root directory containing datasets",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Limit samples for testing"
    )
    args = parser.parse_args()

    # Example 1: Load training data (UniMER-1M)
    print("=" * 80)
    print("LOADING TRAINING DATA")
    print("=" * 80)

    train_dir = Path(args.data_root) / "UniMER-1M" / "images"
    train_file = Path(args.data_root) / "UniMER-1M" / "train.txt"

    if train_dir.exists() and train_file.exists():
        train_dataset = UniMERDataset(
            image_dir=train_dir.absolute().__str__(),
            formula_file=train_file.absolute().__str__(),
            transform=get_transforms(is_train=True),
            max_samples=args.max_samples,
        )
        print(f"Sample 0: {train_dataset[0]['text'][:]}...")
    else:
        print(f"⚠ Training data not found at {train_dir}")

    # Example 2: Load evaluation data (UniMER-Test)
    print("\n" + "=" * 80)
    print("LOADING EVALUATION DATA")
    print("=" * 80)

    eval_configs = [
        (
            "SPE",
            f"{args.data_root}/UniMER-Test/spe",
            f"{args.data_root}/UniMER-Test/spe.txt",
        ),
        (
            "CPE",
            f"{args.data_root}/UniMER-Test/cpe",
            f"{args.data_root}/UniMER-Test/cpe.txt",
        ),
        (
            "SCE",
            f"{args.data_root}/UniMER-Test/sce",
            f"{args.data_root}/UniMER-Test/sce.txt",
        ),
        (
            "HWE",
            f"{args.data_root}/UniMER-Test/hwe",
            f"{args.data_root}/UniMER-Test/hwe.txt",
        ),
    ]

    dataloaders = create_dataloaders(
        eval_configs=eval_configs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Test dataloaders
    for name, loader in dataloaders.items():
        print(f"\n{name} Dataset:")
        batch = next(iter(loader))
        print(f"  - Batch size: {batch['image'].shape[0]}")
        print(f"  - Image shape: {batch['image'].shape}")
        print(f"  - Sample text: {batch['text'][0][:]}...")

    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)

    for name, loader in dataloaders.items():
        total_samples = len(loader.dataset)  # type: ignore
        num_batches = len(loader)
        print(f"{name:4s}: {total_samples:6d} samples, {num_batches:4d} batches")


if __name__ == "__main__":
    r"""
    uv run python loader.py --data_root ../../datasets/UniMER_Dataset 

    ================================================================================
    LOADING TRAINING DATA
    ================================================================================
    Loading formulas: 1061791it [00:00, 1653207.89it/s]                             
    ✓ Loaded 986122 samples from /Users/jasonharris/Documents/workspace/Deep-Learning-Practices/latex_ocr/data/unimer/../../datasets/UniMER_Dataset/UniMER-1M/images
    Sample 0: n \to \infty...

    ================================================================================
    LOADING EVALUATION DATA
    ================================================================================
    Loading formulas: 234884it [00:00, 2158242.04it/s]                              
    ✓ Loaded 6762 samples from ../../datasets/UniMER_Dataset/UniMER-Test/spe
    Loading formulas: 100%|████████████████| 5921/5921 [00:00<00:00, 1027619.23it/s]
    ✓ Loaded 5921 samples from ../../datasets/UniMER_Dataset/UniMER-Test/cpe
    Loading formulas: 6708it [00:00, 2050684.49it/s]                                
    ✓ Loaded 4742 samples from ../../datasets/UniMER_Dataset/UniMER-Test/sce
    Loading formulas: 100%|████████████████| 6332/6332 [00:00<00:00, 1998068.98it/s]
    ✓ Loaded 6332 samples from ../../datasets/UniMER_Dataset/UniMER-Test/hwe

    SPE Dataset:
      - Batch size: 32
      - Image shape: torch.Size([32, 3, 192, 672])
      - Sample formula: S \sim \tilde { \psi } Q _ { o } \tilde { \psi } + g _ { s } ^ { 1 / 2 } \tilde { \psi } ^ { 3 } + \tilde { \phi } Q _ { c } \tilde { \phi } + g _ { s } \tilde { \phi } ^ { 3 } + \tilde { \phi } B ( g _ { s } ^ { 
    1 / 2 } \tilde { \psi } ) + \cdots ....

    CPE Dataset:
      - Batch size: 32
      - Image shape: torch.Size([32, 3, 192, 672])
      - Sample formula: \begin{array} { r l } { \mathcal { L } ( \{ \mathbf { u , v , w , z , x } \} , \{ \boldsymbol { \kappa , \lambda , \mu , \nu } \} ) = \frac { 1 } { 2 } \| \mathbf { y - C u } \| _ { 2 } ^ { 2 } } & { + \tau _ { 1 
    } \| \mathbf { v } \| _ { 1 } + \tau _ { 2 } \| \mathbf { w } \| _ { 1 } } \\ & { + \frac { \rho _ { 1 } } { 2 } \| \mathbf { A x - u } \| _ { 2 } ^ { 2 } + \boldsymbol { \kappa } ^ { \top } ( \mathbf { A x - u } ) } \\ & { + \frac {
     \rho _ { 2 } } { 2 } \| \mathbf { x - v } \| _ { 2 } ^ { 2 } + \boldsymbol { \lambda } ^ { \top } ( \mathbf { x - v } ) } \\ & { + \frac { \rho _ { 3 } } { 2 } \| \mathbf { D x - w } \| _ { 2 } ^ { 2 } + \boldsymbol { \mu } ^ { \top
     } ( \mathbf { D x - w } ) } \\ & { + \frac { \rho _ { 4 } } { 2 } \| \mathbf { x - z } \| _ { 2 } ^ { 2 } + \boldsymbol { \nu } ^ { \top } ( \mathbf { x - z } ) } \\ & { + \mathcal { I } _ { + } ( \mathbf { z } ) } \end{array}...

    SCE Dataset:
      - Batch size: 32
      - Image shape: torch.Size([32, 3, 192, 672])
      - Sample formula: F _ { i } [ z ] ( x , y ) = f _ { i } ( x , y , z ) \ i = 1 , \ldots , n ,...

    HWE Dataset:
      - Batch size: 32
      - Image shape: torch.Size([32, 3, 192, 672])
      - Sample formula: b _ { n + 1 } - b _ { n } = - 1...

    ================================================================================
    DATASET STATISTICS
    ================================================================================
    SPE :   6762 samples,  212 batches
    CPE :   5921 samples,  186 batches
    SCE :   4742 samples,  149 batches
    HWE :   6332 samples,  198 batches


    """
    main()
