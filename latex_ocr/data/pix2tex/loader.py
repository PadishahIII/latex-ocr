"""
LaTeX-OCR Dataset Loader
"""

from __future__ import annotations

import argparse
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union, cast

import torch
from datasets import Dataset as HFDataset
from datasets import IterableDataset as HFIterableDataset
from datasets import load_dataset
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

Split = Literal["train", "validation"]
HFData = Union[HFDataset, HFIterableDataset]


class LaTeXOCRDataset(Dataset):
    """PyTorch dataset wrapper around `lukbl/LaTeX-OCR-dataset`.

    Each sample in the underlying Hugging Face dataset has two fields:

    - ``image``: a PIL image containing the rendered formula.
    - ``text``: the corresponding LaTeX string.

    This wrapper exposes items as a dictionary with keys ``"image"`` and ``"text"``,
    where ``"image"`` is a float tensor in ``[0, 1]`` by default. You can override
    the image preprocessing pipeline by providing a custom ``transform``.
    """

    def __init__(
        self,
        split: Split = "train",
        *,
        transform: Optional[Callable] = None,
        cache_dir: Optional[str] = None,
        hf_kwargs: Optional[Dict] = None,
    ) -> None:
        """Create a LaTeX-OCR dataset loader.

        Parameters
        ----------
        split:
            Which split of the dataset to load. Must be ``"train"`` or
            ``"validation"`` as defined by the dataset card.
        transform:
            Optional torchvision-style transform applied to the PIL image.
            If ``None``, a default ``ToTensor`` transform is used.
        cache_dir:
            Optional path where the Hugging Face dataset should be cached.
        hf_kwargs:
            Additional keyword arguments forwarded to ``datasets.load_dataset``.
        """

        if split not in ("train", "validation"):
            raise ValueError(
                f"Unsupported split: {split!r}. Use 'train' or 'validation'."
            )

        self.split: Split = split

        extra: Dict = hf_kwargs or {}
        if cache_dir is not None:
            extra.setdefault("cache_dir", cache_dir)

        hf_ds = load_dataset("lukbl/LaTeX-OCR-dataset", split=split, **extra)
        self._dataset: HFData = cast(HFData, hf_ds)

        # Default transform converts to float tensor in [0, 1].
        self.transform: Callable = transform or transforms.ToTensor()

        # Some HF iterable datasets do not define __len__.
        self._has_len: bool = hasattr(self._dataset, "__len") or hasattr(
            self._dataset, "__len__"
        )

    def __len__(self) -> int:  # type: ignore[override]
        if not self._has_len:
            raise TypeError("Underlying Hugging Face dataset does not support __len__.")
        return len(self._dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> Dict:  # type: ignore[override]
        ex = self._dataset[index]  # type: ignore[index]
        image = ex["image"]
        text = ex["text"]

        # Convert grayscale to RGB if needed for consistency with other datasets
        if image.mode == "L":  # type: ignore
            image = image.convert("RGB")  # type: ignore

        img_tensor = self.transform(image)
        if not isinstance(img_tensor, torch.Tensor):
            raise TypeError("Transform must return a torch.Tensor.")

        return {
            "image": img_tensor,
            "text": str(text),
        }

    @staticmethod
    def get_formulas_only(
        split: Split = "train",
        cache_dir: Optional[str] = None,
        hf_kwargs: Optional[Dict] = None,
    ) -> List[str]:
        """
        Load only the formula text without loading images.
        This is much faster for generating synthetic datasets where we only need the LaTeX strings.

        Parameters
        ----------
        split:
            Which split of the dataset to load. Must be ``"train"`` or ``"validation"``.
        cache_dir:
            Optional path where the Hugging Face dataset should be cached.
        hf_kwargs:
            Additional keyword arguments forwarded to ``datasets.load_dataset``.

        Returns
        -------
        List[str]
            List of LaTeX formula strings
        """
        if split not in ("train", "validation"):
            raise ValueError(
                f"Unsupported split: {split!r}. Use 'train' or 'validation'."
            )

        extra: Dict = hf_kwargs or {}
        if cache_dir is not None:
            extra.setdefault("cache_dir", cache_dir)

        print(f"Loading LaTeX-OCR {split} formulas (text only, no images)...")
        hf_ds = load_dataset("lukbl/LaTeX-OCR-dataset", split=split, **extra)

        # Extract only the text field without loading/decoding images
        formulas: List[str] = []
        dataset_cast = cast(HFData, hf_ds)

        # Use with_format to prevent image decoding
        # This is much faster as it skips image loading entirely
        dataset_cast.set_format(type=None, columns=["text"])  # type: ignore

        for item in dataset_cast:  # type: ignore
            if isinstance(item, dict) and "text" in item:
                text = item["text"]
                if text:
                    formulas.append(str(text))

        # Reset format
        dataset_cast.reset_format()  # type: ignore

        print(f"✓ Loaded {len(formulas)} formulas from LaTeX-OCR {split} split")
        return formulas


def get_transforms(image_size: Tuple[int, int] = (192, 672)) -> Callable:
    """Return a deterministic transform similar to UniMER loader.

    The dataset images are grayscale (mode "L"), but ToTensor handles this and
    produces a 1xHxW tensor. Callers can adapt this if they need 3 channels.
    """

    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
    )


def main() -> None:
    """Inspect LaTeX-OCR HF dataset and print simple statistics.

    Example:
        uv run python loader.py --batch_size 32 --num_workers 4
    """

    parser = argparse.ArgumentParser(description="LaTeX-OCR HF Dataset Inspector")
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
        "--cache_dir",
        type=str,
        default=None,
        help="Optional cache directory for the HF dataset",
    )
    args = parser.parse_args()

    print("=" * 80)
    print(f"LOADING LaTeX-OCR DATASET ({args.split} split)")
    print("=" * 80)

    dataset = LaTeXOCRDataset(
        split=args.split, transform=get_transforms(), cache_dir=args.cache_dir
    )

    try:
        num_samples = len(dataset)
        print(f"✓ Loaded {num_samples} samples from lukbl/LaTeX-OCR-dataset")
    except TypeError:
        num_samples = -1
        print("Loaded streaming dataset (length unknown)")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Peek at a single batch
    batch = next(iter(dataloader))
    images = batch["image"]
    texts = batch["text"]
    print("\nSample batch:")
    print(f"  - Batch size: {images.shape[0]}")
    print(f"  - Image tensor shape: {images.shape}")
    print(f"  - Sample text: {texts[0][:80]}...")

    # Dataset statistics
    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)

    if num_samples > 0:
        num_batches = len(dataloader)
        print(
            f"Split {args.split!r}: {num_samples:7d} samples, {num_batches:4d} batches "
            f"(batch_size={args.batch_size})"
        )
    else:
        print(f"Split {args.split!r}: streaming dataset, batch_size={args.batch_size}")


if __name__ == "__main__":
    main()
