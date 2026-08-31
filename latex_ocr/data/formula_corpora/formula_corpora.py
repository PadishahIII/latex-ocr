"""
Formula Corpus Dataset Loader

This module provides PyTorch Dataset classes to load LaTeX formula corpora from disk.
The corpus files are JSON files containing lists of LaTeX formula strings extracted
from various datasets (UniMER, pix2tex, etc.).

Example usage:

    from latex_ocr.data.formula_corpora.formula_corpora import (
        FormulaCorpusDataset,
        load_all_corpus_files,
    )

    # Load a single corpus file
    dataset = FormulaCorpusDataset("unimer_train.json")

    # Load multiple corpus files
    dataset = FormulaCorpusDataset(["unimer_train.json", "pix2tex_train.json"])

    # Load all available corpus files
    all_formulas = load_all_corpus_files()
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset


class FormulaCorpusDataset(Dataset):
    """
    PyTorch Dataset for loading LaTeX formula corpora from JSON files.

    Each sample returns a dictionary with:
    - "text": str, the LaTeX formula
    - "source": str, the source dataset name (e.g., "unimer_train")
    - "index": int, the index within the source file
    """

    def __init__(
        self,
        corpus_files: Union[str, List[str]],
        corpus_dir: Optional[Path] = None,
        max_samples: Optional[int] = None,
        include_metadata: bool = True,
    ):
        """
        Args:
            corpus_files: Single filename or list of filenames to load
            corpus_dir: Directory containing corpus files.
                       Defaults to latex_ocr/datasets/formula_corpora
            max_samples: Optional limit on total number of formulas to load
            include_metadata: If True, include source and index in returned dict
        """
        if corpus_dir is None:
            # Default to latex_ocr/datasets/formula_corpora
            corpus_dir = (
                Path(__file__).parent.parent.parent / "datasets" / "formula_corpora"
            )

        self.corpus_dir = Path(corpus_dir)
        self.include_metadata = include_metadata

        # Ensure corpus_files is a list
        if isinstance(corpus_files, str):
            corpus_files = [corpus_files]

        self.formulas: List[str] = []
        self.metadata: List[Dict[str, Union[str, int]]] = []

        # Load all specified corpus files
        for filename in corpus_files:
            self._load_corpus_file(filename)

        # Apply max_samples limit if specified
        if max_samples is not None and max_samples < len(self.formulas):
            self.formulas = self.formulas[:max_samples]
            self.metadata = self.metadata[:max_samples]

        print(
            f"✓ Loaded {len(self.formulas)} formulas from {len(corpus_files)} file(s)"
        )

    def _load_corpus_file(self, filename: str) -> None:
        """Load formulas from a single corpus JSON file."""
        filepath = self.corpus_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Corpus file not found: {filepath}")

        print(f"Loading formulas from {filename}...")
        with open(filepath, "r", encoding="utf-8") as f:
            formulas = json.load(f)

        if not isinstance(formulas, list):
            raise ValueError(f"Expected list in {filename}, got {type(formulas)}")

        # Extract source name from filename (e.g., "unimer_train.json" -> "unimer_train")
        source_name = filepath.stem

        # Add formulas and metadata
        start_idx = len(self.formulas)
        for idx, formula in enumerate(formulas):
            if not isinstance(formula, str):
                print(f"⚠️  Skipping non-string formula at index {idx} in {filename}")
                continue

            self.formulas.append(formula)
            self.metadata.append(
                {
                    "source": source_name,
                    "index": idx,
                }
            )

        print(f"  ✓ Loaded {len(formulas)} formulas from {filename}")

    def __len__(self) -> int:
        return len(self.formulas)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, int]]:
        """
        Get a formula by index.

        Returns:
            Dict with keys:
            - "text": str, the LaTeX formula
            - "source": str, source dataset name (if include_metadata=True)
            - "index": int, index within source file (if include_metadata=True)
        """
        if idx < 0 or idx >= len(self.formulas):
            raise IndexError(f"Index {idx} out of range [0, {len(self.formulas)})")

        result: Dict[str, Union[str, int]] = {"text": self.formulas[idx]}

        if self.include_metadata:
            result.update(self.metadata[idx])  # type: ignore

        return result

    @staticmethod
    def get_available_corpus_files(corpus_dir: Optional[Path] = None) -> List[str]:
        """
        Get list of available corpus JSON files in the corpus directory.

        Args:
            corpus_dir: Directory to search. Defaults to latex_ocr/datasets/formula_corpora

        Returns:
            List of corpus filenames (without path)
        """
        if corpus_dir is None:
            corpus_dir = (
                Path(__file__).parent.parent.parent / "datasets" / "formula_corpora"
            )

        corpus_dir = Path(corpus_dir)

        if not corpus_dir.exists():
            return []

        # Find all JSON files in the corpus directory
        json_files = sorted([f.name for f in corpus_dir.glob("*.json")])

        return json_files


def load_all_corpus_files(
    corpus_dir: Optional[Path] = None,
    max_samples: Optional[int] = None,
    include_metadata: bool = True,
) -> FormulaCorpusDataset:
    """
    Load all available corpus files from the corpus directory.

    Args:
        corpus_dir: Directory containing corpus files.
                   Defaults to latex_ocr/datasets/formula_corpora
        max_samples: Optional limit on total number of formulas to load
        include_metadata: If True, include source and index in returned dict

    Returns:
        FormulaCorpusDataset with all formulas loaded
    """
    available_files = FormulaCorpusDataset.get_available_corpus_files(corpus_dir)

    if not available_files:
        raise FileNotFoundError(f"No corpus files found in {corpus_dir}")

    print(f"Found {len(available_files)} corpus files:")
    for f in available_files:
        print(f"  - {f}")
    print()

    return FormulaCorpusDataset(
        corpus_files=available_files,
        corpus_dir=corpus_dir,
        max_samples=max_samples,
        include_metadata=include_metadata,
    )


def main():
    """Example usage and testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Formula Corpus Dataset Loader")
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default=None,
        help="Directory containing corpus files",
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        default=None,
        help="Specific corpus files to load (e.g., unimer_train.json pix2tex_train.json)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to load",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available corpus files and exit",
    )
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else None

    # List available files if requested
    if args.list:
        print("=" * 80)
        print("AVAILABLE CORPUS FILES")
        print("=" * 80)
        available = FormulaCorpusDataset.get_available_corpus_files(corpus_dir)
        if available:
            for f in available:
                print(f"  - {f}")
        else:
            print("  No corpus files found")
        return

    # Load dataset
    print("=" * 80)
    print("LOADING FORMULA CORPUS DATASET")
    print("=" * 80)

    if args.files:
        # Load specific files
        dataset = FormulaCorpusDataset(
            corpus_files=args.files,
            corpus_dir=corpus_dir,
            max_samples=args.max_samples,
        )
    else:
        # Load all files
        dataset = load_all_corpus_files(
            corpus_dir=corpus_dir,
            max_samples=args.max_samples,
        )

    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    print(f"Total formulas: {len(dataset)}")

    # Show some samples
    print("\n" + "=" * 80)
    print("SAMPLE FORMULAS")
    print("=" * 80)

    import random

    sample_indices = random.sample(range(len(dataset)), min(5, len(dataset)))

    for i, idx in enumerate(sample_indices, 1):
        sample = dataset[idx]
        formula = sample["text"]
        source = sample.get("source", "unknown")
        formula_str = str(formula)
        print(f"\nSample {i} (from {source}):")
        print(f"  {formula_str[:100]}{'...' if len(formula_str) > 100 else ''}")

    # Show source distribution
    print("\n" + "=" * 80)
    print("SOURCE DISTRIBUTION")
    print("=" * 80)

    from collections import Counter

    sources = [dataset[i].get("source", "unknown") for i in range(len(dataset))]
    source_counts = Counter(sources)

    for source, count in sorted(source_counts.items()):
        percentage = 100 * count / len(dataset)
        print(f"  {source:20s}: {count:8d} formulas ({percentage:5.2f}%)")


if __name__ == "__main__":
    """
    Example usage:
    
    # List available corpus files
    PYTHONPATH=. uv run python latex_ocr/data/formula_corpora/formula_corpora.py --list
    
    # Load all corpus files
    PYTHONPATH=. uv run python latex_ocr/data/formula_corpora/formula_corpora.py
    
    # Load specific files
    PYTHONPATH=. uv run python latex_ocr/data/formula_corpora/formula_corpora.py --files unimer_train.json pix2tex_train.json
    
    # Load with sample limit
    PYTHONPATH=. uv run python latex_ocr/data/formula_corpora/formula_corpora.py --max-samples 10000
    """
    main()
