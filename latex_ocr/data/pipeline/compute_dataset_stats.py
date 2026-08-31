"""
Compute statistics for the synthetic LaTeX OCR dataset.

Usage:
    PYTHONPATH=. uv run python latex_ocr/data/pipeline/compute_dataset_stats.py \
        --plain_root latex_ocr/datasets/synth/plain \
        --styled_root latex_ocr/datasets/synth/styled

    # With histogram output
    PYTHONPATH=. uv run python latex_ocr/data/pipeline/compute_dataset_stats.py \
        --plain_root latex_ocr/datasets/synth/plain \
        --styled_root latex_ocr/datasets/synth/styled \
        --histogram_output length_distribution.png
"""

import argparse
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
from tqdm import tqdm


def compute_split_stats(data_root: Path, split: str) -> Optional[Dict[str, Any]]:
    """Compute statistics for a single split.
    
    Returns:
        Dictionary with statistics including lengths list for histogram plotting,
        formula entries for top-N analysis, or None if the split data is not found.
    """
    jsonl_file = data_root / split / "metadata.jsonl"
    split_dir = data_root / split

    if not jsonl_file.exists():
        print(f"⚠️  File not found: {jsonl_file}")
        return None

    # Store tuples of (formula, file_name) for top-N analysis
    formula_entries: List[Dict[str, str]] = []
    missing_images = []
    total_entries = 0
    print(f"Loading {split} split from {jsonl_file.name}...")

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Reading {split}"):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                total_entries += 1

                # Validate image file exists
                file_name = entry.get("file_name", "")
                if file_name:
                    image_path = split_dir / file_name
                    if not image_path.exists():
                        missing_images.append(file_name)
                        continue  # Skip entries with missing images

                formula = entry.get("text", "")
                formula_entries.append({
                    "formula": formula,
                    "file_name": file_name,
                    "image_path": str(split_dir / file_name),
                })
            except json.JSONDecodeError:
                continue

    # Report missing images
    if missing_images:
        print(f"⚠️  {len(missing_images)} missing images in {split} split")
        if len(missing_images) <= 5:
            for img in missing_images:
                print(f"    - {img}")
        else:
            for img in missing_images[:3]:
                print(f"    - {img}")
            print(f"    ... and {len(missing_images) - 3} more")

    if not formula_entries:
        return None

    # Compute statistics
    lengths = [len(e["formula"]) for e in formula_entries]
    lengths_arr = np.array(lengths)

    # Sort entries by formula length (descending) for top-N
    sorted_entries = sorted(formula_entries, key=lambda x: len(x["formula"]), reverse=True)

    stats = {
        "num_samples": len(formula_entries),
        "total_entries": total_entries,
        "missing_images": len(missing_images),
        "avg_length": float(np.mean(lengths_arr)),
        "median_length": float(np.median(lengths_arr)),
        "min_length": int(np.min(lengths_arr)),
        "max_length": int(np.max(lengths_arr)),
        "std_length": float(np.std(lengths_arr)),
        "p95_length": float(np.percentile(lengths_arr, 95)),
        "p99_length": float(np.percentile(lengths_arr, 99)),
        "lengths": lengths,  # Keep raw lengths for histogram
        "top_entries": sorted_entries[:10],  # Top 10 longest formulas
    }

    return stats


def plot_histogram(
    results: Dict[str, Dict[str, Optional[Dict[str, Any]]]],
    output_path: Path,
    bins: int = 50,
) -> None:
    """Plot length distribution histograms for all datasets.
    
    Args:
        results: Dictionary containing stats for plain/styled train/validation splits
        output_path: Path to save the histogram image
        bins: Number of histogram bins
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for saving
    except ImportError:
        print("⚠️  matplotlib not installed. Install with: uv pip install matplotlib")
        return

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Formula Length Distribution", fontsize=16, fontweight="bold")

    plot_configs = [
        ("plain", "train", axes[0, 0], "Plain Train"),
        ("plain", "validation", axes[0, 1], "Plain Validation"),
        ("styled", "train", axes[1, 0], "Styled Train"),
        ("styled", "validation", axes[1, 1], "Styled Validation"),
    ]

    for dataset_type, split, ax, title in plot_configs:
        stats = results.get(dataset_type, {}).get(split)
        
        if stats is None or "lengths" not in stats:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=12)
            ax.set_title(title)
            continue

        lengths = stats["lengths"]
        
        # Plot histogram
        ax.hist(lengths, bins=bins, color="steelblue", edgecolor="white", alpha=0.7)
        
        # Add vertical lines for percentiles
        ax.axvline(
            stats["median_length"],
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {stats['median_length']:.0f}",
        )
        ax.axvline(
            stats["p95_length"],
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"P95: {stats['p95_length']:.0f}",
        )
        ax.axvline(
            stats["p99_length"],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"P99: {stats['p99_length']:.0f}",
        )

        ax.set_title(f"{title} (n={stats['num_samples']:,})")
        ax.set_xlabel("Formula Length (chars)")
        ax.set_ylabel("Count")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n📊 Histogram saved to: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compute dataset statistics")
    parser.add_argument(
        "--plain_root",
        type=str,
        required=True,
        help="Root directory for plain dataset",
    )
    parser.add_argument(
        "--styled_root",
        type=str,
        required=True,
        help="Root directory for styled dataset",
    )
    parser.add_argument(
        "--histogram_output",
        type=str,
        default=None,
        help="Output path for length distribution histogram (e.g., histogram.png)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins (default: 50)",
    )
    args = parser.parse_args()

    plain_root = Path(args.plain_root)
    styled_root = Path(args.styled_root)

    print("=" * 80)
    print("COMPUTING DATASET STATISTICS")
    print("=" * 80)

    # Compute stats for all splits
    results = {
        "plain": {
            "train": compute_split_stats(plain_root, "train"),
            "validation": compute_split_stats(plain_root, "validation"),
        },
        "styled": {
            "train": compute_split_stats(styled_root, "train"),
            "validation": compute_split_stats(styled_root, "validation"),
        },
    }

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    for dataset_type in ["plain", "styled"]:
        print(f"\n{dataset_type.upper()} Dataset:")
        print("-" * 60)

        for split in ["train", "validation"]:
            stats = results[dataset_type][split]
            if stats:
                print(f"\n  {split.capitalize()} Split:")
                print(f"    Total Entries:  {stats['total_entries']:>10,}")
                print(f"    Valid Samples:  {stats['num_samples']:>10,}")
                print(f"    Missing Images: {stats['missing_images']:>10,}")
                print(f"    Avg Length:     {stats['avg_length']:>10.1f} chars")
                print(f"    Median Length:  {stats['median_length']:>10.1f} chars")
                print(f"    Min Length:     {stats['min_length']:>10,} chars")
                print(f"    Max Length:     {stats['max_length']:>10,} chars")
                print(f"    Std Length:     {stats['std_length']:>10.1f} chars")
                print(f"    P95 Length:     {stats['p95_length']:>10.1f} chars")
                print(f"    P99 Length:     {stats['p99_length']:>10.1f} chars")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    header = f"{'Split':<12} {'Samples':<12} {'Avg':<10} {'Median':<10} {'P95':<10} {'P99':<10}"
    separator = "-" * 64

    print("\nPLAIN Dataset:")
    print(header)
    print(separator)
    for split in ["train", "validation"]:
        stats = results["plain"][split]
        if stats:
            print(
                f"{split:<12} {stats['num_samples']:<12,} "
                f"{stats['avg_length']:<10.1f} {stats['median_length']:<10.1f} "
                f"{stats['p95_length']:<10.1f} {stats['p99_length']:<10.1f}"
            )

    print("\nSTYLED Dataset:")
    print(header)
    print(separator)
    for split in ["train", "validation"]:
        stats = results["styled"][split]
        if stats:
            print(
                f"{split:<12} {stats['num_samples']:<12,} "
                f"{stats['avg_length']:<10.1f} {stats['median_length']:<10.1f} "
                f"{stats['p95_length']:<10.1f} {stats['p99_length']:<10.1f}"
            )

    # Total counts
    plain_total = sum(s["num_samples"] for s in results["plain"].values() if s)
    styled_total = sum(s["num_samples"] for s in results["styled"].values() if s)

    print("\n" + "=" * 80)
    print(
        f"TOTAL: Plain={plain_total:,} | Styled={styled_total:,} | Combined={plain_total + styled_total:,}"
    )
    print("=" * 80)

    # Print top-10 longest formulas for each dataset/split
    print("\n" + "=" * 80)
    print("TOP-10 LONGEST FORMULAS")
    print("=" * 80)

    for dataset_type in ["plain", "styled"]:
        for split in ["train", "validation"]:
            stats = results[dataset_type][split]
            if stats and "top_entries" in stats:
                print(f"\n{dataset_type.upper()} - {split.capitalize()}:")
                print("-" * 80)
                for i, entry in enumerate(stats["top_entries"], 1):
                    formula = entry["formula"]
                    length = len(formula)
                    image_path = entry["image_path"]
                    
                    # Truncate formula for display
                    display_formula = formula[:80] + "..." if len(formula) > 80 else formula
                    
                    print(f"\n  #{i} (length={length} chars)")
                    print(f"     Image: {image_path}")
                    print(f"     Formula: {display_formula}")

    # Generate histogram if requested
    if args.histogram_output:
        output_path = Path(args.histogram_output)
        plot_histogram(results, output_path, bins=args.bins)


if __name__ == "__main__":
    main()
