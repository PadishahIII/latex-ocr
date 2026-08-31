"""
Detect duplicate image references in the synthetic LaTeX OCR dataset.

This script checks for:
1. Multiple JSONL entries pointing to the same image file
2. Different formulas mapped to the same image (data corruption)
3. Orphaned images (images without JSONL entries)
4. Missing images (JSONL entries without corresponding images)

Usage:
    PYTHONPATH=. uv run python latex_ocr/data/pipeline/detect_duplicates.py \
        --data_root latex_ocr/datasets/synth

    # Check specific dataset type
    PYTHONPATH=. uv run python latex_ocr/data/pipeline/detect_duplicates.py \
        --data_root latex_ocr/datasets/synth \
        --dataset plain

    # With verbose output
    PYTHONPATH=. uv run python latex_ocr/data/pipeline/detect_duplicates.py \
        --data_root latex_ocr/datasets/synth \
        --verbose
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from tqdm import tqdm


def check_split(
    split_dir: Path,
    split_name: str,
    verbose: bool = False,
) -> Dict:
    """Check a single split for duplicates and issues.
    
    Args:
        split_dir: Path to the split directory (e.g., synth/plain/train)
        split_name: Name of the split for reporting
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary with detection results
    """
    jsonl_file = split_dir / "metadata.jsonl"
    images_dir = split_dir / "images"
    
    if not jsonl_file.exists():
        return {"error": f"JSONL file not found: {jsonl_file}"}
    
    # Track image -> list of (formula, line_number)
    image_to_formulas: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    
    # Track all referenced images
    referenced_images: Set[str] = set()
    
    # Parse JSONL file
    total_entries = 0
    parse_errors = 0
    
    print(f"\n📂 Checking {split_name}...")
    print(f"   JSONL: {jsonl_file}")
    
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(tqdm(f, desc=f"   Reading {split_name}"), 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
                total_entries += 1
                
                file_name = entry.get("file_name", "")
                formula = entry.get("text", "")
                
                if file_name:
                    image_to_formulas[file_name].append((formula, line_num))
                    referenced_images.add(file_name)
                    
            except json.JSONDecodeError as e:
                parse_errors += 1
                if verbose:
                    print(f"   ⚠️  JSON parse error at line {line_num}: {e}")
    
    # Find duplicates (same image, multiple entries)
    duplicates: List[Dict] = []
    same_formula_duplicates = 0
    different_formula_duplicates = 0
    
    for image_path, formula_list in image_to_formulas.items():
        if len(formula_list) > 1:
            formulas = [f[0] for f in formula_list]
            unique_formulas = set(formulas)
            
            dup_entry = {
                "image": image_path,
                "count": len(formula_list),
                "line_numbers": [f[1] for f in formula_list],
                "unique_formulas": len(unique_formulas),
                "formulas": formulas if verbose else formulas[:3],
            }
            duplicates.append(dup_entry)
            
            if len(unique_formulas) == 1:
                same_formula_duplicates += 1
            else:
                different_formula_duplicates += 1
    
    # Check for orphaned images (exist on disk but not in JSONL)
    orphaned_images: List[str] = []
    if images_dir.exists():
        all_images = set(f"images/{f.name}" for f in images_dir.glob("*.png"))
        orphaned_images = list(all_images - referenced_images)
    
    # Check for missing images (in JSONL but not on disk)
    missing_images: List[str] = []
    for image_path in referenced_images:
        full_path = split_dir / image_path
        if not full_path.exists():
            missing_images.append(image_path)
    
    results = {
        "split": split_name,
        "total_entries": total_entries,
        "unique_images": len(image_to_formulas),
        "parse_errors": parse_errors,
        "duplicate_count": len(duplicates),
        "same_formula_duplicates": same_formula_duplicates,
        "different_formula_duplicates": different_formula_duplicates,
        "orphaned_images": len(orphaned_images),
        "missing_images": len(missing_images),
        "duplicates": duplicates,
        "orphaned_list": orphaned_images[:10] if not verbose else orphaned_images,
        "missing_list": missing_images[:10] if not verbose else missing_images,
    }
    
    return results


def print_results(results: Dict, verbose: bool = False) -> None:
    """Print detection results for a split."""
    if "error" in results:
        print(f"   ❌ {results['error']}")
        return
    
    split = results["split"]
    
    print(f"\n   📊 {split} Results:")
    print(f"      Total JSONL entries:     {results['total_entries']:,}")
    print(f"      Unique images referenced: {results['unique_images']:,}")
    
    if results["parse_errors"] > 0:
        print(f"      ⚠️  JSON parse errors:    {results['parse_errors']:,}")
    
    # Duplicates
    if results["duplicate_count"] > 0:
        print(f"\n      🔴 DUPLICATES FOUND: {results['duplicate_count']}")
        print(f"         Same formula duplicates:      {results['same_formula_duplicates']}")
        print(f"         Different formula duplicates: {results['different_formula_duplicates']}")
        
        if results["different_formula_duplicates"] > 0:
            print(f"\n      ⚠️  WARNING: Different formulas point to the same image!")
            print(f"         This indicates data corruption from multiple runs.")
        
        if verbose and results["duplicates"]:
            print(f"\n      Duplicate details (showing up to 10):")
            for i, dup in enumerate(results["duplicates"][:10], 1):
                print(f"\n         [{i}] {dup['image']}")
                print(f"             Referenced {dup['count']} times at lines: {dup['line_numbers']}")
                print(f"             Unique formulas: {dup['unique_formulas']}")
                if dup['unique_formulas'] > 1:
                    print(f"             Formulas:")
                    for j, formula in enumerate(dup['formulas'][:3], 1):
                        truncated = formula[:60] + "..." if len(formula) > 60 else formula
                        print(f"               {j}. {truncated}")
    else:
        print(f"\n      ✅ No duplicates found")
    
    # Orphaned images
    if results["orphaned_images"] > 0:
        print(f"\n      ⚠️  Orphaned images (not in JSONL): {results['orphaned_images']}")
        if verbose and results["orphaned_list"]:
            for img in results["orphaned_list"][:5]:
                print(f"         - {img}")
    
    # Missing images
    if results["missing_images"] > 0:
        print(f"\n      ❌ Missing images (in JSONL but not on disk): {results['missing_images']}")
        if verbose and results["missing_list"]:
            for img in results["missing_list"][:5]:
                print(f"         - {img}")


def main():
    parser = argparse.ArgumentParser(
        description="Detect duplicate image references in synthetic LaTeX OCR dataset"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory of the synthetic dataset (e.g., datasets/synth)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["plain", "styled", "both"],
        default="both",
        help="Which dataset to check (default: both)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "validation", "both"],
        default="both",
        help="Which split to check (default: both)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed information about duplicates",
    )
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    
    if not data_root.exists():
        print(f"❌ Data root not found: {data_root}")
        return 1
    
    print("=" * 80)
    print("DUPLICATE DETECTION REPORT")
    print("=" * 80)
    print(f"Data root: {data_root}")
    
    # Determine which datasets and splits to check
    datasets = ["plain", "styled"] if args.dataset == "both" else [args.dataset]
    splits = ["train", "validation"] if args.split == "both" else [args.split]
    
    all_results = {}
    total_duplicates = 0
    total_different_formula_dups = 0
    
    for dataset in datasets:
        print(f"\n{'=' * 80}")
        print(f"📁 {dataset.upper()} DATASET")
        print("=" * 80)
        
        all_results[dataset] = {}
        
        for split in splits:
            split_dir = data_root / dataset / split
            split_name = f"{dataset}/{split}"
            
            results = check_split(split_dir, split_name, args.verbose)
            all_results[dataset][split] = results
            print_results(results, args.verbose)
            
            if "duplicate_count" in results:
                total_duplicates += results["duplicate_count"]
                total_different_formula_dups += results["different_formula_duplicates"]
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if total_duplicates == 0:
        print("\n✅ No duplicates found in any dataset!")
        print("   The dataset is clean and ready for use.")
    else:
        print(f"\n⚠️  Found {total_duplicates} total duplicate references")
        
        if total_different_formula_dups > 0:
            print(f"\n🔴 CRITICAL: {total_different_formula_dups} images have multiple different formulas!")
            print("   This indicates data corruption, likely from running the synthesis")
            print("   script multiple times without clearing the output directory.")
            print("\n   To fix this:")
            print("   1. Delete the corrupted dataset:")
            print(f"      rm -rf {data_root}/plain {data_root}/styled")
            print("   2. Re-run the synthesis script from scratch")
            print("   3. Or use --resume-from-batch if resuming an interrupted run")
        else:
            print("\n   All duplicates have the same formula (likely harmless).")
            print("   Consider deduplicating to reduce dataset size.")
    
    print("\n" + "=" * 80)
    
    return 1 if total_different_formula_dups > 0 else 0


if __name__ == "__main__":
    exit(main())
