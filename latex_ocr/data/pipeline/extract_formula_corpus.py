import json
from pathlib import Path
from torch.utils.data import ConcatDataset, Dataset
from typing import Dict, Optional
import re
from collections import Counter

import click
from tqdm import tqdm
from latex_ocr.data.synth.loader import SyntheticLaTeXDataset
from latex_ocr.data.unimer.loader import UniMERDataset
from latex_ocr.data.pix2tex.loader import LaTeXOCRDataset


def extract_formula_corpus(
    data_root: Path,
):
    if not data_root.exists():
        raise ValueError(f"Data root path {data_root} does not exist.")

    plain_train_ds = SyntheticLaTeXDataset(
        "plain", local_dir=None, split="train", formula_only=True
    )
    plain_val_ds = SyntheticLaTeXDataset(
        "plain", local_dir=None, split="validation", formula_only=True
    )
    styled_train_ds = SyntheticLaTeXDataset(
        "styled", local_dir=None, split="train", formula_only=True
    )
    styled_val_ds = SyntheticLaTeXDataset(
        "styled", local_dir=None, split="validation", formula_only=True
    )
    ds = ConcatDataset([plain_train_ds, plain_val_ds, styled_train_ds, styled_val_ds])

    f = data_root / "formula_corpus.txt"
    cnt = 0
    skip_cnt = 0
    bar = tqdm(
        total=len(ds),
    )
    with f.open("w", encoding="utf-8") as fout:
        for dp in ds:
            bar.update(1)
            txt = dp["text"]
            txt = txt.strip()
            if not txt:
                print("⚠️  Empty formula found, skipping.")
                skip_cnt += 1
                continue
            fout.write(txt + "\n")
            cnt += 1
    print(f"✅ Extracted {cnt} formulas to {f}. Skipped {skip_cnt} empty formulas.")


def make_macros_statistic(corpus: Path):
    if not corpus.exists():
        raise ValueError(f"Corpus file {corpus} does not exist.")
    user_defined_symbols_f = corpus.parent / "user_defined_symbols.json"

    # Regex pattern to match LaTeX macros: backslash followed by one or more letters
    macro_pattern = re.compile(r"\\[a-zA-Z]+")

    # Use Counter for efficient counting
    macro_counter = Counter()

    # Read corpus file and extract macros
    print(f"📖 Reading corpus from {corpus}...")
    line_num = 0
    with corpus.open("r", encoding="utf-8") as fin:
        for line_num, line in enumerate(fin, 1):
            # Find all macros in the current line
            macros = macro_pattern.findall(line)
            macro_counter.update(macros)

            if line_num % 10000 == 0:
                print(
                    f"   Processed {line_num} lines, found {len(macro_counter)} unique macros..."
                )

    print(f"✅ Found {len(macro_counter)} unique macros across {line_num} formulas.")
    print(f"📊 Total macro occurrences: {sum(macro_counter.values())}")

    # Sort macros by count in descending order
    sorted_macros = [macro for macro, _ in macro_counter.most_common()]

    # Save comma-separated list to output file
    print(f"💾 Saving macro list to {user_defined_symbols_f}...")
    with user_defined_symbols_f.open("w", encoding="utf-8") as fout:
        fout.write(json.dumps(sorted_macros, ensure_ascii=False))

    print(
        f"✅ Macro statistics saved to {user_defined_symbols_f}, total {len(sorted_macros)} macros."
    )
    print(f"   Top 10 most common macros:")
    for macro, count in macro_counter.most_common(10):
        print(f"   {macro}: {count}")
    # test load
    l = json.loads(user_defined_symbols_f.read_text(encoding="utf-8"))
    print(f"Top 100 macros loaded from file: {l[:100]}")

    return macro_counter


def extract_unimer_corpus(
    data_root: Path,
    output_dir: Path,
):
    """
    Extract formula corpus from UniMER dataset.

    Args:
        data_root: Root directory containing UniMER_Dataset folder
        output_dir: Output directory for JSON files (formula_corpora)
    """
    unimer_root = data_root / "UniMER_Dataset"
    if not unimer_root.exists():
        raise ValueError(f"UniMER dataset not found at {unimer_root}")

    # Check for train and test data
    train_image_dir = unimer_root / "UniMER-1M" / "images"
    train_formula_file = unimer_root / "UniMER-1M" / "train.txt"

    if not train_image_dir.exists() or not train_formula_file.exists():
        raise ValueError(
            f"UniMER-1M training data not found at {unimer_root / 'UniMER-1M'}"
        )

    # Extract training formulas
    print("=" * 80)
    print("Extracting UniMER-1M training formulas...")
    print("=" * 80)
    train_formulas = UniMERDataset.get_formulas_only(
        formula_file=str(train_formula_file),
        image_dir=str(train_image_dir),
    )

    # Save train formulas as JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    train_output = output_dir / "unimer_train.json"
    with train_output.open("w", encoding="utf-8") as f:
        json.dump(train_formulas, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(train_formulas)} training formulas to {train_output}")

    # Extract test formulas from all test subsets
    print("\n" + "=" * 80)
    print("Extracting UniMER-Test formulas...")
    print("=" * 80)

    test_subsets = ["spe", "cpe", "sce", "hwe"]
    all_test_formulas = []

    for subset in test_subsets:
        test_image_dir = unimer_root / "UniMER-Test" / subset
        test_formula_file = unimer_root / "UniMER-Test" / f"{subset}.txt"

        if test_image_dir.exists() and test_formula_file.exists():
            print(f"\nProcessing {subset.upper()} subset...")
            subset_formulas = UniMERDataset.get_formulas_only(
                formula_file=str(test_formula_file),
                image_dir=str(test_image_dir),
            )
            all_test_formulas.extend(subset_formulas)
        else:
            print(f"⚠️  {subset.upper()} subset not found, skipping...")

    # Save test formulas as JSON
    test_output = output_dir / "unimer_test.json"
    with test_output.open("w", encoding="utf-8") as f:
        json.dump(all_test_formulas, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved {len(all_test_formulas)} test formulas to {test_output}")

    print("\n" + "=" * 80)
    print(f"UniMER corpus extraction complete!")
    print(f"  - Train: {len(train_formulas)} formulas → {train_output}")
    print(f"  - Test: {len(all_test_formulas)} formulas → {test_output}")
    print("=" * 80)


def extract_pix2tex_corpus(
    output_dir: Path,
    cache_dir: Optional[Path] = None,
):
    """
    Extract formula corpus from pix2tex (LaTeX-OCR) dataset.

    Args:
        output_dir: Output directory for JSON files (formula_corpora)
        cache_dir: Optional cache directory for HuggingFace datasets
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract training formulas
    print("=" * 80)
    print("Extracting pix2tex training formulas...")
    print("=" * 80)
    train_formulas = LaTeXOCRDataset.get_formulas_only(
        split="train",
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    # Save train formulas as JSON
    train_output = output_dir / "pix2tex_train.json"
    with train_output.open("w", encoding="utf-8") as f:
        json.dump(train_formulas, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(train_formulas)} training formulas to {train_output}")

    # Extract validation formulas
    print("\n" + "=" * 80)
    print("Extracting pix2tex validation formulas...")
    print("=" * 80)
    val_formulas = LaTeXOCRDataset.get_formulas_only(
        split="validation",
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    # Save validation formulas as JSON
    val_output = output_dir / "pix2tex_val.json"
    with val_output.open("w", encoding="utf-8") as f:
        json.dump(val_formulas, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(val_formulas)} validation formulas to {val_output}")

    print("\n" + "=" * 80)
    print(f"pix2tex corpus extraction complete!")
    print(f"  - Train: {len(train_formulas)} formulas → {train_output}")
    print(f"  - Val: {len(val_formulas)} formulas → {val_output}")
    print("=" * 80)


@click.group()
def cli():
    """
    CLI for extracting formula corpus and computing macro statistics.

    Usage examples:

    Extract synthetic corpus:
        PYTHONPATH=. uv run python latex_ocr/data/pipeline/extract_formula_corpus.py extract

    Extract UniMER dataset corpus:
        PYTHONPATH=. uv run python latex_ocr/data/pipeline/extract_formula_corpus.py extract-unimer

    Extract pix2tex dataset corpus:
        PYTHONPATH=. uv run python latex_ocr/data/pipeline/extract_formula_corpus.py extract-pix2tex

    Compute macro statistics:
        PYTHONPATH=. uv run python latex_ocr/data/pipeline/extract_formula_corpus.py stats
    """
    pass


@cli.command()
@click.option(
    "--data-root",
    type=click.Path(exists=False, path_type=Path),
    default=None,
    help="Root directory for datasets. Defaults to latex_ocr/datasets",
)
def extract(data_root: Optional[Path]):
    """Extract formula corpus from synthetic LaTeX datasets."""
    if data_root is None:
        data_root = Path(__file__).parent.parent.parent / "datasets"

    extract_formula_corpus(data_root)


@cli.command()
@click.option(
    "--corpus",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to the formula corpus file. Defaults to latex_ocr/datasets/formula_corpus.txt",
)
def stats(corpus: Optional[Path]):
    """Compute macro statistics from the formula corpus."""
    if corpus is None:
        corpus = Path(__file__).parent.parent.parent / "datasets" / "formula_corpus.txt"

    make_macros_statistic(corpus)


@cli.command()
@click.option(
    "--data-root",
    type=click.Path(exists=False, path_type=Path),
    default=None,
    help="Root directory for datasets. Defaults to latex_ocr/datasets",
)
@click.option(
    "--output-dir",
    type=click.Path(exists=False, path_type=Path),
    default=None,
    help="Output directory for formula corpus JSON files. Defaults to latex_ocr/datasets/formula_corpora",
)
def extract_unimer(data_root: Optional[Path], output_dir: Optional[Path]):
    """Extract formula corpus from UniMER dataset (train and test splits)."""
    if data_root is None:
        data_root = Path(__file__).parent.parent.parent / "datasets"

    if output_dir is None:
        output_dir = data_root / "formula_corpora"

    extract_unimer_corpus(data_root, output_dir)


@cli.command()
@click.option(
    "--data-root",
    type=click.Path(exists=False, path_type=Path),
    default=None,
    help="Root directory for datasets. Defaults to latex_ocr/datasets",
)
@click.option(
    "--output-dir",
    type=click.Path(exists=False, path_type=Path),
    default=None,
    help="Output directory for formula corpus JSON files. Defaults to latex_ocr/datasets/formula_corpora",
)
@click.option(
    "--cache-dir",
    type=click.Path(exists=False, path_type=Path),
    default=None,
    help="Cache directory for HuggingFace datasets",
)
def extract_pix2tex(
    data_root: Optional[Path], output_dir: Optional[Path], cache_dir: Optional[Path]
):
    """Extract formula corpus from pix2tex (LaTeX-OCR) dataset (train and validation splits)."""
    if data_root is None:
        data_root = Path(__file__).parent.parent.parent / "datasets"

    if output_dir is None:
        output_dir = data_root / "formula_corpora"

    extract_pix2tex_corpus(output_dir, cache_dir)


if __name__ == "__main__":
    cli()
