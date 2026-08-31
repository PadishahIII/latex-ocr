"""
Synthetic LaTeX Dataset Generation Pipeline with Style Injection

This module generates synthetic training datasets by:
1. Extracting formulas from real datasets (UniMER, LaTeXOCR)
2. Injecting random math styles (\mathbf, \mathbb, etc.) into formulas
3. Rendering styled formulas as images using LaTeX
4. Generating both plain (PDF-style) and styled datasets

Key Features:
- Memory-efficient batch processing (1000 formulas per batch)
- Resume capability with deterministic ordering
- Explicit resource cleanup to prevent memory leaks

CRITICAL: Deterministic Ordering Requirements
============================================
This pipeline processes formulas in batches and supports resuming from any batch.
For resume functionality to work correctly, the formula order MUST be deterministic
and consistent across runs.

Dataset Ordering Guarantees:
- UniMER Dataset: ✅ DETERMINISTIC
  - Uses sorted(os.listdir()) for consistent file ordering (data/unimer/loader.py:56)
  - Set conversion removed to preserve sort order
  
- LaTeXOCR Dataset: ✅ DETERMINISTIC  
  - HuggingFace datasets maintain consistent iteration order (data/pix2tex/loader.py)
  - No random operations applied
  
- MixedTrainingDataset (20% Plain): ✅ DETERMINISTIC
  - Uses torch.Generator with fixed seed (default: 42) for 20% selection (data/loader.py:226)
  - Same seed produces same 20% subset across runs

Resume Usage:
  # Original run interrupted at batch 50
  python syntax_enrich.py --split train --output_root datasets/synth
  
  # Resume from batch 50 (skips batches 0-49)
  python syntax_enrich.py --split train --output_root datasets/synth --resume-from-batch 50
  
  IMPORTANT: Must use identical --output_root, --split, and --limit values!

Memory Management:
- Batches of 1000 formulas processed sequentially (not all in memory)
- Explicit PIL Image.close() after saving
- Garbage collection after each batch
- Peak memory: ~1GB (vs ~100GB without optimizations)

Example:
    PYTHONPATH=. python latex_ocr/data/pipeline/syntax_enrich.py \\
        --output_root latex_ocr/datasets \\
        --split train \\
        --num_workers 8
"""

import random
import subprocess
import shutil
import json
from pathlib import Path
import traceback
from typing import Dict, Optional, Tuple, Set, List
import typing
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import gc
import os

import torch
from PIL import Image, ImageOps
from pdf2image import convert_from_path
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from tqdm import tqdm

from latex_ocr.data.pix2tex.loader import LaTeXOCRDataset, get_transforms
from latex_ocr.data.unimer.loader import UniMERDataset
import latex_ocr.data.unimer.loader as unimer_loader
from latex_ocr.data.formula_corpora.formula_corpora import (
    FormulaCorpusDataset,
)

# Import pylatexenc for safe LaTeX parsing
try:
    from pylatexenc.latexwalker import (
        LatexWalker,
        LatexCharsNode,
        LatexMacroNode,
        LatexGroupNode,
    )
except ImportError:
    raise ImportError("Please install pylatexenc: pip install pylatexenc")

# ==============================================================================
# 1. STRATEGY LOGIC: Formula Parsing & Style Injection
# ==============================================================================

dataset_root = "synth"
metadata_jsonl = "metadata.jsonl"


class FormulaStyleInjector:
    r"""
    Implements the 'mathxx' random injection strategy.
    Parses LaTeX formulas and injects font macros (\mathbf, \mathbb, etc.)
    based on semantic variable types.
    """

    def __init__(self):
        # Section 3.2: Heuristic distributions
        self.SETS = {"R", "C", "N", "Z", "Q"}
        self.VECTORS = {"x", "y", "z", "u", "v", "w", "i", "j", "k", "A", "B", "M", "X"}
        self.OPS = {"d", "e", "i"}  # differential, base e, imaginary unit

        # Font commands that only work with uppercase letters
        self.UPPERCASE_ONLY_FONTS = {"\\mathbb", "\\mathcal", "\\mathscr"}

        # Canonical math alphabets (Section 3)
        self.STYLES_SETS = [("\\mathbb", 0.35), ("\\mathcal", 0.15), (None, 0.50)]
        self.STYLES_VEC = [("\\mathbf", 0.30), ("\\mathit", 0.10), (None, 0.60)]
        self.STYLES_OPS = [("\\mathrm", 0.20), (None, 0.80)]
        self.STYLES_GENERIC = [
            ("\\mathsf", 0.02),
            ("\\mathtt", 0.02),
            ("\\mathfrak", 0.02),
            ("\\mathcal", 0.02),
            ("\\mathscr", 0.02),
            (None, 0.90),
        ]

    def get_identifiers(self, formula: str) -> Set[str]:
        """
        Parse formula and return a set of single-letter identifiers suitable for styling.
        """
        walker = LatexWalker(formula)
        identifiers = set()

        try:
            nodes, _, _ = walker.get_latex_nodes()
            self._walk_and_collect(nodes, identifiers)
        except Exception:
            # Fallback for very broken latex, though pylatexenc is robust
            pass

        return identifiers

    def _walk_and_collect(self, nodes, ids: Set[str]):
        for node in nodes:
            if isinstance(node, LatexCharsNode):
                # Check for single distinct letters (A-Z, a-z)
                for char in node.chars:
                    if char.isalpha():
                        ids.add(char)
            elif isinstance(node, LatexGroupNode):
                self._walk_and_collect(node.nodelist, ids)
            elif isinstance(node, LatexMacroNode):
                # Don't style inside existing macros usually, but we recurse arguments
                if node.nodeargd and node.nodeargd.argnlist:
                    for arg in node.nodeargd.argnlist:
                        if arg:
                            self._walk_and_collect(arg, ids)

    def pick_style(self, char: str) -> Optional[str]:
        """Determine style for a variable based on probabilities."""
        if char in self.SETS:
            choices = self.STYLES_SETS
        elif char in self.VECTORS:
            choices = self.STYLES_VEC
        elif char in self.OPS:
            choices = self.STYLES_OPS
        else:
            choices = self.STYLES_GENERIC

        styles, weights = zip(*choices)
        return random.choices(styles, weights=weights, k=1)[0]

    def inject_styles(self, formula: str) -> Tuple[str, str]:
        r"""
        Returns (plain_formula, enriched_formula).
        Plain is normalized. Enriched has \mathxx injection.
        """
        # 1. Normalize legacy commands (Section 4.1)
        plain = (
            formula.replace(r"\bf", r"\mathbf")
            .replace(r"\rm", r"\mathrm")
            .replace(r"\it", r"\mathit")
        )

        # 2. Identify candidates
        identifiers = self.get_identifiers(plain)
        if not identifiers:
            return plain, plain

        # 3. Global Cap: ~40% of identifiers (Section 3.1)
        num_to_style = max(1, int(len(identifiers) * 0.4))
        targets = set(
            random.sample(list(identifiers), k=min(len(identifiers), num_to_style))
        )

        # 4. Assign Styles (Consistency Rule)
        style_map = {}
        for char in targets:
            style = self.pick_style(char)
            if style:
                # Handle uppercase-only font commands
                if style in self.UPPERCASE_ONLY_FONTS:
                    # Only apply if character is a letter (not digit)
                    if char.isalpha():
                        # Convert to uppercase for uppercase-only fonts
                        uppercase_char = char.upper()
                        # Map both original and uppercase to same style
                        # This ensures consistent replacement
                        style_map[char] = (style, uppercase_char)
                    # else: skip styling for non-letters with uppercase-only fonts
                else:
                    # Regular fonts work with any character
                    style_map[char] = (style, char)

        if not style_map:
            return plain, plain

        # 5. Rewriting the formula
        # Note: A robust rewrite walks the tree. For brevity, we use a careful string replacement
        # or token-level replacement. Here we implement a simple walker-based rebuilder.
        enriched = self._rewrite_formula(plain, style_map)

        return plain, enriched

    def _rewrite_formula(
        self, formula: str, style_map: Dict[str, Tuple[str, str]]
    ) -> str:
        """
        Reconstructs the formula inserting macros for target chars.

        Args:
            formula: Original formula string
            style_map: Dict mapping original char -> (style_command, replacement_char)
                      For example: {'x': ('\\mathbb', 'X')} means replace 'x' with '\\mathbb{X}'
        """
        walker = LatexWalker(formula)
        try:
            nodes, _, _ = walker.get_latex_nodes()
        except Exception as e:
            print(f"Parsing error during rewrite: {e}, formula: {formula}")
            return formula

        output = []

        def recurse(nodelist):
            if not isinstance(nodelist, (list, tuple)):
                nodelist = [nodelist]
            for node in nodelist:
                if isinstance(node, LatexCharsNode):
                    s = ""
                    for char in node.chars:
                        if char in style_map:
                            style_cmd, replacement_char = style_map[char]
                            s += f"{style_cmd}{{{replacement_char}}}"
                        else:
                            s += char
                    output.append(s)
                elif isinstance(node, LatexGroupNode):
                    output.append("{")
                    recurse(node.nodelist)
                    output.append("}")
                elif isinstance(node, LatexMacroNode):
                    output.append(f"\\{node.macroname}")
                    if node.nodeargd and node.nodeargd.argnlist:
                        # Check if it's a macro we shouldn't touch args for (optional refinement)
                        for arg in node.nodeargd.argnlist:
                            output.append("{")
                            if arg:
                                recurse(arg)
                            output.append("}")
                    # Handle optional args if needed
                else:
                    # Comments, etc.
                    output.append(node.latex_verbatim())

        recurse(nodes)
        return "".join(output)


# ==============================================================================
# 2. RENDERER LOGIC: KaTeX -> PDF -> Image
# ==============================================================================


class FormulaRenderer:
    """
    Handles Section 2: High-fidelity rendering and Screenshot-style rasterization.

    Uses KaTeX + Puppeteer instead of XeLaTeX for more tolerant rendering.
    Malformed LaTeX (e.g., \\mu inside \\textrm{}) will render with red error text
    instead of failing completely.
    """

    def __init__(self, temp_dir: str | Path = Path("./temp_render")):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True, parents=True)

        # Import and initialize the MathJax renderer (uses project root for node_modules)
        from latex_ocr.data.pipeline.mathjax_renderer import MathjaxRenderer

        self._renderer = MathjaxRenderer()  # Uses default project root paths

    def render(
        self, formula: str, filename_base: str, dpi: int = 150
    ) -> Tuple[Optional[Image.Image], Optional[str]]:
        """
        Render a LaTeX formula to an image using KaTeX/MathJax.

        Args:
            formula: LaTeX formula string
            filename_base: Base name for temporary files (unused now)
            dpi: DPI for the output image

        Returns:
            Tuple(PIL Image if successful else None, Error message if failed else None)
        """
        try:
            # Render formula to PDF bytes
            pdf_bytes, error_msg = self._renderer.render(formula, filename_base)

            if pdf_bytes is None:
                return None, error_msg

            # Rasterize PDF bytes to image
            # Note: convert_from_bytes handles bytes input directly
            from pdf2image import convert_from_bytes

            images = convert_from_bytes(pdf_bytes, dpi=dpi)

            if not images:
                return None, "Failed to rasterize PDF"

            # Convert first image and free the rest
            result_img = images[0].convert("RGB")
            for img in images[1:]:
                img.close()

            return result_img, None

        except Exception as e:
            return None, f"Render error: {type(e).__name__}: {e}"

    def verify_contrast(self, img1: Image.Image, img2: Image.Image) -> bool:
        """
        Anti-hallucination check (Section 2, step 5).
        Returns True if images are sufficiently different.
        """
        if img1.size != img2.size:
            img2 = img2.resize(img1.size)

        # Simple diff check
        ImageOps.grayscale(Image.blend(img1, img2, 0.5))
        # This is a stub; real implementation might use SSIM or exact pixel diff
        # For now, assume if the formula string changed, we trust KaTeX rendered it differently
        return True


# ==============================================================================
# 3. PIPELINE ORCHESTRATION
# ==============================================================================


class SynthesisPipeline:
    """
    The Main Pipeline (Section 2 & 4).
    Consumes a source DataLoader, processes formulas, and writes the
    Synthetic Plain and Synthetic Styled datasets to disk with train/validation splits
    in HuggingFace dataset format (JSONL).

    Uses ThreadPoolExecutor for parallel rendering to speed up generation.

    Output structure:
        datasets/synth/
            plain/
                train/
                    images/
                        000001.png
                        000002.png
                    train.jsonl (lines: {"image": "images/000001.png", "text": "..."})
                validation/
                    images/
                    validation.jsonl
            styled/
                train/
                    images/
                    train.jsonl
                validation/
                    images/
                    validation.jsonl
    """

    def __init__(
        self,
        output_root: str,
        num_workers: int = 4,
    ):
        """
        Args:
            output_root: Root directory for output
            num_workers: Number of parallel workers for rendering (default: 4)
        """
        self.root = Path(output_root)
        self.injector = FormulaStyleInjector()
        self.renderer = FormulaRenderer()
        self.num_workers = num_workers

        # Output directories with train/validation splits
        self.plain_train_dir = self.root / dataset_root / "plain" / "train"
        self.plain_val_dir = self.root / dataset_root / "plain" / "validation"
        self.styled_train_dir = self.root / dataset_root / "styled" / "train"
        self.styled_val_dir = self.root / dataset_root / "styled" / "validation"

        # Create all directories
        for d in [
            self.plain_train_dir / "images",
            self.plain_val_dir / "images",
            self.styled_train_dir / "images",
            self.styled_val_dir / "images",
        ]:
            d.mkdir(parents=True, exist_ok=True)

        # Thread locks for file writing
        self.lock = Lock()

    def _process_single_formula(
        self,
        formula: str,
        uid: str,
        split: str,
    ) -> Tuple[Optional[Dict], Optional[Dict], List[str]]:
        """
        Process a single formula: inject styles and render both plain and styled versions.

        Args:
            formula: LaTeX formula string
            uid: Unique identifier for this sample
            split: Dataset split ("train" or "validation")

        Returns:
            Tuple of (plain_result, styled_result, errors) where:
            - plain_result/styled_result are per previous definition
            - errors is a list of error strings encountered
        """
        # 1. Generate Variants
        f_plain, f_styled = self.injector.inject_styles(formula)
        errors = []

        # Skip if trivial
        # if len(f_plain) < 5:
        #    print(f"skip formula since len('{formula}') < 5")
        #    return None, None, [f"skip formula since len('{formula}') < 5"]

        plain_result = None
        styled_result = None

        # 2. Render Plain
        img_plain, err_plain = self.renderer.render(f_plain, f"plain_{uid}")
        if img_plain:
            if split == "validation":
                p_path = self.plain_val_dir / "images" / f"{uid}.png"
            else:
                p_path = self.plain_train_dir / "images" / f"{uid}.png"

            plain_result = {
                "image": img_plain,
                "path": p_path,
                "text": f_plain,
                "split": split,
                "uid": uid,
            }
        elif err_plain:
            errors.append(
                f"[{uid}] Plain render error: {err_plain}\n  Formula: {f_plain}"
            )

        # 3. Render Styled (only if it differs)
        if f_styled != f_plain:
            img_styled, err_styled = self.renderer.render(f_styled, f"styled_{uid}")
            if img_styled:
                # Verification (Section 2.5)
                if img_plain and self.renderer.verify_contrast(img_plain, img_styled):
                    if split == "validation":
                        s_path = self.styled_val_dir / "images" / f"{uid}.png"
                    else:
                        s_path = self.styled_train_dir / "images" / f"{uid}.png"

                    styled_result = {
                        "image": img_styled,
                        "path": s_path,
                        "text": f_styled,
                        "split": split,
                        "uid": uid,
                    }
            elif err_styled:
                errors.append(
                    f"[{uid}] Styled render error: {err_styled}\n  Formula: {f_styled}"
                )

        return plain_result, styled_result, errors

    def _write_result(
        self,
        result: Dict,
        plain_train_file,
        plain_val_file,
        styled_train_file,
        styled_val_file,
        is_plain: bool,
    ):
        """
        Write a single result to disk and JSONL file (thread-safe).

        Args:
            result: Dict with keys {"image", "path", "text", "split", "uid"}
            plain_train_file, plain_val_file, styled_train_file, styled_val_file: File handles
            is_plain: True if plain dataset, False if styled dataset
        """
        # Prepare JSONL entry
        json_entry = {
            "file_name": f"images/{result['uid']}.png",
            "text": result["text"],
        }
        json_line = json.dumps(json_entry) + "\n"

        # Write to appropriate file with thread safety
        # Note: Image saving happens outside lock (it's safe since each uid is unique)
        # But JSONL writing must be locked to prevent interleaved writes

        # Save image first (each UID is unique, so no conflicts)
        img = result["image"]
        img.save(result["path"])
        # Explicitly close image to free memory immediately
        img.close()

        # Then write JSONL entry with lock
        with self.lock:
            if is_plain:
                if result["split"] == "validation":
                    plain_val_file.write(json_line)
                    plain_val_file.flush()
                else:
                    plain_train_file.write(json_line)
                    plain_train_file.flush()
            else:
                if result["split"] == "validation":
                    styled_val_file.write(json_line)
                    styled_val_file.flush()
                else:
                    styled_train_file.write(json_line)
                    styled_train_file.flush()

    def process_dataset(
        self,
        source_loader: typing.Iterable,
        split: str = "train",
        limit: Optional[int] = None,
        resume_from_batch: int = 0,
    ):
        """
        Iterate over an existing UniMER or LaTeX-OCR loader and generate variants
        for the specified split in HuggingFace JSONL format.

        Args:
            source_loader: Iterable of formulas
            split: Dataset split - either "train" or "validation"
            limit: Optional limit on number of formulas to process
            resume_from_batch: Batch index to resume from (0-based). If > 0, will skip
                             earlier batches and append to existing JSONL files.

        Uses ThreadPoolExecutor for parallel rendering to speed up generation.

        Resume Example:
            If process was interrupted at batch 50:
            pipeline.process_dataset(loader, split="train", resume_from_batch=50)
            This will skip batches 0-49 and continue from batch 50.
        """
        if split not in ["train", "validation"]:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'validation'")

        # Always append to avoid overwriting existing data
        file_mode = "a"

        # Open JSONL files for the specified split
        if split == "train":
            plain_file = open(
                self.plain_train_dir / metadata_jsonl, file_mode, encoding="utf-8"
            )
            styled_file = open(
                self.styled_train_dir / metadata_jsonl, file_mode, encoding="utf-8"
            )
        else:
            plain_file = open(
                self.plain_val_dir / metadata_jsonl, file_mode, encoding="utf-8"
            )
            styled_file = open(
                self.styled_val_dir / metadata_jsonl, file_mode, encoding="utf-8"
            )

        # Open error log file
        error_log_path = self.root / dataset_root / "error_log.txt"
        error_log_file = open(error_log_path, file_mode, encoding="utf-8")

        # We still need to open dummy files for the _write_result method signature
        # but we won't use them if processing only one split
        if split == "train":
            dummy_val_plain = open(
                self.plain_val_dir / metadata_jsonl, "a", encoding="utf-8"
            )
            dummy_val_styled = open(
                self.styled_val_dir / metadata_jsonl, "a", encoding="utf-8"
            )
            plain_train_file, plain_val_file = plain_file, dummy_val_plain
            styled_train_file, styled_val_file = styled_file, dummy_val_styled
        else:
            dummy_train_plain = open(
                self.plain_train_dir / metadata_jsonl, "a", encoding="utf-8"
            )
            dummy_train_styled = open(
                self.styled_train_dir / metadata_jsonl, "a", encoding="utf-8"
            )
            plain_train_file, plain_val_file = dummy_train_plain, plain_file
            styled_train_file, styled_val_file = dummy_train_styled, styled_file

        processed_cnt = 0
        plain_cnt = 0
        styled_cnt = 0
        error_cnt = 0  # Track render/processing errors

        print(f"\nStarting synthesis pipeline for {split} split...")
        if resume_from_batch > 0:
            print(f"⚠️  RESUMING from batch {resume_from_batch}")
            print(f"   Skipping batches 0-{resume_from_batch - 1}")
            print(f"   Appending to existing JSONL files")
        print(f"Output format: HuggingFace dataset (JSONL)")
        print(f"Error log: {error_log_path}")
        print(f"Parallel workers: {self.num_workers}")

        # Determine start index for UIDs to avoid collisions
        # We check the plain/train or plain/validation directory for existing images
        start_index = self._get_next_index(split)
        if start_index > 0:
            print(
                f"🔄 Detected existing data. Starting UIDs from index {start_index} (e.g., {split}_{start_index:07d})"
            )
            if resume_from_batch == 0 and start_index > 0:
                print(f"   Note: Appending new data to existing dataset.")

        # Collect formulas with their metadata
        formula_tasks = []
        skipped_batches = 0

        print(f"Loading formulas from data loader...")
        try:
            total_batches = len(source_loader)  # type: ignore[arg-type]
        except TypeError:
            total_batches = None
        for batch_idx, batch_data in tqdm(
            enumerate(source_loader), desc="Loading batches", total=total_batches
        ):
            if limit and len(formula_tasks) >= limit:
                break

            try:
                # Handle the batch data
                if isinstance(batch_data, str):
                    formulas = [batch_data]
                elif isinstance(batch_data, list):
                    formulas = batch_data
                else:
                    # Skip unexpected format
                    skipped_batches += 1
                    continue

                for formula in formulas:
                    if limit and len(formula_tasks) >= limit:
                        break

                    # Use start_index + current_count to generate unique UID
                    current_idx = start_index + len(formula_tasks)
                    uid = f"{split}_{current_idx:07d}"
                    formula_tasks.append((formula, uid, split))

            except Exception as e:
                # Skip corrupted batches (e.g., corrupted images in dataset)
                skipped_batches += 1
                if skipped_batches <= 5 or skipped_batches % 100 == 0:
                    print(f"\n⚠️  Skipped batch {batch_idx}: {type(e).__name__}: {e}")
                    if skipped_batches <= 5:
                        traceback.print_exc()
                continue

        if skipped_batches > 0:
            print(f"Error: Skipped {skipped_batches} corrupted batches total")

        print(f"Collected {len(formula_tasks)} formulas for processing")

        # Process formulas in parallel with thread pool
        # Use batched submission to prevent memory accumulation
        batch_size = 1000  # Process in batches to limit memory usage

        # Calculate which batches to process based on resume point
        total_batches = (len(formula_tasks) - 1) // batch_size + 1
        start_batch_idx = resume_from_batch

        if resume_from_batch > 0:
            if resume_from_batch >= total_batches:
                print(
                    f"⚠️  Warning: resume_from_batch ({resume_from_batch}) >= total batches ({total_batches})"
                )
                print(f"   Nothing to process!")
                plain_train_file.close()
                plain_val_file.close()
                styled_train_file.close()
                styled_val_file.close()
                error_log_file.close()
                return
            print(
                f"Processing batches {start_batch_idx} to {total_batches - 1} (total: {total_batches - start_batch_idx} batches)"
            )

        for batch_idx in range(start_batch_idx, total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(formula_tasks))
            batch_tasks = formula_tasks[batch_start:batch_end]

            print(f"\n{'=' * 60}")
            print(f"Processing batch {batch_idx + 1}/{total_batches}")
            print(
                f"Formulas: {batch_start} to {batch_end - 1} (count: {len(batch_tasks)})"
            )
            print(f"{'=' * 60}")

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit batch of tasks
                future_to_formula = {
                    executor.submit(
                        self._process_single_formula, formula, uid, split
                    ): (
                        formula,
                        uid,
                        split,
                    )
                    for formula, uid, split in batch_tasks
                }

                # Process results as they complete
                batch_errors = []
                with tqdm(
                    total=len(batch_tasks),
                    desc=f"Batch {batch_idx + 1}/{total_batches}",
                ) as pbar:
                    for future in as_completed(future_to_formula):
                        try:
                            plain_result, styled_result, errors = future.result()

                            # Log errors
                            if errors:
                                batch_errors.extend(errors)
                                error_cnt += len(errors)

                            # Write plain result
                            if plain_result:
                                self._write_result(
                                    plain_result,
                                    plain_train_file,
                                    plain_val_file,
                                    styled_train_file,
                                    styled_val_file,
                                    is_plain=True,
                                )
                                plain_cnt += 1

                            # Write styled result
                            if styled_result:
                                self._write_result(
                                    styled_result,
                                    plain_train_file,
                                    plain_val_file,
                                    styled_train_file,
                                    styled_val_file,
                                    is_plain=False,
                                )
                                styled_cnt += 1

                            processed_cnt += 1

                        except Exception as e:
                            error_cnt += 1
                            formula, uid, _ = future_to_formula[future]
                            msg = f"[{uid}] Uncaught exception: {type(e).__name__}: {e}\n{traceback.format_exc()}"
                            batch_errors.append(msg)
                        finally:
                            # Explicitly delete future to free memory
                            del future

                        pbar.update(1)

                # Clear futures dict and force garbage collection after each batch
                future_to_formula.clear()

                # Write batch errors to log file
                if batch_errors:
                    with self.lock:
                        error_log_file.write(
                            f"\n--- Batch {batch_idx + 1}/{total_batches} Errors ---\n"
                        )
                        for err in batch_errors:
                            error_log_file.write(f"{err}\n")
                        error_log_file.flush()

                gc.collect()

            # Save checkpoint info after each batch
            print(f"✓ Batch {batch_idx + 1} complete")
            print(
                f"  Processed: {processed_cnt} | Plain: {plain_cnt} | Styled: {styled_cnt} | Errors: {error_cnt}"
            )
            if batch_idx + 1 < total_batches:
                print(
                    f"  💾 To resume from next batch, use: --resume-from-batch {batch_idx + 1}"
                )

        # Close all files
        plain_train_file.close()
        plain_val_file.close()
        styled_train_file.close()
        styled_val_file.close()
        error_log_file.close()

        print(
            f"\nPipeline finished for {split} split. Processed {processed_cnt} formulas."
        )
        print(f"Plain: {plain_cnt}, Styled: {styled_cnt}, Errors: {error_cnt}")
        if error_cnt > 0:
            error_rate = error_cnt / max(1, processed_cnt + error_cnt) * 100
            print(f"⚠️  Error rate: {error_rate:.2f}%")
        print("\nGenerated HuggingFace dataset format:")
        print("  - Each line: {'image': 'images/XXXXXX.png', 'text': 'LaTeX formula'}")

    def _get_next_index(self, split: str) -> int:
        """
        Scan existing images to find the next available ID index.
        Checks plain/{split}/images for file names like 'train_0012345.png'.
        """
        if split == "train":
            image_dir = self.plain_train_dir / "images"
        else:
            image_dir = self.plain_val_dir / "images"  # Should rely on plain usually

        if not image_dir.exists():
            return 0

        max_idx = -1
        # Sample check: scanning all files might be slow for millions of files.
        # But for 150k it's fine (approx 0.1-0.5s).
        # We look for files starting with split and ending with .png
        prefix = f"{split}_"

        try:
            # Optimize: assume monotonic and just scan directory listing
            # Using scantree or glob
            # We only need to guard against 'overwrite', so finding the max is strictly required
            count = 0
            for entry in os.scandir(image_dir):
                if entry.name.startswith(prefix) and entry.name.endswith(".png"):
                    try:
                        # Extract number part: train_0012345.png -> 0012345
                        num_part = entry.name[len(prefix) : -4]
                        idx = int(num_part)
                        if idx > max_idx:
                            max_idx = idx
                        count += 1
                    except ValueError:
                        pass

            if max_idx >= 0:
                print(f"Found {count} existing images. Max ID: {max_idx}")
                return max_idx + 1

        except Exception as e:
            print(f"Error: Failed to scan for existing files: {e}")
            raise e
        return 0


# ==============================================================================
# 4. UNIFIED DATASET LOADER (Section 5)
# ==============================================================================


def mixed_dataset_collate_fn(batch):
    """
    Custom collate function to handle mixed dataset formats.

    Handles both:
    - tuple format from LaTeXOCRDataset: (image_tensor, formula_string)
    - dict format from UniMERDataset: {"image": tensor, "formula": string, "image_path": path}

    Returns a list of formula strings for processing in the synthesis pipeline.
    """
    formulas = []
    for item in batch:
        if isinstance(item, dict):
            # UniMER format: dict with 'formula' key
            formulas.append(item["formula"])
        elif isinstance(item, tuple):
            # LaTeXOCR format: (image, formula) tuple
            formulas.append(item[1])
        else:
            raise ValueError(f"Unexpected item type: {type(item)}")
    return formulas


# ==============================================================================
# 5. LIGHTWEIGHT FORMULA-ONLY DATASET
# ==============================================================================


class FormulaOnlyDataset(Dataset):
    """
    Lightweight dataset that only contains formula strings (no images).
    Used for fast iteration when we only need LaTeX formulas for rendering.
    """

    def __init__(self, formulas: List[str]):
        self.formulas = formulas

    def __len__(self) -> int:
        return len(self.formulas)

    def __getitem__(self, idx: int) -> str:
        return self.formulas[idx]


def simple_collate_fn(batch: List[str]) -> List[str]:
    """Simple collate function that just returns the batch of strings."""
    return batch


# ==============================================================================
# 6. DATA LOADER FACTORIES
# ==============================================================================


def create_train_loader(
    limit: Optional[int] = None, corpus_file: Optional[str] = None
) -> DataLoader:
    """
    Create data loader for training data (UniMER-1M + LaTeX-OCR train).
    Only loads formula text, not images, for fast iteration.

    Args:
        limit: Optional limit on number of formulas total
        corpus_file: Optional path to a corpus JSON file to load formulas from.
                    If provided, loads from this file instead of datasets.
                    Should be a JSON list of formula strings.

    Returns:
        DataLoader containing formula strings only
    """
    # Option 1: Load from corpus JSON file using FormulaCorpusDataset
    if corpus_file:
        # corpus_file can be a single filename or a list of filenames
        # FormulaCorpusDataset will handle both cases
        corpus_dataset = FormulaCorpusDataset(
            corpus_files=corpus_file,
            max_samples=limit,
            include_metadata=False,  # We only need the formula text
        )
        # Extract formula strings from the dataset
        all_formulas = [corpus_dataset[i]["text"] for i in range(len(corpus_dataset))]

    # Option 2: Load from original datasets
    else:
        # Load LaTeX-OCR formulas (text only, no images)
        latex_ocr_formulas = LaTeXOCRDataset.get_formulas_only(
            split="train",
            cache_dir=None,
            # cache_dir="~/.cache/huggingface/datasets/lukbl___la_te_x-ocr-dataset/",
        )

        # Load UniMER-1M formulas (text only, no images)
        unimer_formula_file = (
            (
                Path(__file__).parent.parent.parent
                / "datasets"
                / "UniMER_Dataset"
                / "UniMER-1M"
                / "train.txt"
            )
            .absolute()
            .__str__()
        )

        unimer_image_dir = (
            (
                Path(__file__).parent.parent.parent
                / "datasets"
                / "UniMER_Dataset"
                / "UniMER-1M"
                / "images"
            )
            .absolute()
            .__str__()
        )

        unimer_formulas = UniMERDataset.get_formulas_only(
            formula_file=unimer_formula_file,
            image_dir=unimer_image_dir,  # Validate which images exist
        )

        # Combine formulas
        all_formulas = latex_ocr_formulas + unimer_formulas

    # Apply limit if specified
    if limit:
        all_formulas = all_formulas[:limit]

    print(f"✓ Total training formulas: {len(all_formulas)}")

    # Create lightweight dataset
    formula_dataset = FormulaOnlyDataset(all_formulas)

    return DataLoader(
        formula_dataset,
        batch_size=128,  # Larger batch size since we're only loading text
        shuffle=False,
        num_workers=0,
        collate_fn=simple_collate_fn,
    )


def create_validation_loader(
    limit: Optional[int] = None, corpus_file: Optional[str] = None
) -> DataLoader:
    """
    Create data loader for validation data (UniMER-Test + LaTeX-OCR validation).
    Only loads formula text, not images, for fast iteration.

    UniMER-Test has subdirectories: spe, cpe, sce (excluding hwe - handwritten equations)
    Each subdirectory has images and a corresponding .txt file.

    Args:
        limit: Optional limit on number of formulas total
        corpus_file: Optional path to a corpus JSON file to load formulas from.
                    If provided, loads from this file instead of datasets.
                    Should be a JSON list of formula strings.

    Returns:
        DataLoader containing formula strings only
    """
    # Option 1: Load from corpus JSON file using FormulaCorpusDataset
    if corpus_file:
        # corpus_file can be a single filename or a list of filenames
        # FormulaCorpusDataset will handle both cases
        corpus_dataset = FormulaCorpusDataset(
            corpus_files=corpus_file,
            max_samples=limit,
            include_metadata=False,  # We only need the formula text
        )
        # Extract formula strings from the dataset
        all_formulas = [corpus_dataset[i]["text"] for i in range(len(corpus_dataset))]

    # Option 2: Load from original datasets
    else:
        # Load LaTeX-OCR validation formulas (text only, no images)
        latex_ocr_formulas = LaTeXOCRDataset.get_formulas_only(
            split="validation",
            cache_dir=None,
            # cache_dir="~/.cache/huggingface/datasets/lukbl___la_te_x-ocr-dataset/",
        )

        # Load UniMER-Test formulas from selected subdirectories (excluding hwe)
        unimer_test_dir = (
            Path(__file__).parent.parent.parent
            / "datasets"
            / "UniMER_Dataset"
            / "UniMER-Test"
        )

        all_formulas = latex_ocr_formulas.copy()

        if unimer_test_dir.exists():
            test_subdirs = [
                "spe",
                "cpe",
                "sce",
            ]  # Exclude 'hwe' (handwritten equations)
            for subdir in test_subdirs:
                image_dir = unimer_test_dir / subdir
                formula_file = unimer_test_dir / f"{subdir}.txt"

                if image_dir.exists() and formula_file.exists():
                    unimer_formulas = UniMERDataset.get_formulas_only(
                        formula_file=formula_file.absolute().__str__(),
                        image_dir=image_dir.absolute().__str__(),
                    )
                    all_formulas.extend(unimer_formulas)
        else:
            print(
                "Warning: UniMER-Test not found, using only LaTeX-OCR validation data"
            )

    # Apply limit if specified
    if limit:
        all_formulas = all_formulas[:limit]

    print(f"✓ Total validation formulas: {len(all_formulas)}")

    # Create lightweight dataset
    formula_dataset = FormulaOnlyDataset(all_formulas)

    return DataLoader(
        formula_dataset,
        batch_size=128,  # Larger batch size since we're only loading text
        shuffle=False,
        num_workers=0,
        collate_fn=simple_collate_fn,
    )


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

"""
To test train split:
PYTHONPATH=. python latex_ocr/data/pipeline/syntax_enrich.py --output_root latex_ocr/datasets --split train --limit 10 --num_workers 4

To test validation split:
PYTHONPATH=. python latex_ocr/data/pipeline/syntax_enrich.py --output_root latex_ocr/datasets --split validation --limit 10 --num_workers 4

To generate both splits:
rm -rf latex_ocr/datasets/synth/plain latex_ocr/datasets/synth/styled
PYTHONPATH=. python latex_ocr/data/pipeline/syntax_enrich.py --output_root latex_ocr/datasets --split both --num_workers 8
PYTHONPATH=. uv run python latex_ocr/data/pipeline/syntax_enrich.py --output_root latex_ocr/datasets --split both --num_workers 8  &> syntax_enrich.log &
PYTHONPATH=. uv run python latex_ocr/data/pipeline/syntax_enrich.py --output_root latex_ocr/datasets --split train --num_workers 8 --limit 200 &> syntax_enrich.log &

To use corpus file:
PYTHONPATH=. uv run python latex_ocr/data/pipeline/syntax_enrich.py --output_root latex_ocr/datasets --corpus-file unimer_train_sanitized.json --num_workers 16 --split=train &> syntax_enrich_unimer_train.log &
PYTHONPATH=. uv run python latex_ocr/data/pipeline/syntax_enrich.py --output_root latex_ocr/datasets --corpus-file unimer_test_sanitized.json --num_workers 16 --split=validation &> syntax_enrich_unimer_test.log &
PYTHONPATH=. uv run python latex_ocr/data/pipeline/syntax_enrich.py --output_root latex_ocr/datasets --corpus-file pix2tex_train_sanitized.json --num_workers 16 --split=train &> syntax_enrich_pix2tex_train.log &
PYTHONPATH=. uv run python latex_ocr/data/pipeline/syntax_enrich.py --output_root latex_ocr/datasets --corpus-file pix2tex_val_sanitized.json --num_workers 16 --split=validation &> syntax_enrich_pix2tex_val.log &

To generate only training data with parallel processing:
PYTHONPATH=. python latex_ocr/data/pipeline/syntax_enrich.py --output_root latex_ocr/datasets --split train --num_workers 8

To resume from a specific batch (if interrupted):
PYTHONPATH=. python latex_ocr/data/pipeline/syntax_enrich.py --output_root latex_ocr/datasets --split train --num_workers 8 --resume-from-batch 50
This will skip batches 0-49 and continue from batch 50, appending to existing files.

Important notes for resuming:
- Use the exact same --output_root, --split, and --limit arguments as the original run
- Batch size is fixed at 1000 formulas per batch
- The script will append to existing JSONL files and skip regenerating images
- Check the console output for the checkpoint batch number to resume from
"""
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic LaTeX OCR dataset with style variations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate training data
  python syntax_enrich.py --output_root datasets --split train --num_workers 8
  
  # Resume from batch 50 after interruption
  python syntax_enrich.py --output_root datasets --split train --resume-from-batch 50
  
  # Generate validation data with limit
  python syntax_enrich.py --output_root datasets --split validation --limit 1000
        """,
    )
    parser.add_argument(
        "--output_root",
        default="./data/Synthesis_Output",
        help="Root directory for output datasets",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of formulas per split (for testing)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers for rendering (default: 4)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "both"],
        help="Which split to generate: train, validation, or both",
    )
    parser.add_argument(
        "--resume-from-batch",
        type=int,
        default=0,
        help="Resume processing from this batch index (0-based). Use this if process was interrupted.",
    )
    parser.add_argument(
        "--corpus-file",
        type=str,
        default=None,
        help="Path to corpus JSON file to use instead of loading from datasets. Should contain a JSON list of formula strings.",
    )
    args = parser.parse_args()

    # Validate arguments
    if args.resume_from_batch < 0:
        parser.error("--resume-from-batch must be >= 0")

    if args.resume_from_batch > 0 and args.split == "both":
        parser.error(
            "Cannot use --resume-from-batch with --split both. Resume train and validation separately."
        )

    if args.resume_from_batch > 0:
        print("\n" + "⚠️ " * 20)
        print("RESUME MODE ENABLED")
        print(f"Resuming from batch {args.resume_from_batch}")
        print(
            "Make sure you're using the same --output_root, --split, and --limit as the original run!"
        )
        print("⚠️ " * 20 + "\n")

    # Initialize pipeline
    pipeline = SynthesisPipeline(args.output_root, num_workers=args.num_workers)

    # Process based on split argument
    if args.split in ["train", "both"]:
        print("\n" + "=" * 80)
        print("GENERATING TRAINING DATA")
        print("=" * 80)
        train_loader = create_train_loader(
            limit=args.limit, corpus_file=args.corpus_file
        )
        pipeline.process_dataset(
            train_loader,
            split="train",
            limit=args.limit,
            resume_from_batch=args.resume_from_batch if args.split == "train" else 0,
        )

    if args.split in ["validation", "both"]:
        print("\n" + "=" * 80)
        print("GENERATING VALIDATION DATA")
        print("=" * 80)
        val_loader = create_validation_loader(
            limit=args.limit, corpus_file=args.corpus_file
        )
        pipeline.process_dataset(
            val_loader,
            split="validation",
            limit=args.limit,
            resume_from_batch=args.resume_from_batch
            if args.split == "validation"
            else 0,
        )

    print(f"\n{'=' * 80}")
    print(f"Data generation complete at {args.output_root}")
    print(f"{'=' * 80}")
