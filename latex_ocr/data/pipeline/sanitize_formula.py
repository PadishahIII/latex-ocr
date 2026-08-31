"""
Formula Sanitization Pipeline

Implements the LaTeX space normalization rules from sanitize_formula.md to ensure
robust tokenization and prevent token merging errors.

Key Rules:
1. Protect control word boundaries (e.g., \mathcal A)
2. Preserve spaces inside text-like commands (e.g., \text{in region 1})
3. Preserve spaces after control symbols (e.g., \, x)
4. Never modify verbatim commands
5. Handle array/tabular row separators
6. Remove all other spaces

Usage:
    PYTHONPATH=. uv run python latex_ocr/data/pipeline/sanitize_formula.py --output formula_sanitized.json
"""

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from tqdm import tqdm
import click
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


# Regex patterns for sanitization rules
PATTERN_CONTROL_WORD = re.compile(r"\\[a-zA-Z]+\*?")
PATTERN_CONTROL_SYMBOL = re.compile(r"\\[^a-zA-Z\s]")
PATTERN_TEXT_LIKE = re.compile(
    r"\\(?:"
    r"text[a-zA-Z]*"           # \text, \textbf, \textit, \textnormal, \textsuperscript, etc.
    r"|math(?:rm|bf|it|sf|tt|cal|bb|frak|scr)"  # \mathrm, \mathbf, \mathit, \mathsf, \mathtt, \mathcal, \mathbb, \mathfrak, \mathscr
    r"|operatorname\*?"        # \operatorname and \operatorname*
    r"|[mhf]box"               # \mbox, \hbox, \fbox
    r"|makebox|parbox"         # \makebox, \parbox
    r"|boldsymbol|pmb"         # bold math symbols
    r"|intertext"              # \intertext (amsmath)
    r")\{[^}]*\}"
)
PATTERN_VERBATIM = re.compile(r"\\verb(?:\*)?(.)(?:.*?)(?:\1)")

# Common math-mode-only symbols that cannot appear in text mode
MATH_SYMBOLS = {
    r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon", r"\varepsilon",
    r"\zeta", r"\eta", r"\theta", r"\vartheta", r"\iota", r"\kappa",
    r"\lambda", r"\mu", r"\nu", r"\xi", r"\pi", r"\varpi", r"\rho",
    r"\varrho", r"\sigma", r"\varsigma", r"\tau", r"\upsilon", r"\phi",
    r"\varphi", r"\chi", r"\psi", r"\omega",
    # Uppercase Greek
    r"\Gamma", r"\Delta", r"\Theta", r"\Lambda", r"\Xi", r"\Pi",
    r"\Sigma", r"\Upsilon", r"\Phi", r"\Psi", r"\Omega",
    # Common math operators/symbols
    r"\infty", r"\partial", r"\nabla", r"\forall", r"\exists",
    r"\emptyset", r"\varnothing", r"\neg", r"\wedge", r"\vee",
    r"\cap", r"\cup", r"\in", r"\notin", r"\subset", r"\supset",
    r"\subseteq", r"\supseteq", r"\setminus", r"\times", r"\cdot",
    r"\pm", r"\mp", r"\leq", r"\geq", r"\neq", r"\approx", r"\equiv",
    r"\sim", r"\simeq", r"\propto", r"\perp", r"\parallel",
    r"\rightarrow", r"\leftarrow", r"\Rightarrow", r"\Leftarrow",
    r"\leftrightarrow", r"\Leftrightarrow", r"\mapsto",
    r"\sum", r"\prod", r"\int", r"\oint", r"\bigcup", r"\bigcap",
    r"\lim", r"\limsup", r"\liminf", r"\max", r"\min", r"\sup", r"\inf",
}

# Pattern to match text-mode commands with their content
PATTERN_TEXT_CMD_WITH_CONTENT = re.compile(
    r"(\\(?:textrm|textbf|textit|textnormal|textsf|texttt|text|mbox|hbox|fbox))\{([^}]*)\}"
)


def fix_math_in_text(formula: str) -> str:
    """
    Fix math symbols incorrectly placed inside text-mode commands.
    
    Example:
        \\textrm{\\mu isergodic} -> \\textrm{}\\mu\\textrm{ isergodic}
        \\text{for \\alpha > 0} -> \\text{for }\\alpha\\text{ > 0}
    """
    def replace_text_cmd(match):
        cmd = match.group(1)  # e.g., \textrm
        content = match.group(2)  # e.g., \mu isergodic
        
        # Check if content contains any math symbols
        has_math = False
        for sym in MATH_SYMBOLS:
            if sym in content:
                has_math = True
                break
        
        if not has_math:
            return match.group(0)  # Return unchanged
        
        # Split content around math symbols and rebuild
        result_parts = []
        current_text = ""
        i = 0
        
        while i < len(content):
            # Check if a math symbol starts here
            found_symbol = None
            for sym in MATH_SYMBOLS:
                if content[i:].startswith(sym):
                    # Verify it's a complete command (not part of longer word)
                    end_pos = i + len(sym)
                    if end_pos >= len(content) or not content[end_pos].isalpha():
                        found_symbol = sym
                        break
            
            if found_symbol:
                # Flush accumulated text
                if current_text:
                    result_parts.append(f"{cmd}{{{current_text}}}")
                    current_text = ""
                # Add the math symbol directly (outside text command)
                result_parts.append(found_symbol)
                i += len(found_symbol)
            else:
                current_text += content[i]
                i += 1
        
        # Flush remaining text
        if current_text:
            result_parts.append(f"{cmd}{{{current_text}}}")
        
        return "".join(result_parts)
    
    return PATTERN_TEXT_CMD_WITH_CONTENT.sub(replace_text_cmd, formula)


def find_protected_spans(formula: str) -> List[Tuple[int, int]]:
    """
    Find spans that should be completely protected from modification.

    Returns:
        List of (start, end) tuples for protected regions
    """
    protected = []

    # Protect verbatim commands (Rule 4)
    for match in PATTERN_VERBATIM.finditer(formula):
        protected.append((match.start(), match.end()))

    # Protect text-like command contents (Rule 2)
    for match in PATTERN_TEXT_LIKE.finditer(formula):
        protected.append((match.start(), match.end()))

    return protected


def is_in_protected_span(pos: int, protected_spans: List[Tuple[int, int]]) -> bool:
    """Check if position is within any protected span."""
    return any(start <= pos < end for start, end in protected_spans)


def sanitize_formula(formula: str) -> str:
    """
    Apply LaTeX space normalization rules to a formula.

    This is an optimized implementation that processes the formula in a single pass
    while respecting all protection rules.

    Args:
        formula: Input LaTeX formula string

    Returns:
        Sanitized formula with normalized spaces
    """
    if not formula or not formula.strip():
        return formula

    # Step 0: Fix math symbols inside text-mode commands (e.g., \textrm{\mu})
    formula = fix_math_in_text(formula)

    # Step 1: Find all protected spans (verbatim, text-like commands)
    protected_spans = find_protected_spans(formula)

    # Step 2: Process formula character by character with state machine
    result = []
    i = 0
    length = len(formula)

    while i < length:
        # Check if we're in a protected span - if so, copy verbatim
        in_protected = False
        for start, end in protected_spans:
            if start <= i < end:
                result.append(formula[start:end])
                i = end
                in_protected = True
                break

        if in_protected:
            continue

        char = formula[i]

        # Handle backslash (start of command)
        if char == "\\":
            if i + 1 < length:
                next_char = formula[i + 1]

                # Control symbol (Rule 3: preserve following space)
                if not next_char.isalpha():
                    result.append(char)
                    result.append(next_char)
                    i += 2
                    # Keep any immediately following space
                    if i < length and formula[i] == " ":
                        result.append(" ")
                        i += 1
                    continue

                # Control word - extract the full command
                match = PATTERN_CONTROL_WORD.match(formula, i)
                if match:
                    command = match.group()
                    result.append(command)
                    i = match.end()

                    # Rule 1: Check if next non-space character is alphanumeric
                    # If so, we need to keep exactly one space
                    j = i
                    space_count = 0
                    while j < length and formula[j] in " \t\n\r":
                        space_count += 1
                        j += 1

                    if j < length and (formula[j].isalnum()):
                        # Keep one space to prevent token merging
                        result.append(" ")
                        i = j  # Skip all spaces, we already added one
                    else:
                        # No alphanumeric follows, skip all spaces (Rule 6)
                        i = j

                    continue

            # Lone backslash at end
            result.append(char)
            i += 1
            continue

        # Handle whitespace (Rule 6: remove unless protected)
        if char in " \t\n\r":
            # Skip whitespace - it's either protected by rules 1-5 or should be removed
            i += 1
            continue

        # Handle braces and other structural characters
        if char in "{}":
            result.append(char)
            i += 1
            continue

        # Handle array/matrix separators (Rule 5)
        if char in "&":
            # Remove spaces before and after & in array contexts
            # Since we're already removing spaces, just add the separator
            result.append(char)
            i += 1
            continue

        # All other characters - copy as-is
        result.append(char)
        i += 1

    return "".join(result)


def sanitize_formulas_batch(
    formulas: List[str], show_progress: bool = True
) -> List[str]:
    """
    Sanitize a batch of formulas efficiently.

    Args:
        formulas: List of formula strings
        show_progress: Whether to show progress bar

    Returns:
        List of sanitized formulas
    """
    sanitized = []
    iterator = (
        tqdm(formulas, desc="Sanitizing formulas", ncols=80)
        if show_progress
        else formulas
    )

    for formula in iterator:
        try:
            sanitized.append(sanitize_formula(formula))
        except Exception as e:
            # On error, keep original formula and log warning
            print(f"\n⚠️  Error sanitizing formula: {e}")
            print(f"   Formula: {formula[:100]}...")
            sanitized.append(formula)

    return sanitized


def validate_latex_formula(
    formula: str, timeout: int = 5
) -> Tuple[bool, Optional[str]]:
    """
    Validate that a LaTeX formula can be compiled.

    Args:
        formula: LaTeX formula string
        timeout: Timeout in seconds for compilation

    Returns:
        Tuple of (success, error_message)
    """
    # Create a minimal LaTeX document
    latex_document = (
        r"""\documentclass{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsfonts}
\usepackage{mathtools}
\pagestyle{empty}
\begin{document}
$%s$
\end{document}
"""
        % formula
    )

    # Try to compile with pdflatex
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            tex_file = tmpdir_path / "test.tex"

            # Write LaTeX file
            with open(tex_file, "w", encoding="utf-8") as f:
                f.write(latex_document)

            # Run pdflatex
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "test.tex"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            # Check if PDF was created
            pdf_file = tmpdir_path / "test.pdf"
            if pdf_file.exists():
                return True, None
            else:
                # Extract error message from log
                log_file = tmpdir_path / "test.log"
                if log_file.exists():
                    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                        log_content = f.read()
                        # Find error line
                        for line in log_content.split("\n"):
                            if line.startswith("!"):
                                return False, line[:200]
                return False, "PDF not created"

    except subprocess.TimeoutExpired:
        return False, "Compilation timeout"
    except FileNotFoundError:
        return False, "pdflatex not found"
    except Exception as e:
        return False, f"Error: {str(e)[:100]}"


def validate_formula_wrapper(args: Tuple[int, str]) -> Tuple[int, bool, Optional[str]]:
    """Wrapper for parallel validation."""
    idx, formula = args
    success, error = validate_latex_formula(formula)
    return idx, success, error


def validate_formulas_batch(
    formulas: List[str],
    show_progress: bool = True,
    max_workers: Optional[int] = None,
) -> Dict:
    """
    Validate a batch of formulas in parallel.

    Args:
        formulas: List of formula strings
        show_progress: Whether to show progress bar
        max_workers: Number of parallel workers (default: CPU count)

    Returns:
        Dictionary with validation results
    """
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)

    results = {
        "total": len(formulas),
        "valid": 0,
        "invalid": 0,
        "errors": [],
    }

    print(f"🔍 Validating formulas with {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(validate_formula_wrapper, (i, formula)): i
            for i, formula in enumerate(formulas)
        }

        # Process results with progress bar
        pbar = None
        if show_progress:
            pbar = tqdm(total=len(formulas), desc="Validating", ncols=80)

        for future in as_completed(futures):
            idx, success, error = future.result()

            if success:
                results["valid"] += 1
            else:
                results["invalid"] += 1
                results["errors"].append(
                    {
                        "index": idx,
                        "formula": formulas[idx][:100],
                        "error": error,
                    }
                )

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

    return results


@click.group()
def cli():
    """Formula sanitization pipeline CLI."""
    pass


@click.command()
@click.option(
    "--input",
    type=str,
    required=True,
    help="Input JSON file to sanitize (e.g., unimer_train.json)",
)
@click.option(
    "--corpus-dir",
    type=click.Path(exists=False, path_type=Path),
    default=None,
    help="Directory containing corpus files. Defaults to latex_ocr/datasets/formula_corpora",
)
@click.option(
    "--output",
    type=str,
    default=None,
    help="Output filename (default: <input>_sanitized.json)",
)
@click.option(
    "--max-samples",
    type=int,
    default=None,
    help="Maximum number of formulas to process (for testing)",
)
@click.option(
    "--test",
    is_flag=True,
    help="Run tests on sanitization rules",
)
def sanitize(
    input: str,
    corpus_dir: Optional[Path],
    output: Optional[str],
    max_samples: Optional[int],
    test: bool,
):
    """Sanitize a single formula corpus file and output to JSON."""

    if test:
        run_tests()
        return

    print("=" * 80)
    print("FORMULA SANITIZATION PIPELINE")
    print("=" * 80)

    # Set up paths
    if corpus_dir is None:
        corpus_dir = (
            Path(__file__).parent.parent.parent / "datasets" / "formula_corpora"
        )
    else:
        corpus_dir = Path(corpus_dir)

    input_path = corpus_dir / input

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return

    # Determine output filename
    if output is None:
        # Extract base name and add _sanitized suffix
        input_stem = input_path.stem  # e.g., "unimer_train"
        output = f"{input_stem}_sanitized.json"

    output_path = corpus_dir / output

    # Load formulas
    print(f"\n📖 Loading formulas from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        formulas = json.load(f)

    if not isinstance(formulas, list):
        print(f"❌ Expected list in {input_path}, got {type(formulas)}")
        return

    print(f"✓ Loaded {len(formulas)} formulas")

    # Limit samples if requested
    if max_samples and max_samples < len(formulas):
        print(f"⚠️  Limiting to {max_samples} samples for testing")
        formulas = formulas[:max_samples]

    # Sanitize formulas
    print("\n🧹 Sanitizing formulas...")
    sanitized = sanitize_formulas_batch(formulas, show_progress=True)

    # Calculate statistics
    print("\n📊 Computing statistics...")
    num_changed = sum(1 for orig, san in zip(formulas, sanitized) if orig != san)
    change_rate = 100 * num_changed / len(formulas) if formulas else 0

    avg_length_before = sum(len(f) for f in formulas) / len(formulas) if formulas else 0
    avg_length_after = (
        sum(len(f) for f in sanitized) / len(sanitized) if sanitized else 0
    )
    compression = (
        100 * (avg_length_before - avg_length_after) / avg_length_before
        if avg_length_before > 0
        else 0
    )

    print(f"  - Total formulas: {len(formulas)}")
    print(f"  - Changed: {num_changed} ({change_rate:.2f}%)")
    print(f"  - Avg length before: {avg_length_before:.1f} chars")
    print(f"  - Avg length after: {avg_length_after:.1f} chars")
    print(f"  - Compression: {compression:.2f}%")

    # Save to JSON
    print(f"\n💾 Saving to {output_path}...")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sanitized, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(sanitized)} sanitized formulas to {output_path}")

    # Show some examples
    print("\n" + "=" * 80)
    print("SAMPLE TRANSFORMATIONS")
    print("=" * 80)

    import random

    sample_indices = []
    for _ in range(min(5, len(formulas))):
        for _ in range(100):  # Try to find changed formulas
            idx = random.randint(0, len(formulas) - 1)
            if formulas[idx] != sanitized[idx]:
                sample_indices.append(idx)
                break

    if not sample_indices:
        # If no changes found, just show random samples
        sample_indices = random.sample(range(len(formulas)), min(5, len(formulas)))

    for i, idx in enumerate(sample_indices[:5], 1):
        orig = formulas[idx]
        san = sanitized[idx]
        orig_str = str(orig)
        san_str = str(san)
        changed = "✏️ " if orig_str != san_str else "✓ "

        print(f"\n{changed}Example {i}:")
        print(f"  Before: {orig_str[:100]}{'...' if len(orig_str) > 100 else ''}")
        print(f"  After:  {san_str[:100]}{'...' if len(san_str) > 100 else ''}")


def run_tests():
    """Run tests on sanitization rules."""
    print("=" * 80)
    print("SANITIZATION RULE TESTS")
    print("=" * 80)

    test_cases = [
        # Rule 1: Control word boundaries
        (r"\mathcal A", r"\mathcal A", "Rule 1: Keep space after control word"),
        (r"\mathrm sin", r"\mathrm sin", "Rule 1: Keep space after control word"),
        (r"\frac{1}{2}", r"\frac{1}{2}", "Rule 1: No space needed with braces"),
        (
            r"\alpha  \beta",
            r"\alpha\beta",
            "Rule 6: Remove space between control words",
        ),
        # Rule 2: Text-like commands
        (
            r"\text{in region 1}",
            r"\text{in region 1}",
            "Rule 2: Preserve spaces in \\text",
        ),
        (
            r"\mathrm{max value}",
            r"\mathrm{max value}",
            "Rule 2: Preserve spaces in \\mathrm",
        ),
        (
            r"\operatorname{some op}",
            r"\operatorname{some op}",
            "Rule 2: Preserve spaces in \\operatorname",
        ),
        # Rule 3: Control symbols
        (r"\, x", r"\, x", "Rule 3: Preserve space after control symbol"),
        (r"\; y", r"\; y", "Rule 3: Preserve space after control symbol"),
        # Rule 6: Remove other spaces
        (r"a + b", r"a+b", "Rule 6: Remove spaces between tokens"),
        (r"x  ^  2", r"x^2", "Rule 6: Remove multiple spaces"),
        # Complex cases
        (r"\frac { 1 } { 2 }", r"\frac{1}{2}", "Complex: Remove spaces inside braces"),
        (
            r"\sum _ { i = 1 } ^ { n }",
            r"\sum_{i=1}^{n}",
            "Complex: Remove spaces in subscript/superscript",
        ),
    ]

    passed = 0
    failed = 0

    for i, (input_str, expected, description) in enumerate(test_cases, 1):
        result = sanitize_formula(input_str)
        status = "✅" if result == expected else "❌"

        if result == expected:
            passed += 1
        else:
            failed += 1

        print(f"\n{status} Test {i}: {description}")
        print(f"  Input:    '{input_str}'")
        print(f"  Expected: '{expected}'")
        print(f"  Got:      '{result}'")

    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 80)


@click.command()
@click.option(
    "--input",
    type=str,
    required=True,
    help="Input JSON file with formulas to validate (e.g., unimer_train_sanitized.json)",
)
@click.option(
    "--corpus-dir",
    type=click.Path(exists=False, path_type=Path),
    default=None,
    help="Directory containing corpus files. Defaults to latex_ocr/datasets/formula_corpora",
)
@click.option(
    "--max-samples",
    type=int,
    default=None,
    help="Maximum number of formulas to validate (for testing)",
)
@click.option(
    "--workers",
    type=int,
    default=None,
    help="Number of parallel workers (default: CPU count - 1)",
)
@click.option(
    "--show-errors",
    is_flag=True,
    help="Show detailed error messages for failed formulas",
)
def validate(
    input: str,
    corpus_dir: Optional[Path],
    max_samples: Optional[int],
    workers: Optional[int],
    show_errors: bool,
):
    """Validate that sanitized formulas can be compiled with LaTeX."""

    print("=" * 80)
    print("LATEX FORMULA VALIDATION")
    print("=" * 80)

    # Set up paths
    if corpus_dir is None:
        corpus_dir = (
            Path(__file__).parent.parent.parent / "datasets" / "formula_corpora"
        )
    else:
        corpus_dir = Path(corpus_dir)

    input_path = corpus_dir / input

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return

    # Load formulas
    print(f"\n📖 Loading formulas from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        formulas = json.load(f)

    if not isinstance(formulas, list):
        print(f"❌ Expected list in {input_path}, got {type(formulas)}")
        return

    print(f"✓ Loaded {len(formulas)} formulas")

    # Limit samples if requested
    if max_samples and max_samples < len(formulas):
        print(f"⚠️  Limiting validation to {max_samples} samples")
        formulas = formulas[:max_samples]

    # Validate formulas
    print(f"\n🔍 Validating {len(formulas)} formulas...")
    results = validate_formulas_batch(formulas, show_progress=True, max_workers=workers)

    # Print results
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print(f"Total:   {results['total']}")
    print(
        f"Valid:   {results['valid']} ({100 * results['valid'] / results['total']:.2f}%)"
    )
    print(
        f"Invalid: {results['invalid']} ({100 * results['invalid'] / results['total']:.2f}%)"
    )

    # Show errors if requested
    if show_errors and results["errors"]:
        print("\n" + "=" * 80)
        print(f"ERROR DETAILS (showing first {min(10, len(results['errors']))})")
        print("=" * 80)

        for i, error_info in enumerate(results["errors"][:10], 1):
            print(f"\n❌ Error {i} (index {error_info['index']}):")
            print(f"   Formula: {error_info['formula']}...")
            print(f"   Error:   {error_info['error']}")

    # Summary
    print("\n" + "=" * 80)
    if results["invalid"] == 0:
        print("✅ ALL FORMULAS PASSED VALIDATION!")
    else:
        print(f"⚠️  {results['invalid']} formulas failed validation")
        print("   Use --show-errors to see detailed error messages")
    print("=" * 80)
    print("LATEX FORMULA VALIDATION")
    print("=" * 80)

    # Set up paths
    if corpus_dir is None:
        corpus_dir = (
            Path(__file__).parent.parent.parent / "datasets" / "formula_corpora"
        )
    else:
        corpus_dir = Path(corpus_dir)

    input_path = corpus_dir / input

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return

    # Load formulas
    print(f"\n📖 Loading formulas from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        formulas = json.load(f)

    if not isinstance(formulas, list):
        print(f"❌ Expected list in {input_path}, got {type(formulas)}")
        return

    print(f"✓ Loaded {len(formulas)} formulas")

    # Limit samples if requested
    if max_samples and max_samples < len(formulas):
        print(f"⚠️  Limiting validation to {max_samples} samples")
        formulas = formulas[:max_samples]

    # Validate formulas
    print(f"\n🔍 Validating {len(formulas)} formulas...")
    results = validate_formulas_batch(formulas, show_progress=True, max_workers=workers)

    # Print results
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print(f"Total:   {results['total']}")
    print(
        f"Valid:   {results['valid']} ({100 * results['valid'] / results['total']:.2f}%)"
    )
    print(
        f"Invalid: {results['invalid']} ({100 * results['invalid'] / results['total']:.2f}%)"
    )

    # Show errors if requested
    if show_errors and results["errors"]:
        print("\n" + "=" * 80)
        print(f"ERROR DETAILS (showing first {min(10, len(results['errors']))})")
        print("=" * 80)

        for i, error_info in enumerate(results["errors"][:10], 1):
            print(f"\n❌ Error {i} (index {error_info['index']}):")
            print(f"   Formula: {error_info['formula']}...")
            print(f"   Error:   {error_info['error']}")

    # Summary
    print("\n" + "=" * 80)
    if results["invalid"] == 0:
        print("✅ ALL FORMULAS PASSED VALIDATION!")
    else:
        print(f"⚠️  {results['invalid']} formulas failed validation")
        print("   Use --show-errors to see detailed error messages")
    print("=" * 80)


@click.command()
@click.option(
    "--input",
    type=str,
    required=True,
    help="Input JSON file to count formulas",
)
@click.option(
    "--corpus-dir",
    type=click.Path(exists=False, path_type=Path),
    default=None,
    help="Directory containing corpus files. Defaults to latex_ocr/datasets/formula_corpora",
)
def count(
    input: str,
    corpus_dir: Optional[Path],
):
    """Count the number of formulas in a corpus JSON file."""

    # Set up paths
    if corpus_dir is None:
        corpus_dir = (
            Path(__file__).parent.parent.parent / "datasets" / "formula_corpora"
        )
    else:
        corpus_dir = Path(corpus_dir)

    input_path = corpus_dir / input

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return

    # Load formulas
    print(f"📖 Loading {input_path.name}...")
    with open(input_path, "r", encoding="utf-8") as f:
        formulas = json.load(f)

    if not isinstance(formulas, list):
        print(f"❌ Expected list in {input_path}, got {type(formulas)}")
        return

    # Compute statistics
    num_formulas = len(formulas)
    total_chars = sum(len(str(f)) for f in formulas)
    avg_length = total_chars / num_formulas if num_formulas > 0 else 0

    file_size_mb = input_path.stat().st_size / (1024 * 1024)

    print(f"\n📊 Statistics for {input_path.name}:")
    print(f"  - Total formulas: {num_formulas:,}")
    print(f"  - Total characters: {total_chars:,}")
    print(f"  - Average length: {avg_length:.1f} chars")
    print(f"  - File size: {file_size_mb:.2f} MB")


cli.add_command(sanitize)
cli.add_command(validate)
cli.add_command(count)


if __name__ == "__main__":
    """
    Usage examples:
    
    # Run tests
    PYTHONPATH=. uv run python latex_ocr/data/pipeline/sanitize_formula.py sanitize --input dummy.json --test
    
    # Sanitize a single file (output: unimer_train_sanitized.json)
    PYTHONPATH=. uv run python latex_ocr/data/pipeline/sanitize_formula.py sanitize --input unimer_train.json
    
    # Sanitize with custom output name
    PYTHONPATH=. uv run python latex_ocr/data/pipeline/sanitize_formula.py sanitize --input unimer_train.json --output my_output.json
    
    # Test with limited samples
    PYTHONPATH=. uv run python latex_ocr/data/pipeline/sanitize_formula.py sanitize --input pix2tex_val.json --max-samples 1000
    
    # Validate sanitized formulas
    PYTHONPATH=. uv run python latex_ocr/data/pipeline/sanitize_formula.py validate --input unimer_train_sanitized.json
    
    # Validate with error details
    PYTHONPATH=. uv run python latex_ocr/data/pipeline/sanitize_formula.py validate --input pix2tex_val_sanitized.json --show-errors
    
    # Validate limited samples
    PYTHONPATH=. uv run python latex_ocr/data/pipeline/sanitize_formula.py validate --input unimer_test_sanitized.json --max-samples 100
    
    # Count formulas in a corpus file
    PYTHONPATH=. uv run python latex_ocr/data/pipeline/sanitize_formula.py count --input unimer_train.json
    """
    cli()
