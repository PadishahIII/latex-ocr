"""
Streamlit app to visualize randomly sampled images from synthetic LaTeX OCR dataset.
Loads datasets from HuggingFace Hub (PadishahIIIXXX/latex-ocr-dataset).

Usage:
    # Install streamlit if needed
    uv pip install streamlit pillow datasets
    
    # Run the app
    PYTHONPATH=. uv run streamlit run latex_ocr/data/pipeline/visualize_samples.py -- \
        --num_samples 100 \
        --split train
"""

import argparse
import io
import random
from pathlib import Path
from typing import List, Dict, Tuple

import streamlit as st
from PIL import Image
from datasets import load_dataset


def load_samples_from_hub(
    config_name: str, split: str, num_samples: int = 100, seed: int = 42
) -> List[Dict]:
    """
    Load random samples from HuggingFace Hub.

    Args:
        config_name: Configuration name ("plain" or "styled")
        split: Dataset split ("train" or "validation")
        num_samples: Number of samples to load
        seed: Random seed for reproducibility

    Returns:
        List of dicts with 'image' (PIL Image) and 'formula' keys
    """
    repo_id = "PadishahIIIXXX/latex-ocr-dataset"

    # Load dataset from HuggingFace Hub
    st.info(f"Loading {config_name} dataset from HuggingFace Hub ({split} split)...")

    try:
        hf_dataset = load_dataset(
            repo_id,
            config_name,
            split=split,
            verification_mode="no_checks",
        )

        st.success(f"✓ Loaded {len(hf_dataset)} samples from {repo_id}/{config_name}")  # type: ignore

        # Random sample
        random.seed(seed)
        total_samples = len(hf_dataset)  # type: ignore

        if total_samples > num_samples:
            # Get random indices
            indices = random.sample(range(total_samples), num_samples)
            hf_dataset = hf_dataset.select(indices)  # type: ignore
            st.info(f"Randomly sampled {num_samples} from {total_samples} total")

        # Convert to list of dicts with PIL Images
        samples = []
        for idx in range(len(hf_dataset)):  # type: ignore
            sample = hf_dataset[idx]  # type: ignore

            # Get image from HuggingFace dataset
            image = sample["image"]
            if not isinstance(image, Image.Image):
                # If it's bytes, convert to PIL Image
                if isinstance(image, bytes):
                    image = Image.open(io.BytesIO(image)).convert("RGB")
                else:
                    image = Image.open(io.BytesIO(bytes(image))).convert("RGB")  # type: ignore
            else:
                image = image.convert("RGB")

            samples.append(
                {
                    "image": image,
                    "formula": sample["text"],
                }
            )

        return samples

    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return []


def display_samples(samples: List[Dict], title: str):
    """
    Display samples in a grid layout.

    Args:
        samples: List of sample dicts with 'image' (PIL Image) and 'formula' keys
        title: Title for this section
    """
    st.header(title)
    st.markdown(f"**Total Samples**: {len(samples)}")

    # Add filters
    col1, col2 = st.columns(2)
    with col1:
        min_length = st.slider(
            f"Min formula length ({title})",
            min_value=0,
            max_value=200,
            value=0,
            key=f"min_len_{title}",
        )
    with col2:
        max_length = st.slider(
            f"Max formula length ({title})",
            min_value=0,
            max_value=500,
            value=500,
            key=f"max_len_{title}",
        )

    # Filter samples
    filtered_samples = [
        s for s in samples if min_length <= len(s["formula"]) <= max_length
    ]

    st.markdown(f"**Filtered Samples**: {len(filtered_samples)} (showing up to 100)")

    # Display in grid
    samples_to_show = filtered_samples[:100]  # Limit to 100 for performance

    # Number of columns
    num_cols = st.selectbox(
        f"Columns ({title})",
        options=[1, 2, 3, 4],
        index=2,  # Default 3 columns
        key=f"cols_{title}",
    )

    # Display samples
    for idx in range(0, len(samples_to_show), num_cols):
        cols = st.columns(num_cols)

        for col_idx, col in enumerate(cols):
            sample_idx = idx + col_idx
            if sample_idx >= len(samples_to_show):
                break

            sample = samples_to_show[sample_idx]

            with col:
                try:
                    # Display image (already a PIL Image)
                    st.image(sample["image"], use_container_width=True)

                    # Display formula in expandable section
                    with st.expander(f"Formula #{sample_idx + 1}"):
                        st.code(sample["formula"], language="latex")
                        st.caption(f"Length: {len(sample['formula'])} chars")

                except Exception as e:
                    st.error(f"Error displaying image: {e}")


def main():
    st.set_page_config(
        page_title="Synthetic LaTeX OCR Visualizer",
        page_icon="📐",
        layout="wide",
    )

    st.title("📐 Synthetic LaTeX OCR Dataset Visualizer")
    st.markdown("""
    This app displays randomly sampled images from the synthetic LaTeX OCR dataset.
    Loads datasets from HuggingFace Hub (PadishahIIIXXX/latex-ocr-dataset).
    Use the sidebar to configure sampling parameters.
    """)

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # Get command-line args (if provided)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to load per dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation"],
        help="Dataset split to visualize",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )

    try:
        args = parser.parse_args()
    except SystemExit:
        # Streamlit may pass its own args, use defaults
        args = argparse.Namespace(
            num_samples=100,
            split="train",
            seed=42,
        )

    # Override with sidebar inputs
    split = st.sidebar.selectbox(
        "Split",
        options=["train", "validation"],
        index=0 if args.split == "train" else 1,
    )
    num_samples = st.sidebar.number_input(
        "Number of Samples",
        min_value=10,
        max_value=1000,
        value=args.num_samples,
        step=10,
    )
    seed = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=9999,
        value=args.seed,
        step=1,
    )

    # Load button
    if st.sidebar.button("🔄 Load Samples", type="primary"):
        st.session_state["load_trigger"] = True

    # Initialize session state
    if "plain_samples" not in st.session_state:
        st.session_state["plain_samples"] = []
    if "styled_samples" not in st.session_state:
        st.session_state["styled_samples"] = []
    if "load_trigger" not in st.session_state:
        st.session_state["load_trigger"] = False

    # Load samples
    if st.session_state["load_trigger"]:
        with st.spinner("Loading samples from HuggingFace Hub..."):
            # Load plain samples
            plain_samples = load_samples_from_hub("plain", split, num_samples, seed)

            # Load styled samples
            styled_samples = load_samples_from_hub("styled", split, num_samples, seed)

            st.session_state["plain_samples"] = plain_samples
            st.session_state["styled_samples"] = styled_samples
            st.session_state["load_trigger"] = False

        st.success("✅ Samples loaded successfully!")

    # Display samples
    if st.session_state["plain_samples"] or st.session_state["styled_samples"]:
        # Create tabs
        tab1, tab2, tab3 = st.tabs(
            ["📄 Plain Dataset", "🎨 Styled Dataset", "📊 Statistics"]
        )

        with tab1:
            if st.session_state["plain_samples"]:
                display_samples(
                    st.session_state["plain_samples"],
                    "Plain Dataset (PDF-style rendering)",
                )
            else:
                st.warning(
                    "No plain samples loaded. Click 'Load Samples' in the sidebar."
                )

        with tab2:
            if st.session_state["styled_samples"]:
                display_samples(
                    st.session_state["styled_samples"],
                    "Styled Dataset (Font-enriched rendering)",
                )
            else:
                st.warning(
                    "No styled samples loaded. Click 'Load Samples' in the sidebar."
                )

        with tab3:
            st.subheader("📊 Dataset Statistics")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Plain Dataset")
                if st.session_state["plain_samples"]:
                    plain_lengths = [
                        len(s["formula"]) for s in st.session_state["plain_samples"]
                    ]
                    st.metric("Samples", len(st.session_state["plain_samples"]))
                    st.metric(
                        "Avg Formula Length",
                        f"{sum(plain_lengths) / len(plain_lengths):.1f} chars",
                    )
                    st.metric("Min Length", f"{min(plain_lengths)} chars")
                    st.metric("Max Length", f"{max(plain_lengths)} chars")

                    # Length distribution
                    import pandas as pd

                    df = pd.DataFrame({"Length": plain_lengths})
                    st.bar_chart(df["Length"].value_counts().sort_index())

            with col2:
                st.markdown("### Styled Dataset")
                if st.session_state["styled_samples"]:
                    styled_lengths = [
                        len(s["formula"]) for s in st.session_state["styled_samples"]
                    ]
                    st.metric("Samples", len(st.session_state["styled_samples"]))
                    st.metric(
                        "Avg Formula Length",
                        f"{sum(styled_lengths) / len(styled_lengths):.1f} chars",
                    )
                    st.metric("Min Length", f"{min(styled_lengths)} chars")
                    st.metric("Max Length", f"{max(styled_lengths)} chars")

                    # Length distribution
                    import pandas as pd

                    df = pd.DataFrame({"Length": styled_lengths})
                    st.bar_chart(df["Length"].value_counts().sort_index())

    else:
        st.info(
            "👈 Click 'Load Samples' in the sidebar to start visualizing the dataset."
        )

        # Show example
        st.markdown("---")
        st.markdown("### Example Usage")
        st.code(
            """
# Run with default settings (loads from HuggingFace Hub)
PYTHONPATH=. uv run streamlit run latex_ocr/data/pipeline/visualize_samples.py

# Run with custom settings
PYTHONPATH=. uv run streamlit run latex_ocr/data/pipeline/visualize_samples.py -- \
    --num_samples 200 \
    --split validation \
    --seed 123
        """,
            language="bash",
        )


if __name__ == "__main__":
    main()
