#!/bin/bash
# Install dependencies for the visualizer
# Usage: ./install_visualizer_deps.sh

echo "Installing streamlit and dependencies..."
uv pip install streamlit pillow pandas

echo "✅ Installation complete!"
echo ""
echo "Run the visualizer with:"
echo "  PYTHONPATH=. uv run streamlit run latex_ocr/data/pipeline/visualize_samples.py"
