"""Gradio web UI: upload a formula image, run OCR, display the LaTeX.

    latex-ocr webui --model models/checkpoints/<ckpt>.pth --device cpu
"""

import gradio as gr

from latex_ocr.serve import LatexOCRPredictor

APP_TITLE = "latex-ocr"
APP_DESCRIPTION = (
    "Upload an image of a LaTeX formula — the model returns the LaTeX source. "
    "Runs on CPU."
)


def build_app(
    model_path: str,
    device: str,
    max_length: int,
    beam_size: int,
    max_upload_mb: int,
) -> gr.Blocks:
    """Build the Gradio Blocks UI (lazy model load on first use)."""
    _ = max_upload_mb  # size limits are enforced by the ASGI/server layer
    predictor: LatexOCRPredictor | None = None
    load_error: str | None = None

    def _get_predictor() -> LatexOCRPredictor:
        nonlocal predictor, load_error
        if predictor is None:
            if load_error is not None:
                # Already tried and failed: keep reporting the same error.
                raise RuntimeError(load_error)
            try:
                predictor = LatexOCRPredictor(
                    model_path=model_path,
                    device=device,
                    max_length=max_length,
                    beam_size=beam_size,
                )
            except Exception as exc:  # pragma: no cover - surfaced in the UI
                load_error = f"Failed to load model from {model_path!r}: {exc}"
                raise RuntimeError(load_error) from exc
        return predictor

    def run_ocr(image):
        if image is None:
            raise gr.Error("Please upload an image first.")
        latex = _get_predictor().predict(image)
        return f"$${latex}$$", latex  # display-math for gr.Latex + raw source

    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(f"# {APP_TITLE}\n{APP_DESCRIPTION}")

        with gr.Row():
            with gr.Column():
                image_in = gr.Image(type="pil", label="Formula image")
                with gr.Row():
                    submit_btn = gr.Button("Recognize", variant="primary")
                    clear_btn = gr.ClearButton([image_in], value="Clear")
            with gr.Column():
                rendered_out = gr.Latex(label="Rendered")
                latex_out = gr.Textbox(label="LaTeX source", lines=3, show_copy_button=True)

        submit_btn.click(
            fn=run_ocr,
            inputs=[image_in],
            outputs=[rendered_out, latex_out],
            concurrency_limit=1,
        )
        image_in.upload(
            fn=run_ocr,
            inputs=[image_in],
            outputs=[rendered_out, latex_out],
            concurrency_limit=1,
        )

        gr.Markdown(
            "Model: release-1.0.0 (67M, CoCa) · "
            "[code](https://github.com/PadishahIII/latex-ocr) · "
            "[model card](https://huggingface.co/PadishahIIIXXX/latex-ocr)"
        )

    return demo
