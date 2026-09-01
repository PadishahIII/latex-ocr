"""Facade CLI for latex-ocr.

One entry point (`latex-ocr`) that starts either the Gradio web UI or the
FastAPI inference API server, sharing the same model/inference options:

    latex-ocr webui --model <ckpt.pth> --device cpu
    latex-ocr api   --model <ckpt.pth> --device cpu --port 8000
"""

import click

from latex_ocr.serve import (
    DEFAULT_BEAM_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MAX_UPLOAD_MB,
    DEFAULT_MODEL_PATH,
)


def _shared_model_options(cmd):
    """Attach the model/inference options common to both servers."""

    def _wrap(f):
        for decorator in reversed(
            [
                click.option(
                    "--model",
                    "model_path",
                    default=DEFAULT_MODEL_PATH,
                    show_default=True,
                    help="Path to a full-model (.pth) or state_dict checkpoint.",
                ),
                click.option(
                    "--device",
                    "device",
                    default=DEFAULT_DEVICE,
                    show_default=True,
                    help="torch device, e.g. cpu, cuda, cuda:1.",
                ),
                click.option(
                    "--max-length",
                    default=DEFAULT_MAX_LENGTH,
                    show_default=True,
                    help="Max LaTeX token length to generate.",
                ),
                click.option(
                    "--beam-size",
                    default=DEFAULT_BEAM_SIZE,
                    show_default=True,
                    help="Beam width for decoding (1 = greedy).",
                ),
            ]
        ):
            f = decorator(f)
        return f

    return _wrap(cmd)


@click.group()
@click.version_option(package_name="latex-ocr")
def main() -> None:
    """latex-ocr: run the web UI (`webui`) or the inference API server (`api`)."""


@main.command("webui")
@click.option("--host", default="127.0.0.1", show_default=True,
              help="Bind address for the Gradio UI.")
@click.option("--port", default=7860, show_default=True, type=int)
@click.option(
    "--share/--no-share",
    "share",
    default=False,
    show_default=True,
    help="Create a temporary public gradio.live link.",
)
@click.option("--max-upload-mb", default=DEFAULT_MAX_UPLOAD_MB, show_default=True, type=int)
@_shared_model_options
def webui(
    model_path: str,
    device: str,
    max_length: int,
    beam_size: int,
    max_upload_mb: int,
    host: str,
    port: int,
    share: bool,
) -> None:
    """Start the Gradio web UI (browser: upload an image, get LaTeX)."""
    from latex_ocr.webui import build_app

    demo = build_app(
        model_path=model_path,
        device=device,
        max_length=max_length,
        beam_size=beam_size,
        max_upload_mb=max_upload_mb,
    )
    demo.queue(max_size=16)
    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        show_error=True,
        max_file_size=f"{max_upload_mb}mb",
    )
    demo.block_thread()


@main.command("api")
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
@click.option("--max-upload-mb", default=DEFAULT_MAX_UPLOAD_MB, show_default=True, type=int)
@_shared_model_options
def api(
    model_path: str,
    device: str,
    max_length: int,
    beam_size: int,
    max_upload_mb: int,
    host: str,
    port: int,
) -> None:
    """Start the FastAPI inference API server (HTTP/JSON endpoints)."""
    import uvicorn

    from latex_ocr.serve import create_app

    app = create_app(
        model_path=model_path,
        device=device,
        max_length=max_length,
        beam_size=beam_size,
        max_upload_mb=max_upload_mb,
    )
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
