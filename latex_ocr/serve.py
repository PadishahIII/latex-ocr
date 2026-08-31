"""Inference API server: image in, LaTeX out.

Exposes a small HTTP API around a trained model (CoCa/Swin by default; the
factory supports every architecture in ``latex_ocr.trainers.model``):

    POST /predict        multipart image (or raw image bytes) -> {"latex": ...}
    POST /predict_json   JSON body {"images_b64": [...]}        -> {"results": [...]}
    GET  /health         liveness + loaded-model info

Usage:
    latex-ocr-server --model models/checkpoints/latex-ocr-coca-finetune.pth --port 8000
    # or
    python -m latex_ocr.serve --model ... --port 8000
"""

# NOTE: no `from __future__ import annotations` here — FastAPI resolves the
# UploadFile annotation at route-registration time and needs the real class.

import base64
import binascii
import io
import os
from typing import Optional

import torch
from PIL import Image
from torchvision import transforms

from latex_ocr.models.tokenizer.latex_tokenizer import Tokenizer
from latex_ocr.trainers.config import ModelCfg
from latex_ocr.trainers.model import get_model

# ---------------------------------------------------------------------------
# Defaults (overridable via CLI flags or LATEX_OCR_* environment variables)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = os.getenv(
    "LATEX_OCR_MODEL_PATH", "models/checkpoints/latex-ocr-coca-finetune.pth"
)
DEFAULT_DEVICE = os.getenv("LATEX_OCR_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_MAX_LENGTH = int(os.getenv("LATEX_OCR_MAX_LENGTH", "354"))
DEFAULT_BEAM_SIZE = int(os.getenv("LATEX_OCR_BEAM_SIZE", "4"))
DEFAULT_MAX_UPLOAD_MB = int(os.getenv("LATEX_OCR_MAX_UPLOAD_MB", "10"))


def build_eval_transform(image_size: tuple[int, int] = (192, 672)) -> transforms.Compose:
    """Deterministic transform matching the validation pipeline used in training."""
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class LatexOCRPredictor:
    """Loads a checkpointed model and generates LaTeX from PIL images."""

    def __init__(
        self,
        model_path: str,
        device: str = DEFAULT_DEVICE,
        max_length: int = DEFAULT_MAX_LENGTH,
        beam_size: int = DEFAULT_BEAM_SIZE,
    ):
        self.device = torch.device(device)
        self.max_length = max_length
        self.beam_size = beam_size
        self.transform = build_eval_transform()

        self.tokenizer = Tokenizer()
        self.tokenizer.load(None)  # default packaged sentencepiece model

        print(f"Loading checkpoint: {model_path}")
        obj = torch.load(model_path, weights_only=False, map_location="cpu")

        if isinstance(obj, torch.nn.Module):
            # Full-model checkpoint (e.g. MLflow/pytorch logged CoCaSwinOCR).
            model = obj
        elif isinstance(obj, dict):
            # state_dict checkpoint -> build the architecture from ModelCfg,
            # which must be provided alongside the weights.
            model_cfg = ModelCfg.model_validate(obj.get("model_cfg", {}))
            model = get_model(model_cfg)
            state_dict = obj.get("model", obj.get("model_state_dict", obj))
            model.load_state_dict(state_dict)
        else:
            raise ValueError(f"Unsupported checkpoint format: {type(obj)!r}")

        model.eval()
        self.model = model.to(self.device)

        self.bos_id = getattr(self.tokenizer, "bos_token_id", 1)
        self.eos_id = getattr(self.tokenizer, "eos_token_id", 2)

    # ------------------------------------------------------------------
    def predict(self, image: Image.Image) -> str:
        """Run inference on a single PIL image, returning LaTeX text."""
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        generated = self._generate(tensor)
        ids = generated[0].tolist()
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _generate(self, tensor: torch.Tensor) -> torch.Tensor:
        try:
            if self.beam_size and self.beam_size > 1:
                return self.model.generate(
                    src=tensor,
                    bos_token_id=self.bos_id,
                    eos_token_id=self.eos_id,
                    max_length=self.max_length,
                    beam_size=self.beam_size,
                )
            return self.model.generate(
                src=tensor,
                bos_token_id=self.bos_id,
                eos_token_id=self.eos_id,
                max_length=self.max_length,
            )
        except TypeError:
            # Simpler generate() signatures (e.g. GRU decoder) ignore the extras.
            return self.model.generate(src=tensor, max_len=self.max_length)


def create_app(
    model_path: str = DEFAULT_MODEL_PATH,
    device: str = DEFAULT_DEVICE,
    max_length: int = DEFAULT_MAX_LENGTH,
    beam_size: int = DEFAULT_BEAM_SIZE,
    max_upload_mb: int = DEFAULT_MAX_UPLOAD_MB,
):
    """Build the FastAPI app (lazy model loading on startup event)."""
    try:
        from fastapi import FastAPI, File, HTTPException, Request, UploadFile
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The inference server needs fastapi + uvicorn + python-multipart. "
            "Install them with: uv sync --extra server"
        ) from exc

    predictor: Optional[LatexOCRPredictor] = None
    max_bytes = max_upload_mb * 1024 * 1024

    app = FastAPI(
        title="latex-ocr",
        description="Image-to-LaTeX OCR. POST an image to /predict, get LaTeX back.",
        version="0.1.0",
    )

    @app.on_event("startup")
    def _load_model() -> None:
        nonlocal predictor
        predictor = LatexOCRPredictor(
            model_path=model_path,
            device=device,
            max_length=max_length,
            beam_size=beam_size,
        )

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model_loaded": predictor is not None,
            "device": device,
        }

    def _check_loaded() -> LatexOCRPredictor:
        if predictor is None:
            raise HTTPException(status_code=503, detail="model not loaded yet")
        return predictor

    def _decode_upload(data: bytes) -> Image.Image:
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"image too large (> {max_upload_mb} MB)",
            )
        if not data:
            raise HTTPException(status_code=400, detail="empty upload")
        try:
            return Image.open(io.BytesIO(data))
        except Exception:
            raise HTTPException(status_code=400, detail="invalid image file")

    @app.post("/predict")
    async def predict(request: Request):
        """Image upload: image in, LaTeX out.

        - multipart/form-data with a 'file' field:
            `curl -F file=@formula.png /predict`
        - raw image bytes:
            `curl --data-binary @formula.png /predict`
        """
        content_type = request.headers.get("content-type", "")
        if content_type.startswith("multipart/form-data"):
            form = await request.form()
            upload = form.get("file")
            if upload is None:
                raise HTTPException(
                    status_code=400, detail="multipart form missing 'file' field"
                )
            data = await upload.read()
        else:
            data = await request.body()  # raw bytes
        img = _decode_upload(data)
        pred = _check_loaded()
        try:
            latex = pred.predict(img)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"inference failed: {exc}")
        return JSONResponse({"latex": latex})


    class PredictJSON(BaseModel):
        images_b64: list[str]

    @app.post("/predict_json")
    async def predict_json(body: PredictJSON):
        """Base64 JSON batch: {"images_b64": ["...", ...]} -> {"results": [...]}."""
        pred = _check_loaded()
        results = []
        for b64 in body.images_b64:
            try:
                raw = base64.b64decode(b64, validate=True)
            except (binascii.Error, ValueError):
                raise HTTPException(status_code=400, detail="invalid base64 payload")
            img = _decode_upload(raw)
            try:
                results.append(pred.predict(img))
            except Exception as exc:
                results.append({"error": str(exc)})
        return JSONResponse({"results": results})

    return app


def main() -> None:
    import click
    import uvicorn

    @click.command()
    @click.option("--model", "model_path", default=DEFAULT_MODEL_PATH, show_default=True,
                  help="Path to a full-model (.pth) or state_dict checkpoint.")
    @click.option("--device", "device", default=DEFAULT_DEVICE, show_default=True,
                  help="torch device, e.g. cuda, cuda:1, cpu.")
    @click.option("--host", default="0.0.0.0", show_default=True)
    @click.option("--port", default=8000, show_default=True, type=int)
    @click.option("--max-length", default=DEFAULT_MAX_LENGTH, show_default=True)
    @click.option("--beam-size", default=DEFAULT_BEAM_SIZE, show_default=True)
    @click.option("--max-upload-mb", default=DEFAULT_MAX_UPLOAD_MB, show_default=True)
    def serve(model_path, device, host, port, max_length, beam_size, max_upload_mb):
        """Start the LaTeX OCR inference API server."""
        app = create_app(
            model_path=model_path,
            device=device,
            max_length=max_length,
            beam_size=beam_size,
            max_upload_mb=max_upload_mb,
        )
        uvicorn.run(app, host=host, port=port)

    serve()


if __name__ == "__main__":
    main()
