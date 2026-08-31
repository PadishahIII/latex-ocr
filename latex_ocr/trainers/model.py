import os
from pathlib import Path
import warnings

import torch

from recipe.logging.logger import get_logger_by_name
from recipe.utils.mlflow_util import load_model, load_model_state_dict, setup_mlflow
from latex_ocr.models.cnn_based.xception_with_gru import XceptionGRUCaptioner
from latex_ocr.models.coca.model import CoCaOCRConfig, CoCaSwinOCR

# Disable tokenizers parallelism warning before any imports that use tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress torch.distributed redirect warning on macOS/Windows
warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed")

from latex_ocr.data.synth.data import create_dataloader
from latex_ocr.models.swin_based.swin_with_transformer_decoder import (
    SwinTBase,
)
from latex_ocr.models.swin_based.swin_with_gru import SwinGRUCaptioner
from latex_ocr.trainers.config import (
    CoCaCfg,
    GRUDecoderCfg,
    ImageToSeqModel,
    ModelCfg,
    ModelType,
    TransformerDecoderCfg,
)

from torch import Tensor, nn

logger = get_logger_by_name(__name__)

def get_swin_transformer_model(cfg: ModelCfg) -> SwinTBase:
    """Factory function for Transformer decoder model"""
    if cfg.transformer_decoder_cfg is None:
        cfg.transformer_decoder_cfg = TransformerDecoderCfg()

    return SwinTBase(
        vocab_size=cfg.vocab_size,
        max_seq_length=cfg.max_seq_length,
        dropout=cfg.dropout,
        decoder_ffn_dim=cfg.transformer_decoder_cfg.ffn_dim,
        decoder_nhead=cfg.transformer_decoder_cfg.nhead,
        decoder_layers=cfg.transformer_decoder_cfg.layers,
        encoder_pretrained=cfg.encoder_pretrained,
        decoder_pretrained=cfg.decoder_pretrained,
    )


def get_swin_gru_model(cfg: ModelCfg) -> SwinGRUCaptioner:
    """Factory function for GRU decoder model"""
    if cfg.gru_decoder_cfg is None:
        cfg.gru_decoder_cfg = GRUDecoderCfg()

    return SwinGRUCaptioner(
        vocab_size=cfg.vocab_size,
        bos_id=cfg.bos_id,
        eos_id=cfg.eos_id,
        emb_dim=cfg.gru_decoder_cfg.emb_dim,
        dec_dim=cfg.gru_decoder_cfg.dec_dim,
        attn_dim=cfg.gru_decoder_cfg.attn_dim,
        dropout=cfg.dropout,
        freeze_encoder=cfg.gru_decoder_cfg.freeze_encoder,
        encoder_pretrained=cfg.encoder_pretrained,
        use_gradient_checkpointing=cfg.gru_decoder_cfg.use_gradient_checkpointing,
    )


def get_xception_gru_model(cfg: ModelCfg) -> XceptionGRUCaptioner:
    """Factory function for Xception with GRU decoder model"""
    if cfg.gru_decoder_cfg is None:
        cfg.gru_decoder_cfg = GRUDecoderCfg()
    return XceptionGRUCaptioner(
        vocab_size=cfg.vocab_size,
        bos_id=cfg.bos_id,
        eos_id=cfg.eos_id,
        emb_dim=cfg.gru_decoder_cfg.emb_dim,
        dec_dim=cfg.gru_decoder_cfg.dec_dim,
        attn_dim=cfg.gru_decoder_cfg.attn_dim,
        dropout=cfg.dropout,
        freeze_encoder=cfg.gru_decoder_cfg.freeze_encoder,
        encoder_pretrained=cfg.encoder_pretrained,
        decoder_pretrained=cfg.decoder_pretrained,
        use_gradient_checkpointing=cfg.gru_decoder_cfg.use_gradient_checkpointing,
    )


def get_coca_model(cfg: ModelCfg) -> ImageToSeqModel:
    """Factory function for CoCa (Swin) model."""
    if cfg.coca_cfg is None:
        cfg.coca_cfg = CoCaCfg()

    coca_cfg = CoCaOCRConfig(
        vocab_size=cfg.vocab_size,
        dim=cfg.coca_cfg.dim,
        heads=cfg.coca_cfg.heads,
        unimodal_depth=cfg.coca_cfg.unimodal_depth,
        multimodal_depth=cfg.coca_cfg.multimodal_depth,
        num_img_queries=cfg.coca_cfg.num_img_queries,
        pad_id=cfg.pad_id,
        bos_id=cfg.bos_id,
        eos_id=cfg.eos_id,
        dropout=cfg.dropout,
        dim_latents=cfg.coca_cfg.dim_latents,
    )
    if cfg.load_weight_from_mlflow_run and cfg.mlflow_model_name:
        logger.info(f"Loading CoCaSwinOCR model from MLflow run {cfg.load_weight_from_mlflow_run}, model name {cfg.mlflow_model_name}")
        model = load_model(run_id=cfg.load_weight_from_mlflow_run, model_name=cfg.mlflow_model_name)
        assert isinstance(model, CoCaSwinOCR), f"Loaded model is not of type CoCaSwinOCR but {type(model)}"
        model.set_dropout(cfg.dropout, include_encoder=True)
        return model
    if cfg.coca_pretrained:
        logger.info("Loading pretrained CoCaSwinOCR weights")
        _ckpt = os.getenv(
            "LATEX_OCR_COCA_PRETRAINED_CKPT",
            str(Path(__file__).parent.parent.parent / "models" / "checkpoints" / "latex-ocr-coca-pretrain.pth"),
        )
        if not Path(_ckpt).exists():
            raise FileNotFoundError(
                f"coca_pretrained=True but checkpoint not found: {_ckpt}. "
                "Download it from the Hugging Face releases (see README) or point "
                "LATEX_OCR_COCA_PRETRAINED_CKPT at the file."
            )
        model = torch.load(_ckpt, weights_only=False, map_location=torch.device("cpu"))
        model.set_dropout(cfg.dropout, include_encoder=True)
        return model

    return CoCaSwinOCR(
        cfg=coca_cfg,
        swin_name=cfg.coca_cfg.swin_name,
        pretrained_swin=cfg.encoder_pretrained,
    )


def get_model(cfg: ModelCfg) -> ImageToSeqModel:
    """
    Unified model factory that returns the appropriate model based on config.

    Args:
        cfg: ModelCfg containing model type and configuration

    Returns:
        SwinBasedModel instance (either SwinTBase or SwinGRUCaptioner)

    Raises:
        ValueError: If model_type is not supported
    """
    match cfg.model_type:
        case ModelType.SWIN_TRANSFORMER:
            return get_swin_transformer_model(cfg)
        case ModelType.SWIN_GRU:
            return get_swin_gru_model(cfg)
        case ModelType.XCEPTION_GRU:
            return get_xception_gru_model(cfg)
        case ModelType.COCA:
            return get_coca_model(cfg)
        case _:
            raise ValueError(f"Unsupported model_type: {cfg.model_type}")


if __name__ == "__main__":
    """
    PYTHONPATH=. uv run python latex_ocr/trainers/model.py
    """
    ds, loader = create_dataloader(
        config_name="plain",
        split="train",
        batch_size=2,
        shuffle=False,
    )
    dp = next(iter(loader))

    # Test Transformer decoder model
    print("=== Testing Transformer Decoder Model ===")
    transformer_model = get_model(
        ModelCfg(
            model_type=ModelType.SWIN_TRANSFORMER,
            vocab_size=ds.tokenizer.vocab_size,
            max_seq_length=5000,
            dropout=0.1,
            transformer_decoder_cfg=TransformerDecoderCfg(
                ffn_dim=2048, nhead=8, layers=6
            ),
            encoder_pretrained=True,
        )
    )
    transformer_model.eval()
    logits = transformer_model(
        dp["images"], dp["labels"], tgt_mask=dp["attention_mask"]
    )
    print(f"Transformer logits shape: {logits.shape}")  # [2, 140, 50269]
    loss_fn = nn.CrossEntropyLoss(ignore_index=ds.tokenizer.pad_token_id)
    loss = loss_fn(logits.view(-1, logits.size(-1)), dp["labels"].view(-1))
    print(f"Transformer loss: {loss}")

    # Test GRU decoder model
    print("\n=== Testing GRU Decoder Model ===")
    gru_model = get_model(
        ModelCfg(
            model_type=ModelType.SWIN_GRU,
            vocab_size=ds.tokenizer.vocab_size,
            max_seq_length=5000,
            dropout=0.1,
            gru_decoder_cfg=GRUDecoderCfg(emb_dim=512, dec_dim=512, attn_dim=512),
            encoder_pretrained=True,
            bos_id=getattr(ds.tokenizer, "bos_token_id", 1),
            eos_id=getattr(ds.tokenizer, "eos_token_id", 2),
            pad_id=ds.tokenizer.pad_token_id,
        )
    )
    gru_model.eval()
    logits, attn = gru_model(dp["images"], dp["labels"], tgt_mask=dp["attention_mask"])
    print(f"GRU logits shape: {logits.shape}")
    print(f"GRU attention shape: {attn.shape}")
    loss = loss_fn(logits.view(-1, logits.size(-1)), dp["labels"].view(-1))
    print(f"GRU loss: {loss}")
