from pydantic import Field
import pydantic
from torch import Tensor, nn
from recipe.utils.train_util import get_device
from recipe import config
from typing import Callable, Optional, Protocol, Literal
from enum import Enum


class ModelType(str, Enum):
    """Supported model architectures"""

    SWIN_TRANSFORMER = "swin_transformer"
    SWIN_GRU = "swin_gru"
    XCEPTION_GRU = "xception_gru"
    COCA = "coca"


class ImageToSeqModel(nn.Module):
    """Base protocol for all Swin-based models"""

    def encoder_parameters(self) -> list[nn.Parameter]:
        """Return encoder parameters for optimizer setup"""
        raise NotImplementedError

    def decoder_parameters(self) -> list[nn.Parameter]:
        """Return decoder parameters for optimizer setup"""
        raise NotImplementedError

    def other_parameters(self) -> list[nn.Parameter]:
        """Return other parameters for optimizer setup"""
        raise NotImplementedError

    def forward(self, src: Tensor, tgt: Tensor, tgt_mask: Tensor | None = None):
        """
        Input: src: (B, C, H, W), tgt: (B, seq_len), tgt_mask: Optional[Tensor]
        Output: (B, seq_len, vocab_size) or tuple depending on model
        """
        raise NotImplementedError

    def generate(self, src: Tensor, *args, **kwargs) -> Tensor:
        """
        Generate sequences from images.
        Each model should implement its own generation logic with its own parameters.

        Args:
            src: Input images (B, C, H, W)
            *args: Positional arguments specific to each model
            **kwargs: Keyword arguments specific to each model

        Returns:
            Generated token sequences (B, seq_len)
        """
        raise NotImplementedError


class TransformerDecoderCfg(pydantic.BaseModel):
    """Configuration for Transformer decoder"""

    ffn_dim: int = Field(default=2048, description="Feedforward dimension")
    nhead: int = Field(default=8, description="Number of attention heads")
    layers: int = Field(default=6, description="Number of decoder layers")


class GRUDecoderCfg(pydantic.BaseModel):
    """Configuration for GRU decoder"""

    emb_dim: int = Field(default=512, description="Embedding dimension")
    dec_dim: int = Field(default=512, description="Decoder hidden dimension")
    attn_dim: int = Field(default=512, description="Attention dimension")
    freeze_encoder: bool = Field(default=False, description="Whether to freeze encoder")
    use_gradient_checkpointing: bool = Field(
        default=False, description="Use gradient checkpointing"
    )


class CoCaCfg(pydantic.BaseModel):
    """Configuration for CoCa-style OCR model."""

    dim: int = Field(default=384)
    heads: int = Field(default=6)
    unimodal_depth: int = Field(default=4)
    multimodal_depth: int = Field(default=4)
    num_img_queries: int = Field(default=256)
    dim_latents: int = Field(default=256)
    lambda_cap: float = Field(default=1.0)
    lambda_con: float = Field(default=1.0)
    swin_name: str = Field(default="swin_tiny_patch4_window7_224")
    label_smoothing: float = Field(default=0.0)


class DataCfg(pydantic.BaseModel):
    config_name: str = Field(default="plain")
    both: bool = Field(
        default=False, description="Load both the plain and styled config"
    )
    plain_proportion: float = Field(default=-1)


class ModelCfg(pydantic.BaseModel):
    """Unified model configuration supporting multiple architectures"""

    model_type: ModelType = Field(description="Model architecture type")
    dropout: float = Field(default=0.0, description="Dropout rate")
    vocab_size: int = Field(default=30522, description="Vocabulary size")
    max_seq_length: int = Field(default=512, description="Maximum sequence length")
    encoder_pretrained: bool = Field(
        default=True, description="Use pretrained encoder weights"
    )
    decoder_pretrained: bool = Field(
        default=True, description="Use pretrained decoder weights"
    )
    load_weight_from_mlflow_run: Optional[str] = Field(
        default=None, description="mlflow run id to load weight"
    )
    mlflow_model_name: Optional[str] = Field(
        default=None, description="mlflow model name to load weight"
    )
    coca_pretrained: bool = Field(default=False)

    # Decoder-specific configs (only one will be used based on model_type)
    transformer_decoder_cfg: TransformerDecoderCfg | None = Field(
        default_factory=TransformerDecoderCfg,
        description="Config for transformer decoder (used when model_type=TRANSFORMER_DECODER)",
    )
    gru_decoder_cfg: GRUDecoderCfg | None = Field(
        default=None,
        description="Config for GRU decoder (used when model_type=GRU_DECODER)",
    )
    coca_cfg: CoCaCfg | None = Field(
        default=None, description="Config for CoCa model (used when model_type=COCA)"
    )
    encoder_lr: float = Field(default=1e-5, description="Learning rate for encoder")
    decoder_lr: float = Field(default=1e-4, description="Learning rate for decoder")

    # Special tokens
    bos_id: int = Field(default=1, description="Begin-of-sequence token ID")
    eos_id: int = Field(default=2, description="End-of-sequence token ID")
    pad_id: int = Field(default=0, description="Padding token ID")


class TrainerCfg(config.Config):
    model_cfg: ModelCfg = Field()
    data_cfg: DataCfg = Field()
    model_factory: Optional[Callable[[ModelCfg], ImageToSeqModel]] = Field(default=None)
    label_smoothing: float = Field(default=0.0)
    pin_memory: bool = Field(default=False, description="Pin memory in DataLoader")
