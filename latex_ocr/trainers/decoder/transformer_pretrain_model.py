from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from latex_ocr.models.embeddings.position_encoding import PositionalEncoding
from latex_ocr.trainers.config import ImageToSeqModel


class TransformerDecoder(ImageToSeqModel):
    """Transformer Decoder-only pretraining model.

    This intentionally contains no real encoder - instead uses a learnable
    trial encoder memory (similar to learnable positional embeddings).

    The learnable encoder memory acts as a fixed-size context that the decoder
    can attend to during pretraining, allowing the decoder to learn cross-attention
    patterns before being connected to a real encoder.

    At each timestep, the decoder:
    - Takes token embeddings with positional encoding
    - Attends to the learnable encoder memory via cross-attention
    - Uses causal self-attention for autoregressive generation
    """

    def __init__(
        self,
        vocab_size: int,
        decoder_nhead: int = 8,
        decoder_layers: int = 6,
        decoder_ffn_dim: int = 2048,
        decoder_model_dim:int=768,
        max_seq_length: int = 512,
        encoder_memory_size:Tuple[int,int] = (126, 768),
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__()

        self.vocab_size = int(vocab_size)

        self.decoder_model_dim = int(decoder_model_dim)

        # Token embedding
        self.embed = nn.Embedding(self.vocab_size, self.decoder_model_dim)

        # Positional encoding for decoder inputs
        self.position_encoding = PositionalEncoding(
            d_model=self.decoder_model_dim,
            max_seq_length=max_seq_length,
            dropout=dropout,
        )

        # Learnable encoder memory: shape (1, encoder_memory_size, emb_dim)
        # This will be broadcast to (B, encoder_memory_size, emb_dim)
        self.learnable_memory = nn.Parameter(
            torch.randn(1, encoder_memory_size[0], encoder_memory_size[1])
        )
        nn.init.normal_(self.learnable_memory, mean=0.0, std=0.02)

        # Transformer decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.decoder_model_dim,
                nhead=decoder_nhead,
                dim_feedforward=decoder_ffn_dim,
                dropout=dropout,
                activation=F.gelu,
                batch_first=True,
            ),
            num_layers=decoder_layers,
        )

        # Output projection
        self.out = nn.Linear(self.decoder_model_dim, self.vocab_size)

        self.dropout = nn.Dropout(float(dropout))

    def get_learnable_memory(
        self, batch_size: int,
    ) -> torch.Tensor:
        """Expand learnable encoder memory to batch size."""
        memory = self.learnable_memory
        return memory.expand(batch_size, -1, -1)

    def forward(
        self,
        src: torch.Tensor | None,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for training.

        Args:
            src: Ignored (no real encoder in pretrain mode)
            tgt: Target token sequences (B, seq_len)
            tgt_mask: causal mask for decoder

        Returns:
            logits: (B, seq_len-1, vocab_size) - predicting tokens 1..seq_len-1
        """
        del src  # Not used in decoder-only pretraining
        assert tgt_mask is not None, "tgt_mask must be provided for decoder"

        emb = self.embed(tgt)  # (B, seq_len, emb_dim)
        tgt_input = self.position_encoding(emb)  # (B, seq_len, emb_dim)

        # Get learnable encoder memory
        memory = self.get_learnable_memory(
            tgt.size(0), 
        )  # (B, encoder_memory_size, emb_dim)

        # Decode
        decoder_output = self.decoder(
            tgt_input, memory, tgt_mask=tgt_mask
        )  # (B, seq_len, emb_dim)

        # Project to vocabulary
        logits = self.out(self.dropout(decoder_output))  # (B, seq_len, vocab_size)

        return logits

