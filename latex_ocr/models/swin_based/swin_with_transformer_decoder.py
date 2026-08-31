from functools import partial
from pathlib import Path
from typing import Callable, Optional
import torch
from torch import Tensor, nn
from torchvision.models.swin_transformer import (
    PatchMerging,
    Permute,
    ShiftedWindowAttention,
    SwinTransformerBlock,
)
import torch.hub
import torch.nn.functional as F
from torchtune.generation import sample

from latex_ocr.models.embeddings.position_encoding import PositionalEncoding
from latex_ocr.trainers.config import ImageToSeqModel
from latex_ocr.trainers.decoder.transformer_pretrain_model import TransformerDecoder


class SwinTransformer(nn.Module):
    """
    Implements Swin Transformer as an encoder.
    Args:
        patch_size (List[int]): Patch size.
        embed_dim (int): Patch embedding dimension.
        depths (List(int)): Depth of each Swin Transformer layer.
        num_heads (List(int)): Number of attention heads in different layers.
        window_size (List[int]): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.1.
        num_classes (int): Number of classes for classification head. Default: 1000.
        block (nn.Module, optional): SwinTransformer Block. Default: None.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        downsample_layer (nn.Module): Downsample layer (patch merging). Default: PatchMerging.
    """

    def __init__(
        self,
        patch_size: list[int],
        embed_dim: int,
        depths: list[int],
        num_heads: list[int],
        window_size: list[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        block: Optional[Callable[..., nn.Module]] = None,
        downsample_layer: Callable[..., nn.Module] = PatchMerging,
        attn_layer: Callable[..., nn.Module] = ShiftedWindowAttention,
    ):
        super().__init__()

        if block is None:
            block = SwinTransformerBlock
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        layers: list[nn.Module] = []
        # split image into non-overlapping patches
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    3,
                    embed_dim,
                    kernel_size=(patch_size[0], patch_size[1]),
                    stride=(patch_size[0], patch_size[1]),
                ),  # (192, 672) => (48, 168)
                Permute([0, 2, 3, 1]),  # (B, C, H_p, W_p) => (B, H_p, W_p, C)
                norm_layer(embed_dim),
            )
        )

        total_stage_blocks = sum(depths)
        stage_block_id = 0
        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage: list[nn.Module] = []
            dim = embed_dim * 2**i_stage
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = (
                    stochastic_depth_prob
                    * float(stage_block_id)
                    / (total_stage_blocks - 1)
                )
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[
                            0 if i_layer % 2 == 0 else w // 2 for w in window_size
                        ],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                        attn_layer=attn_layer,
                    )
                )
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            # add patch merging layer
            if i_stage < (len(depths) - 1):
                layers.append(downsample_layer(dim, norm_layer))
        self.features = nn.Sequential(*layers)

        self.num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(self.num_features)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Output shape: (B, H*W, C) where C = embed_dim * 2 ** (len(depths) - 1)
        """
        x = self.features(x)  # (B, H, W, C)
        x = self.norm(x)  # (B, H, W, C)
        # Reshape to sequence format for decoder: (B, H, W, C) -> (B, H*W, C)
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        return x


class SwinTBase(ImageToSeqModel):
    def __init__(
        self,
        vocab_size: int,
        max_seq_length: int,
        dropout: float = 0.0,
        decoder_ffn_dim: int = 384,
        decoder_nhead: int = 12,
        decoder_layers: int = 4,
        encoder_pretrained: bool = False,
        decoder_pretrained:bool = False,
    ):
        """Swin-T with Transformer decoder for image-to-sequence tasks.

        This model combines a Swin Transformer encoder for visual feature extraction
        with a Transformer decoder for sequence generation, making it suitable for
        tasks like LaTeX OCR, image captioning, etc.

        Args:
            vocab_size: Size of the target vocabulary for sequence generation.
            max_seq_length: Maximum length of the generated sequences.
            dropout: Dropout probability applied throughout the model. Default: 0.0.
            decoder_ffn_dim: Dimension of the feedforward network in decoder layers. Default: 2048.
            decoder_nhead: Number of attention heads in the decoder. Default: 8.
            decoder_layers: Number of transformer decoder layers. Default: 6.
            encoder_pretrained: Whether to load pretrained Swin-T weights for the encoder. Default: False.

        Architecture:
            - Encoder: Swin Transformer (Swin-Tiny configuration)
                - patch_size: [4, 4]
                - embed_dim: 96
                - depths: [2, 2, 6, 2] (blocks per stage)
                - num_heads: [3, 6, 12, 24] (heads per stage)
                - window_size: [7, 7]
                - stochastic_depth_prob: 0.2
                - Output: (B, H*W, 768) where 768 is the feature dimension
            - Decoder: Standard Transformer Decoder with cross-attention to encoder outputs

        Input/Output:
            - Input:
                - src: (B, 3, H, W) - RGB images
                - tgt: (B, seq_len) - target token sequences
                - tgt_mask: Optional causal mask for autoregressive decoding
            - Output: (B, seq_len, vocab_size) - logits for each position
        """
        super().__init__()
        self.encoder = SwinTransformer(
            patch_size=[4, 4],
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=[7, 7],
            stochastic_depth_prob=0.2,
            dropout=dropout,
            attn_layer=ShiftedWindowAttention,
        )
        if encoder_pretrained:
            self.encoder.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    "https://download.pytorch.org/models/swin_t-704ceda3.pth",  # refer to https://docs.pytorch.org/vision/main/_modules/torchvision/models/swin_transformer.html#Swin_T_Weights
                    progress=True,
                ),
                strict=False,
            )
        if decoder_pretrained:
            model:TransformerDecoder = torch.load(Path(__file__).parent.parent /"pretrained"/"transformer_decoder.pth", weights_only=False)
            self.position_encoding = model.position_encoding
            self.decoder_embedding = model.embed
            self.decoder = model.decoder
            self.head = model.out
        else:
            self.position_encoding = PositionalEncoding(
                d_model=self.encoder.num_features,
                max_seq_length=max_seq_length,
                dropout=dropout,
            )
            self.decoder_embedding = nn.Embedding(vocab_size, self.encoder.num_features)
            self.decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=self.encoder.num_features,
                    nhead=decoder_nhead,
                    dim_feedforward=decoder_ffn_dim,
                    dropout=dropout,
                    activation=F.gelu,
                    batch_first=True,
                ),
                num_layers=decoder_layers,
            )
            self.head = nn.Linear(self.encoder.num_features, vocab_size)

    def encoder_parameters(self) -> list[nn.Parameter]:
        return self.encoder.parameters()

    def decoder_parameters(self) -> list[nn.Parameter]:
        l = []
        l.extend(self.position_encoding.parameters())
        l.extend(self.decoder_embedding.parameters())
        l.extend(self.decoder.parameters())
        l.extend(self.head.parameters())
        return l

    def other_parameters(self) -> list[nn.Parameter]:
        return []

    def forward(self, src: Tensor, tgt: Tensor, tgt_mask: Tensor | None = None):
        memory = self.encoder(src)  # (B, C, H, W) => (B, N, C)
        tgt_emb = self.decoder_embedding(tgt)  # (B, seq_len) => (B, seq_len, E)
        input = self.position_encoding(tgt_emb)  # (B, seq_len, E)
        decoder_res = self.decoder(
            input, memory, tgt_mask=tgt_mask
        )  # (B, seq_len, E) => (B, seq_len, E)
        logit = self.head(decoder_res)  # (B, seq_len, E) => (B, seq_len, vocab_size)
        return logit

    @torch.no_grad()
    def generate(
        self,
        src: Tensor,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        """Generate sequences autoregressively using nucleus and temperature sampling.

        This method generates token sequences from images using autoregressive decoding
        with support for temperature scaling, top-k sampling, and nucleus (top-p) sampling.

        Args:
            src (Tensor): Input images of shape (B, 3, H, W)
            bos_token_id (int): Beginning-of-sequence token ID to start generation
            eos_token_id (int): End-of-sequence token ID to stop generation
            max_length (int): Maximum sequence length to generate. Default: 512
            temperature (float): Temperature for sampling. Higher values make output more random.
                Default: 1.0
            top_k (Optional[int]): If specified, only sample from the top k tokens by probability.
                Default: None (no top-k filtering)
            top_p (Optional[float]): If specified, only sample from tokens whose cumulative
                probability exceeds this threshold (nucleus sampling). Value should be in (0, 1).
                Default: None (no nucleus sampling)

        Returns:
            Tensor: Generated token sequences of shape (B, seq_len) where seq_len <= max_length

        Example:
            >>> model = SwinTBase(vocab_size=50000, max_seq_length=512)
            >>> images = torch.randn(2, 3, 192, 672)
            >>> sequences = model.generate(
            ...     images,
            ...     bos_token_id=1,
            ...     eos_token_id=2,
            ...     max_length=100,
            ...     temperature=0.8,
            ...     top_p=0.9
            ... )
            >>> print(sequences.shape)  # (2, seq_len) where seq_len <= 100
        """
        batch_size = src.size(0)
        device = src.device

        # Encode the source images once
        memory = self.encoder(src)  # (B, H*W, E)

        # Initialize generated sequences with BOS token
        generated = torch.full(
            (batch_size, 1), bos_token_id, dtype=torch.long, device=device
        )

        # Track which sequences have finished (generated EOS)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length - 1):
            # Create causal mask for current sequence
            seq_len = generated.size(1)
            tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=device
            )

            # Get embeddings and add positional encoding
            tgt_emb = self.decoder_embedding(generated)  # (B, seq_len, E)
            tgt_input = self.position_encoding(tgt_emb)  # (B, seq_len, E)

            # Decode
            decoder_output = self.decoder(
                tgt_input, memory, tgt_mask=tgt_mask
            )  # (B, seq_len, E)

            # Get logits for the last position
            logits = self.head(decoder_output[:, -1, :])  # (B, vocab_size)

            # Apply nucleus (top-p) sampling if specified
            if top_p is not None:
                # Sort logits in descending order
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                # Compute cumulative probabilities
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits / max(temperature, 1e-5), dim=-1), dim=-1
                )
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least one token
                sorted_indices_to_remove[:, 0] = False
                # Scatter back to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits = logits.masked_fill(indices_to_remove, float("-inf"))

            # Sample next token using torchtune's sample function
            next_token = sample(logits, temperature=temperature, top_k=top_k)  # (B, 1)

            # Update finished mask
            finished = finished | (next_token.squeeze(-1) == eos_token_id)

            # Append next token to generated sequences
            generated = torch.cat([generated, next_token], dim=1)

            # Stop if all sequences have finished
            if finished.all():
                break

        return generated


"""
PYTHONPATH=. uv run python latex_ocr/models/swin_based/swin_with_transformer_decoder.py
"""
if __name__ == "__main__":
    # Test forward pass
    model = SwinTBase(vocab_size=1000, max_seq_length=512, encoder_pretrained=True)
    print(f"encoder num features: {model.encoder.num_features}")  # 768
    dummy_img = torch.randn(2, 3, 192, 672)  # (B, C, H, W)
    dummy_tgt = torch.randint(0, 1000, (2, 10))  # (B, seq_len)
    output = model(dummy_img, dummy_tgt)
    print(f"Forward output shape: {output.shape}")  # Expected: (2, 10, 1000)
    memory = model.encoder(dummy_img)  # Test encoder separately
    print(f"Encoder output shape: {memory.shape}")  # Expected: (2, 126, 768)

    # Test generate method
    print("\nTesting generate method:")
    model.eval()
    generated = model.generate(
        dummy_img,
        bos_token_id=1,
        eos_token_id=2,
        max_length=20,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
    )
    print(f"Generated sequences shape: {generated.shape}")
    print(f"Generated tokens:\n{generated}")
