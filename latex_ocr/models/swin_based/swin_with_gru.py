import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from latex_ocr.models.swin_based.swin_with_transformer_decoder import (
    SwinTransformer,
)
from latex_ocr.trainers.config import ImageToSeqModel


class AdditiveAttention(nn.Module):
    """
    Bahdanau/additive attention:
      e_{t,i} = v^T tanh(W_h h_{t-1} + W_a a_i + b)
      alpha = softmax(e)
      c_t = sum_i alpha_i a_i
    This is a global attention over all N encoder tokens. [web:21]
    """

    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int):
        super().__init__()
        self.enc_proj = nn.Linear(enc_dim, attn_dim, bias=False)  # W_a
        self.dec_proj = nn.Linear(dec_dim, attn_dim, bias=False)  # W_h
        self.v = nn.Linear(attn_dim, 1, bias=False)  # v^T
        self.bias = nn.Parameter(torch.zeros(attn_dim))

    def forward(self, enc_tokens, dec_hidden):
        """
        enc_tokens: (B, N, C)
        dec_hidden: (B, D)
        returns:
          context: (B, C)
          attn_weights: (B, N)
        """
        # (B, N, attn_dim)
        enc_term = self.enc_proj(enc_tokens)
        # (B, 1, attn_dim) broadcast across N
        dec_term = self.dec_proj(dec_hidden).unsqueeze(1)

        # (B, N, attn_dim) -> (B, N, 1) -> (B, N)
        scores = self.v(torch.tanh(enc_term + dec_term + self.bias)).squeeze(-1)
        attn = F.softmax(scores, dim=-1)

        # weighted sum: (B, 1, N) @ (B, N, C) -> (B, 1, C) -> (B, C)
        context = torch.bmm(attn.unsqueeze(1), enc_tokens).squeeze(1)
        return context, attn


class SwinGRUCaptioner(ImageToSeqModel):
    """
    Swin encoder -> GRU decoder with global additive attention.
    """

    def __init__(
        self,
        vocab_size: int,
        bos_id: int,
        eos_id: int,
        emb_dim: int = 512,
        dec_dim: int = 512,
        attn_dim: int = 512,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
        encoder_pretrained: bool = True,
        use_gradient_checkpointing: bool = True,  # Trade compute for memory
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # ---- Encoder (Swin) ----
        self.encoder = SwinTransformer(
            patch_size=[4, 4],
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=[7, 7],
            stochastic_depth_prob=0.2,
            dropout=dropout,
        )
        if encoder_pretrained:
            self.encoder.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    "https://download.pytorch.org/models/swin_t-704ceda3.pth",  # refer to https://docs.pytorch.org/vision/main/_modules/torchvision/models/swin_transformer.html#Swin_T_Weights
                    progress=True,
                ),
                strict=False,
            )

        enc_dim = self.encoder.num_features  # C

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # ---- Decoder components ----
        self.embed = nn.Embedding(vocab_size, emb_dim)

        self.attn = AdditiveAttention(
            enc_dim=enc_dim, dec_dim=dec_dim, attn_dim=attn_dim
        )

        # Optional projections to control dimensionality
        self.ctx_to_in = nn.Linear(enc_dim, emb_dim, bias=False)  # context -> emb space
        self.h0_proj = nn.Linear(enc_dim, dec_dim)  # init hidden from pooled tokens

        self.gru_cell = nn.GRUCell(input_size=emb_dim + emb_dim, hidden_size=dec_dim)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(dec_dim + enc_dim, vocab_size)

    def encode(self, pixel_values: torch.Tensor):
        """
        pixel_values: (B, C, H, W), already normalized/resized as Swin expects.
        returns enc_tokens: (B, N, C)
        """
        enc_out = self.encoder(pixel_values)  # # (B, C, H, W) => (B, N, C)
        return enc_out

    def init_hidden(self, enc_tokens: torch.Tensor):
        """
        Mean pool tokens and project to decoder hidden.
        """
        pooled = enc_tokens.mean(dim=1)  # (B, C)
        h0 = torch.tanh(self.h0_proj(pooled))  # (B, D)
        return h0

    def decode_step(self, prev_token_ids, prev_hidden, enc_tokens, return_attn=None):
        """
        One decoding step.
          prev_token_ids: (B,)
          prev_hidden: (B, D)
          enc_tokens: (B, N, C)
          return_attn: Whether to return attention weights (None = auto-detect based on training mode)
        returns:
          logits: (B, V)
          hidden: (B, D)
          attn: (B, N) or None if not needed
        """
        # Don't store attention weights during training to save memory
        if return_attn is None:
            return_attn = not self.training

        # attention uses previous hidden as query
        ctx, attn = self.attn(enc_tokens, prev_hidden)  # (B, C), (B, N)

        emb = self.embed(prev_token_ids)  # (B,) => (B, E)
        ctx_in = self.ctx_to_in(ctx)  # (B, C) => (B, E)

        gru_in = torch.cat([emb, ctx_in], dim=-1)  # (B, 2E)
        hidden = self.gru_cell(self.dropout(gru_in), prev_hidden)  # (B, D)

        # produce logits using both hidden and (unprojected) context
        out_in = torch.cat([hidden, ctx], dim=-1)  # (B, D + C)
        logits = self.out(self.dropout(out_in))  # (B, V)

        # Return None for attention during training to save memory
        return logits, hidden, (attn if return_attn else None)

    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor | None = None
    ):
        """
        Teacher-forcing forward pass.

        src: (B, 3, H, W)
        tgt: (B, T) containing [BOS, w1, w2, ..., w_{T-1}] (targets are typically shifted outside)
        tgt_mask: not used

        returns:
          logits: (B, T-1, V) predicting tokens 1..T-1
        """
        enc_tokens = self.encode(src)
        h = self.init_hidden(enc_tokens)

        B, T = tgt.shape

        # Pre-allocate logits tensor to avoid list accumulation and intermediate allocations
        # This significantly reduces memory usage for long sequences
        logits_all = torch.empty(
            B, T - 1, self.vocab_size, device=src.device, dtype=enc_tokens.dtype
        )

        # predict next token for each prev token position (0..T-2)
        for t in range(T - 1):
            prev_tok = tgt[:, t]  # (B,)

            # Use gradient checkpointing to trade compute for memory (~30% memory savings, ~20% slower)
            # Recomputes activations during backward pass instead of storing them
            if self.training and self.use_gradient_checkpointing:
                # Note: checkpoint works with functions that return tuples
                # type: ignore is needed because checkpoint return type confuses type checker
                logits, h, _ = checkpoint(  # type: ignore[misc]
                    self.decode_step,
                    prev_tok,
                    h,
                    enc_tokens,
                    False,  # Don't return attention during training
                    use_reentrant=False,  # Use newer checkpointing API
                    preserve_rng_state=True,
                )
            else:
                # Don't store attention weights during training (saves ~10-15% memory)
                logits, h, _ = self.decode_step(
                    prev_tok, h, enc_tokens, return_attn=False
                )

            logits_all[:, t, :] = logits  # Direct assignment instead of append

        return logits_all

    @torch.no_grad()
    def generate(self, src: torch.Tensor, max_len: int = 30, **kwargs):
        """
        Greedy decoding.

        Args:
            src: (B, C, H, W) input images
            max_len: maximum sequence length to generate
            **kwargs: additional arguments (unused, for compatibility)

        returns:
          token_ids: (B, <=max_len)  (does not include BOS; stops when EOS is produced or max_len reached)
        """
        enc_tokens = self.encode(src)
        h = self.init_hidden(enc_tokens)

        B = src.size(0)
        prev = torch.full((B,), self.bos_id, dtype=torch.long, device=src.device)

        finished = torch.zeros(B, dtype=torch.bool, device=src.device)
        outputs = []

        for _ in range(max_len):
            # During generation, we can optionally get attention weights
            logits, h, _ = self.decode_step(prev, h, enc_tokens, return_attn=False)
            next_tok = torch.argmax(logits, dim=-1)  # (B,)
            outputs.append(next_tok.unsqueeze(1))

            # once EOS, keep emitting EOS
            finished |= next_tok == self.eos_id
            prev = torch.where(
                finished, torch.tensor(self.eos_id, device=prev.device), next_tok
            )

        return torch.cat(outputs, dim=1)


if __name__ == "__main__":
    """
    PYTHONPATH=. uv run python latex_ocr/models/swin_based/swin_with_gru.py
    """
    from torchinfo import summary

    model = SwinGRUCaptioner(
        vocab_size=100,
        bos_id=1,
        eos_id=2,
        emb_dim=256,
        dec_dim=256,
        attn_dim=256,
        dropout=0.1,
        freeze_encoder=False,
        encoder_pretrained=False,
    )
    print(summary(model))
    for name, module in model.named_modules():
        print(name)
    src = torch.randn(2, 3, 224, 224)
    tgt = torch.randint(0, 100, (2, 10))
    logits = model(src, tgt)
    print("Logits shape:", logits.shape)  # (2, 9, 100)

    # Test generate method
    print("\nTesting generate method:")
    model.eval()
    generated = model.generate(src, max_len=20)
    print(f"Generated sequences shape: {generated.shape}")
    print(f"Generated tokens:\n{generated}")
