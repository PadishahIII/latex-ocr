from pathlib import Path
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchtune.generation import sample
from typing import Any, cast as typing_cast

from recipe.logging.logger import get_logger_by_name

# from recipe.utils.mlflow_util import load_model
from latex_ocr.trainers.config import ImageToSeqModel
from latex_ocr.trainers.decoder.gru_pretrain_model import GRUDecoder

logger = get_logger_by_name("XceptionGRUCaptioner")


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


class XceptionGRUCaptioner(ImageToSeqModel):
    """
    Xception encoder -> GRU decoder with global additive attention.
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
        decoder_pretrained: bool = True,
        use_gradient_checkpointing: bool = True,
        use_encoder_checkpointing: bool = True,
        decoder_checkpoint_chunk: int = 8,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.decoder_checkpoint_chunk = decoder_checkpoint_chunk
        self.use_encoder_checkpointing = use_encoder_checkpointing

        # ---- Encoder (Xception) ----
        # `timm.create_model` stubs are a bit loose; keep this as `Any`.
        self.encoder: Any = timm.create_model(
            "xception41p.ra3_in1k",
            pretrained=encoder_pretrained,
            num_classes=0,  # remove classifier nn.Linear
            features_only=False,
        )

        # Get encoder output dimension
        # For xception41p, the feature dimension is 2048
        enc_dim = self.encoder.num_features

        # Disable inplace operations to avoid gradient checkpointing errors
        # Inplace ops (like ReLU(inplace=True)) modify tensors that are needed
        # for gradient computation when using checkpointing
        if self.use_gradient_checkpointing:
            self._disable_inplace_ops(self.encoder)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # ---- Decoder components ----
        if decoder_pretrained:
            # decoder:GRUDecoder = load_model("2d1c18d3bb554359907ce7687bec428d", model_name="latex-ocr-gru-decoder_epoch00003_final")
            decoder: GRUDecoder = torch.load(
                Path(__file__).parent.parent / "pretrained" / "gru_decoder.pth",
                weights_only=False,
                map_location="cpu",
            )
            self.embed = decoder.embed
            self.gru_cell = decoder.gru_cell
            emb_dim = decoder.emb_dim
            dec_dim = decoder.dec_dim
            logger.warning(
                f"Overwrite emb_dim and dec_dim to {emb_dim}, {dec_dim} from loaded pretrained decoder"
            )
        else:
            self.embed = nn.Embedding(vocab_size, emb_dim)
            self.gru_cell = nn.GRUCell(
                input_size=emb_dim + emb_dim, hidden_size=dec_dim
            )

        self.attn = AdditiveAttention(
            enc_dim=enc_dim, dec_dim=dec_dim, attn_dim=attn_dim
        )

        # Optional projections to control dimensionality
        self.ctx_to_in = nn.Linear(enc_dim, emb_dim, bias=False)  # context -> emb space
        self.h0_proj = nn.Linear(enc_dim, dec_dim)  # init hidden from pooled tokens

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(dec_dim + enc_dim, vocab_size)

    def encoder_parameters(self) -> list[nn.Parameter]:
        return list(self.encoder.parameters())

    def decoder_parameters(self) -> list[nn.Parameter]:
        l = []
        l.extend(self.embed.parameters())
        l.extend(self.gru_cell.parameters())
        return l

    def other_parameters(self) -> list[nn.Parameter]:
        l = []
        l.extend(self.attn.parameters())
        l.extend(self.ctx_to_in.parameters())
        l.extend(self.h0_proj.parameters())
        l.extend(self.out.parameters())
        return l

    def _disable_inplace_ops(self, module: nn.Module):
        """
        Recursively disable inplace operations in all ReLU layers.
        This is necessary when using gradient checkpointing to avoid
        'modified by an inplace operation' errors.
        """
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                child.inplace = False
            else:
                self._disable_inplace_ops(child)

    def _encoder_forward_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Used for gradient checkpointing: must be a standalone function.
        out = self.encoder.forward_features(pixel_values)
        return typing_cast(torch.Tensor, out)

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: (B, C, H, W), already normalized/resized as Xception expects.
        returns enc_tokens: (B, N, C)
        """
        # Get unpooled features: (B, C, H, W)
        if (
            self.training
            and self.use_gradient_checkpointing
            and self.use_encoder_checkpointing
            and pixel_values.requires_grad
        ):
            # Coarse checkpoint (works even if encoder isn't a clean nn.Sequential)
            enc_out = checkpoint(  # type: ignore[misc]
                self._encoder_forward_features,
                pixel_values,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        else:
            enc_out = self.encoder.forward_features(pixel_values)

        # Reshape to (B, C, H*W) then transpose to (B, H*W, C) = (B, N, C)
        B, C, H, W = enc_out.shape
        enc_tokens = enc_out.view(B, C, H * W).transpose(1, 2)  # (B, N, C)

        return enc_tokens

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

    def _decode_chunk(
        self,
        tgt: torch.Tensor,
        start_t: int,
        end_t: int,
        h: torch.Tensor,
        enc_tokens: torch.Tensor,
    ):
        # returns logits_chunk: (B, chunk, V), h_out
        B = tgt.size(0)
        chunk_len = end_t - start_t
        logits_chunk = torch.empty(
            B, chunk_len, self.vocab_size, device=tgt.device, dtype=enc_tokens.dtype
        )
        for i, t in enumerate(range(start_t, end_t)):
            prev_tok = tgt[:, t]
            logits, h, _ = self.decode_step(prev_tok, h, enc_tokens, return_attn=False)
            logits_chunk[:, i, :] = logits
        return logits_chunk, h

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
        logits_all = torch.empty(
            B, T - 1, self.vocab_size, device=src.device, dtype=enc_tokens.dtype
        )

        # decode in chunks
        chunk = max(1, int(self.decoder_checkpoint_chunk))
        t = 0
        while t < (T - 1):
            t2 = min(T - 1, t + chunk)

            if self.training and self.use_gradient_checkpointing:
                logits_chunk, h = checkpoint(
                    self._decode_chunk,
                    tgt,
                    t,
                    t2,
                    h,
                    enc_tokens,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                logits_chunk, h = self._decode_chunk(tgt, t, t2, h, enc_tokens)

            logits_all[:, t:t2, :] = logits_chunk
            t = t2

        return logits_all

    def _apply_top_p_filtering(
        self, logits: torch.Tensor, top_p: float
    ) -> torch.Tensor:
        """
        Apply top-p (nucleus) filtering to logits.

        Args:
            logits: (B, V) raw logits
            top_p: cumulative probability threshold for nucleus sampling

        Returns:
            filtered logits with low-probability tokens set to -inf
        """
        if top_p >= 1.0:
            return logits

        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

        # Compute cumulative probabilities
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above the threshold
        # (keep the first token that exceeds top_p)
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift right to keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Scatter back to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))
        return logits

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        max_len: int = 30,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        beam_size: int | None = None,
        length_penalty: float = 0.0,
        early_stopping: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate sequences with temperature, top-k, and top-p (nucleus) sampling.

        Args:
            src: (B, C, H, W) input images
            max_len: maximum sequence length to generate
            temperature: sampling temperature (default: 1.0)
            top_k: if specified, only sample from top k tokens (default: None, no filtering)
            top_p: nucleus sampling threshold (default: 1.0, no filtering)
            beam_size: if > 1, use beam search (greedy/sampling disabled)
            length_penalty: beam search length penalty (0.0 disables)
            early_stopping: stop when all beams hit EOS
            **kwargs: additional arguments (unused, for compatibility)

        returns:
          token_ids: (B, <=max_len)  (does not include BOS; stops when EOS is produced or max_len reached)
        """
        del kwargs

        enc_tokens = self.encode(src)
        h0 = self.init_hidden(enc_tokens)

        if beam_size is not None and int(beam_size) > 1:
            beam_size_int = int(beam_size)
            if (
                top_k is not None
                or (top_p is not None and top_p < 1.0)
                or temperature != 1.0
            ):
                logger.warning(
                    f"Beam search is not compatible with sampling options (temperature/top_k/top_p), temperature/top_k/top_p would be disabled"
                )

            batch_size = src.size(0)
            device = src.device

            # Per-item beam search (simple + clear, acceptable for small batch sizes).
            sequences: list[torch.Tensor] = []
            for b in range(batch_size):
                enc_b = enc_tokens[b : b + 1]  # (1, N, C)
                h_b0 = h0[b : b + 1]  # (1, D)

                # start with a single beam containing BOS
                beam_tokens = torch.full(
                    (1, 1), self.bos_id, dtype=torch.long, device=device
                )
                beam_hidden = h_b0  # (1, D)
                beam_logprob = torch.zeros(1, device=device)
                beam_finished = torch.zeros(1, dtype=torch.bool, device=device)

                for _ in range(int(max_len)):
                    prev = beam_tokens[:, -1]  # (K,)

                    # advance all beams in parallel
                    enc_rep = enc_b.expand(prev.size(0), -1, -1)  # (K, N, C)
                    logits, next_hidden, _ = self.decode_step(
                        prev, beam_hidden, enc_rep, return_attn=False
                    )
                    log_probs = F.log_softmax(logits, dim=-1)  # (K, V)

                    # once EOS, keep emitting EOS only (force log prob 0 for EOS, -inf else)
                    if beam_finished.any():
                        forced = torch.full_like(log_probs, float("-inf"))
                        forced[:, self.eos_id] = 0.0
                        log_probs = torch.where(
                            beam_finished.unsqueeze(1), forced, log_probs
                        )

                    vocab_size = log_probs.size(-1)
                    candidate_scores = beam_logprob.unsqueeze(1) + log_probs  # (K, V)
                    flat_scores = candidate_scores.view(-1)

                    top_scores, top_indices = torch.topk(
                        flat_scores, k=min(beam_size_int, flat_scores.numel())
                    )
                    next_beam = top_indices // vocab_size
                    next_tok = (top_indices % vocab_size).to(torch.long)

                    beam_tokens = torch.cat(
                        [beam_tokens[next_beam], next_tok.unsqueeze(1)], dim=1
                    )
                    beam_hidden = next_hidden[next_beam]
                    beam_finished = beam_finished[next_beam] | (next_tok == self.eos_id)
                    beam_logprob = top_scores

                    if early_stopping and bool(beam_finished.all()):
                        break

                # length-penalized best hypothesis selection
                if length_penalty != 0.0:
                    # do not count BOS, count up to EOS if present
                    token_ids = beam_tokens[:, 1:]
                    lengths = torch.full(
                        (token_ids.size(0),), token_ids.size(1), device=device
                    )
                    eos_pos = token_ids.eq(self.eos_id)
                    if eos_pos.any():
                        first_eos = eos_pos.float().argmax(dim=1)
                        has_eos = eos_pos.any(dim=1)
                        lengths = torch.where(has_eos, first_eos + 1, lengths)
                    lengths = lengths.clamp(min=1)

                    alpha = float(length_penalty)
                    norm = ((5.0 + lengths.float()) / 6.0) ** alpha
                    scores = beam_logprob / norm
                else:
                    scores = beam_logprob

                best = int(torch.argmax(scores).item())
                sequences.append(beam_tokens[best, 1:])  # strip BOS

            max_out_len = max(seq.numel() for seq in sequences)
            padded = torch.full(
                (batch_size, max_out_len),
                self.eos_id,
                dtype=torch.long,
                device=device,
            )
            for i, seq in enumerate(sequences):
                padded[i, : seq.numel()] = seq
            return padded

        # Sampling / greedy path (existing behavior)
        h = h0
        B = src.size(0)
        prev = torch.full((B,), self.bos_id, dtype=torch.long, device=src.device)

        finished = torch.zeros(B, dtype=torch.bool, device=src.device)
        outputs: list[torch.Tensor] = []

        for _ in range(int(max_len)):
            logits, h, _ = self.decode_step(prev, h, enc_tokens, return_attn=False)

            if top_p < 1.0:
                logits = self._apply_top_p_filtering(logits, top_p)

            next_tok = sample(logits, temperature=temperature, top_k=top_k).squeeze(-1)
            outputs.append(next_tok.unsqueeze(1))

            finished |= next_tok == self.eos_id
            prev = torch.where(
                finished, torch.tensor(self.eos_id, device=prev.device), next_tok
            )

        return torch.cat(outputs, dim=1)


if __name__ == "__main__":
    """
    PYTHONPATH=. uv run python latex_ocr/models/cnn_based/xception_with_gru.py
    """
    from torchinfo import summary

    model = XceptionGRUCaptioner(
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
    # for name, module in model.named_modules():
    #     print(name)
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
