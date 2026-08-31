# coca_swin_ocr.py
# Minimal CoCa-style model for LaTeX OCR: Swin encoder + (unimodal -> multimodal) decoder.

from typing import Any, Optional, Tuple, cast

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel
import timm
from recipe.logging.logger import get_logger_by_name

from latex_ocr.trainers.config import ImageToSeqModel

logger = get_logger_by_name("coca_swin_ocr")

# -------------------------
# Utility modules
# -------------------------


class LayerNormNoBias(nn.Module):
    """LayerNorm with learnable gamma, fixed zero beta (matches common CoCa-style LN usage)."""

    def __init__(self, dim: int):
        super().__init__()
        self.gamma: torch.Tensor = nn.Parameter(torch.ones(dim))
        beta: torch.Tensor = torch.zeros(dim)
        self.register_buffer("beta", beta, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, x.shape[-1:], self.gamma, cast(torch.Tensor, self.beta))


class FeedForward(nn.Module):
    def __init__(self, dim: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        inner = dim * ff_mult
        self.net = nn.Sequential(
            LayerNormNoBias(dim),
            nn.Linear(dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SelfAttentionBlock(nn.Module):
    """Pre-norm causal self-attention + FFN."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm = LayerNormNoBias(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ff = FeedForward(dim, ff_mult=4, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = True,
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        # x: (B, T, D)
        h = self.norm(x)
        attn_mask = None
        if causal:
            T = x.shape[1]
            attn_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
        y, _ = self.attn(
            h,
            h,
            h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + y
        x = x + self.ff(x)
        return x


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attn (text queries attend to image tokens) + FFN."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm_q = LayerNormNoBias(dim)
        self.norm_kv = LayerNormNoBias(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ff = FeedForward(dim, ff_mult=4, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,  # (B, T, D) queries
        context: torch.Tensor,  # (B, N, D) keys/values
        context_padding_mask: Optional[torch.Tensor] = None,
    ):
        q = self.norm_q(x)
        kv = self.norm_kv(context)
        y, _ = self.attn(
            q, kv, kv, key_padding_mask=context_padding_mask, need_weights=False
        )
        x = x + y
        x = x + self.ff(x)
        return x


class AttentionPool(nn.Module):
    """
    Attentional pooling: learned queries cross-attend into token set.
    Returns (B, num_queries, D).
    """

    def __init__(self, dim: int, num_queries: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.norm_q = LayerNormNoBias(dim)
        self.norm_ctx = LayerNormNoBias(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

    def forward(
        self, tokens: torch.Tensor, token_padding_mask: Optional[torch.Tensor] = None
    ):
        # tokens: (B, N, D)
        B, _, D = tokens.shape
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, Q, D)
        q = self.norm_q(q)
        ctx = self.norm_ctx(tokens)
        out, _ = self.attn(
            q, ctx, ctx, key_padding_mask=token_padding_mask, need_weights=False
        )
        return out


# -------------------------
# Swin encoder wrapper
# -------------------------


class SwinTokenEncoder(nn.Module):
    """
    Uses timm SwinTransformer.forward_features which returns NHWC feature grid (B,H,W,C).
    We flatten it to tokens (B, N, C) for cross-attn/pooling.
    """

    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()
        # timm Swin models default to `strict_img_size=True` and will assert if the
        # input resolution differs from the pretraining resolution (usually 224).
        # Our OCR pipeline can produce different sizes (e.g. 192), so disable strict checks.
        self.backbone: Any = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            strict_img_size=False,
            drop_rate=drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
        )
        # timm Swin sets self.backbone.num_features for channel dim
        self.out_dim: int = int(self.backbone.num_features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.forward_features(images)  # (B,H,W,C) in timm Swin [page:2]
        B, H, W, C = feat.shape
        tokens = feat.view(B, H * W, C)
        return tokens  # (B, N, C)


# -------------------------
# CoCa-style OCR model
# -------------------------


class CoCaOCRConfig(BaseModel):
    vocab_size: int
    dim: int = 384
    heads: int = 6
    unimodal_depth: int = 4
    multimodal_depth: int = 4
    num_img_queries: int = 256  # +1 CLS-like query for contrastive image embedding
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    dropout: float = 0.1
    dim_latents: int = 256  # contrastive embedding dim
    label_smoothing: float = 0.0


class CoCaSwinOCR(ImageToSeqModel):
    """
    - Swin encoder -> image tokens
    - Attention pooling -> [img_cls, img_queries...]
    - Text decoder split:
        * unimodal stack: self-attn only (yields text embedding for contrastive)
        * multimodal stack: self-attn + cross-attn to image queries (yields logits for captioning)

    This model implements the `ImageToSeqModel` interface directly (no external adapter).
    """

    def __init__(
        self,
        cfg: CoCaOCRConfig,
        swin_name: str = "swin_tiny_patch4_window7_224",
        pretrained_swin: bool = True,
    ):
        super().__init__()
        self.cfg = cfg

        # vision
        # `SwinTokenEncoder` returns a flattened token grid (B, N, C_swin).
        # We optionally project to `cfg.dim` so the text and vision streams share a single model width.
        self.vision = SwinTokenEncoder(
            swin_name,
            pretrained=pretrained_swin,
            drop_rate=float(cfg.dropout),
            proj_drop_rate=float(cfg.dropout),
            attn_drop_rate=float(cfg.dropout),
        )
        if self.vision.out_dim != cfg.dim:
            self.vision_proj = nn.Linear(self.vision.out_dim, cfg.dim, bias=False)
            logger.warning(
                f"Vision output dim ({self.vision.out_dim}) != config dim ({cfg.dim}). "
                f"Adding projection layer."
            )
        else:
            self.vision_proj = nn.Identity()

        # attentional pooling
        # We learn a small set of query vectors that cross-attend into the full Swin token set.
        # Output shape: (B, 1 + Q, D)
        # - the first query is treated as an image 'CLS' embedding for the contrastive (CLIP-style) loss
        # - the remaining Q queries are the fixed-size visual memory used for text cross-attention
        self.img_pool = AttentionPool(
            dim=cfg.dim, num_queries=cfg.num_img_queries + 1, heads=cfg.heads
        )

        # text
        # Token embedding table for teacher-forced inputs.
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.dim)

        # Learnable "CLS" token for contrastive training.
        # During `encode_text_unimodal` we append it after the last real token so we can
        # read out a single global text embedding that corresponds to the whole sequence.
        self.text_cls = nn.Parameter(torch.randn(cfg.dim) * 0.02)
        self.text_cls_norm = LayerNormNoBias(cfg.dim)

        # unimodal and multimodal decoder stacks
        self.unimodal_layers = nn.ModuleList(
            [
                SelfAttentionBlock(cfg.dim, cfg.heads, dropout=cfg.dropout)
                for _ in range(cfg.unimodal_depth)
            ]
        )

        self.multimodal_self = nn.ModuleList(
            [
                SelfAttentionBlock(cfg.dim, cfg.heads, dropout=cfg.dropout)
                for _ in range(cfg.multimodal_depth)
            ]
        )
        self.multimodal_cross = nn.ModuleList(
            [
                CrossAttentionBlock(cfg.dim, cfg.heads, dropout=cfg.dropout)
                for _ in range(cfg.multimodal_depth)
            ]
        )

        # heads
        self.to_latents_img = nn.Linear(cfg.dim, cfg.dim_latents, bias=False)
        self.to_latents_txt = nn.Linear(cfg.dim, cfg.dim_latents, bias=False)

        self.lm_norm = LayerNormNoBias(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        # Weight tying: share parameters between input embeddings and output projection.
        # This is standard in many language models (fewer parameters; often better generalization).
        # Important: do not include `lm_head.weight` in optimizer params separately, because it is
        # the *same* tensor as `token_emb.weight`.
        self.tie_weights()

        # Learned logit scale parameter for the CLIP-style contrastive loss.
        # In the loss we exponentiate it (after clamping) to ensure a positive scale.
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def tie_weights(self) -> None:
        self.lm_head.weight = self.token_emb.weight

    def set_dropout(self, dropout: float, include_encoder: bool = False) -> None:
        """
        Set dropout rate for all dropout layers in the model.

        Args:
            dropout: New dropout probability (0.0 to 1.0)
            include_encoder: If True, also update dropout in the Swin encoder (default: False)
        """
        self.cfg.dropout = dropout

        # Update dropout in Swin encoder if requested
        if include_encoder:
            # Recursively update all Dropout modules in the Swin backbone
            def update_dropout_recursive(module):
                # Direct nn.Dropout modules
                if isinstance(module, nn.Dropout):
                    module.p = dropout
                # Check all attributes for Dropout instances (handles timm's named dropouts)
                for name, child in module.named_children():
                    if isinstance(child, nn.Dropout):
                        child.p = dropout
                    else:
                        update_dropout_recursive(child)

            update_dropout_recursive(self.vision.backbone)

        # Update dropout in attention pooling
        for module in self.img_pool.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout
            elif isinstance(module, nn.MultiheadAttention):
                # Update internal dropout in MultiheadAttention
                module.dropout = dropout

        # Update dropout in unimodal layers
        for layer in self.unimodal_layers:
            for module in layer.modules():
                if isinstance(module, nn.Dropout):
                    module.p = dropout
                elif isinstance(module, nn.MultiheadAttention):
                    module.dropout = dropout

        # Update dropout in multimodal self-attention layers
        for layer in self.multimodal_self:
            for module in layer.modules():
                if isinstance(module, nn.Dropout):
                    module.p = dropout
                elif isinstance(module, nn.MultiheadAttention):
                    module.dropout = dropout

        # Update dropout in multimodal cross-attention layers
        for layer in self.multimodal_cross:
            for module in layer.modules():
                if isinstance(module, nn.Dropout):
                    module.p = dropout
                elif isinstance(module, nn.MultiheadAttention):
                    module.dropout = dropout

        # Update dropout in unimodal layers
        for layer in self.unimodal_layers:
            for module in layer.modules():
                if isinstance(module, nn.Dropout):
                    module.p = dropout
                elif isinstance(module, nn.MultiheadAttention):
                    module.dropout = dropout

        # Update dropout in multimodal self-attention layers
        for layer in self.multimodal_self:
            for module in layer.modules():
                if isinstance(module, nn.Dropout):
                    module.p = dropout
                elif isinstance(module, nn.MultiheadAttention):
                    module.dropout = dropout

        # Update dropout in multimodal cross-attention layers
        for layer in self.multimodal_cross:
            for module in layer.modules():
                if isinstance(module, nn.Dropout):
                    module.p = dropout
                elif isinstance(module, nn.MultiheadAttention):
                    module.dropout = dropout

    def encoder_parameters(self) -> list[nn.Parameter]:
        return list(self.vision.parameters())

    def decoder_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        params.extend(list(self.token_emb.parameters()))
        params.extend(list(self.unimodal_layers.parameters()))
        params.extend(list(self.multimodal_self.parameters()))
        params.extend(list(self.multimodal_cross.parameters()))
        params.extend(list(self.lm_norm.parameters()))
        params.append(self.text_cls)
        return params

    def other_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        params.extend(list(self.img_pool.parameters()))
        if not isinstance(self.vision_proj, nn.Identity):
            params.extend(list(self.vision_proj.parameters()))
        params.extend(list(self.to_latents_img.parameters()))
        params.extend(list(self.to_latents_txt.parameters()))
        params.append(self.temperature)
        return params

    def encode_image(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        img_tokens = self.vision(images)  # (B,N,Cswin)
        img_tokens = self.vision_proj(img_tokens)  # (B,N,D)
        pooled = self.img_pool(img_tokens)  # (B, 1+Q, D)
        img_cls = pooled[:, 0]  # (B,D)
        img_queries = pooled[:, 1:]  # (B,Q,D)
        return img_cls, img_queries

    def encode_text_unimodal(
        self, token_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        token_ids: (B,T) teacher-forced input tokens (usually shifted right)
        returns:
          txt_cls: (B,D) unimodal text embedding for contrastive
          x: (B,T,D) unimodal hidden states for feeding multimodal half
        """
        B, T = token_ids.shape
        x = self.token_emb(token_ids)  # (B,T,D)

        # append CLS token at end (like reference CoCa impl) [page:1]
        cls = self.text_cls.view(1, 1, -1).expand(B, 1, -1)
        x = torch.cat([x, cls], dim=1)  # (B,T+1,D)

        # key_padding_mask: True = pad positions (MultiheadAttention convention)
        # include cls token as non-pad
        key_padding_mask = token_ids == self.cfg.pad_id
        key_padding_mask = F.pad(key_padding_mask, (0, 1), value=False)  # (B,T+1)

        for layer in self.unimodal_layers:
            x = layer(x, causal=True, key_padding_mask=key_padding_mask)

        txt_cls = self.text_cls_norm(x[:, -1])  # (B,D)
        x = x[:, :-1]  # (B,T,D)
        return txt_cls, x

    def _forward_impl(
        self,
        images: torch.Tensor,
        text_in: torch.Tensor,
        text_labels: Optional[torch.Tensor] = None,
        return_loss: bool = True,
        lambda_cap: float = 1.0,
        lambda_con: float = 1.0,
    ):
        """
        images: (B,3,H,W)
        text_in: (B,T) input tokens (teacher forcing; usually BOS + tokens[:-1])
        text_labels: (B,T) labels (usually tokens[1:] + EOS/pad)
        """
        img_cls, img_queries = self.encode_image(images)
        txt_cls, x = self.encode_text_unimodal(text_in)

        # multimodal half: self-attn then cross-attn to image queries
        for sa, ca in zip(self.multimodal_self, self.multimodal_cross):
            x = sa(x, causal=True, key_padding_mask=(text_in == self.cfg.pad_id))
            x = ca(x, img_queries)

        logits = self.lm_head(self.lm_norm(x))  # (B,T,V)

        if not return_loss:
            return logits

        assert text_labels is not None, "text_labels required when return_loss=True"
        cap_loss = (
            F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                text_labels.reshape(-1),
                ignore_index=self.cfg.pad_id,
                label_smoothing=self.cfg.label_smoothing,
            )
            * lambda_cap
        )

        # contrastive (CLIP-style)
        # Build a similarity matrix (B x B) between image and text embeddings.
        # - we normalize then take dot products (cosine similarity)
        # - `temperature` acts like CLIP's logit_scale; we clamp before exp to avoid overflow
        #   (important when training with AMP/FP16)
        # - we compute similarity in fp32 for stability even if the model runs in lower precision
        # NOTE: cast to float32 BEFORE F.normalize. Inside AMP autocast the forward runs in fp16
        # where eps=1e-12 (the default for F.normalize) rounds to 0, making the divide-by-zero
        # guard a no-op. A near-zero norm vector then produces NaN that kills the run.
        img_lat = F.normalize(self.to_latents_img(img_cls).float(), dim=-1)
        txt_lat = F.normalize(self.to_latents_txt(txt_cls).float(), dim=-1)
        logit_scale = self.temperature.float().clamp(max=math.log(100.0)).exp()
        sim = (img_lat @ txt_lat.t()) * logit_scale
        target = torch.arange(sim.size(0), device=sim.device)
        con_loss = (
            0.5
            * (
                F.cross_entropy(sim, target)
                + F.cross_entropy(sim.t().contiguous(), target)
            )
            * lambda_con
        )

        return {
            "loss": cap_loss + con_loss,
            "cap_loss": cap_loss,
            "con_loss": con_loss,
            "logits": logits,
        }

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        *,
        return_loss: bool = False,
        lambda_cap: float = 1.0,
        lambda_con: float = 1.0,
    ):
        del tgt_mask
        text_in = tgt[:, :-1].contiguous()
        if return_loss:
            text_labels = tgt[:, 1:].contiguous()
            return self._forward_impl(
                src,
                text_in=text_in,
                text_labels=text_labels,
                return_loss=True,
                lambda_cap=lambda_cap,
                lambda_con=lambda_con,
            )
        return self._forward_impl(src, text_in=text_in, return_loss=False)

    def forward_with_loss(
        self,
        images: torch.Tensor,
        tokens: torch.Tensor,
        lambda_cap: float = 1.0,
        lambda_con: float = 1.0,
    ):
        return self.forward(
            images,
            tokens,
            return_loss=True,
            lambda_cap=lambda_cap,
            lambda_con=lambda_con,
        )

    def _apply_top_p_filtering(
        self, logits: torch.Tensor, top_p: float
    ) -> torch.Tensor:
        if top_p >= 1.0:
            return logits

        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        return logits.masked_fill(indices_to_remove, float("-inf"))

    def _next_token_logits(
        self, token_ids: torch.Tensor, img_queries: torch.Tensor
    ) -> torch.Tensor:
        _, x = self.encode_text_unimodal(token_ids)
        for sa, ca in zip(self.multimodal_self, self.multimodal_cross):
            x = sa(x, causal=True, key_padding_mask=(token_ids == self.cfg.pad_id))
            x = ca(x, img_queries)
        return self.lm_head(self.lm_norm(x))[:, -1]  # (B, V)

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        max_length: int = 384,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        beam_size: int | None = None,
        length_penalty: float = 0.0,
        early_stopping: bool = True,
        max_len: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        self.eval()

        if max_len is not None:
            max_length = int(max_len)

        bos_id = self.cfg.bos_id if bos_token_id is None else int(bos_token_id)
        eos_id = self.cfg.eos_id if eos_token_id is None else int(eos_token_id)

        _, img_queries = self.encode_image(src)
        batch_size = src.size(0)
        device = src.device

        if beam_size is not None and int(beam_size) > 1:
            beam_size_int = int(beam_size)
            if (
                top_k is not None
                or (top_p is not None and top_p < 1.0)
                or temperature != 1.0
            ):
                logger.warning(
                    "Beam search is not compatible with sampling options (temperature/top_k/top_p); "
                    "temperature/top_k/top_p are ignored when beam_size > 1."
                )

            sequences: list[torch.Tensor] = []
            for b in range(batch_size):
                img_q_b = img_queries[b : b + 1]  # (1, Q, D)

                beam_tokens = torch.full(
                    (1, 1), bos_id, dtype=torch.long, device=device
                )
                beam_logprob = torch.zeros(1, device=device)
                beam_finished = torch.zeros(1, dtype=torch.bool, device=device)

                for _ in range(int(max_length) - 1):
                    img_rep = img_q_b.expand(beam_tokens.size(0), -1, -1)
                    logits = self._next_token_logits(beam_tokens, img_rep)
                    log_probs = F.log_softmax(logits, dim=-1)  # (K, V)

                    if beam_finished.any():
                        forced = torch.full_like(log_probs, float("-inf"))
                        forced[:, eos_id] = 0.0
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
                    beam_finished = beam_finished[next_beam] | (next_tok == eos_id)
                    beam_logprob = top_scores

                    if early_stopping and bool(beam_finished.all()):
                        break

                if length_penalty != 0.0:
                    token_ids = beam_tokens[:, 1:]
                    lengths = torch.full(
                        (token_ids.size(0),), token_ids.size(1), device=device
                    )
                    eos_pos = token_ids.eq(eos_id)
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
                sequences.append(beam_tokens[best])

            max_out_len = max(seq.numel() for seq in sequences)
            padded = torch.full(
                (batch_size, max_out_len),
                eos_id,
                dtype=torch.long,
                device=device,
            )
            for i, seq in enumerate(sequences):
                padded[i, : seq.numel()] = seq
            return padded

        generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(int(max_length) - 1):
            logits = self._next_token_logits(generated, img_queries)

            if top_p is not None and top_p < 1.0:
                logits = self._apply_top_p_filtering(logits, float(top_p))

            if top_k is None and (top_p is None or top_p >= 1.0) and temperature == 1.0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                from torchtune.generation import sample

                next_token = sample(
                    logits,
                    temperature=max(float(temperature), 1e-6),
                    top_k=top_k,
                )

            if finished.any():
                next_token = torch.where(
                    finished.unsqueeze(1),
                    torch.tensor(eos_id, device=device, dtype=torch.long),
                    next_token,
                )

            generated = torch.cat([generated, next_token], dim=1)
            finished |= next_token.squeeze(-1).eq(eos_id)

            if finished.all():
                break

        return generated


class CoCaSwinOCRCascade(nn.Module):
    def __init__(
        self,
        cfg: CoCaOCRConfig,
        swin_name: str = "swin_tiny_patch4_window7_224",
        pretrained_swin: bool = True,
        n_gen_queries: int = 256,
    ):
        super().__init__()
        self.cfg = cfg

        # vision backbone
        self.vision = SwinTokenEncoder(
            swin_name,
            pretrained=pretrained_swin,
            drop_rate=float(cfg.dropout),
            proj_drop_rate=float(cfg.dropout),
            attn_drop_rate=float(cfg.dropout),
        )
        if self.vision.out_dim != cfg.dim:
            self.vision_proj = nn.Linear(self.vision.out_dim, cfg.dim, bias=False)
        else:
            self.vision_proj = nn.Identity()

        # generative pooler (produces queries for decoder cross-attn)
        self.gen_pool = AttentionPool(
            dim=cfg.dim,
            num_queries=n_gen_queries,
            heads=cfg.heads,
            dropout=cfg.dropout,
        )

        # contrastive pooler sits on top of generative outputs
        self.contrastive_pool = AttentionPool(
            dim=cfg.dim,
            num_queries=1,
            heads=cfg.heads,
            dropout=cfg.dropout,
        )

        # text side
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.text_cls = nn.Parameter(torch.randn(cfg.dim) * 0.02)
        self.text_cls_norm = LayerNormNoBias(cfg.dim)

        self.unimodal_layers = nn.ModuleList(
            [
                SelfAttentionBlock(cfg.dim, cfg.heads, dropout=cfg.dropout)
                for _ in range(cfg.unimodal_depth)
            ]
        )
        self.multimodal_self = nn.ModuleList(
            [
                SelfAttentionBlock(cfg.dim, cfg.heads, dropout=cfg.dropout)
                for _ in range(cfg.multimodal_depth)
            ]
        )
        self.multimodal_cross = nn.ModuleList(
            [
                CrossAttentionBlock(cfg.dim, cfg.heads, dropout=cfg.dropout)
                for _ in range(cfg.multimodal_depth)
            ]
        )

        # heads
        self.to_latents_img = nn.Linear(cfg.dim, cfg.dim_latents, bias=False)
        self.to_latents_txt = nn.Linear(cfg.dim, cfg.dim_latents, bias=False)

        self.lm_norm = LayerNormNoBias(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        # Weight tying: share parameters between input embeddings and output projection.
        # This is standard in many language models (fewer parameters; often better generalization).
        # Important: do not include `lm_head.weight` in optimizer params separately, because it is
        # the *same* tensor as `token_emb.weight`.
        self.tie_weights()

        # Learned logit scale parameter for the CLIP-style contrastive loss.
        # In the loss we exponentiate it (after clamping) to ensure a positive scale.
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def tie_weights(self) -> None:
        self.lm_head.weight = self.token_emb.weight

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        out = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.tie_weights()
        return out

    def encode_image(self, images: torch.Tensor):
        # Swin -> tokens
        img_tokens = self.vision(images)  # (B, N, C_swin)
        img_tokens = self.vision_proj(img_tokens)  # (B, N, D)

        # generative pooling: produces decoder memory
        gen_tokens = self.gen_pool(img_tokens)  # (B, Q_gen, D)

        # contrastive pooling cascaded on top of generative tokens
        img_cls = self.contrastive_pool(gen_tokens)[:, 0]  # (B, D)

        return img_cls, gen_tokens  # CLS for contrastive, Q_gen for decoder

    def encode_text_unimodal(self, token_ids: torch.Tensor):
        B, T = token_ids.shape
        x = self.token_emb(token_ids)  # (B, T, D)

        cls = self.text_cls.view(1, 1, -1).expand(B, 1, -1)
        x = torch.cat([x, cls], dim=1)  # (B, T+1, D)

        key_padding_mask = token_ids == self.cfg.pad_id
        key_padding_mask = F.pad(key_padding_mask, (0, 1), value=False)

        for layer in self.unimodal_layers:
            x = layer(x, causal=True, key_padding_mask=key_padding_mask)

        txt_cls = self.text_cls_norm(x[:, -1])  # (B, D)
        x = x[:, :-1]  # (B, T, D)
        return txt_cls, x

    def forward(
        self,
        images,
        text_in,
        text_labels=None,
        return_loss=True,
        lambda_cap=1.0,
        lambda_con=1.0,
    ):
        img_cls, img_queries = self.encode_image(images)
        txt_cls, x = self.encode_text_unimodal(text_in)

        # multimodal decoder
        pad_mask = text_in == self.cfg.pad_id
        for sa, ca in zip(self.multimodal_self, self.multimodal_cross):
            x = sa(x, causal=True, key_padding_mask=pad_mask)
            x = ca(x, img_queries)

        logits = self.lm_head(self.lm_norm(x))

        if not return_loss:
            return logits

        assert text_labels is not None
        cap_loss = (
            F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                text_labels.reshape(-1),
                ignore_index=self.cfg.pad_id,
            )
            * lambda_cap
        )

        img_lat = F.normalize(self.to_latents_img(img_cls), dim=-1)
        txt_lat = F.normalize(self.to_latents_txt(txt_cls), dim=-1)
        logit_scale = self.temperature.float().clamp(max=math.log(100.0)).exp()
        sim = (img_lat.float() @ txt_lat.float().t()) * logit_scale
        target = torch.arange(sim.size(0), device=sim.device)
        con_loss = (
            0.5
            * (
                F.cross_entropy(sim, target)
                + F.cross_entropy(sim.t().contiguous(), target)
            )
            * lambda_con
        )

        return {
            "loss": cap_loss + con_loss,
            "cap_loss": cap_loss,
            "con_loss": con_loss,
            "logits": logits,
        }


if __name__ == "__main__":
    """
    PYTHONPATH=. uv run python latex_ocr/models/coca/model.py
    """
    torch.manual_seed(0)

    cfg = CoCaOCRConfig(
        vocab_size=128,
        dim=64,
        heads=4,
        unimodal_depth=2,
        multimodal_depth=2,
        num_img_queries=16,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        dropout=0.0,
        dim_latents=32,
    )

    model = CoCaSwinOCR(
        cfg=cfg, swin_name="swin_tiny_patch4_window7_224", pretrained_swin=False
    )
    model.eval()

    B, H, W = 2, 224, 224
    T = 12

    images = torch.randn(B, 3, H, W)
    tokens = torch.randint(3, cfg.vocab_size, (B, T))
    tokens[:, 0] = cfg.bos_id

    # logits-only forward (ImageToSeqModel interface)
    logits = model(images, tokens)
    assert logits.shape == (B, T - 1, cfg.vocab_size), logits.shape

    # forward with internal caption+contrastive loss
    out = model.forward_with_loss(images, tokens=tokens)
    assert out["logits"].shape == (B, T - 1, cfg.vocab_size), out["logits"].shape
    assert out["loss"].ndim == 0, out["loss"].shape

    # generation smoke test
    gen = model.generate(src=images, max_length=8)
    assert gen.shape[0] == B, gen.shape

    print("CoCaSwinOCR random-input test passed")
