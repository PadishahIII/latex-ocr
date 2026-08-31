from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from latex_ocr.trainers.config import ImageToSeqModel


class GRUDecoder(ImageToSeqModel):
    """Decoder-only pretraining model.

    This intentionally contains no attention mechanism and no encoder modules.

    At each timestep, the GRU input is the concatenation of:
    - token embedding
    - a learned null context vector

    The null context is a learnable parameter of shape (1, 1, emb_dim) that is
    broadcast-expanded to (B, T, emb_dim).
    """

    def __init__(
        self,
        vocab_size: int,
        bos_id: int,
        eos_id: int,
        emb_dim: int = 512,
        dec_dim: int = 512,
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__()

        self.vocab_size = int(vocab_size)
        self.bos_id = int(bos_id)
        self.eos_id = int(eos_id)

        self.emb_dim = int(emb_dim)
        self.dec_dim = int(dec_dim)

        self.embed = nn.Embedding(self.vocab_size, self.emb_dim)

        # shape (1, 1, D) broadcasts to (B, T, D)
        self.null_ctx = nn.Parameter(torch.zeros(1, 1, self.emb_dim))

        self.dropout = nn.Dropout(float(dropout))

        self.gru_cell = nn.GRUCell(
            input_size=self.emb_dim * 2,
            hidden_size=self.dec_dim,
        )
        self.out = nn.Linear(self.dec_dim, self.vocab_size)

    def get_null_ctx(
        self, batch_size: int, seq_len: int, device: torch.device | None = None
    ) -> torch.Tensor:
        ctx = self.null_ctx
        if device is not None:
            ctx = ctx.to(device)
        return ctx.expand(batch_size, seq_len, self.emb_dim)

    def forward(
        self,
        src: torch.Tensor | None,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del src, tgt_mask

        batch_size, seq_len = tgt.shape
        if seq_len < 2:
            raise ValueError("tgt must have seq_len >= 2")

        inp = tgt[:, :-1]  # teacher forcing inputs
        emb = self.embed(inp)  # (B, T-1, D)
        ctx = self.get_null_ctx(batch_size, emb.size(1), emb.device)  # (B, T-1, D)
        x = torch.cat([emb, ctx], dim=-1)  # (B, T-1, 2D)

        # Decode step-by-step with GRUCell (matches xception_with_gru.py style).
        h = torch.zeros(batch_size, self.dec_dim, device=emb.device, dtype=emb.dtype)

        logits_all = torch.empty(
            batch_size,
            x.size(1),
            self.vocab_size,
            device=emb.device,
            dtype=emb.dtype,
        )
        for t in range(x.size(1)):
            h = self.gru_cell(self.dropout(x[:, t, :]), h)
            logits_all[:, t, :] = self.out(self.dropout(h))

        return logits_all

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

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor | None,
        max_len: int = 128,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        del src, kwargs

        device = self.null_ctx.device
        batch_size = 1

        prev = torch.full((batch_size, 1), self.bos_id, dtype=torch.long, device=device)
        outputs: list[torch.Tensor] = []

        h: torch.Tensor | None = None
        for _ in range(int(max_len)):
            emb = self.embed(prev)  # (B, 1, D)
            ctx = self.get_null_ctx(batch_size, 1, device)  # (B, 1, D)
            x = torch.cat([emb, ctx], dim=-1)  # (B, 1, 2D)

            # h: (B, dec_dim)
            if h is None:
                h = torch.zeros(
                    batch_size, self.dec_dim, device=device, dtype=emb.dtype
                )
            h = self.gru_cell(x[:, 0, :], h)
            logits = self.out(h)  # (B, V)

            if top_p < 1.0:
                logits = self._apply_top_p_filtering(logits, top_p)

            if temperature <= 0:
                next_tok = torch.argmax(logits, dim=-1)
            else:
                scaled = logits / max(float(temperature), 1e-5)
                if top_k is not None and top_k > 0:
                    topv, topi = torch.topk(scaled, k=int(top_k), dim=-1)
                    probs = F.softmax(topv, dim=-1)
                    next_tok = topi.gather(
                        -1, torch.multinomial(probs, num_samples=1)
                    ).squeeze(-1)
                else:
                    probs = F.softmax(scaled, dim=-1)
                    next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)

            outputs.append(next_tok.unsqueeze(1))
            prev = next_tok.unsqueeze(1)

        return torch.cat(outputs, dim=1)
