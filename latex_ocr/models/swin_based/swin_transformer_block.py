"""
Swin Transformer Block with Flash Attention Support

This module implements the Swin Transformer attention mechanism with optional Flash Attention
for improved memory efficiency and speed.

Flash Attention is used when:
1. The flash_attn package is installed
2. logit_scale is None (Flash Attention doesn't support cosine attention)

To install Flash Attention:
    pip install flash-attn --no-build-isolation

Note: Flash Attention requires:
- CUDA 11.4 or higher
- PyTorch 1.12 or higher
- A compatible NVIDIA GPU (Ampere, Ada, or Hopper architecture recommended)

If Flash Attention is not available, the implementation automatically falls back to
standard PyTorch attention.
"""

import math
import warnings
from typing import Callable, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from flash_attn import flash_attn_func  # type: ignore[import-untyped]

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    flash_attn_func = None  # type: ignore[assignment]
    warnings.warn(
        "flash_attn not available, falling back to PyTorch attention kernels",
        RuntimeWarning,
    )


def _get_relative_position_bias(
    relative_position_bias_table: torch.Tensor,
    relative_position_index: torch.Tensor,
    window_size: list[int],
) -> torch.Tensor:
    """
    Get relative position bias from the table.
    """
    N = window_size[0] * window_size[1]
    relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
    relative_position_bias = relative_position_bias.view(N, N, -1)
    relative_position_bias = (
        relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
    )
    return relative_position_bias


def shifted_window_attention(
    input: Tensor,
    qkv_weight: Tensor,
    proj_weight: Tensor,
    relative_position_bias: Tensor,
    window_size: list[int],
    num_heads: int,
    shift_size: list[int],
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    qkv_bias: Optional[Tensor] = None,
    proj_bias: Optional[Tensor] = None,
    logit_scale: Optional[torch.Tensor] = None,
    training: bool = True,
) -> Tensor:
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Uses Flash Attention for efficient computation when available.

    Args:
        input (Tensor[N, H, W, C]): The input tensor or 4-dimensions.
        qkv_weight (Tensor[in_dim, out_dim]): The weight tensor of query, key, value.
        proj_weight (Tensor[out_dim, out_dim]): The weight tensor of projection.
        relative_position_bias (Tensor): The learned relative position bias added to attention.
        window_size (List[int]): Window size.
        num_heads (int): Number of attention heads.
        shift_size (List[int]): Shift size for shifted window attention.
        attention_dropout (float): Dropout ratio of attention weight. Default: 0.0.
        dropout (float): Dropout ratio of output. Default: 0.0.
        qkv_bias (Tensor[out_dim], optional): The bias tensor of query, key, value. Default: None.
        proj_bias (Tensor[out_dim], optional): The bias tensor of projection. Default: None.
        logit_scale (Tensor[out_dim], optional): Logit scale of cosine attention for Swin Transformer V2. Default: None.
        training (bool, optional): Training flag used by the dropout parameters. Default: True.
    Returns:
        Tensor[N, H, W, C]: The output tensor after shifted window attention.
    """
    B, H, W, C = input.shape
    # pad feature maps to multiples of window size
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    x = F.pad(input, (0, 0, 0, pad_r, 0, pad_b))
    _, pad_H, pad_W, _ = x.shape

    shift_size = shift_size.copy()
    # If window size is larger than feature size, there is no need to shift window
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0

    # cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

    # partition windows
    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
    x = x.view(
        B,
        pad_H // window_size[0],
        window_size[0],
        pad_W // window_size[1],
        window_size[1],
        C,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(
        B * num_windows, window_size[0] * window_size[1], C
    )  # B*nW, Ws*Ws, C

    # multi-head attention
    if logit_scale is not None and qkv_bias is not None:
        qkv_bias = qkv_bias.clone()
        length = qkv_bias.numel() // 3
        qkv_bias[length : 2 * length].zero_()
    qkv = F.linear(x, qkv_weight, qkv_bias)
    head_dim = C // num_heads
    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, head_dim).permute(
        2, 0, 3, 1, 4
    )
    q, k, v = qkv[0], qkv[1], qkv[2]  # (B*nW, num_heads, N, head_dim)

    relative_position_bias = relative_position_bias.to(device=q.device, dtype=q.dtype)

    shift_attn_mask: Optional[Tensor] = None
    if sum(shift_size) > 0:
        img_mask = torch.zeros((pad_H, pad_W), device=q.device, dtype=torch.int64)
        h_slices = (
            (0, -window_size[0]),
            (-window_size[0], -shift_size[0]),
            (-shift_size[0], None),
        )
        w_slices = (
            (0, -window_size[1]),
            (-window_size[1], -shift_size[1]),
            (-shift_size[1], None),
        )
        count = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[h[0] : h[1], w[0] : w[1]] = count
                count += 1

        mask_windows = img_mask.view(
            pad_H // window_size[0],
            window_size[0],
            pad_W // window_size[1],
            window_size[1],
        )
        mask_windows = mask_windows.permute(0, 2, 1, 3).reshape(
            num_windows, window_size[0] * window_size[1]
        )
        shift_attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        shift_attn_mask = shift_attn_mask.ne(0).to(dtype=q.dtype) * (-100.0)
        # (num_windows, N, N) with values {0, -100}

    attn_bias = relative_position_bias.expand(x.size(0), -1, -1, -1)
    if shift_attn_mask is not None:
        shift_attn_mask = shift_attn_mask.repeat(B, 1, 1).unsqueeze(1)
        attn_bias = attn_bias + shift_attn_mask

    can_flash_attn = (
        FLASH_ATTN_AVAILABLE
        and flash_attn_func is not None
        and logit_scale is None
        and q.is_cuda
        # and q.dtype in (torch.float16, torch.bfloat16)
        and head_dim <= 256
        and head_dim % 8 == 0
    )

    print(flash_attn_func is not None)
    print(logit_scale is None)
    print(q.is_cuda)
    print(q.dtype)
    print(head_dim)
    print(head_dim)
    if can_flash_attn:
        print("Using flash attention") # FIXME: dummy
        flash_attn = cast(Callable[..., Tensor], flash_attn_func)

        # flash_attn expects (batch, seqlen, nheads, headdim)
        q_ = q.transpose(1, 2).contiguous()
        k_ = k.transpose(1, 2).contiguous()
        v_ = v.transpose(1, 2).contiguous()

        x = flash_attn(
            q_,
            k_,
            v_,
            dropout_p=attention_dropout if training else 0.0,
            softmax_scale=None,
            causal=False,
            attn_bias=attn_bias,
        )
        # (B*nW, N, num_heads, head_dim) -> (B*nW, N, C)
        x = x.reshape(x.size(0), x.size(1), C)
    else:
        if logit_scale is not None:
            # cosine attention (Swin V2)
            attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
            logit_scale = torch.clamp(logit_scale, max=math.log(100.0)).exp()
            attn = attn * logit_scale
            attn = attn + attn_bias
            attn = F.softmax(attn, dim=-1)
            attn = F.dropout(attn, p=attention_dropout, training=training)
            x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
        else:
            # Let PyTorch pick the best attention kernel.
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_bias,
                dropout_p=attention_dropout if training else 0.0,
                is_causal=False,
            )
            # attn_out: (B*nW, num_heads, N, head_dim)
            x = attn_out.transpose(1, 2).reshape(
                attn_out.shape[0], attn_out.shape[2], C
            )

    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout, training=training)

    # reverse windows
    x = x.view(
        B,
        pad_H // window_size[0],
        pad_W // window_size[1],
        window_size[0],
        window_size[1],
        C,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

    # reverse cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

    # unpad features
    x = x[:, :H, :W, :].contiguous()
    return x


torch.fx.wrap("shifted_window_attention")


class ShiftedWindowAttentionFlashAttn(nn.Module):
    """
    See :func:`shifted_window_attention`.
    """

    def __init__(
        self,
        dim: int,
        window_size: list[int],
        shift_size: list[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError("window_size and shift_size must be of length 2")
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        self.define_relative_position_bias_table()
        self.define_relative_position_index()

    def define_relative_position_bias_table(self):
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                self.num_heads,
            )
        )  # 2*Wh-1 * 2*Ww-1, nH
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def define_relative_position_index(self):
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij")
        )  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).flatten()  # Wh*Ww*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def get_relative_position_bias(self) -> torch.Tensor:
        return _get_relative_position_bias(
            self.relative_position_bias_table,
            self.relative_position_index,  # type: ignore[arg-type]
            self.window_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Tensor with layout of [B, H, W, C]
        Returns:
            Tensor with same layout as input, i.e. [B, H, W, C]
        """
        relative_position_bias = self.get_relative_position_bias()
        return shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            training=self.training,
        )
