"""RCTC reader (Feature-Rearrangement CTC) - paper Sec 3.5 (notebook Cell 7).

Replaces naive height-axis pooling with a learnable query that cross-attends
to the encoder feature map and selectively collapses the vertical dimension
(paper Sec 3.5). For input feature map F of shape [B, C, H, W], the decoder
produces a 1-D sequence of length T = W with channel dim C, then a linear
classifier maps to (|alphabet| + 1) logits.

Pipeline (matches pipeline.pdf "RCTC reader" block):
1. Width-wise self-attention over each row.
2. Query cross-attention down-collapses H -> 1.
3. Linear classifier.
4. Softmax (eval) or log-softmax (train).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from lvt_eg.models.svtrv2_encoder import DropPath, Mlp, SVTRAttention


class RCTCBlock(nn.Module):
    """One self-attention + MLP block applied along the width dimension."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SVTRAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                  qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class RCTCDecoder(nn.Module):
    """RCTC reader: width-wise SA -> query cross-attn (collapse H) -> linear -> logits."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.char_token = nn.Parameter(torch.zeros(1, 1, in_channels), requires_grad=True)
        trunc_normal_(self.char_token, mean=0, std=0.02)
        self.fc = nn.Linear(in_channels, out_channels, bias=True)
        self.fc_kv = nn.Linear(in_channels, 2 * in_channels, bias=True)
        self.w_atten_block = RCTCBlock(
            dim=in_channels, num_heads=in_channels // 32, mlp_ratio=4.0,
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        # Width-wise self-attention: treat each row as an independent length-W sequence.
        x = self.w_atten_block(
            x.permute(0, 2, 3, 1).reshape(-1, W, C)
        ).reshape(B, H, W, C).permute(0, 3, 1, 2)
        # Query cross-attention: char_token attends over the H*W feature grid,
        # collapsing height dimension to produce W sequence elements.
        x_kv = self.fc_kv(x.flatten(2).transpose(1, 2)).reshape(B, H * W, 2, C).permute(2, 0, 3, 1)
        x_k, x_v = x_kv.unbind(0)
        char_token = self.char_token.tile([B, 1, 1])
        attn = char_token @ x_k                         # [B, 1, H*W]
        attn = attn.reshape(-1, 1, H, W)
        attn = F.softmax(attn, dim=2)                   # softmax over H
        attn = attn.permute(0, 3, 1, 2)                 # [B, W, 1, H]
        x_v = x_v.reshape(B, C, H, W)
        feats = attn @ x_v.permute(0, 3, 2, 1)          # [B, W, 1, C]
        feats = feats.squeeze(2)                        # [B, W, C]
        logits = self.fc(feats)                         # [B, W, num_classes]
        if not self.training:
            logits = F.softmax(logits, dim=2)
        return logits
