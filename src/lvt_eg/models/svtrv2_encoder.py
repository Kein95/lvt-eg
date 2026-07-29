"""SVTRv2-Base sequence encoder - paper Sec 3.4 (notebook Cell 7).

Three-stage hierarchical encoder interleaving local convolution blocks
(ConvMixer) with global self-attention blocks. The first conv is modified
from 3-channel to 4-channel to accept the [RGB, edge] concatenation; its
weights (including the new edge channel) use the standard Kaiming
initialization applied to every conv, matching the paper and the notebooks.

For 48x192 input, the encoder produces a 384 x 6 x 48 feature map (paper).
Stochastic depth uses a maximum drop-path rate of 0.2 (paper Sec 3.4).

Code taken verbatim from notebook Cell 7 and split for readability.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

# ---------------------------------------------------------------------------
# Stochastic-depth helpers
# ---------------------------------------------------------------------------

def drop_path_fn(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False,
                 scale_by_keep: bool = True) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path_fn(x, self.drop_prob, self.training)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class SVTRAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dim)
        return self.proj_drop(self.proj(x))


class FGlobalAttention(nn.Module):
    """Per-row (height-wise) global attention, used in stage 2 of SVTRv2-Base."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.dim = dim
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, sz):
        B, N, _ = x.shape
        H, W = sz
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q_h = q.reshape(B, self.num_heads, H, W, self.head_dim).permute(0, 2, 1, 3, 4).reshape(B * H, self.num_heads, W, self.head_dim)
        k_h = k.reshape(B, self.num_heads, H, W, self.head_dim).permute(0, 2, 1, 3, 4).reshape(B * H, self.num_heads, W, self.head_dim)
        v_h = v.reshape(B, self.num_heads, H, W, self.head_dim).permute(0, 2, 1, 3, 4).reshape(B * H, self.num_heads, W, self.head_dim)
        attn_h = (q_h @ k_h.transpose(-2, -1)) * self.scale
        attn_h = attn_h.softmax(dim=-1)
        attn_h = self.attn_drop(attn_h)
        out_h = (attn_h @ v_h).reshape(B, H, self.num_heads, W, self.head_dim).permute(0, 2, 1, 3, 4).reshape(B, self.num_heads, N, self.head_dim)
        x = out_h.transpose(1, 2).reshape(B, N, self.dim)
        return self.proj_drop(self.proj(x))


class ConvMixer(nn.Module):
    def __init__(self, dim, kernel_size=3, num_conv=2, HW=None, num_heads=8,
                 legacy=False):
        super().__init__()
        self.HW = HW
        convs = []
        for _ in range(num_conv):
            if legacy:
                # v60a CCPD notebook variant: grouped conv (groups=num_heads),
                # no GELU/BatchNorm. This is the architecture that trained the
                # paper's CCPD model (~20.12M); the full-conv+BN branch below is
                # ~3.4M heavier and was the v61a recipe for the other datasets.
                convs.append(
                    nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2,
                              groups=num_heads))
            else:
                convs.extend([
                    nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2),
                    nn.GELU(),
                    nn.BatchNorm2d(dim),
                ])
        self.conv = nn.Sequential(*convs)

    def forward(self, x, sz):
        B, N, C = x.shape
        H, W = sz
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.conv(x)
        return x.flatten(2).transpose(1, 2), sz


class SVTRBlock(nn.Module):
    def __init__(self, dim, num_heads, mixer="Global", HW=None, mlp_ratio=4.0,
                 qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 eps=1e-6, num_conv=2, kernel_size=3, legacy_conv_mixer=False):
        super().__init__()
        self.norm1 = norm_layer(dim, eps=eps)
        self.mixer_type = mixer
        if mixer == "Global":
            self.mixer = SVTRAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        elif mixer == "FGlobal":
            self.mixer = FGlobalAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        else:
            self.mixer = ConvMixer(dim, kernel_size, num_conv, HW,
                                   num_heads=num_heads, legacy=legacy_conv_mixer)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim, eps=eps)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, sz):
        if self.mixer_type == "Global":
            x = x + self.drop_path(self.mixer(self.norm1(x)))
        elif self.mixer_type == "FGlobal":
            x = x + self.drop_path(self.mixer(self.norm1(x), sz))
        else:
            cur, sz = self.mixer(self.norm1(x), sz)
            x = x + self.drop_path(cur)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, sz


class POPatchEmbed(nn.Module):
    """4-channel patch embed (RGB + edge). The first conv, including the edge
    channel, is Kaiming-initialized by SVTRv2Encoder._init_weights (paper Sec 3.4)."""

    def __init__(self, in_channels=4, feat_max_size=(8, 32), embed_dim=64,
                 use_pos_embed=True, flatten=True, bias=False):
        super().__init__()
        self.patch = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, 3, 2, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1),
            nn.GELU(),
        )
        self.use_pos_embed = use_pos_embed
        self.flatten = flatten
        if use_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, feat_max_size[0] * feat_max_size[1], embed_dim))
            trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.patch(x)
        sz = [x.shape[2], x.shape[3]]
        x = x.flatten(2).transpose(1, 2)
        if self.use_pos_embed:
            x = x + self.pos_embed[:, :x.shape[1], :]
        return x, sz


class Feat2D(nn.Module):
    def forward(self, x, sz=None):
        if sz is None:
            return x, None
        B, _, C = x.shape
        H, W = sz
        return x.transpose(1, 2).reshape(B, C, H, W), sz


class SubSample(nn.Module):
    def __init__(self, in_channels, out_channels, sub_k=(2, 1),
                 norm_layer=nn.LayerNorm, act=nn.GELU):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm = norm_layer(out_channels)
        self.pool = nn.AvgPool2d(kernel_size=tuple(sub_k), stride=tuple(sub_k)) if (sub_k[0] > 1 or sub_k[1] > 1) else None
        self.act = act()

    def forward(self, x, sz):
        B, _, C = x.shape
        H, W = sz
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.conv(x)
        if self.pool is not None:
            x = self.pool(x)
        sz = [x.shape[2], x.shape[3]]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(self.act(x))
        return x, sz


class SVTRStage(nn.Module):
    def __init__(self, dim, out_dim, depth, mixer, kernel_sizes=None,
                 sub_k=(2, 1), num_heads=8, mlp_ratio=4.0,
                 qkv_bias=False, qk_scale=None, drop_rate=0.0,
                 attn_drop_rate=0.0, drop_path=None,
                 norm_layer=nn.LayerNorm, act=nn.GELU,
                 downsample=True, eps=1e-6, num_conv=None,
                 legacy_conv_mixer=False):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3] * depth
        if num_conv is None:
            num_conv = [2] * depth
        self.blocks = nn.ModuleList([
            SVTRBlock(
                dim=dim, num_heads=num_heads, mixer=mixer[i],
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path[i] if drop_path is not None else 0.0,
                norm_layer=norm_layer, act_layer=act, eps=eps,
                num_conv=num_conv[i], kernel_size=kernel_sizes[i],
                legacy_conv_mixer=legacy_conv_mixer,
            )
            for i in range(depth)
        ])
        self.downsample = SubSample(dim, out_dim, sub_k, norm_layer, act) if (downsample and out_dim > 0) else None

    def forward(self, x, sz):
        for blk in self.blocks:
            x, sz = blk(x, sz)
        if self.downsample is not None:
            x, sz = self.downsample(x, sz)
        return x, sz


# ---------------------------------------------------------------------------
# Top-level SVTRv2-Base encoder
# ---------------------------------------------------------------------------

class SVTRv2Encoder(nn.Module):
    """SVTRv2-Base encoder accepting [RGB+edge] 4-channel input.

    Default config matches notebook constants (V61/V60 = SVTRv2-Base).
    """

    def __init__(
        self,
        max_sz=(32, 128),
        in_channels: int = 4,
        depths=(6, 6, 6),
        dims=(128, 256, 384),
        num_heads=(4, 8, 12),
        # Notebook-authoritative mixer (5 datasets identical): stage 2 has
        # 2 Conv + 1 FGlobal + 3 Global, NOT 2 Conv + 4 FGlobal.
        mixer=(("Conv",) * 6,
               ("Conv", "Conv", "FGlobal", "Global", "Global", "Global"),
               ("Global",) * 6),
        # `[-1,-1]` in notebook = "no downsample at last stage"; the encoder
        # already disables downsample for the last stage via the
        # `downsample=(i != num_stages-1)` rule, so (1,1) here is a benign
        # placeholder that is never used.
        sub_k=((1, 1), (2, 1), (1, 1)),
        use_pos_embed: bool = True,
        drop_path_rate: float = 0.2,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        eps: float = 1e-6,
        feat2d: bool = True,
        num_convs=(("2",) * 6, ("2",) * 2 + ("3",) * 4, ("3",) * 6),
        kernel_sizes=(("3",) * 6, ("3",) * 6, ("3",) * 6),
        # CCPD (notebook v60a) used grouped-conv ConvMixer (no BN); set True to
        # reproduce that ~20.12M model. Default False = v61a full-conv+BN (~23.58M).
        legacy_conv_mixer: bool = False,
    ):
        super().__init__()
        # Convert nested string tuples to lists of ints (python defaults)
        num_convs = [[int(v) for v in stage] for stage in num_convs]
        kernel_sizes = [[int(v) for v in stage] for stage in kernel_sizes]
        mixer = [list(stage) for stage in mixer]

        num_stages = len(depths)
        feat_max_size = (max_sz[0] // 4, max_sz[1] // 4)
        self.pope = POPatchEmbed(
            in_channels=in_channels, feat_max_size=feat_max_size,
            embed_dim=dims[0], use_pos_embed=use_pos_embed,
        )
        dpr = np.linspace(0, drop_path_rate, sum(depths))
        self.stages = nn.ModuleList()
        for i in range(num_stages):
            self.stages.append(SVTRStage(
                dim=dims[i],
                out_dim=dims[i + 1] if i < num_stages - 1 else 0,
                depth=depths[i],
                mixer=mixer[i],
                kernel_sizes=kernel_sizes[i],
                sub_k=sub_k[i],
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[sum(depths[:i]):sum(depths[: i + 1])],
                downsample=(i != num_stages - 1),
                eps=eps,
                num_conv=num_convs[i],
                legacy_conv_mixer=legacy_conv_mixer,
            ))
        if feat2d:
            self.stages.append(Feat2D())
        self.out_channels = dims[-1]
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, sz = self.pope(x)
        for stage in self.stages:
            x, sz = stage(x, sz)
        return x  # [B, C, H', W']
