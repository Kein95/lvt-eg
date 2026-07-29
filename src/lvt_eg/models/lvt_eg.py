"""LVT-EG top-level model - paper Eq. (1), Sec 3.1 (notebook Cell 8 'SVTRv2EdgeOCR').

Pipeline (paper Eq. 1):

    x  -- STN_3ch -->  x'  -- f_theta -->  e_hat
    [x', e_hat]  -- SVTRv2 -->  F  -- RCTC -->  y_hat

The same STN module rectifies both the degraded input crop x and the
undegraded reference crop x_ref (training only). The Visual-Tactile Branch
predicts e_hat from x' and exposes get_gt_edge() to compute the Sobel target
e_GT from the rectified clean reference.

Total parameters depend on the input grid size:
    * Default 48x192 (RodoSol, Ukrainian, LPLC, LP-2025):
        23.58M total = 23.20M (SVTRv2-Base enc+RCTC) + 284K (STN) + 94.4K (VT branch).
        Matches paper Implementation Details (full model param count).
    * CCPD 32x128 (legacy_conv_mixer=True):
        ~20.12M = 17.65M (SVTRv2 enc) + 2.10M (RCTC dec) + 284K + 94.4K.
        Lighter because CCPD trained with the grouped-conv ConvMixer (v60a, no
        BatchNorm), not the full-conv+BN variant used for the other datasets.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lvt_eg.models.rctc_decoder import RCTCDecoder
from lvt_eg.models.stn import STNBlock
from lvt_eg.models.svtrv2_encoder import SVTRv2Encoder
from lvt_eg.models.visual_tactile_branch import VisualTactileBranch


class LVTEGModel(nn.Module):
    """LVT-EG: STN -> Visual-Tactile Branch -> SVTRv2-Base -> RCTC -> CTC."""

    def __init__(
        self,
        num_classes: int = 37,
        img_h: int = 48,
        img_w: int = 192,
        use_stn: bool = True,
        drop_path: float = 0.2,
        use_pos_embed: bool = True,
        legacy_conv_mixer: bool = False,
        use_edge: bool = True,
        stn_order: str = "stn_first",
    ) -> None:
        super().__init__()
        if stn_order not in ("stn_first", "rect_after"):
            raise ValueError(
                f"stn_order must be 'stn_first' or 'rect_after', got {stn_order!r}"
            )
        self.use_stn = use_stn
        self.use_edge = use_edge
        self.stn_order = stn_order

        # Sec 3.2: STN rectifier. Placement decides its input width:
        #   stn_first  -> STN on the 3-channel RGB crop, edge predicted after.
        #   rect_after -> STN on the fused 4-channel [RGB, edge] tensor.
        # The no-edge ablation always rectifies the 3-channel RGB crop.
        if use_stn:
            stn_in = 4 if (use_edge and stn_order == "rect_after") else 3
            self.stn = STNBlock(in_channels=stn_in)

        # Sec 3.3: lightweight CNN edge predictor f_theta (omitted for no-edge).
        if use_edge:
            self.imaginator = VisualTactileBranch(in_channels=3, hidden_channels=64)

        # Sec 3.4: SVTRv2-Base. 4-channel input (RGB + edge) with edge fusion,
        # 3-channel (RGB only) for the no-edge ablation (paper Table VI row 7).
        self.encoder = SVTRv2Encoder(
            max_sz=(img_h, img_w),
            in_channels=4 if use_edge else 3,
            drop_path_rate=drop_path,
            use_pos_embed=use_pos_embed,
            feat2d=True,
            legacy_conv_mixer=legacy_conv_mixer,
        )

        # Sec 3.5: RCTC decoder + linear classifier
        self.decoder = RCTCDecoder(
            in_channels=self.encoder.out_channels,
            out_channels=num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
        hr_images: torch.Tensor | None = None,
        return_all: bool = False,
    ):
        """Forward pass implementing paper Eq. (1).

        Args:
            x:         [B, 3, H, W] degraded LR input.
            hr_images: [B, 3, H, W] STN-input-grid undegraded reference (training only).
            return_all: if True, also return (pred_edge, gt_edge) for the loss.
        """
        # No-edge ablation (paper Table VI row 7): RGB-only 3-channel pipeline.
        if not self.use_edge:
            if self.use_stn:
                x = self.stn(x)
            features = self.encoder(x)
            logits = self.decoder(features)
            log_probs = (F.log_softmax(logits, dim=2) if self.training
                         else torch.log(logits + 1e-8))
            return (log_probs, None, None) if return_all else log_probs

        if self.stn_order == "stn_first":
            # STN-first: rectify RGB (and reference) first, predict edge on the
            # rectified crop; edge target = Sobel of the rectified reference.
            if self.use_stn:
                x = self.stn(x)
                if hr_images is not None:
                    hr_images = self.stn(hr_images)
            pred_edge = self.imaginator(x)
            gt_edge = self.imaginator.get_gt_edge(hr_images) if hr_images is not None else None
            x_4ch = torch.cat([x, pred_edge], dim=1)
        else:
            # rect_after: predict edge on the raw crop, fuse to 4 channels, then
            # rectify the 4-channel tensor. Edge target = Sobel of the raw
            # (un-rectified) reference, matching the ablation notebooks.
            pred_edge = self.imaginator(x)
            gt_edge = self.imaginator.get_gt_edge(hr_images) if hr_images is not None else None
            x_4ch = torch.cat([x, pred_edge], dim=1)
            if self.use_stn:
                x_4ch = self.stn(x_4ch)

        # SVTRv2-Base encode -> 2D feature map -> RCTC reader + classifier.
        features = self.encoder(x_4ch)
        logits = self.decoder(features)

        # During training the decoder returns raw logits; during eval it
        # already applied softmax. Convert both to log-probs for CTC.
        if self.training:
            log_probs = F.log_softmax(logits, dim=2)
        else:
            log_probs = torch.log(logits + 1e-8)

        if return_all:
            return log_probs, pred_edge, gt_edge
        return log_probs
