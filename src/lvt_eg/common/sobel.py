"""Frozen Sobel edge operator - paper Sec 3.6 (notebook Cell 6).

Computes the structural edge target e_GT from the STN-aligned undegraded
reference crop:

    e_GT = Sobel(x'_ref)              (paper Eq. after Eq. 2)

Procedure (paper Sec 3.6):
1. Convert RGB to grayscale via standard ITU-R BT.601 weights.
2. Apply 3x3 horizontal and vertical Sobel kernels.
3. Compute gradient magnitude sqrt(Gx^2 + Gy^2).
4. Per-image normalize to [0, 1] by max value.

The operator is fixed (no learnable parameters) and the target is computed
under no-gradient - the edge loss treats it as a constant within each batch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelEdgeDetector(nn.Module):
    """3x3 Sobel filter producing a per-image-normalized edge magnitude map."""

    def __init__(self) -> None:
        super().__init__()
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("sx", sx.view(1, 1, 3, 3))
        self.register_buffer("sy", sy.view(1, 1, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        # x: [B, 3, H, W] in arbitrary normalized RGB range
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        gx = F.conv2d(gray, self.sx, padding=1)
        gy = F.conv2d(gray, self.sy, padding=1)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        b = mag.size(0)
        # Clamp the per-image max to avoid divide-by-near-zero / NaN under
        # AMP fp16 when the input crop is essentially uniform (e.g. heavily
        # degraded synth-v2 sample).
        per_img_max = mag.view(b, -1).max(dim=1)[0].view(b, 1, 1, 1).clamp(min=1e-6)
        return mag / per_img_max
