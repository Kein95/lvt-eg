"""Spatial Transformer Network - paper Sec 3.2 (notebook Cell 6).

LVT-EG applies an STN BEFORE edge prediction so the RGB input and the
structural supervision are expressed in a rectified coordinate frame.
The localization network is a compact convolutional regressor followed by
a 6-parameter affine head, initialized to the identity transform so
rectification starts from a no-warp mapping.

Parameter count: 284K (paper Sec 3.2). The same STN module - with shared
weights - is applied separately to the degraded input crop and the
undegraded reference crop during training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class STNBlock(nn.Module):
    """3-channel STN that rectifies an RGB plate crop via affine warp."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, 2, 2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4, 8)),
        )
        self.fc_loc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 8, 128),
            nn.ReLU(True),
            nn.Linear(128, 6),
        )
        # Identity-init: affine = [1 0 0; 0 1 0]
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(
            torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = self.localization(x)
        theta = self.fc_loc(xs).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)
