"""Visual-Tactile Branch f_theta - paper Sec 3.3 (notebook Cell 6 'StructureImaginator').

Lightweight 5-layer CNN edge predictor:

    f_theta : R^(3 x H x W)  ->  [0, 1]^(1 x H x W)

Maps the rectified RGB crop x' to the structural edge cue e_hat. Output
goes through sigmoid to lie in [0, 1] like the Sobel target. The predicted
edge map is concatenated with x' channel-wise for the recognizer.

Architecture (Sec 3.3): Conv 3x3 + ReLU stack ending in 1-channel sigmoid.
Total parameters: 94.4K - only 0.4% of the 23.58M LVT-EG model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lvt_eg.common.sobel import SobelEdgeDetector


class VisualTactileBranch(nn.Module):
    """f_theta in paper Eq. (1). Predicts e_hat from rectified RGB crop x'."""

    def __init__(self, in_channels: int = 3, hidden_channels: int = 64) -> None:
        super().__init__()
        # Frozen Sobel - used at training time to compute e_GT from the
        # STN-aligned undegraded reference crop. Always part of the module
        # so we can call .get_gt_edge() without external dependencies.
        self.edge_detector = SobelEdgeDetector()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(hidden_channels // 2, 1, 3, 1, 1), nn.Sigmoid(),
        )

    def get_gt_edge(self, hr_images: torch.Tensor) -> torch.Tensor:
        """Compute Sobel-based e_GT from the (already STN-aligned) clean reference."""
        with torch.no_grad():
            return self.edge_detector(hr_images)

    def forward(self, x_prime: torch.Tensor) -> torch.Tensor:
        # x_prime: [B, 3, H, W] - rectified RGB crop
        # returns:  [B, 1, H, W] - predicted edge cue e_hat in [0, 1]
        return self.net(x_prime)
