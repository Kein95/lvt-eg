"""LVT-EG training objective - paper Sec 3.6 (notebook Cell 8).

    L = L_CTC + lambda * || e_hat - e_GT ||_1,    lambda = 0.1     (Eq. 2)

The edge term is L1 between predicted edge map e_hat (from the Visual-Tactile
Branch) and the Sobel target e_GT (from the STN-aligned undegraded reference).
The CTC term is the standard CTC loss between predicted log-probs and the
plate label sequence. Lambda is fixed at 0.1 across all datasets so that
sequence recognition remains the dominant objective.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LVTEGLoss(nn.Module):
    """Combined CTC + L1-edge loss matching paper Eq. (2)."""

    def __init__(self, edge_l1_weight: float = 0.1) -> None:
        super().__init__()
        self.ctc = nn.CTCLoss(blank=0, zero_infinity=True)
        self.edge_l1_weight = edge_l1_weight

    def forward(
        self,
        log_probs: torch.Tensor,           # [B, T, C]
        labels: torch.Tensor,              # [B, max_label_len]
        input_lengths: torch.Tensor,       # [B]
        target_lengths: torch.Tensor,      # [B]
        pred_edge: torch.Tensor | None,    # [B, 1, H, W] or None at inference
        gt_edge: torch.Tensor | None,      # [B, 1, H, W] or None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (total, ctc, edge_l1)."""
        # nn.CTCLoss expects [T, B, C]; permute once
        loss_ctc = self.ctc(log_probs.permute(1, 0, 2), labels, input_lengths, target_lengths)
        if gt_edge is not None and pred_edge is not None:
            loss_edge = F.l1_loss(pred_edge, gt_edge)
            total = loss_ctc + self.edge_l1_weight * loss_edge
        else:
            loss_edge = torch.tensor(0.0, device=log_probs.device)
            total = loss_ctc
        return total, loss_ctc, loss_edge
