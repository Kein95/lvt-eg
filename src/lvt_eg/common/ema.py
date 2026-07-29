"""Exponential Moving Average of model weights - paper Sec impl (notebook Cell 6).

Default decay 0.999 matches the paper. EMA weights are evaluated at the end
of each epoch and saved as the "best" checkpoint when val plate-RR improves.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn


class EMA:
    """In-memory state-dict EMA. Cheap and keeps weights on the same device as the model."""

    def __init__(self, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: dict[str, Any] | None = None
        self.initialized = False

    def update(self, model: nn.Module) -> None:
        if not self.initialized:
            self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
            self.initialized = True
            return
        assert self.shadow is not None
        for k, v in model.state_dict().items():
            # BatchNorm `num_batches_tracked` is int64 - EMA decay only makes
            # sense for floating-point tensors. Copy non-float buffers as-is.
            if v.is_floating_point():
                self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.detach()
            else:
                self.shadow[k] = v.detach().clone()

    def apply(self, model: nn.Module) -> None:
        """Load EMA weights into `model` (in-place). Caller is responsible for backup if needed."""
        if self.shadow:
            model.load_state_dict(self.shadow)

    def state_dict(self) -> dict[str, Any]:
        return self.shadow or {}
