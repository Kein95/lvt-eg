"""EMA (Exponential Moving Average) regression tests."""

import torch
import torch.nn as nn

from lvt_eg.common.ema import EMA


def test_ema_skips_int_tensors():
    """Verify EMA.update() skips int64 BatchNorm buffers without dtype errors."""
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3),
        nn.BatchNorm2d(16),
        nn.ReLU(),
    )
    ema = EMA(decay=0.999)

    # First update: init shadow
    ema.update(model)
    assert ema.initialized

    # Second update: should NOT crash on num_batches_tracked (int64)
    ema.update(model)

    # Verify shadow exists and num_batches_tracked is preserved
    assert ema.shadow is not None
    bn_tracked_key = "1.num_batches_tracked"
    assert bn_tracked_key in ema.shadow
    assert ema.shadow[bn_tracked_key].dtype == torch.int64


def test_ema_decay_blends_toward_current_weights():
    """Decay blend must move the shadow TOWARD the live weights by (1-decay).

    Governs the CCPD/paper EMA numbers; a flipped blend or wrong decay would
    silently shift every EMA-evaluated result.
    """
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.0)
    ema = EMA(decay=0.9)
    ema.update(model)                       # init: shadow = 0.0
    assert ema.initialized
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema.update(model)                       # 0.9*0 + 0.1*1 = 0.1
    assert abs(ema.shadow["weight"].item() - 0.1) < 1e-6
    ema.update(model)                       # 0.9*0.1 + 0.1*1 = 0.19 (toward 1.0)
    assert abs(ema.shadow["weight"].item() - 0.19) < 1e-6
