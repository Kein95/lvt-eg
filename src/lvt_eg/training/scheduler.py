"""OneCycleLR + AdamW factory - paper Sec impl.

Builds optimizer + scheduler from a YAML config block:

    optimizer:    adamw
    weight_decay: 0.02     (CCPD: 1e-4)
    scheduler:    onecycle
    max_lr:       3.0e-4   (CCPD: 5.0e-4)
    pct_start:    0.3      (matches the notebooks' implicit OneCycleLR default)
    epochs:       30

steps_per_epoch is supplied from the training DataLoader length.
"""

from __future__ import annotations

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR


def build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    name = cfg.get("optimizer", "adamw").lower()
    wd = float(cfg.get("weight_decay", 0.02))
    if name != "adamw":
        raise ValueError(f"Only adamw supported, got {name}")
    # Initial LR is set by OneCycleLR; placeholder here.
    return AdamW(model.parameters(), lr=1e-4, weight_decay=wd)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict,
                    steps_per_epoch: int) -> OneCycleLR:
    name = cfg.get("scheduler", "onecycle").lower()
    if name != "onecycle":
        raise ValueError(f"Only onecycle supported, got {name}")
    return OneCycleLR(
        optimizer,
        max_lr=float(cfg["max_lr"]),
        epochs=int(cfg["epochs"]),
        steps_per_epoch=steps_per_epoch,
        pct_start=float(cfg.get("pct_start", 0.3)),
    )
