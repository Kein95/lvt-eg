"""Save / load training checkpoints (best + last).

Per-run output layout (results/<run_name>/):
    config.yaml   - frozen config copy
    best.pth      - EMA weights at best val plate-RR
    last.pth      - EMA weights at the latest epoch (resumable)
    history.json  - per-epoch metrics
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, model: torch.nn.Module, epoch: int,
                    val_acc: float, ema_state: dict[str, Any] | None = None,
                    extra: dict[str, Any] | None = None) -> None:
    """Save model state-dict (and optional EMA shadow) to `path`."""
    payload: dict[str, Any] = {
        "epoch": epoch,
        "val_acc": val_acc,
        "model": model.state_dict(),
    }
    if ema_state:
        payload["ema"] = ema_state
    if extra:
        payload["extra"] = extra
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    # weights_only=True (torch >= 2.6 default) blocks pickle RCE on
    # third-party checkpoints. Set explicitly for older torch versions.
    return torch.load(path, map_location=map_location, weights_only=True)


def append_history(path: str | Path, entry: dict[str, Any]) -> None:
    """Append a JSON line of per-epoch metrics. Creates the file if missing."""
    p = Path(path)
    history = json.loads(p.read_text()) if p.exists() else []
    history.append(entry)
    p.write_text(json.dumps(history, indent=2))
