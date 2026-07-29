"""Deterministic seeding for Python / NumPy / PyTorch.

Used at the start of every training/eval run to keep results reproducible
across the 2-4 seeds reported in the paper (Sec impl).
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed all RNGs used by the training pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Keep cuDNN deterministic enough for paper-level reporting.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader per-worker seeding so augmentation RNG is reproducible.

    Each worker derives its seed from the main process seed via
    torch.initial_seed(); re-seed NumPy/random so albumentations is
    deterministic across runs with the same config seed.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Return a CPU generator seeded for DataLoader shuffling order."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
