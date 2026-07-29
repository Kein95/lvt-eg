"""Centralized path + environment configuration.

Mirrors the layout of `ViInfographicVQA/src/config.py`.
All paths resolve from env vars with sensible defaults so the code runs
identically on Colab, local Linux, and Windows.
"""

from __future__ import annotations

import os
from pathlib import Path

# -----------------------------------------------------------------------------
# Environment helpers
# -----------------------------------------------------------------------------

def _get_env_path(env_var: str, default: str | None = None) -> str | None:
    """Read path from env, fall back to default. Empty strings count as unset."""
    value = os.environ.get(env_var, "").strip()
    return value or default


# -----------------------------------------------------------------------------
# Colab auto-detection
# -----------------------------------------------------------------------------

IS_COLAB = (
    os.environ.get("LVT_COLAB") == "1"
    or os.path.exists("/content")
    or "COLAB_GPU" in os.environ
)

# -----------------------------------------------------------------------------
# Base directories
# -----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = Path(_get_env_path("LVT_DATA_DIR", str(REPO_ROOT / "data")))
OUTPUT_DIR = Path(_get_env_path("LVT_OUTPUT_DIR", str(REPO_ROOT / "results")))
PRETRAINED_BASE = _get_env_path("LVT_PRETRAINED_BASE")  # may be None

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Per-dataset sub-paths
# -----------------------------------------------------------------------------

# Ordered as in paper Table I: RodoSol -> CCPD -> Ukrainian -> LPLC -> LP-2025.
DATASET_SUBDIRS = {
    "rodosol":   "rodosol-alpr",
    "ccpd2019":  "ccpd-2019",
    "ukrainian": "ukrainian-lp",
    "lplc":      "lplc",
    "lp2025":    "lp-2025",
}


def get_dataset_dir(name: str) -> Path:
    """Return the directory holding the named dataset (created if missing)."""
    if name not in DATASET_SUBDIRS:
        raise ValueError(
            f"Unknown dataset key: {name}. "
            f"Available: {list(DATASET_SUBDIRS.keys())}"
        )
    path = DATA_DIR / DATASET_SUBDIRS[name]
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_run_dir(run_name: str) -> Path:
    """Return the per-run output directory (created if missing)."""
    path = OUTPUT_DIR / run_name
    path.mkdir(parents=True, exist_ok=True)
    return path
