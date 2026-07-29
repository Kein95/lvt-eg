"""Multi-Level Synthetic Degradation - paper Sec 3.7 (notebook Cell 5).

    "Edge-safe": only the RGB input is degraded; the Sobel-based structural
    target is computed from the aligned undegraded reference crop.

Two policies (paper Sec 3.7 last paragraph):
    * v1 - single-level degrade (blur, additive noise, JPEG, moderate
      downscale) -> 2x training data (original + one degraded view).
    * v2 - three-level light/medium/heavy + fog overlay and occasional
      sun-glare spot -> 4x training data (original + three degraded views).
Validation / test sets are never augmented.
"""

from __future__ import annotations

import inspect as _inspect
import random

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from lvt_eg.data.lp_crop_dataset import (
    NORMALIZE,
    crop_plate,
    resolve_lr_aug,
)

# --- Synth v1: light / medium / heavy (blur + noise + JPEG) -----------------

# Albumentations API drift between versions - probe signatures so the same
# code runs on Colab (newer) and local (older) installs.
_GN_NEW = "std_range" in _inspect.signature(A.GaussNoise.__init__).parameters
_IC_NEW = "quality_range" in _inspect.signature(A.ImageCompression.__init__).parameters
_DS_NEW = "scale_range" in _inspect.signature(A.Downscale.__init__).parameters


def _gauss_noise(p):
    if _GN_NEW:
        return A.GaussNoise(std_range=(0.01, 0.03), p=p)
    return A.GaussNoise(var_limit=(6.5, 58.5), p=p)


def _img_compress(lo, hi, p):
    if _IC_NEW:
        return A.ImageCompression(quality_range=(lo, hi), p=p)
    return A.ImageCompression(quality_lower=lo, quality_upper=hi, p=p)


def _downscale(lo, hi, p):
    if _DS_NEW:
        return A.Downscale(scale_range=(lo, hi), p=p)
    return A.Downscale(scale_min=lo, scale_max=hi, p=p)

V1_LIGHT = A.Compose([
    A.GaussianBlur(blur_limit=(3, 3), p=0.5),
    _gauss_noise(p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
    A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=15, p=0.4),
])

V1_MEDIUM = A.Compose([
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.MotionBlur(blur_limit=(3, 5), p=1.0),
    ], p=0.7),
    A.OneOf([
        A.GaussNoise(p=1.0),
        A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
    ], p=0.6),
    _img_compress(30, 60, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=25, val_shift_limit=25, p=0.4),
])

V1_HEAVY = A.Compose([
    A.OneOf([
        A.GaussianBlur(blur_limit=(5, 7), p=1.0),
        A.MotionBlur(blur_limit=(5, 9), p=1.0),
    ], p=0.8),
    A.OneOf([
        A.GaussNoise(p=1.0),
        A.MultiplicativeNoise(multiplier=(0.8, 1.2), p=1.0),
    ], p=0.8),
    _img_compress(15, 40, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=30, p=0.5),
])


# --- Synth v2 additions: downscale + fog + glare ----------------------------

V2_HEAVY_DOWNSCALE = _downscale(0.3, 0.5, p=0.5)


def add_fog_overlay(image: np.ndarray, intensity: float = 0.3) -> np.ndarray:
    """Blend image with white to simulate fog/haze."""
    fog = np.ones_like(image, dtype=np.float32) * 255
    alpha = random.uniform(0.05, intensity)
    blended = cv2.addWeighted(image.astype(np.float32), 1 - alpha, fog, alpha, 0)
    return np.clip(blended, 0, 255).astype(np.uint8)


def add_glare_spot(image: np.ndarray, max_radius: int = 30) -> np.ndarray:
    """Bright elliptical spot to simulate sun glare."""
    h, w = image.shape[:2]
    cx = random.randint(w // 4, 3 * w // 4)
    cy = random.randint(h // 4, 3 * h // 4)
    rx = random.randint(10, max_radius)
    ry = random.randint(5, max_radius // 2)
    overlay = image.copy().astype(np.float32)
    Y, X = np.ogrid[:h, :w]
    mask = ((X - cx) ** 2 / max(rx ** 2, 1) + (Y - cy) ** 2 / max(ry ** 2, 1)) <= 1.0
    overlay[mask] = overlay[mask] * 0.3 + 255 * 0.7
    return np.clip(overlay, 0, 255).astype(np.uint8)


# --- Synth v1: single-level degrade (2x expansion) --------------------------
# Notebook v1 (synthetic_degrade): one moderate pass - blur OR motion blur,
# additive noise, JPEG compression, moderate downscale - for 2x training data.
V1_SINGLE = A.Compose([
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        A.MotionBlur(blur_limit=(3, 7), p=1.0),
    ], p=0.7),
    A.OneOf([
        _gauss_noise(p=1.0),
        A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
    ], p=0.6),
    _img_compress(20, 50, p=0.5),
    _downscale(0.3, 0.5, p=0.5),
])


# --- Policy registry --------------------------------------------------------

POLICIES = {
    "v1": [V1_SINGLE],
    "v2": [V1_LIGHT, V1_MEDIUM, A.Compose([V1_HEAVY, V2_HEAVY_DOWNSCALE])],
}
LEVEL_NAMES = ["light", "medium", "heavy"]


# --- Dataset wrapper --------------------------------------------------------

class MultiLevelSyntheticDataset(Dataset):
    """Wrap an LPCropDataset to produce (1 + n_levels)x samples.

    Expansion is policy-dependent: v1 has one synthetic level (2x total),
    v2 has three light/medium/heavy levels (4x total).

    Index layout for N base samples and L synthetic levels:
        [0,           N)  -> real-degraded LR (base dataset's pipeline)
        [k*N,   (k+1)N)   -> synthetic level k-1  (v2: + occasional fog/glare)

    Edge target e_GT is always derived from the clean HR crop, never degraded.
    """

    def __init__(self, base_dataset, policy: str = "v2", use_synthetic: bool = True,
                 synth_aug: str = "photometric", synth_degrade_at: str = "lr") -> None:
        if policy not in POLICIES:
            raise ValueError(f"Unknown synth policy: {policy}. Available: {list(POLICIES)}")
        if synth_degrade_at not in ("lr", "hr"):
            raise ValueError(f"synth_degrade_at must be 'lr' or 'hr', got {synth_degrade_at}")
        self.base = base_dataset
        self.use_synthetic = use_synthetic and base_dataset.augment
        self.policy_name = policy
        self.levels = POLICIES[policy]
        self.n_levels = len(self.levels)
        self.n_real = len(base_dataset)
        # LR aug applied on top of the level degradation for synthetic views.
        # Notebooks differ: RodoSol/LPLC/LP-2025 add photometric lr_aug; CCPD
        # re-applies its geometric train aug; Ukrainian applies none (the
        # degradation levels already provide the perturbation).
        self.synth_aug = resolve_lr_aug(synth_aug)
        # Where the level degradation is applied: "lr" degrades at the model grid
        # (RodoSol/CCPD/LP-2025/LPLC notebooks); "hr" degrades at the HR crop then
        # resizes down (Ukrainian notebook).
        self.synth_degrade_at = synth_degrade_at

    def __len__(self) -> int:
        return self.n_real * (1 + self.n_levels) if self.use_synthetic else self.n_real

    def __getitem__(self, idx: int):
        if idx < self.n_real or not self.use_synthetic:
            return self.base[idx]

        synth_idx = idx - self.n_real
        level = min(synth_idx // self.n_real, self.n_levels - 1)
        real_idx = synth_idx % self.n_real
        s = self.base.samples[real_idx]

        img = cv2.imread(s["image_path"])
        if img is None:
            return self.base[real_idx]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        hr_crop = crop_plate(img, s.get("corners"), self.base.hr_h, self.base.hr_w)
        # Clean HR reference resampled to the model grid (feeds the edge target).
        hr_img = cv2.resize(hr_crop, (self.base.img_w, self.base.img_h))

        def _degrade(image):
            """Level degradation (+ v2 fog/glare) on an RGB image at its resolution."""
            out = self.levels[level](image=image)["image"]
            if self.policy_name == "v2":
                if level >= 1 and random.random() < 0.3:
                    out = add_fog_overlay(out, intensity=0.2 if level == 1 else 0.35)
                if level >= 2 and random.random() < 0.2:
                    out = add_glare_spot(out, max_radius=20)
            return out

        if self.synth_degrade_at == "hr":
            # Ukrainian notebook: degrade at the HR crop (e.g. 64x256), then resize
            # the degraded result down to the LR model grid.
            lr_img = cv2.resize(_degrade(hr_crop), (self.base.img_w, self.base.img_h))
        else:
            # Default: degrade directly at the LR model grid.
            lr_img = _degrade(hr_img)

        # Per-dataset LR aug on top of degradation, then normalize.
        if self.synth_aug is not None:
            lr_img = self.synth_aug(image=lr_img)["image"]
        lr_tensor = NORMALIZE(image=lr_img)["image"]
        hr_tensor = NORMALIZE(image=hr_img)["image"]

        # Encode label (same logic as base dataset)
        encoded = [self.base.char2idx.get(c, 0) for c in s["plate"].upper()]
        length = len(encoded)
        padded = encoded + [0] * (self.base.max_label_len - length)
        label = torch.tensor(padded[: self.base.max_label_len], dtype=torch.long)
        return lr_tensor, hr_tensor, label, length
