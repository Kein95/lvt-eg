"""LPCropDataset - single-frame plate-crop dataset (notebook Cell 5).

Implements the data-loading invariants of paper Sec 3.2 / 3.7:
    * Perspective-warp the LP region from the source image into a 64x256 HR crop.
    * Resample HR to the model input grid (e.g. 48x192).
    * Optionally apply LR-only augmentation (edge-safe: HR stays clean).

Each sample dict must contain:
    image_path: str   - source full image
    plate:      str   - ground-truth plate string (uppercase)
    corners:    list of 4 (x, y) tuples  - perspective corners
"""

from __future__ import annotations

from collections.abc import Sequence

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

# --- Standard transforms (notebook Cell 5) ----------------------------------

# Light "real-degradation" pipeline applied to LR during baseline training.
# Multi-Level Synthetic v1/v2 lives in `synthetic_degradation.py`.
LR_AUG = A.Compose([
    A.MotionBlur(blur_limit=(3, 7), p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.GaussNoise(p=0.2),
    A.RandomBrightnessContrast(p=0.3),
    A.HueSaturationValue(p=0.2),
])

# Geometric+photometric train aug used by the CCPD and Ukrainian notebooks'
# get_train_transforms (identical op set in both). Applied to the LR image
# before NORMALIZE, so it must NOT include Normalize/ToTensor here.
LR_AUG_GEOMETRIC = A.Compose([
    A.Affine(scale=(0.95, 1.05), translate_percent=(0.05, 0.05),
             rotate=(-5, 5), fill=128, p=0.5),
    A.Perspective(scale=(0.02, 0.05), p=0.3),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
    A.Rotate(limit=10, p=0.3),
    A.ChannelShuffle(p=0.3),
    A.CoarseDropout(num_holes_range=(2, 5), hole_height_range=(4, 8),
                    hole_width_range=(4, 8), p=0.3),
])

# Same normalization for both LR and HR (paper Sec 3.7: edge-safe principle).
NORMALIZE = A.Compose([
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
])

# Per-dataset LR augmentation policy (matches each notebook's train pipeline):
#   photometric - RodoSol / LPLC / LP-2025 (blur/noise/color only)
#   geometric   - CCPD / Ukrainian (Affine/Perspective/Rotate/ChannelShuffle/CoarseDropout)
#   none        - no augmentation (used for the Ukrainian synthetic branch)
_AUG_POLICIES = {"photometric": LR_AUG, "geometric": LR_AUG_GEOMETRIC, "none": None}


def resolve_lr_aug(name: str):
    """Return the albumentations Compose for a named policy (or None)."""
    if name not in _AUG_POLICIES:
        raise ValueError(f"Unknown aug policy: {name}. Available: {list(_AUG_POLICIES)}")
    return _AUG_POLICIES[name]


# --- Geometric helpers ------------------------------------------------------

def crop_plate(img: np.ndarray, corners: Sequence[tuple[int, int]] | None,
               out_h: int, out_w: int) -> np.ndarray:
    """Perspective-warp the LP region defined by 4 corners to a rectangle.

    corners=None (Ukrainian: pre-cropped plates, no polygon) falls back to a
    plain resize; INTER_CUBIC matches the Ukrainian notebook's HR-crop resize.
    """
    if corners is None or len(corners) != 4:
        return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    src = np.float32(corners)
    dst = np.float32([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (out_w, out_h))


def degrade_hr_to_lr(hr_image: np.ndarray, lr_h: int, lr_w: int) -> np.ndarray:
    """Default HR -> LR pipeline: light Gaussian blur then bilinear downsample."""
    blurred = cv2.GaussianBlur(hr_image, (5, 5), 1.5)
    return cv2.resize(blurred, (lr_w, lr_h))


def process_one_frame(image_path: str, corners, hr_h: int, hr_w: int,
                      lr_h: int, lr_w: int, augment: bool = False, aug=LR_AUG
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """Load -> perspective crop (HR) -> degrade -> LR.

    Returns:
        lr_tensor: [3, lr_h, lr_w] normalized to [-1, 1].
        hr_tensor: [3, lr_h, lr_w] normalized to [-1, 1] (HR resampled to LR grid
                   so it shares the same spatial grid before STN - paper Sec 3.2).
    """
    img = cv2.imread(image_path)
    if img is None:
        lr_img = np.zeros((lr_h, lr_w, 3), dtype=np.uint8)
        hr_img = np.zeros((lr_h, lr_w, 3), dtype=np.uint8)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hr_crop = crop_plate(img, corners, hr_h, hr_w)
        lr_img = degrade_hr_to_lr(hr_crop, lr_h, lr_w)
        hr_img = cv2.resize(hr_crop, (lr_w, lr_h))
    if augment and aug is not None:
        lr_img = aug(image=lr_img)["image"]
    lr_tensor = NORMALIZE(image=lr_img)["image"]
    hr_tensor = NORMALIZE(image=hr_img)["image"]
    return lr_tensor, hr_tensor


# --- Dataset ----------------------------------------------------------------

class LPCropDataset(Dataset):
    """Single-frame LP crop dataset returning (lr, hr, label, length).

    `samples` is a list of dicts with keys {image_path, plate, corners}.
    Augmentation is applied only when `augment=True` and only to the LR image.
    """

    def __init__(self, samples, char2idx: dict[str, int], hr_h: int, hr_w: int,
                 img_h: int, img_w: int, augment: bool = False,
                 max_label_len: int = 10, train_aug: str = "photometric") -> None:
        self.samples = samples
        self.char2idx = char2idx
        self.hr_h = hr_h
        self.hr_w = hr_w
        self.img_h = img_h
        self.img_w = img_w
        self.augment = augment
        self.max_label_len = max_label_len
        # LR aug pipeline selected per dataset (only applied when augment=True).
        self.lr_aug = resolve_lr_aug(train_aug)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        lr, hr = process_one_frame(
            s["image_path"], s.get("corners"),
            self.hr_h, self.hr_w, self.img_h, self.img_w, self.augment,
            aug=self.lr_aug,
        )
        encoded = [self.char2idx.get(c, 0) for c in s["plate"].upper()]
        length = len(encoded)
        padded = encoded + [0] * (self.max_label_len - length)
        label = torch.tensor(padded[: self.max_label_len], dtype=torch.long)
        return lr, hr, label, length
