"""Ukrainian LP dataset parser - notebook Cell 4 (Ukrainian LP).

Two on-disk layouts are supported:
  1) Pre-split:  <root>/{train,valid,test}/images/*.png
  2) Flat:       <root>/images/*.png  (auto-split by ratio)

Label is the filename stem (e.g. `AA0000IH.png` -> `AA0000IH`).
Paper Table I: 8K synth train, 1K real val, 1K real test.

Ukrainian plates have 24 character classes. Filenames must contain only
characters from `chars` to be admitted (others are silently dropped).
"""

from __future__ import annotations

import random
from pathlib import Path


def _scan_split(img_dir: Path, chars: str) -> list[dict]:
    samples: list[dict] = []
    for img_path in sorted(img_dir.glob("*.png")):
        label = img_path.stem
        if label and all(c in chars for c in label):
            # No corners for Ukrainian - perspective crop falls back to a
            # plain resize inside `crop_plate()`.
            samples.append({
                "image_path": str(img_path),
                "plate": label,
                "corners": None,
            })
    return samples


def parse_ukrainian_dataset(dataset_dir: str | Path, chars: str,
                            train_ratio: float = 0.8, val_ratio: float = 0.1,
                            seed: int = 42
                            ) -> tuple[list[dict], list[dict], list[dict]]:
    """Parse the Ukrainian LP dataset and return (train, val, test) lists."""
    root = Path(dataset_dir)

    if (root / "train" / "images").exists():
        train = _scan_split(root / "train" / "images", chars)
        val = _scan_split(root / "valid" / "images", chars)
        test = _scan_split(root / "test" / "images", chars)
        return train, val, test

    img_dir = root / "images"
    if not img_dir.exists():
        return [], [], []

    all_samples = _scan_split(img_dir, chars)
    rng = random.Random(seed)
    indices = list(range(len(all_samples)))
    rng.shuffle(indices)
    n_train = int(len(all_samples) * train_ratio)
    n_val = int(len(all_samples) * val_ratio)

    train = [all_samples[i] for i in indices[:n_train]]
    val = [all_samples[i] for i in indices[n_train:n_train + n_val]]
    test = [all_samples[i] for i in indices[n_train + n_val:]]
    return train, val, test
