"""CCPD 2019 dataset parser - paper Sec 4.1 (v2019 protocol).

CCPD encodes the plate label and 4 corners directly in the filename:

    <area>-<tilt>-<bbox>-<corners>-<plate_idx>-<brightness>-<blur>.jpg

where:
    corners:    "x1&y1_x2&y2_x3&y3_x4&y4" (4 polygon points around the plate)
    plate_idx:  "_"-separated indices into the CCPD alphabet (province + 6 chars)

The dataset is partitioned into 8 official subdirectories evaluated separately
in paper Table III: ccpd_base, ccpd_blur, ccpd_challenge, ccpd_db, ccpd_fn,
ccpd_rotate, ccpd_tilt, ccpd_weather.
"""

from __future__ import annotations

from pathlib import Path

# CCPD alphabet (paper Sec 3.5: 68 classes = 67 unique chars + 1 CTC blank).
# Authoritative source: paper Sec 4.1.
# 33 provinces include the standard 31 admin codes + 警 (police) + 学 (student).
CCPD_PROVINCES = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
    "新", "警", "学",
]  # 33
CCPD_ALPHABETS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z",
]  # 24 (no I, no O - to avoid plate-text ambiguity)
CCPD_ADS = CCPD_ALPHABETS + [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
]  # 34 = 24 letters + 10 digits

# Dedup preserves order: 33 provinces + 24 alphabets (already in ADS) + 10 digits = 67 unique.
ALL_CHARS = list(dict.fromkeys(CCPD_PROVINCES + CCPD_ALPHABETS + CCPD_ADS))  # 67
NUM_CLASSES = len(ALL_CHARS) + 1  # 68 (CTC blank at index 0)

CCPD_SUBSETS = [
    "ccpd_base", "ccpd_blur", "ccpd_challenge", "ccpd_db",
    "ccpd_fn", "ccpd_rotate", "ccpd_tilt", "ccpd_weather",
]


def _decode_plate(idx_field: str) -> str | None:
    """Decode the plate-index field of a CCPD filename.

    Filename layout: `<province_idx>_<alpha_idx>_<ads_idx>_<ads_idx>_<ads_idx>_<ads_idx>_<ads_idx>`
    Position 0 indexes into PROVINCES (33), position 1 into ALPHABETS (24),
    positions 2..6 into ADS (34).
    """
    parts = idx_field.split("_")
    if len(parts) < 7:
        return None
    try:
        province = CCPD_PROVINCES[int(parts[0])]
        alpha = CCPD_ALPHABETS[int(parts[1])]
        ads = "".join(CCPD_ADS[int(p)] for p in parts[2:7])
        return province + alpha + ads
    except (ValueError, IndexError):
        return None


def _decode_corners(field: str):
    """Parse 'x1&y1_x2&y2_x3&y3_x4&y4'. Returns 4 (x, y) tuples in [TL,TR,BR,BL]
    order (what crop_plate expects), or None.

    CCPD stores corners in [BR, BL, TL, TR] order; the notebook reorders
    vertices[2,3,0,1] before the perspective warp, so a plate crops upright.
    We apply the same reorder here (used by both the split-manifest reader in
    train.py and parse_ccpd_directory) so every CCPD crop is upright, not
    rotated 180 degrees.
    """
    pts = []
    for pair in field.split("_"):
        try:
            x, y = pair.split("&")
            pts.append((int(x), int(y)))
        except ValueError:
            return None
    if len(pts) != 4:
        return None
    return [pts[2], pts[3], pts[0], pts[1]]  # [BR,BL,TL,TR] -> [TL,TR,BR,BL]


def parse_ccpd_directory(dir_path: str | Path) -> list[dict]:
    """Parse all .jpg files in a CCPD subset directory."""
    dir_path = Path(dir_path)
    samples: list[dict] = []
    for img in sorted(dir_path.glob("*.jpg")):
        fields = img.stem.split("-")
        if len(fields) < 5:
            continue
        corners = _decode_corners(fields[3])
        plate = _decode_plate(fields[4])
        if corners and plate:
            samples.append({"image_path": str(img), "plate": plate, "corners": corners})
    return samples


def parse_ccpd_subsets(dataset_dir: str | Path,
                       subsets: list[str] = None) -> dict[str, list[dict]]:
    """Return {subset_name: samples} for each requested CCPD subset directory."""
    dataset_dir = Path(dataset_dir)
    out: dict[str, list[dict]] = {}
    for sub in subsets or CCPD_SUBSETS:
        sub_dir = dataset_dir / sub
        out[sub] = parse_ccpd_directory(sub_dir) if sub_dir.exists() else []
    return out
