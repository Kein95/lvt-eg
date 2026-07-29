"""LPLC dataset parser + 5-fold stratified CV - paper Sec 4.1, Table IV.

LPLC ships full-scene images under
    images/<image_id>.jpg
and `annotations_formatted.json` (keyed by "<image_id>.jpg") giving, per plate,
the OCR string, legibility (`readable`: 0=illegible..3=perfect), validity /
occlusion flags, and the 4-corner polygon in full-scene pixel coords. The plate
is perspective-warped out of the full scene at load time (LPCropDataset).

Each fold ships as a JSON file with split keys 'train' / 'val' / 'test'; the
values group plate *filenames* of the form
    LPRD_Dataset/all_lps/<bucket>/<image_id>.jpg_<lp_idx>_<ocr>.jpg
by a nested bucket key. Those filenames only ENUMERATE which (image_id, lp_idx)
belong to each split - the loader parses the id/idx out and joins with the
annotation for the authoritative label. We use only RAW filenames (no
`_c_<scale>` author-pre-augmented variants) so each plate appears in exactly one
split per fold. Protocol: 5-fold CV repeated twice on the readable
subset (~11,083 of the 12,687 annotated plates; paper Sec 4.1).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Filename: <image_id>.jpg_<lp_idx>_<ocr>.jpg  (raw only - no _c_X)
_FNAME_RE = re.compile(r"([0-9a-f-]+)\.jpg_(\d+)_([A-Za-z0-9]*)\.jpg$")


def parse_lp_filename(path: str | Path):
    """Returns (image_id, lp_idx, ocr_in_filename) or None on no match."""
    m = _FNAME_RE.search(Path(path).name)
    return (m.group(1), int(m.group(2)), m.group(3)) if m else None


def _xy_to_corners(xy):
    """Convert flat list of 8 ints to [(x,y) x 4]."""
    if xy is None or len(xy) != 8:
        return None
    return [(int(xy[i]), int(xy[i + 1])) for i in range(0, 8, 2)]


def parse_lplc_fold(fold_json: str | Path, annotations_json: str | Path | dict,
                    images_dir: str | Path, target_split: str,
                    min_legibility: int = 1) -> list[dict]:
    """Parse one fold file and join with annotations - mirrors the notebook loader.

    The fold JSON groups plate paths under a nested legibility key, but that
    key is only the detector's bucketing; the *authoritative* legibility, OCR
    label, and corners all come from ``annotations_formatted.json`` (keyed by
    ``"<image_id>.jpg"``). Images are the FULL scenes under
    ``<images_dir>/<image_id>.jpg`` - ``LPCropDataset`` perspective-warps the
    plate out using the annotation corners, which are full-scene pixel coords.

    Filters, applied in the notebook order (paper Sec 4.1):
        * raw filenames only (``parse_lp_filename`` rejects ``_c_<scale>`` variants)
        * dedup by ``(image_id, lp_idx)``
        * annotation ``readable`` >= ``min_legibility`` (1=poor..3=perfect; 0 dropped)
        * annotation ``valid`` is True and ``occluded`` is False
        * OCR length == 7
        * filename OCR (when present) must match the annotation OCR

    Args:
        fold_json:        path to fold_<n>_<iter>.json (split keys train/val/test).
        annotations_json: path to annotations_formatted.json, OR the already
                          loaded dict (so a fold loop can load it once).
        images_dir:       directory of full-scene images (``<root>/images``).
        target_split:     "train" | "val" | "test".
        min_legibility:   keep plates with ``readable`` level >= this.
    """
    fold = json.loads(Path(fold_json).read_text())
    if target_split not in fold:
        raise KeyError(f"split '{target_split}' missing in {Path(fold_json).name}")
    annotations = (annotations_json if isinstance(annotations_json, dict)
                   else json.loads(Path(annotations_json).read_text()))
    images_dir = Path(images_dir)

    seen: set[tuple[str, int]] = set()
    samples: list[dict] = []
    # fold[split] is {leg_level_str: [paths...]}; the bucket key is ignored - the
    # per-plate legibility comes from the annotation's `readable` field below.
    for paths in fold[target_split].values():
        for p in paths:
            parsed = parse_lp_filename(p)
            if parsed is None:  # _c_<scale> variant or malformed -> skip
                continue
            image_id, lp_idx, ocr_fn = parsed
            key = (image_id, lp_idx)
            if key in seen:  # dedup: each raw plate appears once per fold
                continue
            seen.add(key)
            ann_group = annotations.get(f"{image_id}.jpg")
            if ann_group is None:
                continue
            anns = ann_group.get("anns", [])
            if lp_idx >= len(anns):
                continue
            ann = anns[lp_idx]
            try:
                readable = int(ann.get("readable", 0))
            except (ValueError, TypeError):
                readable = 0
            if readable < min_legibility:
                continue
            if not ann.get("valid", False):
                continue
            if ann.get("occluded", False):
                continue
            ocr = (ann.get("ocr") or "").upper().strip()
            if len(ocr) != 7:
                continue
            # Filename OCR (when non-empty) must match the annotation OCR,
            # else lp_idx -> anns[idx] is mis-aligned.
            if ocr_fn and ocr_fn.upper() != ocr:
                continue
            corners = _xy_to_corners(ann.get("xy"))
            if corners is None:
                continue
            img_path = images_dir / f"{image_id}.jpg"
            if not img_path.exists():
                continue
            samples.append({
                "image_path": str(img_path),
                "plate": ocr,
                "corners": corners,
                "legibility": readable,
            })
    return samples


def list_lplc_folds(folds_dir: str | Path, repeats: int = 2,
                    scenario: str = "scen0") -> list[Path]:
    """Return up to 5*repeats fold files, deterministically sorted.

    Accepts both the flat layout (folds/fold_*.json) and the official nested
    layout shipped with the source data (folds/<scenario>/fold_<n>_<iter>.json,
    e.g. folds/scen0/fold_0_1.json .. fold_4_2.json). The k-th entry
    corresponds to cv.fold=k as consumed by train._build_datasets; run_lplc_cv
    iterates these for the full protocol.
    """
    folds_dir = Path(folds_dir)
    files = sorted(folds_dir.glob("fold_*.json"))
    if not files and scenario:
        files = sorted((folds_dir / scenario).glob("fold_*.json"))
    return files[: 5 * repeats]
