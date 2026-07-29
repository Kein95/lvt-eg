"""LP-2025 (Taiwan) dataset parser - notebook Cell 4 (LP-2025).

Layout:
    <root>/<split>/images/<id>.jpg
    <root>/<split>/labels_gd/<id>.txt

Each label .txt may contain multiple plates per image (paper notes 1.87
plates/image avg). Each line:
    PLATE x1 y1 x2 y2 x3 y3 x4 y4
where PLATE is alphanumeric/dash, length 2-8, or '_' for unreadable
(skipped per paper Sec 4.3).
"""

from __future__ import annotations

import re
from pathlib import Path

_PLATE_RE = re.compile(r"^[0-9A-Z\-]{2,8}$")


def parse_lp2025_split(dataset_dir: str | Path, split_name: str) -> list[dict]:
    """Return [{image_path, plate, corners}, ...] for one LP-2025 split."""
    img_dir = Path(dataset_dir) / split_name / "images"
    lbl_dir = Path(dataset_dir) / split_name / "labels_gd"
    samples: list[dict] = []
    if not img_dir.exists() or not lbl_dir.exists():
        return samples

    for lbl_file in sorted(lbl_dir.glob("*.txt")):
        img_path = img_dir / (lbl_file.stem + ".jpg")
        if not img_path.exists():
            continue
        with open(lbl_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                plate = parts[0].upper()
                if plate == "_" or not _PLATE_RE.match(plate):
                    continue
                try:
                    coords = [int(x) for x in parts[1:9]]
                except ValueError:
                    continue
                corners = [
                    (coords[0], coords[1]),
                    (coords[2], coords[3]),
                    (coords[4], coords[5]),
                    (coords[6], coords[7]),
                ]
                samples.append({
                    "image_path": str(img_path),
                    "plate": plate,
                    "corners": corners,
                })
    return samples
