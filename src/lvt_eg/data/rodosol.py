"""RodoSol-ALPR dataset parser - notebook Cell 4 (RodoSol).

Format: each image directory has a sibling .txt with lines:
    plate: ABC1234
    corners: x1,y1 x2,y2 x3,y3 x4,y4

Top-level split.txt assigns each image to {training, validation, testing}.
"""

from __future__ import annotations

from pathlib import Path


def parse_rodosol_split(dataset_dir: str | Path, target_split: str) -> list[dict]:
    """Parse RodoSol split.txt and return [{image_path, plate, corners}, ...].

    Args:
        dataset_dir: directory containing `split.txt` and the image tree.
        target_split: one of "training", "validation", "testing".
    """
    dataset_dir = Path(dataset_dir)
    split_file = dataset_dir / "split.txt"
    samples: list[dict] = []
    with open(split_file) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) != 2 or parts[1] != target_split:
                continue
            img_path = dataset_dir / parts[0].lstrip("./")
            if not img_path.exists():
                continue
            txt_path = img_path.with_suffix(".txt")
            plate, corners = None, None
            if txt_path.exists():
                with open(txt_path) as af:
                    for aline in af:
                        aline = aline.strip()
                        if aline.startswith("plate:"):
                            plate = aline.split(":", 1)[1].strip().upper()
                        elif aline.startswith("corners:"):
                            cs = aline.split(":", 1)[1].strip()
                            pts = []
                            for pt in cs.split():
                                x, y = pt.split(",")
                                pts.append((int(x), int(y)))
                            if len(pts) == 4:
                                corners = pts
            if plate and corners:
                samples.append({"image_path": str(img_path), "plate": plate, "corners": corners})
    return samples


def parse_rodosol_merged(dataset_dir: str | Path, splits: list[str]) -> list[dict]:
    """Merge multiple RodoSol splits into a single sample list.

    Used by the zeroshot-12K config (paper Table IV: "Rodo 12K" =
    training + validation merged into one 12K training pool).
    """
    out: list[dict] = []
    for s in splits:
        out.extend(parse_rodosol_split(dataset_dir, s))
    return out
