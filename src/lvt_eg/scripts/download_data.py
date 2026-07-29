"""Read vendored dataset metadata and print download instructions.

This script never calls a network service for metadata - it reads local
JSON under `data/datahub/` (vendored from the LPR Datasets Hub for
offline-self-contained double-blind review). For license-restricted
datasets it prints the source URL and license terms; the user must
manually accept the license and place the data under the correct path.

Usage:
    python -m lvt_eg.scripts.download_data --dataset rodosol
    python -m lvt_eg.scripts.download_data --list
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lvt_eg.config import get_dataset_dir

# Repo-root-relative path to the vendored hub
REPO_ROOT = Path(__file__).resolve().parents[3]
DATAHUB_DIR = REPO_ROOT / "data" / "datahub"

# Map config keys -> JSON file names. Ordered as in paper Table I.
KEY_TO_JSON = {
    "rodosol":   "1-rodosol-alpr.json",
    "ccpd2019":  "2-ccpd-2019.json",
    "ukrainian": "3-ukrainian-lp.json",
    "lplc":      "4-lplc.json",
    "lp2025":    "5-lp-2025.json",
}


def load_metadata(key: str) -> dict:
    if key not in KEY_TO_JSON:
        raise ValueError(f"Unknown dataset key '{key}'. Available: {list(KEY_TO_JSON)}")
    return json.loads((DATAHUB_DIR / KEY_TO_JSON[key]).read_text())


def print_metadata(key: str) -> None:
    meta = load_metadata(key)
    target_dir = get_dataset_dir(key)
    print(f"Dataset:      {meta['name']}")
    print(f"Country:      {meta['country']}")
    print(f"Size:         {meta['size']}")
    print(f"Year:         {meta.get('year', 'n/a')}")
    print(f"License:      {meta.get('license_type', 'n/a')}")
    print(f"Access:       {meta.get('access', 'n/a')}")
    print(f"Source URL:   {meta['source_url']}")
    print(f"Paper URL:    {meta.get('paper_url', 'n/a')}")
    print(f"Place under:  {target_dir}")
    print("\nBibTeX:")
    print(meta.get("bibtex", "(missing)"))
    if meta.get("access") == "license":
        print("\nThis dataset is license-restricted. Please request access from the source URL")
        print("and place the extracted data under the path above.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(KEY_TO_JSON))
    parser.add_argument("--list", action="store_true",
                        help="List all vendored datasets and exit.")
    args = parser.parse_args()

    if args.list or not args.dataset:
        for key in KEY_TO_JSON:
            meta = load_metadata(key)
            print(f"  {key:<10s} {meta['name']:<20s}  {meta['country']:<12s}  "
                  f"{meta['size']:<14s}  {meta.get('license_type', 'n/a')}")
        return

    print_metadata(args.dataset)


if __name__ == "__main__":
    main()
