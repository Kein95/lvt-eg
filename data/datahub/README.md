# Vendored Dataset Metadata

This directory contains one JSON metadata file per dataset used in the paper. The files are intentionally vendored so reviewers can inspect source URLs, licenses, dataset statistics, and BibTeX entries without network access.

Raw datasets are not included.

| # | File | Dataset | Country | Size | License |
| - | --- | --- | --- | --- | --- |
| 1 | `1-rodosol-alpr.json` | RodoSol-ALPR | Brazil | 20K | Academic Non-Commercial |
| 2 | `2-ccpd-2019.json` | CCPD 2019 | China | 300K+ | MIT |
| 3 | `3-ukrainian-lp.json` | Ukrainian LP | Ukraine | 10K synthetic | CC BY 4.0 |
| 4 | `4-lplc.json` | LPLC | Brazil | 12,687 plates | Academic Non-Commercial |
| 5 | `5-lp-2025.json` | LP-2025 | Taiwan | 67K plates | Academic License |

## Schema

Each file follows this shape:

```json
{
  "name": "...",
  "country": "...",
  "size": "...",
  "year": 2022,
  "access": "...",
  "source_url": "https://...",
  "paper_url": "https://...",
  "license_type": "...",
  "bibtex": "@inproceedings{...}",
  "classes": 36,
  "resolution": "...",
  "annotation": "...",
  "description": "..."
}
```

## Usage

```python
import json
from pathlib import Path

meta = json.loads(Path("data/datahub/1-rodosol-alpr.json").read_text())
print(meta["source_url"])
print(meta["license_type"])
print(meta["bibtex"])
```

The training code reads these files only for metadata and download guidance. Dataset images must be obtained from each original source under its license terms.
