# Datasets

This directory holds local dataset files and vendored metadata for the five datasets used by LVT-EG. The repository includes metadata only; it does not redistribute raw images.

## Layout

After downloading each dataset from its original source, organize the directory as:

```text
data/
├── datahub/              # Vendored JSON metadata, already in repo
├── rodosol-alpr/         # RodoSol-ALPR: split.txt + images/
├── ccpd-2019/            # CCPD 2019: ccpd_base/, ccpd_blur/, ... + splits/{train,val,test}.txt
├── ukrainian-lp/         # Ukrainian LP (synthetic 10k): train/, valid/, test/  (each images/ + labels/)
├── lplc/                 # LPLC: images/<id>.jpg (full scenes) + folds/scen0/fold_<n>_<iter>.json + annotations_formatted.json
└── lp-2025/              # LP-2025: train/, val/, test/
```

The base data path defaults to `data/` and can be overridden with `LVT_DATA_DIR`.

## Required split / fold manifests (not bundled)

Two datasets need pre-built split files that this repo does not redistribute.
Training raises a clear error if they are missing.

- **CCPD 2019**: the v2019 (Laroca-2023 "fair") split lists ship inside the
  CCPD2019 archive: after download + extract they are at `<root>/splits/`.
  Place that folder so the lists live at `ccpd-2019/splits/{train,val,test}.txt`
  (one image path per line, relative to the CCPD root, forward slashes, e.g.
  `ccpd_base/<name>.jpg`). Counts: train 100,000 / val 99,996 / test 141,982.
  `test.txt` holds the six hard subsets (blur, challenge, db, fn, rotate, tilt);
  `ccpd_weather/` (9,999) is scanned separately for the per-subset report,
  giving the paper's 151,981 test total. Training raises a clear error (rather
  than silently loading zero samples) if the lists are missing or resolve to no
  images.
- **LPLC**: the official download already ships everything needed: the
  5-fold × 2 fold JSONs at `lplc/folds/scen0/fold_<n>_<iter>.json` (10 files:
  `fold_0_1.json` .. `fold_4_2.json`), the full-scene images at
  `lplc/images/<image_id>.jpg`, and `lplc/annotations_formatted.json`. A flat
  `lplc/folds/fold_*.json` layout is also accepted. Folds are the fixed
  official partitions, not seed-generated. The loader reads each plate's
  legibility (`readable`), OCR, and 4 corners from the annotations and
  perspective-warps the plate out of the full scene; the fold file only
  enumerates which plates belong to each split.

> The Ukrainian LP set (`ukrainian-lp/`) is the publicly released 10k bundle
> used in the paper (Table I): an 8,000-image diffusion-generated synthetic
> train split plus the 1,000-image val and test splits. The parser uses these
> on-disk `train/valid/test` splits directly.

## Metadata

Full source URLs, licenses, paper URLs, and BibTeX entries are stored in [`datahub/`](datahub/).

| Key | Dataset | JSON | Country | Size | License |
| --- | --- | --- | --- | --- | --- |
| `rodosol` | RodoSol-ALPR | `datahub/1-rodosol-alpr.json` | Brazil | 20K | Academic Non-Commercial |
| `ccpd2019` | CCPD 2019 | `datahub/2-ccpd-2019.json` | China | 300K+ | MIT |
| `ukrainian` | Ukrainian LP | `datahub/3-ukrainian-lp.json` | Ukraine | 10K synthetic | CC BY 4.0 |
| `lplc` | LPLC | `datahub/4-lplc.json` | Brazil | 12,687 plates | Academic Non-Commercial |
| `lp2025` | LP-2025 | `datahub/5-lp-2025.json` | Taiwan | 67K plates | Academic License |

List all metadata from the command line:

```bash
python -m lvt_eg.scripts.download_data --list
```

Print one dataset's source, license, target directory, and BibTeX:

```bash
python -m lvt_eg.scripts.download_data --dataset rodosol
```

The script prints metadata and target paths only. Downloading, accepting licenses, extracting archives, and placing raw files are manual steps.

## Cross-domain protocol: RodoSol → LPLC zero-shot

`run_zeroshot.py` trains one model on the RodoSol source split, then scores that
single checkpoint on every LPLC `scen0` fold-**test** split (5 folds × 2 iters =
10 partitions) and reports mean ± sample std Plate RR (+ Perfect/Good/Poor
strata). No extra manifest is needed; it reuses the same `lplc/folds/scen0/`
fold JSONs, `lplc/images/`, and `lplc/annotations_formatted.json` as same-domain
LPLC. The reported ± band is cross-partition variance of one checkpoint, not
seed variance.

```bash
# 8K variant (paper Table IV cross-domain): 58.75 ± 0.61%
python -m lvt_eg.scripts.run_zeroshot --config configs/6-lvt_eg_rodosol2lplc_zeroshot.yaml
# 12K variant (paper Table IV cross-domain best): 69.51 ± 0.31%
python -m lvt_eg.scripts.run_zeroshot --config configs/7-lvt_eg_rodosol2lplc_zeroshot_12k.yaml
```

## Citation

Always cite the original dataset paper when reporting results. BibTeX entries are embedded in each metadata JSON under the `bibtex` key.
