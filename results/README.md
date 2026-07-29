# Results

Training and evaluation outputs are written under `results/<run_name>/`. The base output path defaults to `results/` and can be overridden with `LVT_OUTPUT_DIR`.

## Output Structure

```text
results/
└── <run_name>/
    ├── config.yaml              # Frozen copy of the training config
    ├── best.pth                 # Best checkpoint, with EMA state when available
    ├── last.pth                 # Latest checkpoint
    ├── history.json             # Per-epoch losses and metrics
    └── eval/
        ├── test_predictions.tsv
        └── test_summary.json
```

`best.pth` is the default checkpoint for `lvt_eg.scripts.evaluate` and can also be passed to `lvt_eg.scripts.run_zeroshot --checkpoint`.

Large checkpoint and log artifacts should stay out of git unless explicitly needed for archival or review.

## Reported Paper Numbers

| # | Run | Plate RR |
| - | --- | --- |
| 1 | `lvt_eg_rodosol` | 97.94 ± 0.13% (4 runs, Table II) |
| 2 | `lvt_eg_ccpd` | 93.53 ± 0.21% (3 runs, Table III, Hard-all incl. Weather) |
| 3 | `lvt_eg_ukrainian` | 99.45 ± 0.07% (2 runs) |
| 4 | `lvt_eg_lplc` | 85.30 ± 2.17% (Table IV same-domain) |
| 5 | `lvt_eg_lp2025` | 84.84 ± 0.23% (2 runs, Table V) |
| 6 | `lvt_eg_rodosol2lplc_zeroshot_8k` | 58.75 ± 0.61% (Table IV cross-domain) |
| 7 | `lvt_eg_rodosol2lplc_zeroshot_12k` | **69.51 ± 0.31%** (Table IV cross-domain best) |

Per-run, per-fold, and CCPD per-subset numbers behind every aggregate above are
in [`provenance/`](provenance/) (see [`provenance/PROVENANCE.md`](provenance/PROVENANCE.md)).
Each aggregate was recomputed from those rows and matches the paper.
