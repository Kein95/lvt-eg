# Training Configs

This directory contains one YAML file per dataset or evaluation setting. Pass a file with `--config` to `lvt_eg.scripts.train` or `lvt_eg.scripts.run_zeroshot`.

Configs are ordered as in the paper dataset table, with RodoSol → LPLC zero-shot settings at the end.

| # | Config | Dataset / setting | Run name |
| - | --- | --- | --- |
| 1 | `1-lvt_eg_rodosol.yaml` | RodoSol-ALPR | `lvt_eg_rodosol` |
| 2 | `2-lvt_eg_ccpd.yaml` | CCPD 2019 v2019 Hard-all | `lvt_eg_ccpd` |
| 3 | `3-lvt_eg_ukrainian.yaml` | Ukrainian LP | `lvt_eg_ukrainian` |
| 4 | `4-lvt_eg_lplc.yaml` | LPLC, 5-fold × 2 | `lvt_eg_lplc` |
| 5 | `5-lvt_eg_lp2025.yaml` | LP-2025 | `lvt_eg_lp2025` |
| 6 | `6-lvt_eg_rodosol2lplc_zeroshot.yaml` | Rodo 8K → LPLC | `lvt_eg_rodosol2lplc_zeroshot_8k` |
| 7 | `7-lvt_eg_rodosol2lplc_zeroshot_12k.yaml` | Rodo 12K → LPLC | `lvt_eg_rodosol2lplc_zeroshot_12k` |

## Common Recipe

| Setting | RodoSol | CCPD | Ukrainian | LPLC | LP-2025 | Zero-shot |
| --- | --- | --- | --- | --- | --- | --- |
| Input size | 48×192 | 32×128 | 48×192 | 48×192 | 48×192 | 48×192 |
| HR reference | 64×256 | 64×256 | 64×256 | 64×256 | 64×256 | 64×256 |
| Batch size | 64 | 512 | 64 | 256 | 200 | 256 |
| Epochs | 50 | 30 | 30 | 30 | 50 | 30 |
| Max LR | 3e-4 | 5e-4 | 3e-4 | 3e-4 | 3e-4 | 3e-4 |
| Weight decay | 0.02 | 1e-4 | 0.02 | 0.02 | 0.02 | 0.02 |
| EMA decay | 0.99 | 0.999 | 0.99 | 0.99 | 0.99 | 0.99 |
| Grad clip | 5.0 | 1.0 | 5.0 | 5.0 | 5.0 | 5.0 |
| Drop path | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 |
| Edge L1 weight | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |

All main configs (1–7) use AdamW, OneCycleLR, AMP, synthetic degradation policy `v2`, and Plate RR as the main evaluation metric; the ablation configs vary the synthetic policy (see the Ablations table below).

## Schema

Regular training configs use the keys below. Keys marked *(declarative)* document
paper intent but are fixed in code, not read from the config.

```yaml
run_name: str
seed: int

dataset:
  key: rodosol | ccpd2019 | ukrainian | lplc | lp2025
  chars: str | null          # null for ccpd2019 → filled from data module
  num_classes: int
  img_h: int
  img_w: int
  hr_h: int
  hr_w: int
  synth_policy: v1 | v2 | none  # none = no synthetic degradation (ablation rows 1-3)
  train_aug: photometric | geometric  # LR aug on the real branch; default photometric
                             # (RodoSol/LPLC/LP-2025). CCPD/Ukrainian use geometric
                             # (Affine/Perspective/Rotate/ChannelShuffle/CoarseDropout).
  synth_aug: photometric | geometric | none  # LR aug on synthetic views; default
                             # photometric (RodoSol/LPLC/LP-2025). CCPD=geometric,
                             # Ukrainian=none (degradation only).
  synth_degrade_at: lr | hr  # where the level degradation is applied; default lr
                             # (degrade at the model grid). Ukrainian=hr (degrade at
                             # the 64x256 HR crop, then resize down).
  splits: {train, val, test} # ccpd2019 only: manifest paths under the data root
  subsets: [str]             # ccpd2019 only: subset dirs for per-subset report
  cv:                        # lplc only
    folds: int               # 5
    repeats: int             # 2  → 5*2 = 10 fold files
    fold: int                # which fold to train (set by run_lplc_cv)
    scenario: str            # fold subdir, default scen0 (source ships folds/scen0/fold_<n>_<iter>.json)

model:
  arch: lvt_eg               # (declarative)
  use_stn: bool
  drop_path: float
  use_pos_embed: bool        # default true; CCPD uses false
  legacy_conv_mixer: bool    # default false; CCPD true (grouped-conv, ~20.12M)
  use_edge: bool             # default true; false = no-edge ablation (3-ch enc, Table VI row 7)
  stn_order: stn_first | rect_after  # default stn_first; rect_after = Table VI "Rect. after" rows

loss:
  ctc_weight: float          # (declarative: fixed to 1.0)
  edge_l1_weight: float      # paper Eq. (2): lambda = 0.1

train:
  epochs: int
  batch_size: int
  optimizer: adamw           # (declarative)
  weight_decay: float
  scheduler: onecycle        # (declarative)
  max_lr: float
  pct_start: float
  amp: bool                  # (declarative: AMP always on)
  ema_decay: float
  grad_clip: float
  patience: int              # early-stop patience (0 = off)
  ema_warmup_epochs: int     # validate on EMA only after this many epochs
  ema_update_after_warmup: bool  # default false; CCPD true = also gate EMA
                             # accumulation to epoch > ema_warmup_epochs
  early_stop_metric: str     # (declarative: fixed to val Plate RR)

eval:
  metric: plate_rr           # (declarative)
  per_subset_report: bool    # ccpd2019: emit per-subset breakdown (paper Table III)
  per_legibility_report: bool # lplc (declarative)
```

Zero-shot configs replace `dataset:` with `source:` and `target:` blocks for the RodoSol → LPLC protocol.

## Multi-run and cross-validation

Paper numbers are mean ± std over multiple runs. Reproduce them with:

```bash
# Single-split datasets (RodoSol/CCPD/Ukrainian/LP-2025): train N seeds and
# print mean ± sample std (+ per-subset for CCPD). Each seed writes a separate
# results/<run_name>_seed<N>/.
python -m lvt_eg.scripts.run_multiseed --config configs/1-lvt_eg_rodosol.yaml --n-seeds 3
python -m lvt_eg.scripts.run_multiseed --config configs/2-lvt_eg_ccpd.yaml --seeds 42 43 44

# LPLC 5-fold × 2: trains the 10 official fold JSONs (scen0/fold_0_1.json ..
# fold_4_2.json) and prints mean ± std.
python -m lvt_eg.scripts.run_lplc_cv --config configs/4-lvt_eg_lplc.yaml
```

CCPD per-subset Plate RR (paper Table III) is written to the eval summary when
`eval.per_subset_report: true` (already set in `2-lvt_eg_ccpd.yaml`). LPLC
per-legibility Plate RR / Char.Acc (paper Table IV Perfect/Good/Poor) is
written when `eval.per_legibility_report: true` (already set in
`4-lvt_eg_lplc.yaml`) and aggregated across folds by `run_lplc_cv`.

> The reported ± bands for the single-split datasets (RodoSol/CCPD/Ukrainian/
> LP-2025) reflect run-to-run GPU nondeterminism in the original Colab runs:
> every run (lan 1/2/3/4) used the **same seed 42** and relied on
> non-deterministic cuDNN ops for the spread. The release seeds deterministically
> (`cudnn.deterministic=True`), so repeating seed 42 gives zero spread;
> `run_multiseed` therefore varies the seed as a reproducible proxy. The mean
> lands near the paper number; the ± is seed variance, not a byte-exact replay
> of the notebook's fixed-seed nondeterministic spread.

## Ablations (paper Table VI)

`ablations/` holds one config per non-trivial RodoSol ablation row, using the
`model.use_edge`, `model.stn_order`, and `dataset.synth_policy` knobs:

| Row | Config | dp | Synth | Edge | Rect. |
| --- | --- | --- | --- | --- | --- |
| 1 | `ablations/rodosol_abl1_dp01_nosynth_rectafter.yaml` | 0.1 | none | ✓ | after |
| 2 | `ablations/rodosol_abl2_dp02_nosynth_rectafter.yaml` | 0.2 | none | ✓ | after |
| 3 | `ablations/rodosol_abl3_dp02_nosynth_stnfirst.yaml` | 0.2 | none | ✓ | first |
| 4 | `ablations/rodosol_abl4_v1_rectafter.yaml` | 0.2 | v1 | ✓ | after |
| 5 | `ablations/rodosol_abl5_v1_stnfirst.yaml` | 0.2 | v1 | ✓ | first |
| 6 | `ablations/rodosol_abl6_v2_rectafter.yaml` | 0.2 | v2 | ✓ | after |
| 7 | `ablations/rodosol_abl7_v2_noedge_stnfirst.yaml` | 0.2 | v2 | - | first |
| 8 | `1-lvt_eg_rodosol.yaml` (full model) | 0.2 | v2 | ✓ | first |
