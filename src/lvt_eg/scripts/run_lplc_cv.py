"""Drive the LPLC 5-fold × 2 cross-validation protocol (paper Table IV).

Mirrors the notebook fold loop: train one run per fold file
(scen0/fold_0_1.json .. fold_4_2.json), evaluate each on its own test split, then
report mean ± sample (n-1) std Plate RR over all folds, matching the paper
aggregation recorded in results/provenance/.

Usage:
    python -m lvt_eg.scripts.run_lplc_cv --config configs/4-lvt_eg_lplc.yaml
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import yaml

from lvt_eg.config import get_dataset_dir, get_run_dir
from lvt_eg.data.lplc import list_lplc_folds
from lvt_eg.scripts.evaluate import evaluate_model
from lvt_eg.scripts.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="LPLC k-fold CV driver.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override config train.epochs (e.g. 1 for a smoke run).")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if cfg["dataset"]["key"] != "lplc":
        raise ValueError("run_lplc_cv expects a key=lplc config.")
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs

    cv = cfg["dataset"].get("cv", {})
    repeats = int(cv.get("repeats", 2))
    n_folds = int(cv.get("folds", 5)) * repeats
    folds_dir = get_dataset_dir("lplc") / "folds"
    scenario = cv.get("scenario", "scen0")
    fold_files = list_lplc_folds(folds_dir, repeats=repeats, scenario=scenario)[:n_folds]
    if not fold_files:
        raise FileNotFoundError(
            f"No LPLC fold files under {folds_dir} (flat fold_*.json or "
            f"{scenario}/fold_*.json). Place the {n_folds} official fold JSONs there."
        )

    def _mean_std(xs: list[float]) -> tuple[float, float]:
        return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)

    base_run = cfg["run_name"]
    base_seed = int(cfg["seed"])
    per_fold: list[float] = []
    per_fold_char: list[float] = []
    per_leg: dict[str, list[float]] = {}
    for k, fold_file in enumerate(fold_files):
        cfg["dataset"].setdefault("cv", {})["fold"] = k
        cfg["run_name"] = f"{base_run}_fold{k}"
        # Notebook reseeds each run as set_seed(SEED + run_idx) with run_idx
        # 1-based (1..10), so fold k (0-based) -> seed base_seed + k + 1 (43..52).
        cfg["seed"] = base_seed + k + 1
        print(f"\n=== Fold {k + 1}/{len(fold_files)}: {fold_file.name} ===")
        best = run_training(cfg)
        summary = evaluate_model(cfg, best)
        per_fold.append(summary["plate_rr"])
        per_fold_char.append(summary["char_acc"])
        for name, d in (summary.get("per_legibility") or {}).items():
            per_leg.setdefault(name, []).append(d["plate_rr"])
        print(f"Fold {k}: Plate RR = {summary['plate_rr'] * 100:.2f}% "
              f"Char = {summary['char_acc'] * 100:.2f}%")

    mean, std = _mean_std(per_fold)  # sample (n-1) std
    char_mean, char_std = _mean_std(per_fold_char)
    agg = {
        "n_folds": len(per_fold),
        "plate_rr_mean": mean,
        "plate_rr_std": std,
        "char_acc_mean": char_mean,
        "char_acc_std": char_std,
        "per_fold": per_fold,
    }
    if per_leg:
        agg["per_legibility"] = {
            name: dict(zip(("mean", "std"), _mean_std(v), strict=True))
            for name, v in per_leg.items()
        }
    out = get_run_dir(base_run) / "cv_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(agg, indent=2))
    strata = "".join(
        f" {name.capitalize()} {agg['per_legibility'][name]['mean'] * 100:.2f}"
        for name in ("perfect", "good", "poor") if name in agg.get("per_legibility", {})
    )
    print(f"\n=== LPLC CV: {mean * 100:.2f} ± {std * 100:.2f}% "
          f"over {len(per_fold)} folds | Char {char_mean * 100:.2f}%{strata} ===")


if __name__ == "__main__":
    main()
