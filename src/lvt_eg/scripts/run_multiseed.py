"""Multi-seed mean +/- std driver for the single-split datasets
(RodoSol / CCPD / Ukrainian / LP-2025).

The paper's +/- bands for these datasets come from run-to-run GPU
nondeterminism at a FIXED seed: every notebook (lan 1/2/3/4) uses SEED=42 and
relies on non-deterministic cuDNN ops for the spread. The release seeds
deterministically (cudnn.deterministic=True), so repeating seed 42 would give
zero spread. This driver instead trains N DISTINCT seeds as a reproducible
proxy: the mean lands near the paper number, and the +/- is seed variance
(NOT a byte-exact replay of the notebook's fixed-seed nondeterministic spread).

For LPLC use run_lplc_cv (fold cross-validation), not this driver.

Usage:
    # 3 seeds starting from the config seed (42, 43, 44)
    python -m lvt_eg.scripts.run_multiseed --config configs/1-lvt_eg_rodosol.yaml --n-seeds 3
    # explicit seed list
    python -m lvt_eg.scripts.run_multiseed --config configs/2-lvt_eg_ccpd.yaml --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import yaml

from lvt_eg.config import get_run_dir
from lvt_eg.scripts.evaluate import evaluate_model
from lvt_eg.scripts.train import run_training


def _mean_std(xs: list[float]) -> tuple[float, float]:
    """Mean and sample (n-1) std; std=0 for a single value."""
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-seed mean/std driver for single-split datasets.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Explicit seed list; overrides --n-seeds.")
    parser.add_argument("--n-seeds", type=int, default=3,
                        help="Number of seeds from the config seed (default 3).")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override config train.epochs (e.g. 1 for a smoke run).")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if cfg["dataset"]["key"] == "lplc":
        raise ValueError("Use run_lplc_cv for LPLC fold CV, not run_multiseed.")
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs

    base_run = cfg["run_name"]
    base_seed = int(cfg["seed"])
    seeds = args.seeds if args.seeds else [base_seed + i for i in range(args.n_seeds)]

    per_seed_rr: list[float] = []
    per_seed_char: list[float] = []
    per_subset: dict[str, list[float]] = {}
    rows: list[dict] = []
    for seed in seeds:
        cfg["seed"] = seed
        cfg["run_name"] = f"{base_run}_seed{seed}"
        print(f"\n=== seed {seed} ===")
        best = run_training(cfg)
        summary = evaluate_model(cfg, best)
        per_seed_rr.append(summary["plate_rr"])
        per_seed_char.append(summary["char_acc"])
        for name, d in (summary.get("per_subset") or {}).items():
            per_subset.setdefault(name, []).append(d["plate_rr"])
        rows.append({"seed": seed, "plate_rr": summary["plate_rr"],
                     "char_acc": summary["char_acc"]})
        print(f"seed {seed}: Plate RR = {summary['plate_rr'] * 100:.2f}% "
              f"Char = {summary['char_acc'] * 100:.2f}%")

    rr_mean, rr_std = _mean_std(per_seed_rr)
    char_mean, char_std = _mean_std(per_seed_char)
    agg = {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "plate_rr_mean": rr_mean,
        "plate_rr_std": rr_std,
        "char_acc_mean": char_mean,
        "char_acc_std": char_std,
        "per_seed": rows,
    }
    if per_subset:
        agg["per_subset"] = {
            name: dict(zip(("mean", "std"), _mean_std(v), strict=True))
            for name, v in per_subset.items()
        }
    out = get_run_dir(base_run) / "multiseed_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(agg, indent=2))
    print(f"\n=== {base_run}: {rr_mean * 100:.2f} +/- {rr_std * 100:.2f}% "
          f"over {len(seeds)} seeds | Char {char_mean * 100:.2f}% ===")
    for name in sorted(agg.get("per_subset", {})):
        m = agg["per_subset"][name]
        print(f"  {name}: {m['mean'] * 100:.2f} +/- {m['std'] * 100:.2f}%")


if __name__ == "__main__":
    main()
