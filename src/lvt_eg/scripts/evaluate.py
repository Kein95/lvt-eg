"""Evaluate a trained LVT-EG checkpoint on the test split.

Usage:
    python -m lvt_eg.scripts.evaluate \
        --config configs/1-lvt_eg_rodosol.yaml \
        --checkpoint results/lvt_eg_rodosol/best.pth
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from lvt_eg.common.ctc_decode import decode_ctc
from lvt_eg.common.metrics import character_accuracy, plate_recognition_rate
from lvt_eg.config import get_dataset_dir, get_run_dir
from lvt_eg.data.ccpd import CCPD_SUBSETS, parse_ccpd_directory
from lvt_eg.data.lp_crop_dataset import LPCropDataset
from lvt_eg.models.lvt_eg import LVTEGModel
from lvt_eg.scripts.train import _build_datasets, _build_idx_maps, resolve_chars
from lvt_eg.training.checkpoint import load_checkpoint


def _ccpd_per_subset(samples: list[dict], preds: list[str], targets: list[str]
                     ) -> dict[str, dict]:
    """Plate RR / char acc per CCPD subset (paper Table III breakdown).

    Each subset name appears in the sample's image path (notebook Cell 17).
    Empty subsets (e.g. ccpd_base, absent from the hard test split) are skipped.
    Values are fractions in [0, 1], matching the top-level plate_rr scale.
    """
    out: dict[str, dict] = {}
    for sub in CCPD_SUBSETS:
        idxs = [i for i, s in enumerate(samples)
                if sub in s["image_path"].replace("\\", "/")]
        if not idxs:
            continue
        sp = [preds[i] for i in idxs]
        st = [targets[i] for i in idxs]
        out[sub] = {
            "plate_rr": plate_recognition_rate(sp, st),
            "char_acc": character_accuracy(sp, st),
            "n": len(idxs),
        }
    return out


_LEGIBILITY_NAMES = {3: "perfect", 2: "good", 1: "poor"}


def _lplc_per_legibility(samples: list[dict], preds: list[str], targets: list[str]
                         ) -> dict[str, dict]:
    """Plate RR / char acc per LPLC legibility level (paper Table IV strata).

    Uses the per-sample 'legibility' field kept by parse_lplc_fold
    (3=perfect, 2=good, 1=poor). Requires the loader to run with shuffle=False
    so preds/targets align with samples. Values are fractions in [0, 1].
    """
    out: dict[str, dict] = {}
    for lvl, name in _LEGIBILITY_NAMES.items():
        idxs = [i for i, s in enumerate(samples) if s.get("legibility") == lvl]
        if not idxs:
            continue
        sp = [preds[i] for i in idxs]
        st = [targets[i] for i in idxs]
        out[name] = {
            "plate_rr": plate_recognition_rate(sp, st),
            "char_acc": character_accuracy(sp, st),
            "n": len(idxs),
        }
    return out


@torch.no_grad()
def _infer(model: torch.nn.Module, loader: DataLoader,
           idx2char: dict[int, str], device: torch.device
           ) -> tuple[list[str], list[str]]:
    """Greedy-decode a loader into aligned (preds, targets) string lists."""
    preds_all: list[str] = []
    targets_all: list[str] = []
    for lr, _, labels, lengths in loader:
        lr = lr.to(device)
        log_probs = model(lr)
        preds_all.extend(decode_ctc(log_probs, idx2char))
        for i in range(labels.size(0)):
            n = lengths[i].item() if isinstance(lengths, torch.Tensor) else int(lengths[i])
            targets_all.append("".join(idx2char.get(c.item(), "") for c in labels[i][:n]))
    return preds_all, targets_all


def _ccpd_extra_subsets(cfg: dict, model, char2idx, idx2char, device) -> dict[str, dict]:
    """Score the two CCPD subsets that the test.txt manifest cannot cover.

    ccpd_weather is present only as a directory (never in test.txt) so it is
    globbed; ccpd_base is reported as the validation-split score (val.txt is
    100% base), matching the notebook's 'Val (ccpd_base)' line.
    """
    ds = cfg["dataset"]
    grid = {"char2idx": char2idx, "hr_h": ds.get("hr_h", 64), "hr_w": ds.get("hr_w", 256),
            "img_h": ds["img_h"], "img_w": ds["img_w"]}
    bs = cfg["train"]["batch_size"]
    out: dict[str, dict] = {}

    weather_dir = get_dataset_dir("ccpd2019") / "ccpd_weather"
    weather_samples = parse_ccpd_directory(weather_dir) if weather_dir.exists() else []
    if weather_samples:
        loader = DataLoader(LPCropDataset(weather_samples, augment=False, **grid),
                            batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
        wp, wt = _infer(model, loader, idx2char, device)
        out["ccpd_weather"] = {"plate_rr": plate_recognition_rate(wp, wt),
                               "char_acc": character_accuracy(wp, wt), "n": len(wt)}

    _, val_ds, _ = _build_datasets(cfg, char2idx)
    if val_ds is not None:
        loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=2, pin_memory=True)
        bp, bt = _infer(model, loader, idx2char, device)
        out["ccpd_base"] = {"plate_rr": plate_recognition_rate(bp, bt),
                            "char_acc": character_accuracy(bp, bt), "n": len(bt)}
    return out


@torch.no_grad()
def evaluate_model(cfg: dict, checkpoint, device: torch.device | None = None,
                   write_outputs: bool = True) -> dict:
    """Evaluate a checkpoint on the config's test split; return the summary dict.

    For CCPD with eval.per_subset_report, adds a per-subset breakdown. Reused by
    run_lplc_cv for per-fold scoring (write_outputs left on for per-fold logs).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    char2idx, idx2char = _build_idx_maps(resolve_chars(cfg))
    _, _, test_ds = _build_datasets(cfg, char2idx)
    if test_ds is None:
        raise RuntimeError("No test split available for this config.")

    model = LVTEGModel(
        num_classes=cfg["dataset"]["num_classes"],
        img_h=cfg["dataset"]["img_h"], img_w=cfg["dataset"]["img_w"],
        use_stn=cfg["model"]["use_stn"],
        drop_path=cfg["model"]["drop_path"],
        use_pos_embed=cfg["model"].get("use_pos_embed", True),
        legacy_conv_mixer=cfg["model"].get("legacy_conv_mixer", False),
        use_edge=cfg["model"].get("use_edge", True),
        stn_order=cfg["model"].get("stn_order", "stn_first"),
    ).to(device).eval()
    ckpt = load_checkpoint(Path(checkpoint), map_location=device)
    state = ckpt.get("ema") or ckpt["model"]
    model.load_state_dict(state)

    loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"],
                        shuffle=False, num_workers=2, pin_memory=True)
    preds_all, targets_all = _infer(model, loader, idx2char, device)

    summary = {
        "checkpoint": str(checkpoint),
        "epoch": ckpt.get("epoch"),
        "n_test": len(targets_all),
        "plate_rr": plate_recognition_rate(preds_all, targets_all),
        "char_acc": character_accuracy(preds_all, targets_all),
    }
    if (cfg["dataset"]["key"] == "ccpd2019"
            and cfg.get("eval", {}).get("per_subset_report", False)):
        # 6 hard subsets live in the test split; weather (dir-only) + base
        # (=val split) are added so all 8 Table-4 subsets are reproducible.
        per_subset = _ccpd_per_subset(test_ds.samples, preds_all, targets_all)
        per_subset.update(_ccpd_extra_subsets(cfg, model, char2idx, idx2char, device))
        summary["per_subset"] = per_subset
        # Paper Hard-all folds the near-saturated Weather subset into the
        # 6-subset test-split score to match the baseline aggregate protocol
        # (Laroca / ViTLPR / LPRNet+ all include Weather). The bare plate_rr
        # above stays the 6-subset value.
        weather = per_subset.get("ccpd_weather")
        if weather:
            n6 = summary["n_test"]
            summary["hard_all_incl_weather"] = (
                summary["plate_rr"] * n6 + weather["plate_rr"] * weather["n"]
            ) / (n6 + weather["n"])
    if (cfg["dataset"]["key"] == "lplc"
            and cfg.get("eval", {}).get("per_legibility_report", False)):
        summary["per_legibility"] = _lplc_per_legibility(
            test_ds.samples, preds_all, targets_all)

    if write_outputs:
        out_dir = get_run_dir(cfg["run_name"]) / "eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "test_summary.json").write_text(json.dumps(summary, indent=2))
        (out_dir / "test_predictions.tsv").write_text(
            "\n".join(f"{p}\t{t}\t{int(p == t)}"
                      for p, t in zip(preds_all, targets_all, strict=False))
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LVT-EG on a test split.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    summary = evaluate_model(cfg, args.checkpoint)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
