"""Cross-domain (RodoSol -> LPLC) zero-shot evaluation.

Trains LVT-EG on the source domain defined under `source:` in the YAML
config, then evaluates on the target domain defined under `target:`.

Usage:
    # 8K variant (paper Table IV cross-domain)
    python -m lvt_eg.scripts.run_zeroshot \
        --config configs/6-lvt_eg_rodosol2lplc_zeroshot.yaml

    # 12K variant (paper Table IV, best cross-domain)
    python -m lvt_eg.scripts.run_zeroshot \
        --config configs/7-lvt_eg_rodosol2lplc_zeroshot_12k.yaml
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import yaml
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from lvt_eg.common.ctc_decode import decode_ctc
from lvt_eg.common.ema import EMA
from lvt_eg.common.losses import LVTEGLoss
from lvt_eg.common.metrics import character_accuracy, plate_recognition_rate
from lvt_eg.common.seeding import set_seed
from lvt_eg.config import get_dataset_dir, get_run_dir
from lvt_eg.data.lp_crop_dataset import LPCropDataset
from lvt_eg.data.lplc import list_lplc_folds, parse_lplc_fold
from lvt_eg.data.rodosol import parse_rodosol_merged, parse_rodosol_split
from lvt_eg.data.synthetic_degradation import MultiLevelSyntheticDataset
from lvt_eg.models.lvt_eg import LVTEGModel
from lvt_eg.scripts.evaluate import _lplc_per_legibility
from lvt_eg.scripts.train import _build_idx_maps
from lvt_eg.training.checkpoint import append_history, load_checkpoint, save_checkpoint
from lvt_eg.training.scheduler import build_optimizer, build_scheduler
from lvt_eg.training.trainer import train_epoch, validate


def _build_source_split(source_cfg: dict, char2idx: dict[str, int]):
    """Build train + val datasets from the `source:` block (RodoSol only for now)."""
    if source_cfg["dataset"] != "rodosol":
        raise NotImplementedError("Only rodosol -> X is implemented.")

    root = get_dataset_dir("rodosol")
    common = {
        "char2idx": char2idx,
        "hr_h": source_cfg["hr_h"],
        "hr_w": source_cfg["hr_w"],
        "img_h": source_cfg["img_h"],
        "img_w": source_cfg["img_w"],
    }

    splits = source_cfg["splits"]
    if "merge" in splits.get("train", {}):
        train_s = parse_rodosol_merged(root, splits["train"]["merge"])
    else:
        train_s = parse_rodosol_split(root, splits["train"]["filter"])
    val_s = parse_rodosol_split(root, splits["val"]["filter"])
    return (LPCropDataset(train_s, augment=True, **common),
            LPCropDataset(val_s, augment=False, **common))


def _build_lplc_fold_test(fold_file: Path, annotations, images_dir: Path,
                          char2idx: dict[str, int], src: dict,
                          min_legibility: int) -> tuple[LPCropDataset, list[dict]]:
    """Build one LPLC fold's TEST dataset for cross-domain scoring.

    Returns (dataset, samples) so the caller can compute per-legibility strata.
    The target crop grid matches the source (RodoSol) recipe - the same trained
    checkpoint is scored on every fold, so the input pipeline must be identical.
    """
    samples = parse_lplc_fold(fold_file, annotations, images_dir, "test",
                              min_legibility=min_legibility)
    ds = LPCropDataset(samples, char2idx=char2idx,
                       hr_h=src["hr_h"], hr_w=src["hr_w"],
                       img_h=src["img_h"], img_w=src["img_w"], augment=False)
    return ds, samples


@torch.no_grad()
def _infer_preds(model: torch.nn.Module, loader: DataLoader,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cross-domain zero-shot evaluation.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path,
                        help="If given, skip training and only evaluate this checkpoint.")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    set_seed(int(cfg.get("seed", 42)))
    run_dir = get_run_dir(cfg["run_name"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    char2idx, idx2char = _build_idx_maps(
        # Source/target alphabets must match (RodoSol & LPLC: both 36 alnum).
        # Use source.dataset chars from a sibling file or fall back to default.
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )
    src = cfg["source"]
    tgt = cfg["target"]

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        train_ds, val_ds = _build_source_split(src, char2idx)
        if src.get("synth_policy", "v2") in ("v1", "v2"):
            train_ds = MultiLevelSyntheticDataset(train_ds, policy=src.get("synth_policy", "v2"))

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["train"]["batch_size"],
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        model = LVTEGModel(
            num_classes=37,
            img_h=src["img_h"],
            img_w=src["img_w"],
            use_stn=cfg["model"]["use_stn"],
            drop_path=cfg["model"]["drop_path"],
            use_pos_embed=cfg["model"].get("use_pos_embed", True),
        ).to(device)
        criterion = LVTEGLoss(edge_l1_weight=cfg["loss"]["edge_l1_weight"])
        optimizer = build_optimizer(model, cfg["train"])
        scheduler = build_scheduler(optimizer, cfg["train"], steps_per_epoch=len(train_loader))
        scaler = GradScaler(device.type)
        ema = EMA(decay=cfg["train"].get("ema_decay", 0.999))

        best_val = -1.0
        best_val_loss = float("inf")
        patience = int(cfg["train"].get("patience", 0))
        ema_warmup = int(cfg["train"].get("ema_warmup_epochs", 10))
        patience_counter = 0
        for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
            train_loss, train_ctc, train_edge, train_acc, train_char = train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                criterion,
                scaler,
                ema,
                epoch,
                idx2char=idx2char,
                device=device,
                grad_clip=float(cfg["train"].get("grad_clip", 1.0)),
            )
            log = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_ctc": train_ctc,
                "train_edge": train_edge,
                "train_acc": train_acc,
                "train_char": train_char,
            }

            use_ema = epoch > ema_warmup and ema.initialized
            online_state = {k: v.clone() for k, v in model.state_dict().items()} if use_ema else None
            if use_ema:
                ema.apply(model)
            val_loss, val_ctc, val_edge, val_acc, val_char = validate(
                model,
                val_loader,
                criterion,
                idx2char=idx2char,
                device=device,
            )
            log.update(
                {
                    "val_loss": val_loss,
                    "val_ctc": val_ctc,
                    "val_edge": val_edge,
                    "val_acc": val_acc,
                    "val_char": val_char,
                    "use_ema": use_ema,
                }
            )

            improved = (val_acc > best_val) or (val_acc == best_val == 0.0 and val_loss < best_val_loss)
            if improved:
                best_val = val_acc
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(run_dir / "best.pth", model, epoch, val_acc, ema_state=ema.state_dict())
            else:
                patience_counter += 1
            if online_state is not None:
                model.load_state_dict(online_state)

            save_checkpoint(run_dir / "last.pth", model, epoch, val_acc, ema_state=ema.state_dict())
            append_history(run_dir / "history.json", log)
            print(json.dumps(log))
            if patience > 0 and patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}: no improvement for {patience} epochs.")
                break

        checkpoint_path = run_dir / "best.pth"

    # Load the trained model once, then score it on every LPLC scen0 fold-TEST
    # split (5 folds x 2 iters = 10 partitions). The ± band is cross-partition
    # variance of ONE checkpoint, matching the notebook zero-shot protocol.
    if tgt["dataset"] != "lplc":
        raise NotImplementedError("Only X -> lplc cross-domain is implemented.")
    model = LVTEGModel(
        num_classes=37, img_h=src["img_h"], img_w=src["img_w"],
        use_stn=cfg["model"]["use_stn"],
        drop_path=cfg["model"]["drop_path"],
        use_pos_embed=cfg["model"].get("use_pos_embed", True),
    ).to(device).eval()
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    state = ckpt.get("ema") or ckpt["model"]
    model.load_state_dict(state)

    lplc_root = get_dataset_dir("lplc")
    scenario = tgt.get("scenario", "scen0")
    repeats = int(tgt.get("repeats", 2))
    min_leg = int(tgt.get("min_legibility", 1))
    fold_files = list_lplc_folds(lplc_root / "folds", repeats=repeats, scenario=scenario)
    if not fold_files:
        raise FileNotFoundError(
            f"No LPLC fold files under {lplc_root / 'folds'} "
            f"(looked for {scenario}/fold_*.json and flat fold_*.json)."
        )
    annotations = json.loads((lplc_root / "annotations_formatted.json").read_text())
    images_dir = lplc_root / "images"
    report_leg = bool(tgt.get("per_legibility_report", False))

    per_fold: list[dict] = []
    plate_rrs: list[float] = []
    char_accs: list[float] = []
    per_leg: dict[str, list[float]] = {}
    for k, fold_file in enumerate(fold_files):
        ds, samples = _build_lplc_fold_test(fold_file, annotations, images_dir,
                                            char2idx, src, min_leg)
        loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"],
                            shuffle=False, num_workers=2)
        preds, targets = _infer_preds(model, loader, idx2char, device)
        rr = plate_recognition_rate(preds, targets)
        ca = character_accuracy(preds, targets)
        plate_rrs.append(rr)
        char_accs.append(ca)
        row = {"fold": fold_file.name, "n_test": len(targets),
               "plate_rr": rr, "char_acc": ca}
        if report_leg:
            strata = _lplc_per_legibility(samples, preds, targets)
            row["per_legibility"] = strata
            for name, d in strata.items():
                per_leg.setdefault(name, []).append(d["plate_rr"])
        per_fold.append(row)
        print(f"--- {fold_file.name} ({k + 1}/{len(fold_files)}) -> "
              f"Plate RR {rr * 100:.2f}% | Char {ca * 100:.2f}% ---")

    def _mean_std(xs: list[float]) -> tuple[float, float]:
        return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)

    rr_mean, rr_std = _mean_std(plate_rrs)
    char_mean, char_std = _mean_std(char_accs)
    summary = {
        "source": src["dataset"],
        "target": tgt["dataset"],
        "scenario": scenario,
        "n_folds": len(per_fold),
        "plate_rr_mean": rr_mean,
        "plate_rr_std": rr_std,
        "char_acc_mean": char_mean,
        "char_acc_std": char_std,
        "per_fold": per_fold,
    }
    if per_leg:
        summary["per_legibility"] = {
            name: dict(zip(("mean", "std"), _mean_std(v), strict=True))
            for name, v in per_leg.items()
        }
    out_dir = get_run_dir(cfg["run_name"]) / "zeroshot"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    strata_str = "".join(
        f" {name.capitalize()} {summary['per_legibility'][name]['mean'] * 100:.2f}"
        for name in ("perfect", "good", "poor")
        if name in summary.get("per_legibility", {})
    )
    print(f"\n=== {src['dataset']} -> {tgt['dataset']} zero-shot: "
          f"{rr_mean * 100:.2f} ± {rr_std * 100:.2f}% over {len(per_fold)} folds | "
          f"Char {char_mean * 100:.2f}%{strata_str} ===")


if __name__ == "__main__":
    main()
