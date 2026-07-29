"""Train LVT-EG from a YAML config.

Usage:
    python -m lvt_eg.scripts.train --config configs/1-lvt_eg_rodosol.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from lvt_eg.common.ema import EMA
from lvt_eg.common.losses import LVTEGLoss
from lvt_eg.common.seeding import make_generator, seed_worker, set_seed
from lvt_eg.config import get_dataset_dir, get_run_dir
from lvt_eg.data.ccpd import ALL_CHARS, _decode_corners, _decode_plate
from lvt_eg.data.lp2025 import parse_lp2025_split
from lvt_eg.data.lp_crop_dataset import LPCropDataset
from lvt_eg.data.lplc import list_lplc_folds, parse_lplc_fold
from lvt_eg.data.rodosol import parse_rodosol_split
from lvt_eg.data.synthetic_degradation import MultiLevelSyntheticDataset
from lvt_eg.data.ukrainian import parse_ukrainian_dataset
from lvt_eg.models.lvt_eg import LVTEGModel
from lvt_eg.training.checkpoint import append_history, save_checkpoint
from lvt_eg.training.scheduler import build_optimizer, build_scheduler
from lvt_eg.training.trainer import train_epoch, validate


def _build_idx_maps(chars: str) -> tuple[dict[str, int], dict[int, str]]:
    char2idx = {c: i + 1 for i, c in enumerate(chars)}  # 0 reserved for blank
    idx2char = {i + 1: c for i, c in enumerate(chars)}
    return char2idx, idx2char


def _build_datasets(cfg: dict, char2idx: dict[str, int]):
    """Dispatch to the right per-dataset parser based on cfg['dataset']['key']."""
    ds = cfg["dataset"]
    key = ds["key"]
    root = get_dataset_dir(key)
    img_h, img_w = ds["img_h"], ds["img_w"]
    hr_h, hr_w = ds.get("hr_h", 64), ds.get("hr_w", 256)

    if key == "rodosol":
        train_s = parse_rodosol_split(root, "training")
        val_s = parse_rodosol_split(root, "validation")
        test_s = parse_rodosol_split(root, "testing")
    elif key == "lp2025":
        train_s = parse_lp2025_split(root, "train")
        val_s = parse_lp2025_split(root, "val")
        test_s = parse_lp2025_split(root, "test")
    elif key == "ukrainian":
        train_s, val_s, test_s = parse_ukrainian_dataset(root, ds["chars"], seed=cfg["seed"])
    elif key == "ccpd2019":
        # v2019 protocol uses pre-built manifests (notebook Cell 2):
        #   <root>/splits/{train,val,test}.txt - one image path per line.
        splits_cfg = ds.get("splits", {})
        def _load_manifest(name: str) -> list[dict]:
            rel = splits_cfg.get(name)
            if not rel:
                raise ValueError(
                    f"CCPD config missing dataset.splits.{name}; expected a path "
                    f"to the v2019 split manifest (e.g. splits/{name}.txt)."
                )
            manifest = root / rel
            if not manifest.exists():
                raise FileNotFoundError(
                    f"CCPD {name} manifest not found: {manifest}. Place the v2019 "
                    f"(Laroca-2023 fair) split file there - the release does not "
                    f"bundle CCPD split lists (see data/README.md)."
                )
            samples: list[dict] = []
            for line in manifest.read_text().splitlines():
                # Notebook load_manifest parity: first tab-field, forward slashes.
                p = line.strip().split("\t")[0].replace("\\", "/")
                if not p:
                    continue
                # Each line points to a CCPD image - re-use directory parser
                # on the file's parent to extract corners + plate.
                fp = (root / p) if not Path(p).is_absolute() else Path(p)
                if not fp.exists():
                    continue
                fields = fp.stem.split("-")
                if len(fields) < 5:
                    continue
                corners = _decode_corners(fields[3])
                plate = _decode_plate(fields[4])
                if corners and plate:
                    samples.append({"image_path": str(fp), "plate": plate, "corners": corners})
            if not samples:
                raise FileNotFoundError(
                    f"CCPD {name} manifest {manifest} resolved 0 usable images. "
                    f"Each line must be a path like 'ccpd_base/<name>.jpg' relative to "
                    f"the CCPD root (forward slashes). Check the split file format."
                )
            return samples
        train_s = _load_manifest("train")
        val_s = _load_manifest("val")
        test_s = _load_manifest("test")
    elif key == "lplc":
        # 5-fold cross-validation: pick fold via cfg["dataset"]["cv"]["fold"]
        # (default 0). Caller scripts loop over folds for the full protocol.
        cv = ds.get("cv", {})
        fold_idx = int(cv.get("fold", 0))
        folds_dir = root / "folds"
        ann_path = root / "annotations_formatted.json"
        # Full-scene images; parse_lplc_fold warps out each plate via corners.
        images_dir = root / "images"
        fold_files = list_lplc_folds(folds_dir, repeats=int(cv.get("repeats", 2)),
                                     scenario=cv.get("scenario", "scen0"))
        if not fold_files:
            raise FileNotFoundError(
                f"No LPLC fold files under {folds_dir} (looked for flat "
                f"fold_*.json and {cv.get('scenario', 'scen0')}/fold_*.json). "
                f"Place the official 5x2 fold JSONs there."
            )
        if fold_idx >= len(fold_files):
            raise IndexError(
                f"cv.fold={fold_idx} out of range ({len(fold_files)} folds found)."
            )
        fold_path = fold_files[fold_idx]
        train_s = parse_lplc_fold(fold_path, ann_path, images_dir, "train")
        val_s = parse_lplc_fold(fold_path, ann_path, images_dir, "val")
        test_s = parse_lplc_fold(fold_path, ann_path, images_dir, "test")
    else:
        raise ValueError(f"Unknown dataset key: {key}")

    common = {"char2idx": char2idx, "hr_h": hr_h, "hr_w": hr_w, "img_h": img_h, "img_w": img_w,
              "train_aug": ds.get("train_aug", "photometric")}
    train_ds = LPCropDataset(train_s, augment=True, **common)
    val_ds = LPCropDataset(val_s, augment=False, **common) if val_s else None
    test_ds = LPCropDataset(test_s, augment=False, **common) if test_s else None
    return train_ds, val_ds, test_ds


def resolve_chars(cfg: dict) -> str:
    """Resolve the character set, filling CCPD's authoritative list when null."""
    chars = cfg["dataset"]["chars"]
    if chars is None and cfg["dataset"]["key"] == "ccpd2019":
        # CCPD authoritative chars come from the data module (33 + 24 + 10 dedup).
        chars = "".join(ALL_CHARS)
    if chars is None:
        raise ValueError(f"dataset.chars must be set for key={cfg['dataset']['key']}")
    return chars


def run_training(cfg: dict, device: torch.device | None = None) -> Path:
    """Train one run from an in-memory config; return the path to best.pth.

    Shared by the CLI (single run) and run_lplc_cv (per-fold loop), so the
    config is dumped from memory rather than copied from a file path.
    """
    set_seed(cfg["seed"])
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = get_run_dir(cfg["run_name"])
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    chars = resolve_chars(cfg)
    char2idx, idx2char = _build_idx_maps(chars)
    train_ds, val_ds, _ = _build_datasets(cfg, char2idx)
    if cfg["dataset"].get("synth_policy", "v2") in ("v1", "v2"):
        train_ds = MultiLevelSyntheticDataset(
            train_ds, policy=cfg["dataset"]["synth_policy"],
            synth_aug=cfg["dataset"].get("synth_aug", "photometric"),
            synth_degrade_at=cfg["dataset"].get("synth_degrade_at", "lr"))

    generator = make_generator(cfg["seed"])
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                              shuffle=True, num_workers=2, pin_memory=True,
                              worker_init_fn=seed_worker, generator=generator)
    val_loader = (DataLoader(val_ds, batch_size=cfg["train"]["batch_size"],
                             shuffle=False, num_workers=2, pin_memory=True)
                  if val_ds else None)

    model = LVTEGModel(
        num_classes=cfg["dataset"]["num_classes"],
        img_h=cfg["dataset"]["img_h"], img_w=cfg["dataset"]["img_w"],
        use_stn=cfg["model"]["use_stn"],
        drop_path=cfg["model"]["drop_path"],
        use_pos_embed=cfg["model"].get("use_pos_embed", True),
        legacy_conv_mixer=cfg["model"].get("legacy_conv_mixer", False),
        use_edge=cfg["model"].get("use_edge", True),
        stn_order=cfg["model"].get("stn_order", "stn_first"),
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
    # CCPD notebook gates EMA *accumulation* to start after warmup; the other
    # datasets accumulate from epoch 1. 0 = accumulate every epoch.
    ema_update_warmup = ema_warmup if cfg["train"].get("ema_update_after_warmup", False) else 0
    patience_counter = 0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss, train_ctc, train_edge, train_acc, train_char = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, scaler, ema, epoch,
            idx2char=idx2char, device=device,
            grad_clip=cfg["train"].get("grad_clip", 1.0),
            ema_warmup=ema_update_warmup,
        )
        log = {"epoch": epoch, "train_loss": train_loss, "train_ctc": train_ctc,
               "train_edge": train_edge, "train_acc": train_acc, "train_char": train_char}
        if val_loader is not None:
            # Notebook recipe: validate on EMA only after warmup epochs,
            # otherwise validate on online (training) weights.
            use_ema = epoch > ema_warmup and ema.initialized
            online_state = {k: v.clone() for k, v in model.state_dict().items()} if use_ema else None
            if use_ema:
                ema.apply(model)
            val_loss, val_ctc, val_edge, val_acc, val_char = validate(
                model, val_loader, criterion, idx2char=idx2char, device=device,
            )
            log.update({"val_loss": val_loss, "val_ctc": val_ctc, "val_edge": val_edge,
                        "val_acc": val_acc, "val_char": val_char, "use_ema": use_ema})
            improved = (val_acc > best_val) or (val_acc == best_val == 0.0 and val_loss < best_val_loss)
            if improved:
                best_val = val_acc
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(run_dir / "best.pth", model, epoch, val_acc,
                                ema_state=ema.state_dict())
            else:
                patience_counter += 1
            if online_state is not None:
                model.load_state_dict(online_state)
        save_checkpoint(run_dir / "last.pth", model, epoch, log.get("val_acc", 0.0),
                        ema_state=ema.state_dict())
        append_history(run_dir / "history.json", log)
        print(json.dumps(log))
        if patience > 0 and patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}: no improvement for {patience} epochs.")
            break

    return run_dir / "best.pth"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LVT-EG from a YAML config.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override config train.epochs (e.g. 1 for parity smoke).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override config seed; appends _seed<N> to run_name "
                             "so multi-run mean±std results do not overwrite.")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.seed is not None:
        cfg["seed"] = args.seed
        cfg["run_name"] = f"{cfg['run_name']}_seed{args.seed}"
    run_training(cfg)


if __name__ == "__main__":
    main()
