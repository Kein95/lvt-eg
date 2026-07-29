"""train_epoch + validate - paper Sec impl (notebook Cell 10).

Standard mixed-precision training loop using AMP + gradient clipping + EMA,
with edge-safe loss (LVTEGLoss) and greedy CTC decoding for online metrics.

Returned metrics per pass:
    avg_loss, avg_ctc, avg_edge, plate_rr_pct, char_acc_pct
"""

from __future__ import annotations

import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from lvt_eg.common.ctc_decode import decode_ctc
from lvt_eg.common.ema import EMA


def _epoch_metrics(loss_sum: float, ctc_sum: float, edge_sum: float,
                   correct: int, samples: int, ch_correct: int, ch_total: int,
                   n_batches: int) -> tuple[float, float, float, float, float]:
    return (
        loss_sum / max(1, n_batches),
        ctc_sum / max(1, n_batches),
        edge_sum / max(1, n_batches),
        100 * correct / max(1, samples),
        100 * ch_correct / max(1, ch_total),
    )


def _decode_targets(labels: torch.Tensor, lengths, idx2char: dict[int, str]) -> list[str]:
    out: list[str] = []
    for i in range(labels.size(0)):
        n = lengths[i].item() if isinstance(lengths, torch.Tensor) else int(lengths[i])
        out.append("".join(idx2char.get(c.item(), "") for c in labels[i][:n]))
    return out


def train_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: torch.nn.Module,
    scaler: GradScaler,
    ema: EMA,
    epoch: int,
    idx2char: dict[int, str],
    device: torch.device,
    grad_clip: float = 1.0,
    ema_warmup: int = 0,
) -> tuple[float, float, float, float, float]:
    """One training epoch. Returns metrics tuple.

    ema_warmup: skip EMA accumulation while epoch <= ema_warmup (0 = accumulate
    from epoch 1). Only CCPD gates accumulation; other datasets pass 0.
    """
    if hasattr(loader.dataset, "set_epoch"):
        loader.dataset.set_epoch(epoch)
    model.train()
    loss_sum = ctc_sum = edge_sum = 0.0
    correct = samples = ch_correct = ch_total = 0

    pbar = tqdm(loader, desc=f"Train E{epoch}")
    for lr_images, hr_images, labels, lengths in pbar:
        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device.type):
            log_probs, pred_edge, gt_edge = model(lr_images, hr_images, return_all=True)
            T = log_probs.size(1)
            input_lengths = torch.full((log_probs.size(0),), T, dtype=torch.long, device=device)
            target_lengths = (lengths.long() if isinstance(lengths, torch.Tensor)
                              else torch.tensor(lengths, dtype=torch.long)).to(device)
            loss, loss_ctc, loss_edge = criterion(
                log_probs, labels, input_lengths, target_lengths, pred_edge, gt_edge,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if epoch > ema_warmup and not torch.isnan(loss):
            ema.update(model)

        # Online metrics
        with torch.no_grad():
            preds = decode_ctc(log_probs, idx2char)
            targets = _decode_targets(labels, lengths, idx2char)
            for p, t in zip(preds, targets, strict=False):
                if p == t:
                    correct += 1
                ch_total += max(len(p), len(t))
                for a, b in zip(p, t, strict=False):
                    if a == b:
                        ch_correct += 1
            samples += log_probs.size(0)

        loss_sum += loss.item()
        ctc_sum += loss_ctc.item()
        edge_sum += loss_edge.item()
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            ctc=f"{loss_ctc.item():.4f}",
            acc=f"{100 * correct / max(1, samples):.1f}%",
        )

    return _epoch_metrics(loss_sum, ctc_sum, edge_sum, correct, samples,
                          ch_correct, ch_total, len(loader))


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader,
    criterion: torch.nn.Module,
    idx2char: dict[int, str],
    device: torch.device,
) -> tuple[float, float, float, float, float]:
    """Validation pass. Returns same metrics tuple as train_epoch()."""
    model.eval()
    loss_sum = ctc_sum = edge_sum = 0.0
    correct = samples = ch_correct = ch_total = 0

    for lr_images, hr_images, labels, lengths in tqdm(loader, desc="Val"):
        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)
        labels = labels.to(device)

        with autocast(device.type):
            log_probs, pred_edge, gt_edge = model(lr_images, hr_images, return_all=True)
            T = log_probs.size(1)
            input_lengths = torch.full((log_probs.size(0),), T, dtype=torch.long, device=device)
            target_lengths = (lengths.long() if isinstance(lengths, torch.Tensor)
                              else torch.tensor(lengths, dtype=torch.long)).to(device)
            loss, loss_ctc, loss_edge = criterion(
                log_probs, labels, input_lengths, target_lengths, pred_edge, gt_edge,
            )

        loss_sum += loss.item()
        ctc_sum += loss_ctc.item()
        edge_sum += loss_edge.item()

        preds = decode_ctc(log_probs, idx2char)
        targets = _decode_targets(labels, lengths, idx2char)
        for p, t in zip(preds, targets, strict=False):
            if p == t:
                correct += 1
            ch_total += max(len(p), len(t))
            for a, b in zip(p, t, strict=False):
                if a == b:
                    ch_correct += 1
        samples += log_probs.size(0)

    return _epoch_metrics(loss_sum, ctc_sum, edge_sum, correct, samples,
                          ch_correct, ch_total, len(loader))
