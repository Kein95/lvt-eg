"""Recognition metrics - paper Sec 4 reports plate-RR (exact match) and char acc.

Plate RR (Recognition Rate): fraction of predictions that exactly match the
ground-truth plate string after greedy CTC decoding. This is the primary
metric in every results table of the paper.
"""

from __future__ import annotations

from collections.abc import Iterable


def plate_recognition_rate(preds: Iterable[str], targets: Iterable[str]) -> float:
    """Exact-match plate-level recognition rate in [0, 1]."""
    preds = list(preds)
    targets = list(targets)
    if not preds:
        return 0.0
    correct = sum(1 for p, t in zip(preds, targets, strict=False) if p == t)
    return correct / len(preds)


def character_accuracy(preds: Iterable[str], targets: Iterable[str]) -> float:
    """Per-character accuracy in [0, 1] (paper 'Char. Acc.' columns)."""
    preds = list(preds)
    targets = list(targets)
    total = 0
    matched = 0
    for p, t in zip(preds, targets, strict=False):
        n = max(len(p), len(t))
        if n == 0:
            continue
        total += n
        for a, b in zip(p, t, strict=False):
            if a == b:
                matched += 1
    return matched / total if total else 0.0
