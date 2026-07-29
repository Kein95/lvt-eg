"""Greedy CTC decoding - paper Sec 3.5 (notebook Cell 6).

Plate strings are short left-to-right sequences, so greedy CTC is sufficient
(paper Sec 3.5: "greedy CTC decoding is used at inference"). Beam search is
not used and not needed.

Algorithm:
1. Take argmax over class dim per timestep.
2. Collapse consecutive duplicate indices.
3. Remove the blank token (index 0).
4. Map remaining indices back to characters via the alphabet.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch


def decode_ctc(log_probs: torch.Tensor, idx2char: dict[int, str]) -> list[str]:
    """Greedy CTC decode a batch of log-probability sequences.

    Args:
        log_probs: tensor of shape [B, T, C] - class log-probabilities per timestep.
        idx2char: mapping {1..|C|} -> single-character string. Index 0 is the CTC blank.

    Returns:
        List of decoded plate strings, length B.
    """
    pred_indices = log_probs.argmax(dim=2)
    results: list[str] = []
    for b in range(pred_indices.size(0)):
        raw = pred_indices[b].tolist()
        collapsed: list[int] = []
        prev: int | None = None
        for idx in raw:
            if idx != prev:
                collapsed.append(idx)
            prev = idx
        text = "".join(idx2char.get(i, "") for i in collapsed if i != 0)
        results.append(text)
    return results


def encode_label(text: str, char2idx: dict[str, int], max_len: int) -> tuple[list[int], int]:
    """Encode a plate string for CTC (right-pad with 0=blank). Returns (padded, true_len)."""
    encoded = [char2idx.get(c, 0) for c in text.upper()]
    length = len(encoded)
    padded = encoded + [0] * (max_len - length)
    return padded[:max_len], length


def char_accuracy(preds: Iterable[str], targets: Iterable[str]) -> float:
    """Per-character accuracy across a batch (counts matched chars / max(len) per pair)."""
    total_chars = 0
    matched = 0
    for p, t in zip(preds, targets, strict=False):
        ml = max(len(p), len(t))
        if ml == 0:
            continue
        total_chars += ml
        for a, b in zip(p, t, strict=False):
            if a == b:
                matched += 1
    return matched / total_chars if total_chars else 0.0
