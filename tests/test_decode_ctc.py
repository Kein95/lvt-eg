"""Greedy CTC decoding correctness."""

import torch

from lvt_eg.common.ctc_decode import decode_ctc, encode_label


def _make_log_probs(seq: list[int], num_classes: int) -> torch.Tensor:
    """Build a synthetic log-prob tensor that argmax-decodes to `seq`."""
    T = len(seq)
    logits = torch.full((1, T, num_classes), -10.0)
    for t, idx in enumerate(seq):
        logits[0, t, idx] = 0.0
    return logits


def test_collapse_and_blank_strip():
    # Alphabet: blank=0, A=1, B=2
    # Raw sequence: [A A blank B B] -> "AB"
    idx2char = {1: "A", 2: "B"}
    log_probs = _make_log_probs([1, 1, 0, 2, 2], num_classes=3)
    assert decode_ctc(log_probs, idx2char) == ["AB"]


def test_empty_sequence():
    idx2char = {1: "A"}
    log_probs = _make_log_probs([0, 0, 0], num_classes=2)
    assert decode_ctc(log_probs, idx2char) == [""]


def test_encode_label_padding():
    char2idx = {"A": 1, "B": 2, "C": 3}
    padded, length = encode_label("AB", char2idx, max_len=5)
    assert padded == [1, 2, 0, 0, 0]
    assert length == 2
