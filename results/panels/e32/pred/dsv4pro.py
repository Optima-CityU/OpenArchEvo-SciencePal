# -*- coding: utf-8 -*-
"""Cheap predictor of 30-epoch validation PPL from a 2-epoch run + program source.

Mechanistic hypothesis:
    The gap between train and validation PPL at epoch 2 (train_ppl_ep2 / val_ppl_ep2)
    measures how much optimization headroom remains. A model whose train PPL is still
    far above its val PPL at epoch 2 is still fitting hard and will keep improving by
    epoch 30, whereas a model already close to its val PPL has plateaued. Combining
    this headroom with the absolute val level predicts 30-epoch quality better than
    val_ppl_ep2 alone. Removing the short-convolution (local-context) path is a known
    divergence risk, so architectures without it get a small penalty.

Formula (1 fitted constant, pen=5):
    ratio   = train_ppl_ep2 / val_ppl_ep2
    score   = val_ppl_ep2 / ratio + pen * (1 - short_conv)
            = val_ppl_ep2**2 / train_ppl_ep2 + 5 * (1 - short_conv)
    lower score = predicted better (lower) 30-epoch PPL.
"""
import re


def _has_short_conv(code: str) -> bool:
    """Detect whether the program keeps the short-convolution (local context) path."""
    m = re.search(r"use_short_conv\s*:\s*bool\s*=\s*(\w+)", code)
    if m:
        return m.group(1) == "True"
    return "ShortConvolution(" in code


def score(row: dict, code: str) -> float:
    """Return a lower-is-better prediction of the 30-epoch validation PPL.

    row: dict with test.csv columns as strings (val_ppl_ep2, train_ppl_ep2, ...).
    code: the program source defining the SpikingFLABlock.
    """
    v2 = float(row["val_ppl_ep2"])
    t2 = float(row["train_ppl_ep2"])
    ratio = t2 / v2
    pen = 5.0
    sc = 1.0 if _has_short_conv(code) else 0.0
    return v2 / ratio + pen * (1.0 - sc)
