"""Cheap smoke tests — no model load, no data dependency."""
from __future__ import annotations

import numpy as np
import torch

from src.losses import BCEDiceLoss, DiceLoss, FocalBCELoss, TverskyLoss, build_loss
from src.metrics import dice_score, iou_score, precision_recall
from src.utils import slugify_prompt


def test_slug():
    assert slugify_prompt("Segment Crack!") == "segment_crack"
    assert slugify_prompt("segment taping area") == "segment_taping_area"


def test_metrics_perfect():
    p = torch.zeros(1, 4, 4)
    t = torch.zeros(1, 4, 4)
    p[0, 1:3, 1:3] = 1
    t[0, 1:3, 1:3] = 1
    assert iou_score(p.bool(), t.bool()) == 1.0
    assert dice_score(p.bool(), t.bool()) == 1.0
    pr, rc = precision_recall(p.bool(), t.bool())
    assert pr == 1.0 and rc == 1.0


def test_losses_run():
    logits = torch.randn(2, 1, 8, 8)
    target = (torch.rand(2, 8, 8) > 0.7).float()
    for L in (DiceLoss(), TverskyLoss(), FocalBCELoss(), BCEDiceLoss()):
        v = L(logits, target)
        assert torch.isfinite(v) and v.item() >= 0.0


def test_build_loss():
    cfg = {"type": "bce_dice", "bce_weight": 0.5, "dice_weight": 0.5, "pos_weight": 5.0}
    L = build_loss(cfg)
    v = L(torch.randn(1, 1, 4, 4), torch.zeros(1, 4, 4))
    assert torch.isfinite(v)


if __name__ == "__main__":
    test_slug(); test_metrics_perfect(); test_losses_run(); test_build_loss()
    print("smoke tests passed")
