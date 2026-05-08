"""Segmentation metrics: IoU, Dice, Precision, Recall (binary).

All operate on tensors of identical shape, post-thresholded for hard metrics."""
from __future__ import annotations

import torch


@torch.no_grad()
def confusion_counts(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7):
    pred = pred.bool()
    target = target.bool()
    tp = (pred & target).sum().float()
    fp = (pred & ~target).sum().float()
    fn = (~pred & target).sum().float()
    tn = (~pred & ~target).sum().float()
    return tp, fp, fn, tn


@torch.no_grad()
def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> float:
    tp, fp, fn, _ = confusion_counts(pred, target)
    return float((tp + eps) / (tp + fp + fn + eps))


@torch.no_grad()
def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> float:
    tp, fp, fn, _ = confusion_counts(pred, target)
    return float((2 * tp + eps) / (2 * tp + fp + fn + eps))


@torch.no_grad()
def precision_recall(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> tuple[float, float]:
    tp, fp, fn, _ = confusion_counts(pred, target)
    p = float((tp + eps) / (tp + fp + eps))
    r = float((tp + eps) / (tp + fn + eps))
    return p, r


@torch.no_grad()
def betti_numbers(mask: torch.Tensor) -> tuple[int, int]:
    """Compute (beta_0, beta_1) of a binary 2D mask via scipy.ndimage.label.

    beta_0 = #connected components of foreground.
    beta_1 = #handles. For 2D binary: beta_1 = beta_0(bg_inside) — i.e., number
    of background components that are *enclosed* by foreground (= #holes).
    """
    import numpy as np
    from scipy.ndimage import label
    m = mask.detach().cpu().numpy().astype(bool)
    if m.ndim == 3:
        m = m[0]
    b0 = int(label(m)[1])
    # holes: bg components that don't touch border
    bg = ~m
    lab_bg, n_bg = label(bg)
    if n_bg == 0:
        return b0, 0
    border_labels = set()
    border_labels.update(lab_bg[0, :].tolist())
    border_labels.update(lab_bg[-1, :].tolist())
    border_labels.update(lab_bg[:, 0].tolist())
    border_labels.update(lab_bg[:, -1].tolist())
    border_labels.discard(0)
    b1 = n_bg - len(border_labels)
    return b0, b1


@torch.no_grad()
def betti_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """|beta_0(pred) - beta_0(gt)| + |beta_1(pred) - beta_1(gt)|."""
    p0, p1 = betti_numbers(pred)
    g0, g1 = betti_numbers(target)
    return float(abs(p0 - g0) + abs(p1 - g1))


class MetricAccumulator:
    """Accumulates per-image metrics across a dataset (mean over images)."""

    def __init__(self, compute_betti: bool = False):
        self.iou: list[float] = []
        self.dice: list[float] = []
        self.precision: list[float] = []
        self.recall: list[float] = []
        self.betti: list[float] = []
        self.compute_betti = compute_betti

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        self.iou.append(iou_score(pred, target))
        self.dice.append(dice_score(pred, target))
        p, r = precision_recall(pred, target)
        self.precision.append(p)
        self.recall.append(r)
        if self.compute_betti:
            self.betti.append(betti_error(pred, target))

    def summary(self) -> dict[str, float]:
        def mean(xs):
            return float(sum(xs) / max(1, len(xs)))
        out = {
            "miou": mean(self.iou),
            "dice": mean(self.dice),
            "precision": mean(self.precision),
            "recall": mean(self.recall),
            "n": len(self.iou),
        }
        if self.compute_betti and self.betti:
            out["betti_err"] = mean(self.betti)
        return out
