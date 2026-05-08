"""Segmentation losses for thin-structure / class-imbalanced binary masks.

All losses expect logits of shape (B,1,H,W) or (B,H,W) and float targets in [0,1]."""
from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gudhi  # noqa: F401
    _HAS_GUDHI = True
except Exception:
    _HAS_GUDHI = False


def _flatten_logits_target(logits: torch.Tensor, target: torch.Tensor):
    if logits.dim() == 4 and logits.shape[1] == 1:
        logits = logits.squeeze(1)
    if target.dim() == 4 and target.shape[1] == 1:
        target = target.squeeze(1)
    return logits, target.float()


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits, target = _flatten_logits_target(logits, target)
        prob = torch.sigmoid(logits)
        # per-image dice
        dims = (1, 2)
        inter = (prob * target).sum(dims)
        denom = prob.sum(dims) + target.sum(dims)
        dice = (2 * inter + self.eps) / (denom + self.eps)
        return 1 - dice.mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6):
        super().__init__()
        self.alpha, self.beta, self.eps = alpha, beta, eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits, target = _flatten_logits_target(logits, target)
        p = torch.sigmoid(logits)
        dims = (1, 2)
        tp = (p * target).sum(dims)
        fp = (p * (1 - target)).sum(dims)
        fn = ((1 - p) * target).sum(dims)
        t = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return 1 - t.mean()


class FocalBCELoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: float | None = None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits, target = _flatten_logits_target(logits, target)
        pw = None
        if self.pos_weight is not None:
            pw = torch.tensor([self.pos_weight], device=logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pw)
        p = torch.sigmoid(logits)
        pt = p * target + (1 - p) * (1 - target)
        focal = ((1 - pt) ** self.gamma) * bce
        return focal.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, pos_weight: float | None = None):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = pos_weight
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits, target = _flatten_logits_target(logits, target)
        pw = None
        if self.pos_weight is not None:
            pw = torch.tensor([self.pos_weight], device=logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
        dice = self.dice(logits, target)
        return self.bce_weight * bce + self.dice_weight * dice


class TaskAwareBCEDiceLoss(nn.Module):
    """BCE+Dice with per-task `pos_weight`. Forward signature:

        loss = fn(logits, target, tasks=[task_name_per_sample])

    Per-image BCE uses that image's pos_weight (looked up by task name).
    Dice is unweighted. Final loss = bce_weight * mean(per-image BCE) +
    dice_weight * Dice. Backwards-compatible with BCEDiceLoss when `tasks`
    is omitted — falls back to a single shared pos_weight `default_pos_weight`.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5,
                 pos_weights: dict[str, float] | None = None,
                 default_pos_weight: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weights = pos_weights or {}
        self.default_pos_weight = default_pos_weight
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                tasks: list[str] | None = None) -> torch.Tensor:
        logits, target = _flatten_logits_target(logits, target)
        B = logits.shape[0]
        if tasks is None:
            tasks = [None] * B
        per_image_bce = []
        for i in range(B):
            pw = self.pos_weights.get(tasks[i], self.default_pos_weight) if tasks[i] is not None \
                 else self.default_pos_weight
            pw_t = torch.tensor([pw], device=logits.device)
            per_image_bce.append(
                F.binary_cross_entropy_with_logits(logits[i], target[i], pos_weight=pw_t)
            )
        bce = torch.stack(per_image_bce).mean()
        dice = self.dice(logits, target)
        return self.bce_weight * bce + self.dice_weight * dice


# ============================================================
# Soft clDice (Shit et al. 2021) — cheap topology proxy
# ============================================================

def _soft_erode(x):
    return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)


def _soft_dilate(x):
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def _soft_open(x):
    return _soft_dilate(_soft_erode(x))


def soft_skeletonize(x: torch.Tensor, iters: int = 3) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(1)
    img = x
    skel = F.relu(img - _soft_open(img))
    for _ in range(iters):
        img = _soft_erode(img)
        opened = _soft_open(img)
        delta = F.relu(img - opened)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice_loss(prob: torch.Tensor, target: torch.Tensor, iters: int = 3, eps: float = 1e-6) -> torch.Tensor:
    """1 - clDice (lower = better). Inputs in [0,1], shape (B,1,H,W) or (B,H,W)."""
    if prob.dim() == 3:
        prob = prob.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    target = target.float()
    sp = soft_skeletonize(prob, iters)
    st = soft_skeletonize(target, iters)
    tprec = ((sp * target).sum(dim=(1, 2, 3)) + eps) / (sp.sum(dim=(1, 2, 3)) + eps)
    tsens = ((st * prob).sum(dim=(1, 2, 3)) + eps) / (st.sum(dim=(1, 2, 3)) + eps)
    cl = 2.0 * tprec * tsens / (tprec + tsens + eps)
    return (1.0 - cl).mean()


# ============================================================
# Topological loss (TopoNet, Hu et al. 2019) — patch-based PH
# ============================================================

def _persistence_pairs(arr2d, dim: int):
    """Return list of (birth, death) for 2D cubical complex on values arr2d (HxW).

    Uses gudhi superlevel filtration via -arr (so high-prob = born early).
    """
    import gudhi
    cc = gudhi.CubicalComplex(top_dimensional_cells=arr2d.tolist())
    cc.compute_persistence()
    cof = cc.cofaces_of_persistence_pairs()
    # cof returns ([reg_pairs], [essential]) per dim. reg_pairs[d] is array of (birth_idx, death_idx).
    # Easier: use persistence_intervals_in_dimension which gives (birth, death) values directly.
    pairs = cc.persistence_intervals_in_dimension(dim)
    out = []
    for b, d in pairs:
        if d == float("inf"):
            d = 1.0  # cap essential to global max
        out.append((float(b), float(d)))
    return out


class TopologicalLoss(nn.Module):
    """Patch-based persistent-homology Wasserstein loss (TopoNet, Hu et al. 2019).

    Differentiable proxy: for each predicted patch, compute persistence pairs
    (b, d) over the *probability* map (using gudhi cubical complex). Match
    against gt persistence pairs greedily; for matched pair, push pred birth/
    death values toward gt values via mean-squared error on the *pixel values*
    at the critical locations of the prediction.

    NOTE: gudhi is non-differentiable. We get the gradient by using the pred
    value at the birth/death pixels (gudhi tells us only the values, not pixel
    indices, so we recover pixels by argmax-style lookup of the value). To
    keep it simple we re-run a vectorized lookup: for each persistence value
    v, find the pixel where prob == v (within eps) and use that pixel as the
    critical location. Backprop flows through `prob[crit_pixel]`.
    """

    def __init__(self, patch_size: int = 64, n_patches: int = 4, dims: tuple = (0, 1),
                 fg_min_frac: float = 0.005, eps: float = 1e-6):
        super().__init__()
        if not _HAS_GUDHI:
            raise RuntimeError("gudhi not installed — pip install gudhi")
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.dims = tuple(dims)
        self.fg_min_frac = fg_min_frac
        self.eps = eps

    @staticmethod
    def _crit_pixel_value(prob_patch: torch.Tensor, value: float, used: set) -> torch.Tensor:
        """Locate pixel in prob_patch with value closest to `value` (not yet used).
        Returns a 0-d tensor (the prob[pixel]) so gradients flow through it."""
        flat = prob_patch.flatten()
        diffs = (flat.detach() - value).abs()
        # mask out already-used pixels
        for idx in used:
            diffs[idx] = float("inf")
        i = int(diffs.argmin().item())
        used.add(i)
        return flat[i]

    def _topo_for_patch(self, prob_patch: torch.Tensor, target_patch: torch.Tensor) -> torch.Tensor:
        """prob_patch, target_patch: (H, W). Returns scalar topo loss for this patch."""
        # Convert to superlevel filtration via negation: gudhi computes sub-level on values.
        # A high prob = early birth → use 1 - prob for cubical complex.
        f = (1.0 - prob_patch).detach().cpu().numpy()
        g = (1.0 - target_patch.float()).detach().cpu().numpy()

        loss = prob_patch.new_zeros(())
        for d in self.dims:
            pairs_f = _persistence_pairs(f, d)
            pairs_g = _persistence_pairs(g, d)
            n_g = len(pairs_g)
            # Sort pred pairs by persistence (death-birth) descending; first n_g matched, rest pushed to diag.
            pairs_f_sorted = sorted(pairs_f, key=lambda p: -(p[1] - p[0]))
            used = set()
            # Matched: push toward gt
            for i, (bf, df) in enumerate(pairs_f_sorted[:n_g]):
                bg, dg = pairs_g[i] if i < n_g else (bf, df)
                # value in 1-prob space; we need to push prob value toward (1-bg), (1-dg)
                vb_pred = self._crit_pixel_value(1.0 - prob_patch, bf, used)
                vd_pred = self._crit_pixel_value(1.0 - prob_patch, df, used)
                loss = loss + (vb_pred - bg) ** 2 + (vd_pred - dg) ** 2
            # Unmatched (noise) → push to diagonal: (b+d)/2
            for (bf, df) in pairs_f_sorted[n_g:]:
                target_val = 0.5 * (bf + df)
                vb_pred = self._crit_pixel_value(1.0 - prob_patch, bf, used)
                vd_pred = self._crit_pixel_value(1.0 - prob_patch, df, used)
                loss = loss + (vb_pred - target_val) ** 2 + (vd_pred - target_val) ** 2
        return loss

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                per_image_weight: torch.Tensor | None = None) -> torch.Tensor:
        logits, target = _flatten_logits_target(logits, target)
        prob = torch.sigmoid(logits)  # (B, H, W)
        B, H, W = prob.shape
        ps = min(self.patch_size, H, W)
        if ps < 8:
            return prob.sum() * 0.0  # too small
        # cast to float32 for PH stability if AMP active
        prob32 = prob.float()
        target32 = target.float()
        total = prob.new_zeros(())
        n_used = 0
        for b in range(B):
            w_b = float(per_image_weight[b]) if per_image_weight is not None else 1.0
            if w_b <= 0:
                continue
            tgt = target32[b]
            # Try foreground-biased crops
            for _ in range(self.n_patches):
                # sample crop
                y = random.randint(0, H - ps)
                x = random.randint(0, W - ps)
                tp = tgt[y:y + ps, x:x + ps]
                if tp.sum() / (ps * ps) < self.fg_min_frac:
                    continue
                pp = prob32[b, y:y + ps, x:x + ps]
                total = total + w_b * self._topo_for_patch(pp, tp)
                n_used += 1
        if n_used == 0:
            return prob.sum() * 0.0
        return total / n_used


# ============================================================
# Combined loss: BCE + Dice + clDice + Topo (task-aware)
# ============================================================

class TaskAwareBCEDiceClDiceTopoLoss(TaskAwareBCEDiceLoss):
    """Extends TaskAwareBCEDice with optional clDice + Topo terms.

    `topo_per_task` and `cldice_per_task` are dicts mapping task name -> weight.
    Topo loss respects `topo_warmup_epochs`: returns 0 until `current_epoch >= warmup`.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5,
                 pos_weights=None, default_pos_weight=1.0,
                 cldice_weight=0.0, cldice_per_task=None, cldice_iters=3,
                 topo_weight=0.0, topo_per_task=None,
                 topo_patch_size=64, topo_n_patches=4, topo_dims=(0, 1),
                 topo_warmup_epochs=5, topo_fg_min_frac=0.005):
        super().__init__(bce_weight, dice_weight, pos_weights, default_pos_weight)
        self.cldice_weight = cldice_weight
        self.cldice_per_task = cldice_per_task or {}
        self.cldice_iters = cldice_iters
        self.topo_weight = topo_weight
        self.topo_per_task = topo_per_task or {}
        self.topo_warmup_epochs = topo_warmup_epochs
        self._cur_epoch = 0
        self.topo = (TopologicalLoss(patch_size=topo_patch_size, n_patches=topo_n_patches,
                                     dims=topo_dims, fg_min_frac=topo_fg_min_frac)
                     if topo_weight > 0 else None)

    def set_epoch(self, epoch: int):
        self._cur_epoch = int(epoch)

    def forward(self, logits, target, tasks=None):
        base = super().forward(logits, target, tasks=tasks)
        loss = base

        # clDice term
        if self.cldice_weight > 0:
            logits_f, target_f = _flatten_logits_target(logits, target)
            prob = torch.sigmoid(logits_f.float())
            if tasks is not None and self.cldice_per_task:
                # per-image weight, then scale clDice by mean weight (uniform within batch ok)
                w = torch.tensor([self.cldice_per_task.get(t, 1.0) for t in tasks],
                                 device=prob.device, dtype=prob.dtype)
                cl = soft_cldice_loss(prob, target_f, iters=self.cldice_iters)
                loss = loss + self.cldice_weight * float(w.mean()) * cl
            else:
                cl = soft_cldice_loss(prob, target_f, iters=self.cldice_iters)
                loss = loss + self.cldice_weight * cl

        # Topo term (after warmup)
        if self.topo is not None and self._cur_epoch >= self.topo_warmup_epochs:
            B = logits.shape[0]
            if tasks is not None and self.topo_per_task:
                pw = torch.tensor([self.topo_per_task.get(t, 0.0) for t in tasks],
                                  device=logits.device, dtype=torch.float32)
            else:
                pw = torch.ones(B, device=logits.device, dtype=torch.float32)
            tl = self.topo(logits, target, per_image_weight=pw)
            loss = loss + self.topo_weight * tl

        return loss


def build_loss(cfg: dict) -> nn.Module:
    t = cfg.get("type", "bce_dice").lower()
    if t == "bce":
        pw = cfg.get("pos_weight")
        return FocalBCELoss(gamma=0.0, pos_weight=pw)
    if t == "focal":
        return FocalBCELoss(gamma=cfg.get("gamma", 2.0), pos_weight=cfg.get("pos_weight"))
    if t == "dice":
        return DiceLoss()
    if t == "tversky":
        return TverskyLoss(alpha=cfg.get("alpha", 0.3), beta=cfg.get("beta", 0.7))
    if t == "bce_dice":
        return BCEDiceLoss(
            bce_weight=cfg.get("bce_weight", 0.5),
            dice_weight=cfg.get("dice_weight", 0.5),
            pos_weight=cfg.get("pos_weight"),
        )
    if t == "task_aware_bce_dice":
        return TaskAwareBCEDiceLoss(
            bce_weight=cfg.get("bce_weight", 0.5),
            dice_weight=cfg.get("dice_weight", 0.5),
            pos_weights=cfg.get("pos_weights_per_task"),
            default_pos_weight=cfg.get("pos_weight", 1.0),
        )
    if t == "task_aware_bce_dice_cldice_topo":
        return TaskAwareBCEDiceClDiceTopoLoss(
            bce_weight=cfg.get("bce_weight", 0.5),
            dice_weight=cfg.get("dice_weight", 0.5),
            pos_weights=cfg.get("pos_weights_per_task"),
            default_pos_weight=cfg.get("pos_weight", 1.0),
            cldice_weight=cfg.get("cldice_weight", 0.0),
            cldice_per_task=cfg.get("cldice_per_task"),
            cldice_iters=cfg.get("cldice_iters", 3),
            topo_weight=cfg.get("topo_weight", 0.0),
            topo_per_task=cfg.get("topo_per_task"),
            topo_patch_size=cfg.get("topo_patch_size", 64),
            topo_n_patches=cfg.get("topo_n_patches", 4),
            topo_dims=tuple(cfg.get("topo_dims", [0, 1])),
            topo_warmup_epochs=cfg.get("topo_warmup_epochs", 5),
            topo_fg_min_frac=cfg.get("topo_fg_min_frac", 0.005),
        )
    if t == "focal_dice":
        focal = FocalBCELoss(gamma=cfg.get("gamma", 2.0), pos_weight=cfg.get("pos_weight"))
        dice = DiceLoss()
        bw = cfg.get("bce_weight", 0.5)
        dw = cfg.get("dice_weight", 0.5)

        class FD(nn.Module):
            def forward(self, logits, target):
                return bw * focal(logits, target) + dw * dice(logits, target)

        return FD()
    raise ValueError(f"unknown loss type: {t}")
