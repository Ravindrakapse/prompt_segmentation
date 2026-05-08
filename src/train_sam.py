"""Fine-tune SAM mask decoder on (image, GT bboxes, GT mask) tuples.

Bboxes come from connected-components on the GT mask itself — SAM learns to
refine bbox-prompted masks within the drywall domain. Image encoder and
prompt encoder are frozen; only the small mask decoder (~4M params) trains.

Usage:
    python -m src.train_sam --config configs/grounded_sam.yaml
"""
from __future__ import annotations

import argparse
import math
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor, SamModel

from .data.grounded_sam_dataset import SAMTrainDataset, sam_collate
from .utils import device_auto, load_config, save_json, set_seed


def cosine_lr(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * p))


def freeze_except_mask_decoder(model: SamModel):
    n_train, n_total = 0, 0
    for name, p in model.named_parameters():
        n_total += p.numel()
        if "mask_decoder" not in name:
            p.requires_grad = False
        if p.requires_grad:
            n_train += p.numel()
    return n_train, n_total


def _bce_dice(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """logits, target both (B, H, W) float."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(-1, -2))
    denom = prob.sum(dim=(-1, -2)) + target.sum(dim=(-1, -2))
    dice = 1 - ((2 * inter + eps) / (denom + eps)).mean()
    return 0.5 * bce + 0.5 * dice


def train(cfg_path: str):
    cfg = load_config(cfg_path)
    if "tasks" not in cfg:
        raise ValueError("config must define `tasks`")
    set_seed(cfg.get("seed", 42))
    device = device_auto()

    gs = cfg.get("grounded_sam", {})
    sam_pre = gs.get("sam_pretrained", "facebook/sam-vit-base")
    train_cfg = cfg.get("grounded_sam_train", {}).get("sam", {})
    epochs = train_cfg.get("epochs", 6)
    batch_size = train_cfg.get("batch_size", 2)
    lr = train_cfg.get("lr", 1.0e-4)
    weight_decay = train_cfg.get("weight_decay", 1.0e-4)
    warmup = train_cfg.get("warmup_steps", 200)
    log_every = train_cfg.get("log_every", 20)
    ckpt_dir = Path(train_cfg.get("ckpt_dir", "outputs/checkpoints/sam_finetuned"))
    log_dir = Path(train_cfg.get("log_dir", "outputs/logs/sam_finetuned"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    processor = AutoProcessor.from_pretrained(sam_pre)
    model = SamModel.from_pretrained(sam_pre).to(device)
    n_train, n_total = freeze_except_mask_decoder(model)
    print(f"[sam] params total={n_total/1e6:.2f}M trainable={n_train/1e6:.2f}M (mask decoder only)")

    ds = SAMTrainDataset(cfg["tasks"], split="train")
    print(f"[sam] train samples={len(ds)}")
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=cfg["data"].get("num_workers", 4),
        collate_fn=partial(sam_collate, processor=processor),
        drop_last=True,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    total_steps = epochs * max(1, len(loader))
    best_loss = float("inf")
    global_step = 0
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        ep_loss = 0.0
        n_b = 0
        for it, batch in enumerate(loader):
            if not batch:
                continue
            cur_lr = cosine_lr(global_step, total_steps, warmup, lr)
            for g in optim.param_groups:
                g["lr"] = cur_lr
            pixel_values = batch["pixel_values"].to(device)
            input_boxes = batch["input_boxes"].to(device)
            gt_masks = batch["gt_masks"]   # list of (H, W) float

            # Frozen vision encoder: cache embeddings under no_grad so the
            # backward pass only touches the (small) mask decoder.
            with torch.no_grad():
                image_embeddings = model.get_image_embeddings(pixel_values)
            out = model(image_embeddings=image_embeddings, input_boxes=input_boxes,
                        multimask_output=False)
            # post-process predicted low-res masks -> per-image mask logits at orig size.
            # The SAM image processor returns per-image masks already binarized via
            # threshold; for training we need raw logits at original size. Use
            # nn.functional.interpolate from low-res mask logits directly.
            pred_low = out.pred_masks            # (B, N, 1, h, w) sigmoid logits
            if pred_low.dim() == 5:
                pred_low = pred_low.squeeze(2)   # (B, N, h, w)
            losses = []
            box_counts = batch["box_counts"]
            for i, gt in enumerate(gt_masks):
                gt = gt.to(device)
                k = box_counts[i]
                logits_i = pred_low[i, :k].mean(dim=0, keepdim=True)   # (1, h, w)
                up = F.interpolate(logits_i.unsqueeze(0), size=gt.shape, mode="bilinear",
                                   align_corners=False).squeeze(0).squeeze(0)
                losses.append(_bce_dice(up, gt))
            loss = torch.stack(losses).mean()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            ep_loss += loss.item()
            n_b += 1
            if global_step % log_every == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", cur_lr, global_step)
                print(f"ep {epoch} it {it} step {global_step} loss {loss.item():.4f} lr {cur_lr:.2e}")
            global_step += 1
        avg = ep_loss / max(1, n_b)
        writer.add_scalar("epoch/train_loss", avg, epoch)
        print(f"[ep {epoch}] avg_loss={avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            model.save_pretrained(ckpt_dir / "best")
            processor.save_pretrained(ckpt_dir / "best")
            save_json({"epoch": epoch, "loss": avg}, ckpt_dir / "best_summary.json")
            print(f"[ckpt] saved best to {ckpt_dir/'best'} (loss {avg:.4f})")

    elapsed = time.time() - t0
    save_json({"train_seconds": elapsed, "best_loss": best_loss},
              ckpt_dir / "train_summary.json")
    writer.close()
    print(f"[done] sam train {elapsed/60:.1f} min best_loss={best_loss:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
