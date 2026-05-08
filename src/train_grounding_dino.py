"""Fine-tune Grounding DINO on (image, prompt, bboxes) derived from the
unified task masks. Backbone (Swin) frozen; transformer + heads trained.

Usage:
    python -m src.train_grounding_dino --config configs/grounded_sam.yaml
"""
from __future__ import annotations

import argparse
import math
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor, GroundingDinoForObjectDetection

from .data.grounded_sam_dataset import GDTrainDataset, gd_collate
from .utils import device_auto, load_config, save_json, set_seed


def cosine_lr(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * p))


def freeze_backbone(model: GroundingDinoForObjectDetection):
    n_train, n_total = 0, 0
    for name, p in model.named_parameters():
        n_total += p.numel()
        if "backbone" in name or "text_backbone" in name:
            p.requires_grad = False
        if p.requires_grad:
            n_train += p.numel()
    return n_train, n_total


def train(cfg_path: str):
    cfg = load_config(cfg_path)
    if "tasks" not in cfg:
        raise ValueError("config must define `tasks`")
    set_seed(cfg.get("seed", 42))
    device = device_auto()

    gs = cfg.get("grounded_sam", {})
    gd_pre = gs.get("gd_pretrained", "IDEA-Research/grounding-dino-tiny")
    train_cfg = cfg.get("grounded_sam_train", {}).get("gd", {})
    epochs = train_cfg.get("epochs", 6)
    batch_size = train_cfg.get("batch_size", 4)
    lr = train_cfg.get("lr", 1.0e-5)
    weight_decay = train_cfg.get("weight_decay", 1.0e-4)
    warmup = train_cfg.get("warmup_steps", 200)
    log_every = train_cfg.get("log_every", 20)
    ckpt_dir = Path(train_cfg.get("ckpt_dir", "outputs/checkpoints/gd_finetuned"))
    log_dir = Path(train_cfg.get("log_dir", "outputs/logs/gd_finetuned"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    processor = AutoProcessor.from_pretrained(gd_pre)
    model = GroundingDinoForObjectDetection.from_pretrained(gd_pre).to(device)
    n_train, n_total = freeze_backbone(model)
    print(f"[gd] params total={n_total/1e6:.2f}M trainable={n_train/1e6:.2f}M")

    train_ds = GDTrainDataset(cfg["tasks"], split="train")
    print(f"[gd] train samples={len(train_ds)}")
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=cfg["data"].get("num_workers", 4),
        collate_fn=partial(gd_collate, processor=processor),
        drop_last=True,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    total_steps = epochs * max(1, len(train_loader))
    global_step = 0
    t0 = time.time()
    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        ep_loss = 0.0
        n_batches = 0
        for it, batch in enumerate(train_loader):
            if not batch:
                continue
            cur_lr = cosine_lr(global_step, total_steps, warmup, lr)
            for g in optim.param_groups:
                g["lr"] = cur_lr
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            if "labels" in batch:
                batch["labels"] = [{k: v.to(device) for k, v in lb.items()} for lb in batch["labels"]]
            out = model(**batch)
            loss = out.loss
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            ep_loss += loss.item()
            n_batches += 1
            if global_step % log_every == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", cur_lr, global_step)
                print(f"ep {epoch} it {it} step {global_step} loss {loss.item():.4f} lr {cur_lr:.2e}")
            global_step += 1
        avg = ep_loss / max(1, n_batches)
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
    print(f"[done] gd train {elapsed/60:.1f} min best_loss={best_loss:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
