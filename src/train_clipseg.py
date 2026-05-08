"""CLIPSeg unified fine-tune training loop.

One model, multiple text prompts. Cross-task negative sampling forces
prompt-conditioning so a prompt swap actually changes the output.

Usage:
    python -m src.train_clipseg --config configs/clipseg.yaml
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from .data.dataset import PromptedSegDataset, UnifiedPromptedSegDataset, collate_prompted
from .data.transforms import build_train_transforms, build_val_transforms
from .losses import TaskAwareBCEDiceLoss, build_loss
from .metrics import MetricAccumulator
from .models.clipseg_wrapper import CLIPSegFT
from .utils import count_params, device_auto, load_config, save_json, set_seed


def cosine_lr(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * p))


def evaluate(model: CLIPSegFT, loader: DataLoader, eval_prompt: str, device, threshold: float) -> dict:
    model.eval()
    acc = MetricAccumulator()
    val_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            m = batch["mask"].to(device, non_blocking=True)
            prompts = [eval_prompt] * x.size(0)
            logits = model(x, prompts)
            if logits.shape[-2:] != m.shape[-2:]:
                logits = F.interpolate(logits, size=m.shape[-2:], mode="bilinear", align_corners=False)
            prob = torch.sigmoid(logits).squeeze(1)
            pred = (prob > threshold)
            for i in range(x.size(0)):
                acc.update(pred[i].cpu(), (m[i] > 0.5).cpu())
            val_loss += F.binary_cross_entropy(prob.clamp(1e-6, 1 - 1e-6), m.float()).item()
            n_batches += 1
    summary = acc.summary()
    summary["val_loss"] = val_loss / max(1, n_batches)
    return summary


def evaluate_unified(model: CLIPSegFT, tasks: list, val_tf, cfg, device, threshold: float) -> dict:
    """Per-task val on the union; returns dict with per-task metrics + means."""
    out: dict = {}
    dices, ious = [], []
    for t in tasks:
        ds = PromptedSegDataset(t["dataset_dir"], "val", [t["eval_prompt"]], val_tf,
                                prompt_sampling="first")
        loader = DataLoader(
            ds, batch_size=max(1, cfg["train"]["batch_size"] // 2), shuffle=False,
            num_workers=cfg["data"]["num_workers"], pin_memory=cfg["data"]["pin_memory"],
            collate_fn=collate_prompted,
        )
        s = evaluate(model, loader, t["eval_prompt"], device, threshold)
        for k, v in s.items():
            out[f"{t['name']}/{k}"] = v
        dices.append(s.get("dice", 0.0))
        ious.append(s.get("miou", 0.0))
    out["dice_mean"] = sum(dices) / max(1, len(dices))
    out["iou_mean"] = sum(ious) / max(1, len(ious))
    return out


def train(cfg_path: str):
    cfg = load_config(cfg_path)
    if "tasks" not in cfg:
        raise ValueError("config must define `tasks` (unified multi-task is the only supported mode)")
    set_seed(cfg.get("seed", 42))
    device = device_auto()

    ckpt_dir = Path(cfg["train"]["ckpt_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg["train"]["log_dir"]); log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    train_tf = build_train_transforms(cfg["augmentation"])
    val_tf = build_val_transforms(cfg["augmentation"])
    tasks = cfg["tasks"]
    neg_prob = cfg["train"].get("neg_prob", 0.3)
    train_ds = UnifiedPromptedSegDataset(tasks, "train", train_tf, neg_prob=neg_prob)
    print(f"[unified] train samples={len(train_ds)} tasks={[t['name'] for t in tasks]} neg_prob={neg_prob}")

    sampler = None
    use_balanced = cfg["train"].get("balanced_sampler", False)
    if use_balanced:
        # Inverse-frequency sample weights per source task -> ~50/50 per batch.
        from collections import Counter
        task_idx_per_sample = [s[2] for s in train_ds.samples]
        task_counts = Counter(task_idx_per_sample)
        weights = [1.0 / task_counts[ti] for ti in task_idx_per_sample]
        sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
        n_per = {tasks[k]["name"]: v for k, v in task_counts.items()}
        print(f"[unified] balanced sampler ON — task counts={n_per}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"],
        shuffle=(sampler is None), sampler=sampler,
        num_workers=cfg["data"]["num_workers"], pin_memory=cfg["data"]["pin_memory"],
        collate_fn=collate_prompted, drop_last=True, persistent_workers=True,
    )

    mcfg = cfg["model"]
    model = CLIPSegFT(
        pretrained=mcfg["pretrained"],
        freeze_clip=mcfg.get("freeze_clip", True),
        unfreeze_decoder=mcfg.get("unfreeze_decoder", True),
        unfreeze_film=mcfg.get("unfreeze_film", True),
        unfreeze_visual_adapter=mcfg.get("unfreeze_visual_adapter", False),
    ).to(device)
    total, train_n = count_params(model)
    print(f"[model] total={total/1e6:.2f}M trainable={train_n/1e6:.2f}M")

    loss_fn = build_loss(cfg["loss"]).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    use_amp = cfg["train"].get("amp", True) and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    epochs = cfg["train"]["epochs"]
    steps_per_epoch = max(1, len(train_loader))
    total_steps = epochs * steps_per_epoch
    warmup = cfg["train"].get("warmup_steps", 200)
    base_lr = cfg["train"]["lr"]

    best_metric = -1.0
    monitor_key = cfg["train"].get("monitor", "val/dice_mean").split("/")[-1]
    patience = cfg["train"].get("early_stop_patience", 8)
    bad = 0
    threshold = cfg["inference"].get("threshold", 0.5)

    global_step = 0
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        if hasattr(loss_fn, "set_epoch"):
            loss_fn.set_epoch(epoch)
        for it, batch in enumerate(train_loader):
            lr = cosine_lr(global_step, total_steps, warmup, base_lr)
            for g in optim.param_groups:
                g["lr"] = lr

            x = batch["image"].to(device, non_blocking=True)
            m = batch["mask"].to(device, non_blocking=True)
            prompts = batch["prompt"]

            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                dtype=amp_dtype, enabled=use_amp):
                logits = model(x, prompts)
                if logits.shape[-2:] != m.shape[-2:]:
                    logits = F.interpolate(logits, size=m.shape[-2:], mode="bilinear", align_corners=False)
                if isinstance(loss_fn, TaskAwareBCEDiceLoss):
                    loss = loss_fn(logits, m, tasks=batch.get("task"))
                else:
                    loss = loss_fn(logits, m)

            optim.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optim.step()

            if global_step % cfg["train"].get("log_every", 20) == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", lr, global_step)
                print(f"ep {epoch} it {it}/{steps_per_epoch} step {global_step} loss {loss.item():.4f} lr {lr:.2e}")
            global_step += 1

        val = evaluate_unified(model, tasks, val_tf, cfg, device, threshold)
        for k, v in val.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        print(f"[val] epoch {epoch} {val}")

        cur = val.get(monitor_key, val.get("dice_mean", 0.0))
        if cur > best_metric:
            best_metric = cur
            bad = 0
            ckpt = {
                "model": model.state_dict(),
                "epoch": epoch,
                "best_metric": best_metric,
                "config": cfg,
            }
            torch.save(ckpt, ckpt_dir / "best.pt")
            save_json(val, ckpt_dir / "best_val.json")
            print(f"[ckpt] saved best.pt at epoch {epoch} ({monitor_key}={cur:.4f})")
        else:
            bad += 1
            if bad >= patience:
                print(f"[early-stop] no improvement for {patience} epochs.")
                break

    elapsed = time.time() - t0
    save_json({"train_seconds": elapsed, "best_metric": best_metric}, ckpt_dir / "train_summary.json")
    writer.close()
    print(f"[done] train time {elapsed/60:.1f} min best {monitor_key}={best_metric:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
