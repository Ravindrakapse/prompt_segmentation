"""Compute IoU, Dice, Precision, Recall on a split given a model checkpoint.

Usage:
    python -m src.eval --config configs/clipseg_crack.yaml \\
        --ckpt outputs/checkpoints/crack/best.pt --split test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .data.transforms import (
    CLIP_MEAN,
    CLIP_STD,
    letterbox_pair,
    unletterbox_mask,
)
from .metrics import MetricAccumulator
from .models.clipseg_wrapper import CLIPSegFT
from .utils import device_auto, load_config, save_json, set_seed


def _normalize_image(img_u8: np.ndarray) -> torch.Tensor:
    img = img_u8.astype(np.float32) / 255.0
    mean = np.array(CLIP_MEAN, dtype=np.float32)
    std = np.array(CLIP_STD, dtype=np.float32)
    img = (img - mean) / std
    return torch.from_numpy(np.transpose(img, (2, 0, 1)))


def _resolve_task(cfg: dict, task_name: str) -> dict:
    if "tasks" not in cfg:
        raise ValueError("config must define `tasks` (unified mode is the only supported mode)")
    for t in cfg["tasks"]:
        if t["name"] == task_name:
            return t
    raise ValueError(f"task {task_name} not found in cfg.tasks")


def evaluate_split(cfg_path: str, ckpt_path: str, split: str, task: str,
                   threshold: float | None = None, out_path: str | None = None) -> dict:
    cfg = load_config(cfg_path)
    set_seed(cfg.get("seed", 42))
    device = device_auto()

    model = CLIPSegFT(pretrained=cfg["model"]["pretrained"]).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    t = _resolve_task(cfg, task)
    size = cfg["data"]["input_size"]
    if threshold is None:
        threshold = cfg["inference"].get("threshold", 0.5)
    use_tta = cfg["inference"].get("tta", True)
    use_multi = cfg["inference"].get("multi_prompt_avg", True)
    prompts = t["prompts"]
    eval_prompt = t["eval_prompt"]

    img_dir = Path(t["dataset_dir"]) / "images" / split
    msk_dir = Path(t["dataset_dir"]) / "masks" / split

    acc = MetricAccumulator()
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
            continue
        gt_p = msk_dir / (p.stem + ".png")
        if not gt_p.exists():
            continue
        img_bgr = cv2.imread(str(p))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt = (cv2.imread(str(gt_p), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        img_pad, _, meta = letterbox_pair(img_rgb, None, size)
        x = _normalize_image(img_pad).unsqueeze(0).to(device)

        with torch.no_grad():
            if use_multi:
                prob = model.predict(x, [list(prompts)], out_size=(size, size),
                                     tta=use_tta, multi_prompt_avg=True)
            else:
                prob = model.predict(x, [eval_prompt], out_size=(size, size), tta=use_tta)
        prob_np = prob.squeeze().cpu().numpy()
        mask_pad = (prob_np > threshold).astype(np.uint8)
        pred = unletterbox_mask(mask_pad, meta)
        acc.update(torch.from_numpy(pred.astype(bool)),
                   torch.from_numpy(gt.astype(bool)))

    summary = acc.summary()
    summary["threshold"] = threshold
    summary["split"] = split
    print(json.dumps(summary, indent=2))
    if out_path:
        save_json(summary, out_path)
    return summary


def sweep_threshold(cfg_path: str, ckpt_path: str, task: str, split: str = "val",
                    grid=(0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6)) -> dict:
    """Find best threshold by val Dice."""
    best = {"dice": -1.0}
    for t in grid:
        s = evaluate_split(cfg_path, ckpt_path, split, task=task, threshold=t)
        if s["dice"] > best.get("dice", -1.0):
            best = s
    print("[sweep] best:", best)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", required=True, help="task name from cfg.tasks (e.g. crack, taping)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sweep", action="store_true", help="sweep threshold on val first")
    args = ap.parse_args()
    if args.sweep:
        best = sweep_threshold(args.config, args.ckpt, args.task, "val")
        evaluate_split(args.config, args.ckpt, args.split, args.task,
                       threshold=best["threshold"], out_path=args.out)
    else:
        evaluate_split(args.config, args.ckpt, args.split, args.task,
                       threshold=args.threshold, out_path=args.out)


if __name__ == "__main__":
    main()
