"""Save side-by-side (orig | GT | pred) panels for visual report."""
from __future__ import annotations

import argparse
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
from .models.clipseg_wrapper import CLIPSegFT
from .utils import device_auto, load_config


def overlay(img_rgb: np.ndarray, mask: np.ndarray, color=(255, 60, 60), alpha: float = 0.5) -> np.ndarray:
    out = img_rgb.copy()
    color_layer = np.zeros_like(out)
    color_layer[mask > 0] = color
    return cv2.addWeighted(out, 1.0, color_layer, alpha, 0)


def panel(orig: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    gt_rgb = cv2.cvtColor((gt * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB) if gt.dtype != np.uint8 else cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB)
    pred_rgb = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)
    h = orig.shape[0]
    sep = np.full((h, 4, 3), 255, dtype=np.uint8)
    return np.concatenate([orig, sep, gt_rgb, sep, pred_rgb], axis=1)


def _normalize(img_u8: np.ndarray) -> torch.Tensor:
    img = img_u8.astype(np.float32) / 255.0
    img = (img - np.array(CLIP_MEAN, dtype=np.float32)) / np.array(CLIP_STD, dtype=np.float32)
    return torch.from_numpy(np.transpose(img, (2, 0, 1)))


def _resolve_task(cfg: dict, task_name: str) -> dict:
    if "tasks" not in cfg:
        raise ValueError("config must define `tasks` (unified mode is the only supported mode)")
    for t in cfg["tasks"]:
        if t["name"] == task_name:
            return t
    raise ValueError(f"task {task_name} not found in cfg.tasks")


def visualize(cfg_path: str, ckpt_path: str, split: str, out_dir: str, task: str,
              n: int = 4, threshold: float | None = None):
    cfg = load_config(cfg_path)
    device = device_auto()
    model = CLIPSegFT(pretrained=cfg["model"]["pretrained"]).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    model.eval()

    t = _resolve_task(cfg, task)
    size = cfg["data"]["input_size"]
    if threshold is None:
        threshold = cfg["inference"].get("threshold", 0.5)
    prompts = t["prompts"]
    img_dir = Path(t["dataset_dir"]) / "images" / split
    msk_dir = Path(t["dataset_dir"]) / "masks" / split
    out_dir_p = Path(out_dir); out_dir_p.mkdir(parents=True, exist_ok=True)

    paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg")])
    paths = paths[:n]
    for p in paths:
        gt_p = msk_dir / (p.stem + ".png")
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        gt = (cv2.imread(str(gt_p), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8) * 255 if gt_p.exists() else np.zeros(img.shape[:2], dtype=np.uint8)
        img_pad, _, meta = letterbox_pair(img, None, size)
        x = _normalize(img_pad).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = model.predict(x, [list(prompts)], out_size=(size, size), tta=True, multi_prompt_avg=True)
        mask_pad = (prob.squeeze().cpu().numpy() > threshold).astype(np.uint8)
        pred = (unletterbox_mask(mask_pad, meta) * 255).astype(np.uint8)
        pan = panel(img, gt, pred)
        cv2.imwrite(str(out_dir_p / f"{p.stem}__panel.png"), cv2.cvtColor(pan, cv2.COLOR_RGB2BGR))
        ovl = overlay(img, pred)
        cv2.imwrite(str(out_dir_p / f"{p.stem}__overlay.png"), cv2.cvtColor(ovl, cv2.COLOR_RGB2BGR))
    print(f"[viz] wrote {len(paths)} panels to {out_dir_p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--task", required=True, help="task name from cfg.tasks (e.g. crack, taping)")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()
    visualize(args.config, args.ckpt, args.split, args.out, args.task,
              n=args.n, threshold=args.threshold)


if __name__ == "__main__":
    main()
