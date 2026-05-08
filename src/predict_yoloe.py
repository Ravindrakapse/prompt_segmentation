"""YOLOE predict + eval + visualize per task. Auto-detects fine-tuned ckpt
at outputs/checkpoints/yoloe/best.pt; otherwise falls back to the
zero-shot weights specified in cfg.yoloe.weights.

Usage:
    python -m src.predict_yoloe --config configs/yoloe.yaml \\
        --task crack --split test \\
        --out-pred outputs/predictions_yoloe/crack \\
        --out-viz  outputs/visualizations_yoloe/crack \\
        --out-metrics outputs/predictions_yoloe/crack/metrics.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .metrics import MetricAccumulator
from .models.yoloe_wrapper import YOLOEPredictor
from .utils import device_auto, load_config, save_json, set_seed, slugify_prompt
from .visualize import overlay, panel


def _resolve_task_idx(cfg: dict, task_name: str) -> int:
    if "tasks" not in cfg:
        raise ValueError("config must define `tasks`")
    for i, t in enumerate(cfg["tasks"]):
        if t["name"] == task_name:
            return i
    raise ValueError(f"task {task_name} not found")


def run(cfg_path: str, task: str, split: str = "test",
        out_pred: str | None = None, out_viz: str | None = None,
        out_metrics: str | None = None, n_viz: int = 6,
        weights_override: str | None = None, conf: float | None = None,
        prompt_override: str | None = None):
    cfg = load_config(cfg_path)
    set_seed(cfg.get("seed", 42))
    device = device_auto()
    yolo_cfg = cfg.get("yoloe", {})
    train_cfg = cfg.get("yoloe_train", {})

    # auto-pick fine-tuned ckpt if exists
    ft_ckpt = Path(train_cfg.get("ckpt_dir", "outputs/checkpoints/yoloe")) / "best.pt"
    if weights_override:
        weights = weights_override
        is_ft = "finetuned" in weights or "best.pt" in weights
    elif ft_ckpt.exists():
        weights = str(ft_ckpt)
        is_ft = True
    else:
        weights = yolo_cfg.get("weights", "yoloe-11l-seg.pt")
        is_ft = False

    imgsz = yolo_cfg.get("imgsz", 640)
    conf = conf if conf is not None else yolo_cfg.get("conf", 0.25)

    task_idx = _resolve_task_idx(cfg, task)
    t = cfg["tasks"][task_idx]
    # If --prompt given, override that task's class name with the user's text;
    # YOLOEPredictor will encode it via MobileCLIP at load time. The other
    # task's name is left as-is so its class index stays valid.
    class_names = [tk["name"] for tk in cfg["tasks"]]
    if prompt_override:
        class_names[task_idx] = prompt_override
        print(f"[yoloe] prompt override for task {task}: '{prompt_override}'")
    eval_prompt = prompt_override or t["eval_prompt"]
    out_slug = slugify_prompt(eval_prompt)

    img_dir = Path(t["dataset_dir"]) / "images" / split
    msk_dir = Path(t["dataset_dir"]) / "masks" / split
    if not img_dir.exists():
        raise FileNotFoundError(img_dir)

    print(f"[yoloe] task={task} split={split} class_idx={task_idx}")
    print(f"[yoloe] class_names={class_names}")
    print(f"[yoloe] weights={weights}{' (fine-tuned)' if is_ft else ' (zero-shot)'}")
    predictor = YOLOEPredictor(weights, class_names, device=str(device), imgsz=imgsz, conf=conf)

    pred_dir = Path(out_pred) if out_pred else None
    viz_dir = Path(out_viz) if out_viz else None
    if pred_dir:
        pred_dir.mkdir(parents=True, exist_ok=True)
    if viz_dir:
        viz_dir.mkdir(parents=True, exist_ok=True)

    acc = MetricAccumulator()
    times = []
    n = 0
    n_viz_done = 0
    img_paths = sorted([p for p in img_dir.iterdir()
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")])
    for p in img_paths:
        gt_p = msk_dir / (p.stem + ".png")
        img_bgr = cv2.imread(str(p))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        t0 = time.time()
        mask_bool = predictor.predict_one(img_rgb, target_class=task_idx)
        times.append(time.time() - t0)

        mask_u8 = mask_bool.astype(np.uint8) * 255
        if pred_dir:
            cv2.imwrite(str(pred_dir / f"{p.stem}__{out_slug}.png"), mask_u8)
        if gt_p.exists():
            gt = (cv2.imread(str(gt_p), cv2.IMREAD_GRAYSCALE) > 127).astype(bool)
            acc.update(torch.from_numpy(mask_bool), torch.from_numpy(gt))
        if viz_dir and n_viz_done < n_viz:
            gt_arr = (cv2.imread(str(gt_p), cv2.IMREAD_GRAYSCALE) if gt_p.exists()
                      else np.zeros(img_rgb.shape[:2], dtype=np.uint8))
            gt_u8 = (gt_arr > 127).astype(np.uint8) * 255
            pan = panel(img_rgb, gt_u8, mask_u8)
            cv2.imwrite(str(viz_dir / f"{p.stem}__panel.png"), cv2.cvtColor(pan, cv2.COLOR_RGB2BGR))
            ovl = overlay(img_rgb, mask_u8)
            cv2.imwrite(str(viz_dir / f"{p.stem}__overlay.png"), cv2.cvtColor(ovl, cv2.COLOR_RGB2BGR))
            n_viz_done += 1
        n += 1

    summary = acc.summary()
    summary.update({
        "task": task,
        "split": split,
        "n": n,
        "avg_inference_ms": (sum(times) / max(1, len(times))) * 1000,
        "weights": weights,
        "finetuned": is_ft,
        "imgsz": imgsz,
        "conf": conf,
    })
    print(json.dumps(summary, indent=2))
    if out_metrics:
        save_json(summary, out_metrics)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/yoloe.yaml")
    ap.add_argument("--task", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-pred", default=None)
    ap.add_argument("--out-viz", default=None)
    ap.add_argument("--out-metrics", default=None)
    ap.add_argument("--n-viz", type=int, default=6)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--prompt", default=None,
                    help="override the class prompt for this task at predict time")
    args = ap.parse_args()
    run(args.config, args.task, split=args.split, out_pred=args.out_pred,
        out_viz=args.out_viz, out_metrics=args.out_metrics, n_viz=args.n_viz,
        weights_override=args.weights, conf=args.conf,
        prompt_override=args.prompt)


if __name__ == "__main__":
    main()
