"""Grounded-SAM zero-shot baseline: predict + eval + visualize per task.

No training. Runs Grounding DINO -> bboxes -> SAM/SAM2 -> binary mask, with
prompt = task's paraphrase pool from the unified config.

Usage:
    python -m src.predict_grounded_sam --config configs/grounded_sam.yaml \\
        --task crack --split test \\
        --out-pred outputs/predictions_grounded_sam/crack \\
        --out-viz  outputs/visualizations_grounded_sam/crack \\
        --out-metrics outputs/predictions_grounded_sam/crack/metrics.json
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
from .models.grounded_sam import GroundedSAM
from .utils import device_auto, load_config, save_json, set_seed, slugify_prompt
from .visualize import overlay, panel


def _resolve_task(cfg: dict, task_name: str) -> dict:
    if "tasks" not in cfg:
        raise ValueError("config must define `tasks`")
    for t in cfg["tasks"]:
        if t["name"] == task_name:
            return t
    raise ValueError(f"task {task_name} not found in cfg.tasks")


def run(
    cfg_path: str,
    task: str,
    split: str = "test",
    out_pred: str | None = None,
    out_viz: str | None = None,
    out_metrics: str | None = None,
    n_viz: int = 6,
    box_threshold: float | None = None,
    text_threshold: float | None = None,
):
    cfg = load_config(cfg_path)
    set_seed(cfg.get("seed", 42))
    device = device_auto()
    gs_cfg = cfg.get("grounded_sam", {})
    box_thr = box_threshold if box_threshold is not None else gs_cfg.get("box_threshold", 0.25)
    text_thr = text_threshold if text_threshold is not None else gs_cfg.get("text_threshold", 0.20)
    multi = gs_cfg.get("multi_prompt_union", True)

    t = _resolve_task(cfg, task)
    prompts = t["prompts"] if multi else [t["eval_prompt"]]
    eval_prompt = t["eval_prompt"]
    out_slug = slugify_prompt(eval_prompt)

    img_dir = Path(t["dataset_dir"]) / "images" / split
    msk_dir = Path(t["dataset_dir"]) / "masks" / split
    if not img_dir.exists():
        raise FileNotFoundError(f"missing {img_dir}")

    # Prefer fine-tuned local checkpoints if present in cfg.grounded_sam_train.
    train_cfg = cfg.get("grounded_sam_train", {})
    gd_ckpt = train_cfg.get("gd", {}).get("ckpt_dir")
    sam_ckpt = train_cfg.get("sam", {}).get("ckpt_dir")
    gd_path = (Path(gd_ckpt) / "best") if gd_ckpt and (Path(gd_ckpt) / "best").exists() else None
    sam_path = (Path(sam_ckpt) / "best") if sam_ckpt and (Path(sam_ckpt) / "best").exists() else None
    gd_pre = str(gd_path) if gd_path else gs_cfg.get("gd_pretrained", "IDEA-Research/grounding-dino-tiny")
    sam_pre = str(sam_path) if sam_path else gs_cfg.get("sam_pretrained", "facebook/sam-vit-base")
    print(f"[gs] task={task} split={split} prompts={prompts}")
    print(f"[gs] thresholds box={box_thr} text={text_thr}")
    print(f"[gs] gd={gd_pre}{' (fine-tuned)' if gd_path else ''}")
    print(f"[gs] sam={sam_pre}{' (fine-tuned)' if sam_path else ''}")
    model = GroundedSAM(
        gd_pretrained=gd_pre,
        sam_pretrained=sam_pre,
        device=device,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )

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
        mask_bool = model.predict_one(img_rgb, prompts, box_threshold=box_thr, text_threshold=text_thr)
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
        "box_threshold": box_thr,
        "text_threshold": text_thr,
        "gd_pretrained": gd_pre,
        "sam_pretrained": sam_pre,
        "gd_finetuned": bool(gd_path),
        "sam_finetuned": bool(sam_path),
    })
    print(json.dumps(summary, indent=2))
    if out_metrics:
        save_json(summary, out_metrics)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grounded_sam.yaml")
    ap.add_argument("--task", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-pred", default=None)
    ap.add_argument("--out-viz", default=None)
    ap.add_argument("--out-metrics", default=None)
    ap.add_argument("--n-viz", type=int, default=6)
    ap.add_argument("--box-threshold", type=float, default=None)
    ap.add_argument("--text-threshold", type=float, default=None)
    args = ap.parse_args()
    run(
        args.config,
        args.task,
        split=args.split,
        out_pred=args.out_pred,
        out_viz=args.out_viz,
        out_metrics=args.out_metrics,
        n_viz=args.n_viz,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )


if __name__ == "__main__":
    main()
