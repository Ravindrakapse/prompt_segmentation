"""Generate prediction PNG masks for a split.

Filename: <image_id>__<prompt_slug>.png  (e.g. 123__segment_crack.png)
Mask values {0, 255}, single-channel, same size as source image.

Usage:
    python -m src.predict_clipseg --config configs/clipseg.yaml \\
        --ckpt outputs/checkpoints/clipseg/best.pt --task crack --split test \\
        --out outputs/predictions_clipseg/crack
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .data.transforms import (
    CLIP_MEAN,
    CLIP_STD,
    letterbox_pair,
    unletterbox_mask,
)
from .models.clipseg_wrapper import CLIPSegFT
from .utils import device_auto, load_config, set_seed, slugify_prompt


def _normalize_image(img_u8: np.ndarray) -> torch.Tensor:
    img = img_u8.astype(np.float32) / 255.0
    mean = np.array(CLIP_MEAN, dtype=np.float32)
    std = np.array(CLIP_STD, dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img)


def _resolve_task(cfg: dict, task_name: str) -> dict:
    if "tasks" not in cfg:
        raise ValueError("config must define `tasks` (unified mode is the only supported mode)")
    for t in cfg["tasks"]:
        if t["name"] == task_name:
            return t
    raise ValueError(f"task {task_name} not found in cfg.tasks")


def predict_split(cfg_path: str, ckpt_path: str, split: str, out_dir: str, task: str,
                  prompt_override: str | None = None, threshold: float | None = None):
    cfg = load_config(cfg_path)
    set_seed(cfg.get("seed", 42))
    device = device_auto()

    model = CLIPSegFT(
        pretrained=cfg["model"]["pretrained"],
        freeze_clip=True, unfreeze_decoder=True, unfreeze_film=True,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    t = _resolve_task(cfg, task)
    size = cfg["data"]["input_size"]
    if threshold is None:
        threshold = cfg["inference"].get("threshold", 0.5)
    use_tta = cfg["inference"].get("tta", True)
    use_multi = cfg["inference"].get("multi_prompt_avg", True)
    prompts = t["prompts"] if use_multi else [t["eval_prompt"]]
    eval_prompt = prompt_override or t["eval_prompt"]
    out_slug = slugify_prompt(eval_prompt)

    images_dir = Path(t["dataset_dir"]) / "images" / split
    out_dir_p = Path(out_dir); out_dir_p.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n = 0
    times = []
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
            continue
        img_bgr = cv2.imread(str(p))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pad, _, meta = letterbox_pair(img_rgb, None, size)
        x = _normalize_image(img_pad).unsqueeze(0).to(device)

        t1 = time.time()
        with torch.no_grad():
            if use_multi:
                # build (B=1, paraphrases) — wrapper expects per-image list-of-list when multi
                prob = model.predict(x, [list(prompts)], out_size=(size, size),
                                     tta=use_tta, multi_prompt_avg=True)
            else:
                prob = model.predict(x, [eval_prompt], out_size=(size, size), tta=use_tta)
        times.append(time.time() - t1)

        prob_np = prob.squeeze().cpu().numpy()
        mask_pad = (prob_np > threshold).astype(np.uint8) * 255
        mask = unletterbox_mask(mask_pad, meta)
        out_name = f"{p.stem}__{out_slug}.png"
        cv2.imwrite(str(out_dir_p / out_name), mask)
        n += 1

    elapsed = time.time() - t0
    avg_inf = sum(times) / max(1, len(times))
    print(f"[predict] {n} images in {elapsed:.1f}s. avg inference/image: {avg_inf*1000:.1f} ms. -> {out_dir_p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", required=True, help="task name from cfg.tasks (e.g. crack, taping)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()
    predict_split(args.config, args.ckpt, args.split, args.out, args.task,
                  prompt_override=args.prompt, threshold=args.threshold)


if __name__ == "__main__":
    main()
