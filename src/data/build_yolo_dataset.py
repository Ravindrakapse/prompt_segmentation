"""Convert the processed (image, binary-mask) layout into YOLO-segmentation
format expected by Ultralytics' YOLOE training:

    out_root/
        dataset.yaml
        images/{train,val,test}/<id>.<ext>
        labels/{train,val,test}/<id>.txt    # one polygon per line:
                                            #   <class_id> x1 y1 x2 y2 ... (normalized)

Multi-task: each task in the unified config becomes one YOLO class id.
Image and label filenames mirror the source split.

Usage:
    python -m src.data.build_yolo_dataset --config configs/yoloe.yaml \\
        --out data/processed/yolo_unified
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

from .dataset import _list_pairs


def mask_to_polygons(mask: np.ndarray, min_area: int = 16, epsilon_frac: float = 0.003):
    """Per connected blob -> approxPolyDP -> normalized polygon (xy pairs in [0,1])."""
    H, W = mask.shape[:2]
    bin_mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polys = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area or cnt.shape[0] < 3:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, max(1.0, epsilon_frac * peri), True)
        if approx.shape[0] < 3:
            continue
        pts = approx.reshape(-1, 2).astype(np.float32)
        pts[:, 0] /= W
        pts[:, 1] /= H
        pts = np.clip(pts, 0.0, 1.0)
        polys.append(pts)
    return polys


def build(cfg: dict, out_root: Path, splits=("train", "val", "test"), min_area: int = 16):
    out_root = Path(out_root)
    for sp in splits:
        (out_root / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / sp).mkdir(parents=True, exist_ok=True)

    class_names = [t["name"] for t in cfg["tasks"]]
    counts = {sp: 0 for sp in splits}
    empty_label = {sp: 0 for sp in splits}
    for class_id, task in enumerate(cfg["tasks"]):
        root = Path(task["dataset_dir"])
        task_name = task["name"]
        for sp in splits:
            img_dir = root / "images" / sp
            msk_dir = root / "masks" / sp
            if not img_dir.exists():
                continue
            for ip, mp in _list_pairs(img_dir, msk_dir):
                # collisions across tasks: prefix with task name
                stem = f"{task_name}__{ip.stem}"
                # copy image
                dst_img = out_root / "images" / sp / f"{stem}{ip.suffix.lower()}"
                if not dst_img.exists():
                    shutil.copy2(ip, dst_img)
                mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                polys = mask_to_polygons(mask, min_area=min_area)
                lbl_path = out_root / "labels" / sp / f"{stem}.txt"
                with open(lbl_path, "w") as f:
                    if not polys:
                        empty_label[sp] += 1
                    for poly in polys:
                        flat = poly.flatten().tolist()
                        f.write(f"{class_id} " + " ".join(f"{v:.6f}" for v in flat) + "\n")
                counts[sp] += 1

    # Write dataset.yaml
    yaml_path = out_root / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump({
            "path": str(out_root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {i: n for i, n in enumerate(class_names)},
        }, f, sort_keys=False)
    print("counts:", counts)
    print("empty-label images:", empty_label)
    print("dataset.yaml ->", yaml_path)
    return yaml_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="data/processed/yolo_unified")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    build(cfg, Path(args.out))


if __name__ == "__main__":
    main()
