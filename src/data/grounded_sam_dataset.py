"""Datasets for fine-tuning Grounding DINO and SAM on the unified task pool.

Both pipelines need (image, mask) → derive (image, text, bboxes) for GD or
(image, bboxes, mask) for SAM. Bboxes are extracted via connected-components
on the binary GT mask (one bbox per blob, area-filtered).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import _list_pairs


def mask_to_bboxes(mask: np.ndarray, min_area: int = 16) -> np.ndarray:
    """Connected components -> (N, 4) xyxy ints. Empty = (0, 4) array."""
    bin_mask = (mask > 0).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        boxes.append([x, y, x + w, y + h])
    return np.asarray(boxes, dtype=np.int64) if boxes else np.zeros((0, 4), dtype=np.int64)


def _build_samples(tasks: Sequence[dict], split: str) -> list[tuple[Path, Path, dict]]:
    out: list[tuple[Path, Path, dict]] = []
    for t in tasks:
        root = Path(t["dataset_dir"])
        img_dir = root / "images" / split
        msk_dir = root / "masks" / split
        if not img_dir.exists():
            continue
        for ip, mp in _list_pairs(img_dir, msk_dir):
            out.append((ip, mp, t))
    return out


class GDTrainDataset(Dataset):
    """Yields raw items for a custom collate that runs the HF processor.

    Returns dict with: image (np.uint8 HxWx3 RGB), prompt (str), boxes_xyxy
    (np.float32 [N,4] in pixel coords), task (str)."""

    def __init__(self, tasks: Sequence[dict], split: str, min_area: int = 16,
                 drop_empty: bool = True):
        self.samples = _build_samples(tasks, split)
        self.min_area = min_area
        self.drop_empty = drop_empty

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, msk_path, task = self.samples[idx]
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)
        boxes = mask_to_bboxes(mask, min_area=self.min_area).astype(np.float32)
        # If empty and drop_empty: caller should resample. We mark with empty boxes.
        prompt = random.choice(task["prompts"])
        return {
            "image": image,
            "boxes_xyxy": boxes,
            "prompt": prompt,
            "task": task["name"],
            "image_id": img_path.stem,
        }


class SAMTrainDataset(Dataset):
    """Yields (image, gt_boxes_xyxy, gt_mask) for SAM mask-decoder fine-tune.
    Boxes are derived from the GT mask itself — SAM learns to refine
    bbox-prompted masks within the domain."""

    def __init__(self, tasks: Sequence[dict], split: str, min_area: int = 16,
                 max_boxes_per_image: int = 16):
        self.samples = _build_samples(tasks, split)
        self.min_area = min_area
        self.max_boxes_per_image = max_boxes_per_image

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, msk_path, task = self.samples[idx]
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)
        boxes = mask_to_bboxes(mask, min_area=self.min_area).astype(np.float32)
        if boxes.shape[0] > self.max_boxes_per_image:
            idxs = np.random.choice(boxes.shape[0], self.max_boxes_per_image, replace=False)
            boxes = boxes[idxs]
        return {
            "image": image,
            "mask": (mask > 127).astype(np.uint8),
            "boxes_xyxy": boxes,
            "task": task["name"],
            "image_id": img_path.stem,
        }


def gd_collate(batch: list[dict], processor) -> dict:
    """Run the HF GD processor on a batch and pack labels in DETR format
    (cxcywh normalized). Single-class per image -> class_labels all 0."""
    # Filter out empties
    batch = [b for b in batch if b["boxes_xyxy"].shape[0] > 0]
    if not batch:
        return {}
    images = [b["image"] for b in batch]
    prompts = [b["prompt"].lower().rstrip(".") + "." for b in batch]
    enc = processor(images=images, text=prompts, return_tensors="pt", padding=True)
    labels = []
    for b in batch:
        h, w = b["image"].shape[:2]
        bx = b["boxes_xyxy"].copy()
        cx = (bx[:, 0] + bx[:, 2]) / 2.0 / w
        cy = (bx[:, 1] + bx[:, 3]) / 2.0 / h
        bw = (bx[:, 2] - bx[:, 0]) / w
        bh = (bx[:, 3] - bx[:, 1]) / h
        boxes_cxcywh = np.stack([cx, cy, bw, bh], axis=1).astype(np.float32)
        labels.append({
            "class_labels": torch.zeros(boxes_cxcywh.shape[0], dtype=torch.long),
            "boxes": torch.from_numpy(boxes_cxcywh),
        })
    enc["labels"] = labels
    return enc


def sam_collate(batch: list[dict], processor) -> dict:
    """Run SAM processor on a batch. Pads boxes to common N within the batch
    (SAM processor needs a uniform N). Returns dict with model inputs + a
    separate `gt_masks` list (per-image mask u8) and box counts per image."""
    batch = [b for b in batch if b["boxes_xyxy"].shape[0] > 0]
    if not batch:
        return {}
    images = [b["image"] for b in batch]
    counts = [int(b["boxes_xyxy"].shape[0]) for b in batch]
    max_n = max(counts)
    padded = []
    for b in batch:
        bx = b["boxes_xyxy"].astype(float).tolist()
        pad_n = max_n - len(bx)
        bx.extend([[0.0, 0.0, 1.0, 1.0]] * pad_n)
        padded.append(bx)
    enc = processor(images=images, input_boxes=padded, return_tensors="pt")
    enc["gt_masks"] = [torch.from_numpy(b["mask"]).float() for b in batch]
    enc["box_counts"] = counts
    return enc
