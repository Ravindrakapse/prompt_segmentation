"""Albumentations-based train/val transforms.

Cracks/seams = thin elongated structures. Use rotations (full), flips, mild
photometric. Mask resize must be NEAREST. Letterbox preserves aspect ratio."""
from __future__ import annotations

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2

# CLIP normalization (CLIPSeg processor uses these)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_train_transforms(cfg: dict) -> A.Compose:
    size = cfg.get("crop_size", 352)
    return A.Compose([
        A.LongestMaxSize(max_size=int(size * 1.25), interpolation=cv2.INTER_AREA),
        A.PadIfNeeded(min_height=int(size * 1.25), min_width=int(size * 1.25),
                      border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
        A.RandomCrop(height=size, width=size),
        A.HorizontalFlip(p=cfg.get("hflip_p", 0.5)),
        A.VerticalFlip(p=cfg.get("vflip_p", 0.2)),
        A.RandomRotate90(p=0.5),
        A.Affine(rotate=(-cfg.get("rotate_limit", 15), cfg.get("rotate_limit", 15)),
                 scale=(0.85, 1.15), translate_percent=(0.0, 0.05),
                 border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.RandomBrightnessContrast(p=cfg.get("brightness_contrast_p", 0.3)),
        A.CLAHE(clip_limit=2.0, p=0.2),
        A.GaussNoise(p=cfg.get("noise_p", 0.1)),
        A.GaussianBlur(blur_limit=(3, 5), p=cfg.get("blur_p", 0.1)),
        A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ToTensorV2(),
    ])


def build_val_transforms(cfg: dict) -> A.Compose:
    size = cfg.get("crop_size", 352)
    return A.Compose([
        A.LongestMaxSize(max_size=size, interpolation=cv2.INTER_AREA),
        A.PadIfNeeded(min_height=size, min_width=size,
                      border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
        A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ToTensorV2(),
    ])


def letterbox_pair(image: np.ndarray, mask: np.ndarray | None, size: int):
    """Letterbox (preserve AR) image+mask to (size,size). Returns padded arrays
    plus metadata for unpadding at inference."""
    h, w = image.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img_r = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    mask_r = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST) if mask is not None else None
    pad_h, pad_w = size - nh, size - nw
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left
    img_p = cv2.copyMakeBorder(img_r, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    mask_p = cv2.copyMakeBorder(mask_r, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0) if mask_r is not None else None
    meta = {"orig_h": h, "orig_w": w, "scale": scale, "pad_top": top, "pad_left": left, "nh": nh, "nw": nw}
    return img_p, mask_p, meta


def unletterbox_mask(mask_pad: np.ndarray, meta: dict) -> np.ndarray:
    """Reverse letterbox: crop padding, resize back to original."""
    top, left = meta["pad_top"], meta["pad_left"]
    nh, nw = meta["nh"], meta["nw"]
    crop = mask_pad[top:top + nh, left:left + nw]
    return cv2.resize(crop, (meta["orig_w"], meta["orig_h"]), interpolation=cv2.INTER_NEAREST)
