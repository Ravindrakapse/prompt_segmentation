"""Generic prompted-segmentation dataset.

Expects processed dir layout:
    dataset_dir/
        images/{train,val,test}/<id>.jpg
        masks/{train,val,test}/<id>.png   # binary {0,255}
        splits.json                       # optional metadata
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")


def _list_pairs(images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXT:
            continue
        mask = masks_dir / (p.stem + ".png")
        if mask.exists():
            pairs.append((p, mask))
    return pairs


class PromptedSegDataset(Dataset):
    """Yields (image_tensor, mask_tensor, prompt_str, image_id, orig_size)."""

    def __init__(
        self,
        dataset_dir: str | os.PathLike,
        split: str,
        prompts: Sequence[str],
        transform=None,
        prompt_sampling: str = "random",  # "random" | "first" | "all"
        return_original: bool = False,
    ):
        self.root = Path(dataset_dir)
        self.split = split
        self.prompts = list(prompts)
        self.transform = transform
        self.prompt_sampling = prompt_sampling
        self.return_original = return_original
        images_dir = self.root / "images" / split
        masks_dir = self.root / "masks" / split
        if not images_dir.exists():
            raise FileNotFoundError(f"missing {images_dir}")
        self.pairs = _list_pairs(images_dir, masks_dir)
        if not self.pairs:
            raise RuntimeError(f"no image/mask pairs in {images_dir}")

    def __len__(self) -> int:
        return len(self.pairs)

    def _pick_prompt(self) -> str:
        if self.prompt_sampling == "first" or len(self.prompts) == 1:
            return self.prompts[0]
        return random.choice(self.prompts)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)  # binarize
        orig_h, orig_w = image.shape[:2]

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image_t = out["image"]
            mask_t = out["mask"].float()
        else:
            image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask_t = torch.from_numpy(mask).float()

        prompt = self._pick_prompt()
        sample = {
            "image": image_t,
            "mask": mask_t,
            "prompt": prompt,
            "image_id": img_path.stem,
            "orig_h": orig_h,
            "orig_w": orig_w,
        }
        if self.return_original:
            sample["orig_image"] = image
            sample["orig_mask"] = mask
        return sample


def collate_prompted(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    masks = torch.stack([b["mask"] for b in batch], dim=0)
    prompts = [b["prompt"] for b in batch]
    image_ids = [b["image_id"] for b in batch]
    orig_h = [b["orig_h"] for b in batch]
    orig_w = [b["orig_w"] for b in batch]
    out = {
        "image": images,
        "mask": masks,
        "prompt": prompts,
        "image_id": image_ids,
        "orig_h": orig_h,
        "orig_w": orig_w,
    }
    if "task" in batch[0]:
        out["task"] = [b["task"] for b in batch]
    if "is_negative" in batch[0]:
        out["is_negative"] = torch.tensor([b["is_negative"] for b in batch], dtype=torch.bool)
    return out


class UnifiedPromptedSegDataset(Dataset):
    """Multi-task prompted segmentation with cross-task negative sampling.

    Each `task` entry: {name, dataset_dir, prompts}.

    During training (neg_prob > 0): with probability neg_prob, replace the
    sample's task prompt with a paraphrase from a *different* task and zero
    out the mask. This forces the decoder to actually condition on the prompt
    rather than segmenting any salient structure regardless of text.

    During eval (neg_prob = 0): every sample uses its own task's eval prompt
    and its real mask — equivalent to the per-task PromptedSegDataset.
    """

    def __init__(
        self,
        tasks: Sequence[dict],
        split: str,
        transform=None,
        neg_prob: float = 0.0,
        eval_mode: bool = False,
    ):
        self.tasks = list(tasks)
        self.split = split
        self.transform = transform
        self.neg_prob = neg_prob
        self.eval_mode = eval_mode
        self.samples: list[tuple[Path, Path, int]] = []
        for ti, t in enumerate(self.tasks):
            root = Path(t["dataset_dir"])
            img_dir = root / "images" / split
            msk_dir = root / "masks" / split
            if not img_dir.exists():
                raise FileNotFoundError(f"missing {img_dir}")
            for ip, mp in _list_pairs(img_dir, msk_dir):
                self.samples.append((ip, mp, ti))
        if not self.samples:
            raise RuntimeError(f"no samples for split={split} across tasks")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path, task_idx = self.samples[idx]
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)
        orig_h, orig_w = image.shape[:2]

        is_neg = False
        task = self.tasks[task_idx]
        if (
            not self.eval_mode
            and len(self.tasks) > 1
            and self.neg_prob > 0.0
            and random.random() < self.neg_prob
        ):
            other = [i for i in range(len(self.tasks)) if i != task_idx]
            other_task = self.tasks[random.choice(other)]
            prompt = random.choice(other_task["prompts"])
            mask = np.zeros_like(mask)
            is_neg = True
        else:
            if self.eval_mode:
                prompt = task.get("eval_prompt", task["prompts"][0])
            else:
                prompt = random.choice(task["prompts"])

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image_t = out["image"]
            mask_t = out["mask"].float()
        else:
            image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask_t = torch.from_numpy(mask).float()

        return {
            "image": image_t,
            "mask": mask_t,
            "prompt": prompt,
            "image_id": img_path.stem,
            "task": task["name"],
            "is_negative": is_neg,
            "orig_h": orig_h,
            "orig_w": orig_w,
        }
