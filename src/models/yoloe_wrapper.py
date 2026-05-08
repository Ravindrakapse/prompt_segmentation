"""YOLOE wrapper: text-prompted segmentation -> binary union mask.

Used for both zero-shot and fine-tuned inference. Ultralytics' YOLOE returns
per-instance masks at model resolution; we resize each to the original image
size and union by class.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
import torch
from ultralytics import YOLOE


class YOLOEPredictor:
    def __init__(self, weights: str, class_names: Sequence[str], device: str = "cuda",
                 imgsz: int = 640, conf: float = 0.25):
        self.model = YOLOE(weights)
        self.class_names = list(class_names)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        # set_classes encodes the prompt names; required before predict.
        # Fine-tuned PE-style ckpts may not need it (classes baked in); for
        # those `get_text_pe` may not exist and we skip silently. For the
        # zero-shot path, surface any error.
        try:
            text_pe = self.model.get_text_pe(self.class_names)
        except AttributeError:
            text_pe = None
        if text_pe is not None:
            self.model.set_classes(self.class_names, text_pe)

    @torch.no_grad()
    def predict_one(self, image_rgb: np.ndarray, target_class: int) -> np.ndarray:
        """Return binary union mask (H, W) bool for instances of target_class."""
        H, W = image_rgb.shape[:2]
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        results = self.model.predict(
            bgr, imgsz=self.imgsz, conf=self.conf, device=self.device,
            verbose=False,
        )
        r = results[0]
        union = np.zeros((H, W), dtype=bool)
        if r.masks is None or r.boxes is None:
            return union
        cls = r.boxes.cls.cpu().numpy().astype(int)
        masks = r.masks.data.cpu().numpy()  # (N, h, w) float
        for m, c in zip(masks, cls):
            if int(c) != target_class:
                continue
            m_full = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
            union |= (m_full > 0.5)
        return union
