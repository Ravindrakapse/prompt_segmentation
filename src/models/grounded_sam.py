"""Grounded-SAM zero-shot wrapper: text -> bboxes (Grounding DINO) -> masks (SAM / SAM2).

No fine-tuning. Used to compare against the CLIPSeg fine-tune baseline.

Pipeline:
    image, "segment crack" -> Grounding DINO -> N bboxes
    bboxes -> SAM (or SAM2) -> N per-box masks
    union of masks -> binary mask (H, W) bool
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    GroundingDinoForObjectDetection,
)


def _load_sam(pretrained: str, device: torch.device, dtype: torch.dtype):
    """Load SAM or SAM2, returning (processor, model, is_sam2)."""
    if "sam2" in pretrained.lower():
        from transformers import Sam2Model, Sam2Processor
        proc = Sam2Processor.from_pretrained(pretrained)
        model = Sam2Model.from_pretrained(pretrained, dtype=dtype).to(device)
        return proc, model, True
    from transformers import SamModel
    proc = AutoProcessor.from_pretrained(pretrained)
    model = SamModel.from_pretrained(pretrained, dtype=dtype).to(device)
    return proc, model, False


class GroundedSAM:
    def __init__(
        self,
        gd_pretrained: str = "IDEA-Research/grounding-dino-tiny",
        sam_pretrained: str = "facebook/sam-vit-base",
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        # Grounding DINO has internal float dtype mismatches under fp16/bf16
        # (image features fp16, text-side fp32 -> linear-layer dtype error).
        # Keep GD fp32 — it's already fast enough on A100 (~50 ms / image).
        self.gd_dtype = torch.float32
        self.sam_dtype = dtype
        self.gd_processor = AutoProcessor.from_pretrained(gd_pretrained)
        self.gd_model = GroundingDinoForObjectDetection.from_pretrained(
            gd_pretrained
        ).to(self.device)
        self.gd_model.eval()
        self.sam_processor, self.sam_model, self.is_sam2 = _load_sam(
            sam_pretrained, self.device, self.sam_dtype
        )
        self.sam_model.eval()

    @staticmethod
    def _format_prompt(prompt: str) -> str:
        text = prompt.lower().strip()
        if not text.endswith("."):
            text += "."
        return text

    @torch.no_grad()
    def _detect(self, image: Image.Image, prompt: str,
                box_threshold: float, text_threshold: float) -> torch.Tensor:
        text = self._format_prompt(prompt)
        inputs = self.gd_processor(images=image, text=text, return_tensors="pt").to(self.device)
        out = self.gd_model(**inputs)
        results = self.gd_processor.post_process_grounded_object_detection(
            out,
            inputs["input_ids"],
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        return results["boxes"].float().cpu()  # (N, 4) xyxy

    @torch.no_grad()
    def _segment(self, image: Image.Image, boxes: torch.Tensor) -> np.ndarray:
        if boxes.numel() == 0:
            w, h = image.size
            return np.zeros((h, w), dtype=bool)
        sam_inputs = self.sam_processor(
            image,
            input_boxes=[boxes.tolist()],
            return_tensors="pt",
        ).to(self.device, self.sam_dtype)
        out = self.sam_model(**sam_inputs, multimask_output=False)
        masks = self.sam_processor.image_processor.post_process_masks(
            out.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu(),
        )[0]  # (N, 1, H, W) bool
        if masks.dim() == 4:
            masks = masks.squeeze(1)
        union = masks.any(dim=0).numpy().astype(bool)
        return union

    def predict_one(
        self,
        image_rgb: np.ndarray,
        prompts: Sequence[str] | str,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
    ) -> np.ndarray:
        """Return binary union mask (H, W) bool. Multiple prompts are OR'd
        (boxes from all paraphrases pooled before SAM)."""
        if isinstance(prompts, str):
            prompts = [prompts]
        pil = Image.fromarray(image_rgb)
        all_boxes = []
        for p in prompts:
            b = self._detect(pil, p, box_threshold, text_threshold)
            if b.numel() > 0:
                all_boxes.append(b)
        if not all_boxes:
            h, w = image_rgb.shape[:2]
            return np.zeros((h, w), dtype=bool)
        boxes = torch.cat(all_boxes, dim=0)
        return self._segment(pil, boxes)
