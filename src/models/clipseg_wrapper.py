"""CLIPSeg fine-tune wrapper.

Wraps HuggingFace `CLIPSegForImageSegmentation` with:
  * configurable freezing
  * forward(images, prompts) -> logits @ input_size
  * inference(images, prompts) -> logits, with optional TTA & multi-prompt avg
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor


class CLIPSegFT(nn.Module):
    def __init__(
        self,
        pretrained: str = "CIDAS/clipseg-rd64-refined",
        freeze_clip: bool = True,
        unfreeze_decoder: bool = True,
        unfreeze_film: bool = True,
        unfreeze_visual_adapter: bool = False,
    ):
        super().__init__()
        self.model = CLIPSegForImageSegmentation.from_pretrained(pretrained)
        self.processor = CLIPSegProcessor.from_pretrained(pretrained)
        self._configure_freezing(
            freeze_clip=freeze_clip,
            unfreeze_decoder=unfreeze_decoder,
            unfreeze_film=unfreeze_film,
            unfreeze_visual_adapter=unfreeze_visual_adapter,
        )

    def _configure_freezing(self, freeze_clip, unfreeze_decoder, unfreeze_film, unfreeze_visual_adapter):
        if freeze_clip:
            for p in self.model.clip.parameters():
                p.requires_grad = False
        if unfreeze_decoder:
            for p in self.model.decoder.parameters():
                p.requires_grad = True
        # FiLM modules and visual adapter live under decoder in HF impl;
        # explicit toggles for safety
        for name, p in self.model.named_parameters():
            if "film" in name.lower():
                p.requires_grad = bool(unfreeze_film)
            if "visual_projection" in name and unfreeze_visual_adapter:
                p.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep CLIP backbone in eval mode even during training (BN/Dropout safety).
        try:
            self.model.clip.eval()
        except AttributeError:
            pass
        return self

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _tokenize_text(self, prompts: Sequence[str], device: torch.device):
        tok = self.processor.tokenizer(list(prompts), padding=True, return_tensors="pt")
        return {k: v.to(device) for k, v in tok.items()}

    def forward(
        self,
        pixel_values: torch.Tensor,    # (B,3,H,W) already normalized
        prompts: Sequence[str],        # len B
    ) -> torch.Tensor:
        assert len(prompts) == pixel_values.size(0), "prompt count must match batch"
        text = self._tokenize_text(prompts, pixel_values.device)
        out = self.model(
            pixel_values=pixel_values,
            input_ids=text["input_ids"],
            attention_mask=text.get("attention_mask"),
        )
        logits = out.logits
        # Normalize shape to (B,1,H,W)
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
        elif logits.dim() == 4 and logits.shape[1] != 1:
            # (B, N_prompts, H, W) — collapse if singleton prompts mistakenly batched
            logits = logits[:, :1]
        return logits

    @torch.no_grad()
    def predict(
        self,
        pixel_values: torch.Tensor,
        prompts: Sequence[str] | str,
        out_size: tuple[int, int] | None = None,
        tta: bool = False,
        multi_prompt_avg: bool = False,
    ) -> torch.Tensor:
        """Return sigmoid probabilities (B,1,H,W) at out_size (or input size).

        - prompts: per-image prompt OR single string applied to all
        - tta: 4-way (orig + hflip + vflip + hv-flip) averaged in prob space
        - multi_prompt_avg: if `prompts` is list-of-list, avg logits across paraphrases
        """
        self.eval()
        B = pixel_values.size(0)
        if isinstance(prompts, str):
            prompts = [prompts] * B

        def fwd(x, p):
            return self.forward(x, p)

        def maybe_tta(x, p):
            if not tta:
                return torch.sigmoid(fwd(x, p))
            outs = [torch.sigmoid(fwd(x, p))]
            outs.append(torch.flip(torch.sigmoid(fwd(torch.flip(x, dims=[-1]), p)), dims=[-1]))
            outs.append(torch.flip(torch.sigmoid(fwd(torch.flip(x, dims=[-2]), p)), dims=[-2]))
            outs.append(torch.flip(torch.sigmoid(fwd(torch.flip(x, dims=[-1, -2]), p)), dims=[-1, -2]))
            return torch.stack(outs, 0).mean(0)

        # multi-prompt averaging path: treat `prompts` as list-of-list when first elem is list
        if multi_prompt_avg and isinstance(prompts[0], (list, tuple)):
            prob_sum = None
            n = 0
            n_paras = len(prompts[0])
            for k in range(n_paras):
                p_k = [pl[k] for pl in prompts]
                pr = maybe_tta(pixel_values, p_k)
                prob_sum = pr if prob_sum is None else prob_sum + pr
                n += 1
            prob = prob_sum / max(1, n)
        else:
            prob = maybe_tta(pixel_values, list(prompts))

        if out_size is not None:
            prob = F.interpolate(prob, size=out_size, mode="bilinear", align_corners=False)
        return prob
