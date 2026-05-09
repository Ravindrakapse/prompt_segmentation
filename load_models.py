"""Load each fine-tuned model from HuggingFace Hub and run inference on one image.

End-to-end demo of all three pipelines. Run:

    python load_models.py --image path/to/img.jpg --prompt "segment crack" \\
        --model clipseg --out mask.png

Choices for --model: clipseg | grounded_sam | yoloe | all
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

HF_USER = "ravindrakapse"


# ---------------- CLIPSeg ----------------
def load_clipseg():
    from huggingface_hub import hf_hub_download
    from src.models.clipseg_wrapper import CLIPSegFT

    ckpt = hf_hub_download(repo_id=f"{HF_USER}/drywall-clipseg", filename="best.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPSegFT(pretrained="CIDAS/clipseg-rd64-refined").to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, device


def predict_clipseg(model, device, image_bgr, prompt, threshold=0.5, size=352):
    from src.data.transforms import CLIP_MEAN, CLIP_STD, letterbox_pair, unletterbox_mask

    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_pad, _, meta = letterbox_pair(img_rgb, None, size)
    arr = img_pad.astype(np.float32) / 255.0
    mean = np.array(CLIP_MEAN, dtype=np.float32)
    std = np.array(CLIP_STD, dtype=np.float32)
    arr = (arr - mean) / std
    x = torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = model.predict(x, [prompt], out_size=(size, size), tta=True)
    prob_np = prob.squeeze().cpu().numpy()
    mask_pad = (prob_np > threshold).astype(np.uint8) * 255
    return unletterbox_mask(mask_pad, meta)


# ---------------- Grounded-SAM ----------------
def load_grounded_sam():
    from huggingface_hub import snapshot_download
    from transformers import (
        AutoProcessor,
        AutoModelForZeroShotObjectDetection,
        SamProcessor,
        SamModel,
    )
    local = snapshot_download(repo_id=f"{HF_USER}/drywall-grounded-sam")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gd_proc = AutoProcessor.from_pretrained(f"{local}/gd_finetuned")
    gd = AutoModelForZeroShotObjectDetection.from_pretrained(f"{local}/gd_finetuned").to(device).eval()
    sam_proc = SamProcessor.from_pretrained(f"{local}/sam_finetuned")
    sam = SamModel.from_pretrained(f"{local}/sam_finetuned").to(device).eval()
    return (gd_proc, gd, sam_proc, sam, device)


def predict_grounded_sam(bundle, image_bgr, prompt, box_threshold=0.25, text_threshold=0.20):
    gd_proc, gd, sam_proc, sam, device = bundle
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]
    text = prompt.lower().strip().rstrip(".") + "."

    inputs = gd_proc(images=img_rgb, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = gd(**inputs)
    res = gd_proc.post_process_grounded_object_detection(
        out, inputs.input_ids, box_threshold=box_threshold,
        text_threshold=text_threshold, target_sizes=[(H, W)])[0]
    boxes = res["boxes"].detach().cpu().numpy()
    if len(boxes) == 0:
        return np.zeros((H, W), dtype=np.uint8)

    sam_in = sam_proc(images=img_rgb, input_boxes=[boxes.tolist()], return_tensors="pt").to(device)
    with torch.no_grad():
        sam_out = sam(**sam_in, multimask_output=False)
    masks = sam_proc.image_processor.post_process_masks(
        sam_out.pred_masks.cpu(), sam_in["original_sizes"].cpu(),
        sam_in["reshaped_input_sizes"].cpu())[0]  # (n_boxes, 1, H, W) bool
    union = masks.squeeze(1).any(dim=0).numpy().astype(np.uint8) * 255
    return union


# ---------------- YOLOE ----------------
def load_yoloe():
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLOE
    ckpt = hf_hub_download(repo_id=f"{HF_USER}/drywall-yoloe", filename="best.pt")
    return YOLOE(ckpt)


def predict_yoloe(model, image_bgr, prompt, imgsz=640, conf=0.05):
    """YOLOE was trained with 2 classes (crack, taping). The head's softmax
    dimensionality is fixed at 2, so we must always register exactly 2 class
    names. The user's prompt is bound to whichever slot semantically matches
    (crack-like → slot 0, taping-like → slot 1); the other slot is filled
    with a benign placeholder. Detections of the unmatched slot are filtered
    out before unioning the masks.
    """
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    # Decide which trained slot the user's prompt maps to.
    p_low = prompt.lower()
    if any(k in p_low for k in ("taping", "tape", "seam", "joint", "drywall")):
        target_idx = 1
        classes = ["crack", prompt]            # slot 0 unused but kept consistent
    else:
        target_idx = 0
        classes = [prompt, "taping"]           # slot 1 unused

    text_pe = model.get_text_pe(classes)
    model.set_classes(classes, text_pe)
    res = model.predict(img_rgb, imgsz=imgsz, conf=conf, verbose=False)[0]
    if res.masks is None or len(res.masks) == 0:
        return np.zeros((H, W), dtype=np.uint8)
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    keep = cls_ids == target_idx
    if not keep.any():
        return np.zeros((H, W), dtype=np.uint8)
    m = res.masks.data.cpu().numpy()[keep]     # (n_keep, h, w)
    union = (m.sum(axis=0) > 0).astype(np.uint8)
    union = cv2.resize(union, (W, H), interpolation=cv2.INTER_NEAREST) * 255
    return union


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--model", default="all", choices=["clipseg", "grounded_sam", "yoloe", "all"])
    ap.add_argument("--out", default="mask.png")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.out).stem

    if args.model in ("clipseg", "all"):
        m, dev = load_clipseg()
        mask = predict_clipseg(m, dev, img, args.prompt)
        cv2.imwrite(str(out_dir / f"{stem}_clipseg.png"), mask)
        print(f"saved {stem}_clipseg.png ({mask.sum()/255:.0f} pos px)")

    if args.model in ("grounded_sam", "all"):
        b = load_grounded_sam()
        mask = predict_grounded_sam(b, img, args.prompt)
        cv2.imwrite(str(out_dir / f"{stem}_grounded_sam.png"), mask)
        print(f"saved {stem}_grounded_sam.png ({mask.sum()/255:.0f} pos px)")

    if args.model in ("yoloe", "all"):
        y = load_yoloe()
        mask = predict_yoloe(y, img, args.prompt)
        cv2.imwrite(str(out_dir / f"{stem}_yoloe.png"), mask)
        print(f"saved {stem}_yoloe.png ({mask.sum()/255:.0f} pos px)")


if __name__ == "__main__":
    main()
