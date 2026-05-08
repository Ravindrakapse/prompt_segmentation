"""Push best checkpoints to HuggingFace Hub.

Creates 3 model repos under user `ravindrakapse`:
  - drywall-clipseg          : single .pt + config + README
  - drywall-grounded-sam     : gd_finetuned/ + sam_finetuned/ + config + README
  - drywall-yoloe            : single .pt + config + README

Assumes you're already logged in: `huggingface-cli login` (or hf auth login).

Usage:
    python -m src.hf_upload --model clipseg
    python -m src.hf_upload --model grounded_sam
    python -m src.hf_upload --model yoloe
    python -m src.hf_upload --model all
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo


HF_USER = "ravindrakapse"

REPOS = {
    "clipseg": {
        "repo_id": f"{HF_USER}/drywall-clipseg",
        "ckpt": "outputs/checkpoints/clipseg/best.pt",
        "config": "configs/clipseg.yaml",
        "card": """---
license: apache-2.0
tags:
  - segmentation
  - prompted-segmentation
  - clipseg
  - drywall
---
# drywall-clipseg

CLIPSeg fine-tune (HF `CIDAS/clipseg-rd64-refined` backbone, decoder + FiLM
unfrozen) for prompted binary segmentation on drywall imagery. Single
checkpoint covers two classes selected by text prompt:

- `"segment crack"` → wall-crack mask
- `"segment taping area"` → drywall taping-seam mask

## Test metrics (focal_dice loss, threshold 0.6)

| Task | Dice | mIoU | Precision | Recall |
| --- | --- | --- | --- | --- |
| Crack  | 0.672 | 0.531 | — | — |
| Taping | 0.727 | 0.587 | — | — |

## Load + predict

```python
from huggingface_hub import hf_hub_download
import torch
from src.models.clipseg_wrapper import CLIPSegFT

ckpt_path = hf_hub_download(repo_id="ravindrakapse/drywall-clipseg", filename="best.pt")
model = CLIPSegFT(pretrained="CIDAS/clipseg-rd64-refined").cuda()
state = torch.load(ckpt_path, map_location="cuda")
model.load_state_dict(state["model"])
model.eval()
```

See [`load_models.py`](https://github.com/Ravindrakapse/prompt_segmentation/blob/main/load_models.py) for the full inference pipeline (letterbox + TTA + un-letterbox).
""",
    },
    "grounded_sam": {
        "repo_id": f"{HF_USER}/drywall-grounded-sam",
        "dirs": [
            ("outputs/checkpoints/gd_finetuned/best", "gd_finetuned"),
            ("outputs/checkpoints/sam_finetuned/best", "sam_finetuned"),
        ],
        "config": "configs/grounded_sam.yaml",
        "card": """---
license: apache-2.0
tags:
  - segmentation
  - grounding-dino
  - sam
  - drywall
---
# drywall-grounded-sam

Two-stage prompted segmentation: Grounding DINO (`grounding-dino-tiny`) emits
bboxes from a text prompt → SAM (`sam-vit-base`) converts boxes → masks → union.
Both stages fine-tuned on the unified drywall pool (3 epochs each).

## Test metrics

| Task | Dice | mIoU | Precision | Recall |
| --- | --- | --- | --- | --- |
| Crack  | 0.601 | 0.463 | 0.637 | 0.725 |
| Taping | 0.515 | 0.393 | 0.491 | 0.703 |

## Load + predict

```python
from huggingface_hub import snapshot_download
local = snapshot_download(repo_id="ravindrakapse/drywall-grounded-sam")
# local/gd_finetuned/  -> AutoProcessor + AutoModelForZeroShotObjectDetection
# local/sam_finetuned/ -> SamProcessor + SamModel
```

See [`load_models.py`](https://github.com/Ravindrakapse/prompt_segmentation/blob/main/load_models.py) for the full pipeline.
""",
    },
    "yoloe": {
        "repo_id": f"{HF_USER}/drywall-yoloe",
        "ckpt": "outputs/checkpoints/yoloe/best.pt",
        "config": "configs/yoloe.yaml",
        "card": """---
license: apache-2.0
tags:
  - segmentation
  - yolo
  - yoloe
  - ultralytics
  - drywall
---
# drywall-yoloe

YOLOE (open-vocab YOLO11 + text-prompted seg head) fine-tuned via Ultralytics
`YOLOEPESegTrainer`. `yoloe-11s-seg` backbone, imgsz 640, 30 epochs, prompt
augmentation (paraphrase per batch).

## Test metrics

| Task | Dice | mIoU | Precision | Recall | ms/img |
| --- | --- | --- | --- | --- | --- |
| Crack  | 0.681 | 0.547 | 0.690 | 0.757 | ~30 |
| Taping | 0.859 | 0.774 | 0.807 | 0.956 | ~30 |

Best of all three pipelines on Dice + latency.

## Load + predict

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLOE

ckpt = hf_hub_download(repo_id="ravindrakapse/drywall-yoloe", filename="best.pt")
model = YOLOE(ckpt)
results = model.predict("image.jpg", imgsz=640, conf=0.05)
```

See [`load_models.py`](https://github.com/Ravindrakapse/prompt_segmentation/blob/main/load_models.py).
""",
    },
}


def upload_one(model: str):
    info = REPOS[model]
    repo_id = info["repo_id"]
    api = HfApi()
    create_repo(repo_id, exist_ok=True, repo_type="model")
    print(f"[hf-upload] repo: {repo_id}")

    # Upload README
    card_path = Path(f"/tmp/{model}_README.md")
    card_path.write_text(info["card"])
    api.upload_file(path_or_fileobj=str(card_path),
                    path_in_repo="README.md",
                    repo_id=repo_id, repo_type="model")

    # Upload config
    api.upload_file(path_or_fileobj=info["config"],
                    path_in_repo=Path(info["config"]).name,
                    repo_id=repo_id, repo_type="model")

    # Upload weights
    if "ckpt" in info:
        if not Path(info["ckpt"]).exists():
            raise FileNotFoundError(f"missing ckpt: {info['ckpt']}")
        api.upload_file(path_or_fileobj=info["ckpt"],
                        path_in_repo="best.pt",
                        repo_id=repo_id, repo_type="model")
        print(f"  -> uploaded {info['ckpt']}")
    if "dirs" in info:
        for src_dir, dst_subdir in info["dirs"]:
            if not Path(src_dir).exists():
                raise FileNotFoundError(f"missing dir: {src_dir}")
            api.upload_folder(folder_path=src_dir,
                              path_in_repo=dst_subdir,
                              repo_id=repo_id, repo_type="model")
            print(f"  -> uploaded {src_dir} -> {dst_subdir}/")
    print(f"[hf-upload] done: https://huggingface.co/{repo_id}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["clipseg", "grounded_sam", "yoloe", "all"])
    args = ap.parse_args()
    targets = list(REPOS.keys()) if args.model == "all" else [args.model]
    for m in targets:
        upload_one(m)


if __name__ == "__main__":
    main()
