# Prompt Segmentation for Drywall QA

Text-conditioned binary segmentation. Given an RGB image and a natural-language
prompt, emits a `{0, 255}` mask at the original resolution.

- `"segment crack"` → wall-crack mask
- `"segment taping area"` → drywall taping-seam mask

Three pipelines compared on the same task:

| Pipeline | What it is | Trainable | Test Dice (crack / taping) | ms/img |
| --- | --- | --- | --- | --- |
| **CLIPSeg** | HF CLIPSeg, decoder + FiLM unfrozen, focal-Dice loss | 1.13 M | 0.672 / 0.727 | ~240 |
| **Grounded-SAM** | Grounding DINO bboxes → SAM masks → union, both fine-tuned | 39.9 M | 0.601 / 0.515 | ~520 |
| **YOLOE** | Open-vocab YOLO11 (yoloe-11s-seg) + text-prompted seg head | 35.2 M | **0.681** / **0.859** | **~30** |

YOLOE is best on Dice and latency. CLIPSeg is the lightest. Grounded-SAM is the
two-stage open-vocab baseline.

---

## Repo layout

```
configs/
  clipseg.yaml              CLIPSeg fine-tune config
  grounded_sam.yaml         Grounded-SAM (GD + SAM) config
  yoloe.yaml                YOLOE config
src/
  data/                     dataset + transforms + Roboflow downloader + YOLO seg builder
  models/                   model wrappers (clipseg, grounded_sam, yoloe)
  train_clipseg.py          CLIPSeg fine-tune
  predict_clipseg.py
  train_grounding_dino.py   Grounded-SAM stage 1
  train_sam.py              Grounded-SAM stage 2
  predict_grounded_sam.py
  train_yoloe.py            YOLOE fine-tune (Ultralytics YOLOEPESegTrainer)
  predict_yoloe.py
  eval.py | losses.py | metrics.py | utils.py | visualize.py
  hf_upload.py              push best ckpts to HF Hub
  hf_download.py            pull best ckpts from HF Hub
scripts/
  setup_env.sh              create conda env (one-time)
  download_data.sh          fetch Roboflow datasets + build YOLO seg copy
  train_clipseg.sh          train CLIPSeg
  predict_clipseg.sh        predict + eval + visualize
  train_grounded_sam.sh     fine-tune GD + SAM
  predict_grounded_sam.sh
  train_yoloe.sh            fine-tune YOLOE
  predict_yoloe.sh
  upload_to_hf.sh           push all 3 best ckpts to HF
  download_ckpts_from_hf.sh pull all 3 best ckpts from HF (skip training)
load_models.py              load each model from HF + run inference on one image
tests/test_smoke.py         cheap unit tests (no model load)
```

`data/` and `outputs/` are generated and gitignored.

---

## Pretrained checkpoints

All three best checkpoints are hosted on HuggingFace Hub:

| Model | Repo |
| --- | --- |
| CLIPSeg | [`ravindrakapse/drywall-clipseg`](https://huggingface.co/ravindrakapse/drywall-clipseg) |
| Grounded-SAM | [`ravindrakapse/drywall-grounded-sam`](https://huggingface.co/ravindrakapse/drywall-grounded-sam) |
| YOLOE | [`ravindrakapse/drywall-yoloe`](https://huggingface.co/ravindrakapse/drywall-yoloe) |

Pull all into `outputs/checkpoints/` in one shot:

```bash
bash scripts/download_ckpts_from_hf.sh
```

Then run any `scripts/predict_*.sh` without training.

---

## Quick inference (single image)

```bash
python load_models.py \
  --image path/to/wall.jpg \
  --prompt "segment crack" \
  --model yoloe \
  --out mask.png
```

`--model all` runs all three pipelines and writes one mask per pipeline.

---

## Reproduce from scratch

```bash
# 1) conda env (one-time)
bash scripts/setup_env.sh

# 2) Roboflow API key
export ROBOFLOW_API_KEY=...

# 3) datasets -> data/processed/{crack,taping}/ + data/processed/yolo_unified/
bash scripts/download_data.sh

# 4) train each model (any subset; ~6 min CLIPSeg, ~75 min GD+SAM, ~1 hr YOLOE on A100 80GB)
bash scripts/train_clipseg.sh
bash scripts/train_grounded_sam.sh
bash scripts/train_yoloe.sh

# 5) predict + eval + visualize
bash scripts/predict_clipseg.sh
bash scripts/predict_grounded_sam.sh
bash scripts/predict_yoloe.sh

# 6) (optional) push best ckpts to your own HF Hub
bash scripts/upload_to_hf.sh
```

Artifacts:

| What | Path |
| --- | --- |
| Checkpoints | `outputs/checkpoints/{clipseg,gd_finetuned,sam_finetuned,yoloe}/` |
| Predictions | `outputs/predictions_{clipseg,grounded_sam,yoloe}/<task>/<id>__<prompt>.png` |
| Metrics JSON | `outputs/predictions_*/<task>/metrics.json` |
| Visual panels | `outputs/visualizations_*/<task>/<id>__panel.png` |
| TensorBoard | `outputs/logs/` |

---

## Datasets

| Task | Source | Split |
| --- | --- | --- |
| Crack  | Roboflow [`ravindra-kapse/cracks-3ii36-fdqni`](https://universe.roboflow.com/ravindra-kapse/cracks-3ii36-fdqni) v1 | train 5146 / val 103 / test 102 |
| Taping | Roboflow [`objectdetect-pu6rn/drywall-join-detect`](https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect) v2 | train 820 / val 101 / test 101 |

Downloaders (`src/data/download.py`) deterministically rebalance val:test 50/50
(seed 42) so each task has a meaningful test split.

---

## Why a single unified CLIPSeg model

Two per-task ckpts would each ignore the prompt. To make the prompt actually
pick the class, training mixes in **cross-task negatives**:

> with probability 0.3, swap a sample's prompt with one from the *other* task
> and zero out the mask.

Forces the decoder to listen to text. See `UnifiedPromptedSegDataset` in
[`src/data/dataset.py`](src/data/dataset.py) and `train.neg_prob` in
[`configs/clipseg.yaml`](configs/clipseg.yaml).

---

## Reproducibility

- Global seed `42` (`src.utils.set_seed`).
- Each CLIPSeg checkpoint embeds its full YAML config in `state["config"]`.
- Versions pinned in `environment.yml` / `requirements.txt`.

---

## Tests

```bash
PYTHONPATH=. pytest tests/
```

Cheap — no model load, no data dependency.

---

## License

Apache-2.0.
