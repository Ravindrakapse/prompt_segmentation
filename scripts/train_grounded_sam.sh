#!/usr/bin/env bash
# Fine-tune Grounding DINO + SAM mask decoder (independently).
# After both, scripts/predict_grounded_sam.sh auto-picks fine-tuned ckpts.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prompt_seg

mkdir -p outputs/logs

echo "=== fine-tune Grounding DINO ==="
python -u -m src.train_grounding_dino --config configs/grounded_sam.yaml \
  2>&1 | tee outputs/logs/gd_finetune_run.log

echo "=== fine-tune SAM mask decoder ==="
python -u -m src.train_sam --config configs/grounded_sam.yaml \
  2>&1 | tee outputs/logs/sam_finetune_run.log
