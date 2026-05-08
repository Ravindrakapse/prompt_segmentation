#!/usr/bin/env bash
# Run Grounded-SAM (Grounding DINO + SAM) per task. Auto-picks fine-tuned
# ckpts at outputs/checkpoints/{gd,sam}_finetuned/best/ if present, else zero-shot.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prompt_seg

CFG=configs/grounded_sam.yaml

for TASK in crack taping; do
  echo "=== grounded-sam $TASK ==="
  python -m src.predict_grounded_sam \
    --config "$CFG" --task "$TASK" --split test \
    --out-pred "outputs/predictions_grounded_sam/$TASK" \
    --out-viz  "outputs/visualizations_grounded_sam/$TASK" \
    --out-metrics "outputs/predictions_grounded_sam/$TASK/metrics.json" \
    --n-viz 6
done
