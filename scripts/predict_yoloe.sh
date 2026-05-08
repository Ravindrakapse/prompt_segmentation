#!/usr/bin/env bash
# YOLOE per-task predict + eval + visualize. Auto-picks fine-tuned ckpt at
# outputs/checkpoints/yoloe/best.pt; else falls back to cfg.yoloe.weights (zero-shot).
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prompt_seg

CFG=configs/yoloe.yaml

if [ -f outputs/checkpoints/yoloe/best.pt ]; then
  TAG="ft"
else
  TAG="zs"
fi

for TASK in crack taping; do
  echo "=== yoloe $TASK ($TAG) ==="
  python -m src.predict_yoloe --config "$CFG" --task "$TASK" --split test \
    --out-pred "outputs/predictions_yoloe/$TASK" \
    --out-viz  "outputs/visualizations_yoloe/$TASK" \
    --out-metrics "outputs/predictions_yoloe/$TASK/metrics.json" \
    --n-viz 6
done
