#!/usr/bin/env bash
# Per-task predict + threshold-sweep eval + visualize, single CLIPSeg ckpt.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prompt_seg

CFG=configs/clipseg.yaml
CKPT=outputs/checkpoints/clipseg/best.pt

if [ ! -f "$CKPT" ]; then
  echo "checkpoint not found: $CKPT — run scripts/train_clipseg.sh first (or scripts/download_ckpts_from_hf.sh)" >&2
  exit 1
fi

for TASK in crack taping; do
  case "$TASK" in
    crack)  PROMPT="segment crack" ;;
    taping) PROMPT="segment taping area" ;;
  esac
  echo "=== predict $TASK ==="
  python -m src.predict_clipseg --config "$CFG" --ckpt "$CKPT" --task "$TASK" \
    --split test --out "outputs/predictions_clipseg/$TASK" --prompt "$PROMPT"

  echo "=== eval $TASK ==="
  python -m src.eval --config "$CFG" --ckpt "$CKPT" --task "$TASK" \
    --split test --sweep --out "outputs/predictions_clipseg/$TASK/metrics.json"

  echo "=== visualize $TASK ==="
  python -m src.visualize --config "$CFG" --ckpt "$CKPT" --task "$TASK" \
    --split test --out "outputs/visualizations_clipseg/$TASK" --n 6
done
