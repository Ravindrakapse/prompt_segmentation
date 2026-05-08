#!/usr/bin/env bash
# Fine-tune YOLOE-seg on unified task pool.
# 1. (one-time) build YOLO seg dataset from processed masks
# 2. fine-tune via ultralytics YOLOEPESegTrainer
# 3. best.pt copied to outputs/checkpoints/yoloe/best.pt
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prompt_seg

mkdir -p outputs/logs

CFG=configs/yoloe.yaml
DATA_YAML=data/processed/yolo_unified/dataset.yaml

if [ ! -f "$DATA_YAML" ]; then
  echo "=== building YOLO seg dataset (one-time) ==="
  python -m src.data.build_yolo_dataset --config "$CFG" --out data/processed/yolo_unified
fi

echo "=== fine-tune YOLOE ==="
python -u -m src.train_yoloe --config "$CFG" 2>&1 | tee outputs/logs/yoloe_finetune_run.log
