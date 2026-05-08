#!/usr/bin/env bash
# Train CLIPSeg fine-tune (single ckpt, both classes via prompt).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prompt_seg

mkdir -p outputs/logs
python -u -m src.train_clipseg --config configs/clipseg.yaml \
  2>&1 | tee outputs/logs/clipseg_run.log
