#!/usr/bin/env bash
# Download all best ckpts from HF -> local outputs/checkpoints/.
# Use this if you want to skip training and only run prediction.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prompt_seg

python -m src.hf_download
