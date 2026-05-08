#!/usr/bin/env bash
# Push best ckpts to HF Hub. Requires `hf auth login` first.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prompt_seg

python -m src.hf_upload --model all
