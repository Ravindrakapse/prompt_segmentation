#!/usr/bin/env bash
# Setup conda env. Run once on a GPU box.
set -euo pipefail

ENV_NAME="prompt_seg"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | grep -q "^${ENV_NAME}\b"; then
  echo "[setup] env ${ENV_NAME} exists. activating."
else
  echo "[setup] creating env ${ENV_NAME} from environment.yml"
  conda env create -f environment.yml
fi

conda activate "${ENV_NAME}"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import transformers; print('transformers', transformers.__version__)"

echo "[setup] done."
