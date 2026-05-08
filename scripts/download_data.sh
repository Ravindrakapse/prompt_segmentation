#!/usr/bin/env bash
# Fetch both source datasets via Roboflow SDK.
#   - crack:  ravindra-kapse/cracks-3ii36-fdqni v1
#   - taping: objectdetect-pu6rn/drywall-join-detect v2
# Output: data/processed/<task>/{images,masks}/{train,val,test}/
# Also builds YOLO-seg copy at data/processed/yolo_unified/ (for yoloe).
# Requires: ROBOFLOW_API_KEY env var.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prompt_seg

if [ -z "${ROBOFLOW_API_KEY:-}" ]; then
  echo "set ROBOFLOW_API_KEY first (export ROBOFLOW_API_KEY=...)" >&2
  exit 1
fi

echo "=== crack: SDK download ==="
python -m src.data.download --task crack  --out-root data/processed

echo "=== taping: SDK download ==="
python -m src.data.download --task taping --out-root data/processed

echo "=== build YOLO seg dataset (yolo_unified) ==="
rm -rf data/processed/yolo_unified
python -m src.data.build_yolo_dataset --config configs/yoloe.yaml \
  --out data/processed/yolo_unified
