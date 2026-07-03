#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-./data/raw}"
MODEL_PATH="${2:-./pickle/model.pkl}"
OUTPUT_PATH="${3:-./output/predictions.csv}"

echo "====================================="
echo "ForecastIQ Prediction Pipeline"
echo "====================================="

python src/predict.py \
    --data-dir "$DATA_DIR" \
    --model "$MODEL_PATH" \
    --output "$OUTPUT_PATH"

echo ""
echo "Finished."
echo "Predictions written to $OUTPUT_PATH"