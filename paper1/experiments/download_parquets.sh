#!/usr/bin/env bash
# Download the official HF parquet files with resume/retry support.
#
# Usage:
#   bash paper1/experiments/download_parquets.sh [OUT_DIR]
set -euo pipefail

OUT_DIR="${1:-data/iiit_hindi_parquet}"
REPO="c3rl/IIIT-INDIC-HW-WORDS-Hindi"
SHA="2a27244ff5f5f5eaaf86aa4b9411beb356921f51"
BASE_URL="https://huggingface.co/datasets/${REPO}/resolve/${SHA}/data"

mkdir -p "$OUT_DIR"

files=(
  "train-00000-of-00003.parquet"
  "train-00001-of-00003.parquet"
  "train-00002-of-00003.parquet"
  "validation-00000-of-00001.parquet"
  "test-00000-of-00001.parquet"
)

for file in "${files[@]}"; do
  echo "=== ${file} ==="
  curl -L \
    --fail \
    --continue-at - \
    --retry 20 \
    --retry-all-errors \
    --retry-delay 5 \
    --connect-timeout 30 \
    --output "${OUT_DIR}/${file}" \
    "${BASE_URL}/${file}"
done

echo "Downloaded parquet files to ${OUT_DIR}"
