#!/usr/bin/env bash
# Paper 1 ablation sweep: LoRA rank x target modules, plus full fine-tune
# and the unadapted base-checkpoint baseline. Run on a GPU machine (Kaggle: use the notebook).
#
# Usage: bash paper1/experiments/run_ablation.sh [EPOCHS]
set -euo pipefail

cd "$(dirname "$0")/../.."

EPOCHS="${1:-3}"
SEED=42
BASE_MODEL="${TROCR_BASE_MODEL:-paudelanil/trocr-devanagari-2}"
OUT_ROOT="ablation_runs"
PYTHON_BIN="${PYTHON:-python}"

train() {
  local name="$1"; shift
  echo "=== [$name] ==="
  "$PYTHON_BIN" backend/train_trocr.py \
    --base-model "$BASE_MODEL" \
    --output-dir "$OUT_ROOT/$name" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --eval-test \
    "$@"
}

# --- LoRA rank sweep (attention-only modules) ---
for r in 4 8 16 32; do
  train "lora_r${r}_attn" --lora-r "$r" --target-modules attn
done

# --- Module-set sweep at the best-practice rank ---
train "lora_r16_attn_ffn" --lora-r 16 --target-modules attn-ffn
train "lora_r16_legacy"   --lora-r 16 --target-modules legacy

# --- Full fine-tune upper bound ---
train "full_ft" --full-finetune --learning-rate 2e-5

# --- Evaluate everything with the standalone harness (adds CI + AER + per-sample dumps) ---
"$PYTHON_BIN" -m paper1.experiments.evaluate --run-name base_checkpoint --base-model "$BASE_MODEL"
for name in lora_r4_attn lora_r8_attn lora_r16_attn lora_r32_attn lora_r16_attn_ffn lora_r16_legacy; do
  "$PYTHON_BIN" -m paper1.experiments.evaluate --run-name "$name" \
    --base-model "$BASE_MODEL" --adapter-path "$OUT_ROOT/$name"
done
"$PYTHON_BIN" -m paper1.experiments.evaluate --run-name full_ft --full-model-path "$OUT_ROOT/full_ft"

"$PYTHON_BIN" -m paper1.experiments.make_tables
