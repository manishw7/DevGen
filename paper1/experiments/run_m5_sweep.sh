#!/usr/bin/env bash
# Paper 1 full sweep on Apple Silicon (M5 Pro, MPS), fully sequential.
# All runs use --preprocess app (crop/pad parity with the base checkpoint) and
# local parquet data. Ordered so the most paper-critical results land first.
#
# Usage: nohup caffeinate -i bash paper1/experiments/run_m5_sweep.sh > paper1/runs/sweep.log 2>&1 &
set -uo pipefail

cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-.venv/bin/python}"
PARQUET_DIR="data/iiit_hindi_parquet"
BASE_MODEL="paudelanil/trocr-devanagari-2"
OUT_ROOT="paper1/runs"
LOG_DIR="$OUT_ROOT/logs"
EPOCHS=2
SEED=42
mkdir -p "$LOG_DIR"

# Keep going if a single run dies; record failures.
FAILURES=()

train() {
  local name="$1"; shift
  if [ -f "$OUT_ROOT/$name/run_config.json" ] && grep -q '"preprocess": "app"' "$OUT_ROOT/$name/run_config.json"; then
    echo "=== [$name] already trained with preprocess=app, skipping ==="
    return 0
  fi
  echo "=== [$name] training start $(date) ==="
  if ! "$PYTHON" backend/train_trocr.py \
      --parquet-dir "$PARQUET_DIR" \
      --base-model "$BASE_MODEL" \
      --output-dir "$OUT_ROOT/$name" \
      --epochs "$EPOCHS" --seed "$SEED" \
      --batch-size 8 --eval-batch-size 16 --grad-accum 2 \
      --eval-strategy epoch --save-strategy epoch \
      --eval-limit 2000 \
      --generation-max-length 32 \
      --preprocess app \
      "$@" > "$LOG_DIR/$name.train.log" 2>&1; then
    echo "!!! [$name] TRAINING FAILED — see $LOG_DIR/$name.train.log"
    FAILURES+=("$name")
    return 1
  fi
  echo "=== [$name] training done $(date) ==="
}

evaluate() {
  local run_name="$1"; shift
  if [ -f "paper1/experiments/results/${run_name}.json" ]; then
    echo "=== [$run_name] already evaluated, skipping ==="
    return 0
  fi
  echo "=== [$run_name] eval start $(date) ==="
  if ! "$PYTHON" -m paper1.experiments.evaluate \
      --run-name "$run_name" \
      --parquet-dir "$PARQUET_DIR" \
      --base-model "$BASE_MODEL" \
      --batch-size 16 --max-length 32 \
      --app-preprocess \
      "$@" > "$LOG_DIR/$run_name.eval.log" 2>&1; then
    echo "!!! [$run_name] EVAL FAILED — see $LOG_DIR/$run_name.eval.log"
    FAILURES+=("eval:$run_name")
    return 1
  fi
  "$PYTHON" -m paper1.experiments.error_analysis \
    --results "paper1/experiments/results/${run_name}.json" \
    > "$LOG_DIR/$run_name.analysis.log" 2>&1 || true
  echo "=== [$run_name] eval done $(date) ==="
}

# --- 0. Baseline: unadapted base checkpoint, correct preprocessing ---
evaluate base_prep

# --- 1. Headline LoRA configs ---
train lora_r16_attn   --lora-r 16 --target-modules attn   && evaluate lora_r16_attn   --adapter-path "$OUT_ROOT/lora_r16_attn"
train lora_r16_legacy --lora-r 16 --target-modules legacy && evaluate lora_r16_legacy --adapter-path "$OUT_ROOT/lora_r16_legacy"

# --- 2. Rank sweep ---
train lora_r4_attn  --lora-r 4  --target-modules attn && evaluate lora_r4_attn  --adapter-path "$OUT_ROOT/lora_r4_attn"
train lora_r8_attn  --lora-r 8  --target-modules attn && evaluate lora_r8_attn  --adapter-path "$OUT_ROOT/lora_r8_attn"
train lora_r32_attn --lora-r 32 --target-modules attn && evaluate lora_r32_attn --adapter-path "$OUT_ROOT/lora_r32_attn"

# --- 3. Module-set sweep ---
train lora_r16_attn_ffn --lora-r 16 --target-modules attn-ffn && evaluate lora_r16_attn_ffn --adapter-path "$OUT_ROOT/lora_r16_attn_ffn"

# --- 4. Full fine-tune upper bound (slowest; 1 epoch) ---
EPOCHS=1 train full_ft --full-finetune --learning-rate 2e-5 && evaluate full_ft --full-model-path "$OUT_ROOT/full_ft"

# --- 5. Tables ---
"$PYTHON" -m paper1.experiments.make_tables > "$LOG_DIR/make_tables.log" 2>&1 || true

echo "=== SWEEP COMPLETE $(date) ==="
if [ "${#FAILURES[@]}" -gt 0 ]; then
  echo "FAILED RUNS: ${FAILURES[*]}"
  exit 1
fi
