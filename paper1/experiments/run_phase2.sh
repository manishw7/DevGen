#!/usr/bin/env bash
# Phase 2 (run AFTER run_m5_sweep.sh completes — needs the GPU free and the
# best adapter from phase 1):
#   1. LDM synthetic training pool (7k images, ~6-10h on MPS)
#   2. Paper 2 eval sets (IV / OOV-seen / OOV-unseen-conjunct, 600 images)
#   3. FID + content fidelity for Paper 2
#   4. Augmentation dose-response for Paper 1 (+2.5/5/10% synthetic)
#   5. Refresh tables + paper number macros
#
# Usage: nohup caffeinate -i bash paper1/experiments/run_phase2.sh > paper1/runs/phase2.log 2>&1 &
set -uo pipefail

cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-.venv/bin/python}"
PARQUET_DIR="data/iiit_hindi_parquet"
BASE_MODEL="paudelanil/trocr-devanagari-2"
OUT_ROOT="paper1/runs"
LOG_DIR="$OUT_ROOT/logs"
POOL_DIR="data_synth/ldm_pool"
POOL_SIZE=7000
JUDGE_ADAPTER="${JUDGE_ADAPTER:-$OUT_ROOT/lora_r16_attn}"  # best phase-1 adapter
SEED=42
mkdir -p "$LOG_DIR"

# --- 1. Synthetic training pool ---
if [ ! -f "$POOL_DIR/labels.csv" ]; then
  echo "=== generating $POOL_SIZE-image synthetic pool $(date) ==="
  "$PYTHON" -m paper1.experiments.generate_synthetic \
    --count "$POOL_SIZE" --out-dir "$POOL_DIR" --seed "$SEED" \
    > "$LOG_DIR/gen_pool.log" 2>&1 || echo "!!! pool generation failed"
fi

# --- 2. Paper 2 evaluation sets (one image per vocab word) ---
for stratum in iv oov_seen oov_unseen_conjunct; do
  if [ ! -f "data_synth/eval_${stratum}/labels.csv" ]; then
    echo "=== generating eval set: $stratum $(date) ==="
    "$PYTHON" -m paper1.experiments.generate_synthetic \
      --vocab-file "paper2/experiments/vocab/${stratum}.txt" \
      --one-per-word --out-dir "data_synth/eval_${stratum}" \
      --seed "$SEED" --save-control \
      > "$LOG_DIR/gen_${stratum}.log" 2>&1 || echo "!!! $stratum generation failed"
  fi
done

# --- 3. Paper 2 metrics ---
for stratum in iv oov_seen oov_unseen_conjunct; do
  echo "=== content fidelity: $stratum $(date) ==="
  "$PYTHON" -m paper2.experiments.content_fidelity \
    --generated-dir "data_synth/eval_${stratum}" \
    --base-model "$BASE_MODEL" --adapter-path "$JUDGE_ADAPTER" \
    --run-name "fidelity_${stratum}" \
    > "$LOG_DIR/fidelity_${stratum}.log" 2>&1 || echo "!!! fidelity $stratum failed"
done
echo "=== FID $(date) ==="
"$PYTHON" -m paper2.experiments.compute_fid \
  --generated-dir "data_synth/eval_iv/images" --run-name eval_iv \
  --parquet-dir "$PARQUET_DIR" --real-limit 5000 \
  > "$LOG_DIR/fid.log" 2>&1 || echo "!!! FID failed"

# --- 4. Augmentation dose-response (Paper 1) ---
for pct in 25 50 100; do  # % of the 7k pool => +2.5/5/10% of 70k real
  limit=$((POOL_SIZE * pct / 100))
  name="lora_r16_attn_synth${pct}"
  if [ -f "$OUT_ROOT/$name/run_config.json" ]; then
    echo "=== [$name] already trained, skipping ==="
  else
    echo "=== [$name] training with $limit synthetic $(date) ==="
    "$PYTHON" backend/train_trocr.py \
      --parquet-dir "$PARQUET_DIR" \
      --base-model "$BASE_MODEL" \
      --output-dir "$OUT_ROOT/$name" \
      --epochs 2 --seed "$SEED" \
      --batch-size 8 --eval-batch-size 16 --grad-accum 2 \
      --eval-strategy epoch --save-strategy epoch --eval-limit 2000 \
      --generation-max-length 32 --preprocess app \
      --lora-r 16 --target-modules attn \
      --synthetic-dir "$POOL_DIR" --synthetic-limit "$limit" \
      > "$LOG_DIR/$name.train.log" 2>&1 || { echo "!!! $name failed"; continue; }
  fi
  "$PYTHON" -m paper1.experiments.evaluate \
    --run-name "$name" --parquet-dir "$PARQUET_DIR" \
    --base-model "$BASE_MODEL" --adapter-path "$OUT_ROOT/$name" \
    --batch-size 16 --max-length 32 --app-preprocess \
    > "$LOG_DIR/$name.eval.log" 2>&1 || echo "!!! $name eval failed"
done

# --- 5. Tables + paper macros ---
"$PYTHON" -m paper1.experiments.make_tables > "$LOG_DIR/make_tables2.log" 2>&1 || true
"$PYTHON" -m paper1.experiments.inject_numbers >> "$LOG_DIR/make_tables2.log" 2>&1 || true

echo "=== PHASE 2 COMPLETE $(date) ==="
