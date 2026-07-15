# Paper 1 — Parameter-Efficient TrOCR Adaptation for Devanagari Handwritten Word Recognition

Everything needed to produce the experimental section of Paper 1. All metrics
are computed on the **official test split** of
[IIIT-INDIC-HW-WORDS-Hindi](https://huggingface.co/datasets/c3rl/IIIT-INDIC-HW-WORDS-Hindi)
(local parquet mirror: `data/iiit_hindi_parquet`, fetch with
`download_parquets.sh`), fixed seed 42, NFC Unicode normalization.

## The three claims

1. **First systematic PEFT study for Devanagari word recognition** — LoRA
   rank × module-set ablation vs full fine-tune and unadapted bounds.
2. **Input-distribution parity finding** — fine-tuning on raw wide word
   strips while the base checkpoint expects crop/pad-to-square inputs
   *degrades* the model below baseline (word acc 0.12 → 0.02) despite
   converging training loss; predictions keep consonant skeletons but drop
   matras. With parity (`--preprocess app`) the same setup works. Raw-regime
   evidence archived in `results/raw_pipeline_archive/`.
3. **Akshara-level, script-aware evaluation** — AER + conjunct/matra/sign
   error taxonomy that codepoint CER hides.
4. *(links to Paper 2)* **Synthetic augmentation from the Devanagari
   ControlNet-LDM** — dose-response of mixing generated words into training.

## Metrics

| Metric | Definition |
|---|---|
| CER | Levenshtein / reference length over NFC codepoints, mean over samples, 95% bootstrap CI |
| AER | Same, over **akshara clusters** (`akshara.py`) — the unit readers perceive |
| Word accuracy | Exact-match after NFC normalization |

## Pipeline

```
backend/train_trocr.py          training (LoRA presets, full-FT, --preprocess app,
                                parquet source, synthetic mixing, --eval-test)
paper1/experiments/
  evaluate.py                   test-split harness → results/<run>.json
  error_analysis.py             results JSON → conjunct/matra/sign breakdown
  generate_synthetic.py         LDM → synthetic sets (--one-per-word for eval sets)
  make_tables.py                results/*.json → tables.md + tables.tex
  inject_numbers.py             results/*.json → paper1/paper/numbers.tex macros
  run_m5_sweep.sh               PHASE 1: full ablation on Apple Silicon (~20h)
  run_phase2.sh                 PHASE 2: synthetic pool + Paper-2 evals + augmentation
paper2/experiments/
  build_eval_vocab.py           IV / OOV-seen / OOV-unseen-conjunct vocabularies
  content_fidelity.py           recognizer-as-judge CER on generated images
  compute_fid.py                clean-fid vs real test images
```

## Experiment matrix (phase 1)

| Run | What it shows |
|---|---|
| `base_prep` | Unadapted checkpoint, correct preprocessing (real baseline) |
| `lora_r{4,8,16,32}_attn` | Rank vs quality at fixed placement |
| `lora_r16_attn_ffn` | Do FFN adapters help at equal rank? |
| `lora_r16_legacy` | Shipped-adapter config (suffix-matched q/k/v/dense) |
| `full_ft` | Upper bound (1 epoch) |
| archived `base_checkpoint_m16` | Raw-input baseline for the parity ablation |

Phase 2 adds `lora_r16_attn_synth{25,50,100}` (+2.5/5/10% synthetic) and the
Paper-2 fidelity/FID results.

## Running (M5 Pro, MPS — measured ~80 min/LoRA epoch, 14.6 samples/s)

```bash
# Phase 1 (~20h):
nohup caffeinate -i bash paper1/experiments/run_m5_sweep.sh > paper1/runs/sweep.log 2>&1 &

# Phase 2 (after phase 1; ~20h, mostly LDM generation):
nohup caffeinate -i bash paper1/experiments/run_phase2.sh > paper1/runs/phase2.log 2>&1 &

# Refresh paper artifacts anytime:
.venv/bin/python -m paper1.experiments.make_tables
.venv/bin/python -m paper1.experiments.inject_numbers
```

Both scripts are resumable: completed runs are detected and skipped.

## Publication checklist

- [ ] Seed 42 everywhere; bootstrap CIs reported. 3-seed repeat of the best config = stronger.
- [ ] Compare against IIIT-INDIC-HW-WORDS paper baselines; cite DLoRA-TrOCR and state the delta.
- [ ] Trainable-param counts next to every CER (auto via `inject_numbers.py`).
- [ ] Error-taxonomy table + 2–3 qualitative failure images (conjunct confusions).
- [ ] Preprocessing stated exactly (`--preprocess app`); parity ablation included.
- [ ] Synthetic data: example grid real vs generated; FID reported.
- [ ] Release code + adapters; mention in paper.
- [ ] Venues: ICDAR/ICFHR (workshops), DAS, IJDAR; regional fallbacks.
