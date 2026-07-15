# Paper 1 — Experiment Log

Complete record of every experiment, bug, fix, and decision behind
*"Parameter-Efficient Adaptation of TrOCR for Devanagari Handwritten Word Recognition"*.
Kept for reproducibility, methods-section writing, and reviewer rebuttals.

## Setup

- **Base model**: `paudelanil/trocr-devanagari-2` (VisionEncoderDecoder: ViT-base encoder + byte-BPE decoder, 184M params). Special tokens: cls/bos=0 `<s>`, pad=1, sep/eos=2 `</s>`.
- **Dataset**: `c3rl/IIIT-INDIC-HW-WORDS-Hindi`, pinned HF revision `2a27244ff5f5f5eaaf86aa4b9411beb356921f51`. Official splits: ~70k train / 12,869 val / 12,869 test. Local parquets: `data/iiit_hindi_parquet/`.
- **Hardware**: final runs on Kaggle T4; development/diagnostics on M5 Pro Mac (MPS). Environment pins: `transformers==4.57.6`, peft, torchao **uninstalled** (0.10 breaks peft dispatch on Kaggle).
- **Metrics**: CER (NFC-normalized codepoints, 95% percentile bootstrap, 1000 resamples), AER (akshara error rate; segmentation `(C virama)* C nukta? matra? sign*`), word accuracy. All on full official test split, beam=4, max length 32.
- **Training**: seed 42, AdamW. LoRA: lr 5e-5, batch 16, 2 epochs, α=2r, dropout 0.05. Full FT: lr 2e-5, batch 8, 1 epoch.

## Timeline

### 2026-07-02 — Initial sweep attempts (Mac) — POISONED, archived

- First LoRA runs *degraded* the model (WAcc 12% → 2%, matras dropped, e.g. अनाथों→अनथथय) despite converging training loss.
- Archived: `experiments/results/raw_pipeline_archive/`.

### 2026-07-03 — Bug #1: preprocessing parity

- Base checkpoint expects **crop-to-foreground / pad-to-square** inputs (AR≤1.55 crop; AR>2.2 crop + 384-square; else raw) — same as app serving path.
- Raw strips fed to ViT processor = distribution shift. Base zero-shot: raw CER **0.5297** vs preprocessed CER **0.1695** (full test).
- Fix: `--preprocess app` (default) in `backend/train_trocr.py`, `preprocess_pil_for_ocr` in `backend/preprocessing.py`. → Paper contribution #2.

### 2026-07-03 — Bug #2: transformers 4.57.6 double-shift loss

- Even with preprocessing fixed, all runs collapsed (CER 0.53+, AER>1.3). Diagnosis: `VisionEncoderDecoderModel(labels=)` routes through `ForCausalLMLoss` which shifts targets a **second** time after decoder_input_ids were already shifted → model trained to predict *previous* token.
- Evidence: manual aligned teacher-forced loss **0.4535** vs internal **15.7503** on identical batch.
- Fix: custom `compute_loss` in `TrOCRSeq2SeqTrainer` (manual decoder_input_ids + CE against unshifted labels). Post-fix smoke (1ep/2k samples): test CER 0.128 < base 0.169 ✓.
- Note: `eval_loss` during training still reads ~17 (internal buggy path in prediction_step) — cosmetic only; CER/WER metrics correct.
- Poisoned artifacts archived: `experiments/results/doubleshift_archive/`.
- Kaggle-specific fixes: `pip uninstall torchao` before peft import; DataParallel unwrap loop in compute_loss (T4×2 hides `model.config`).

### 2026-07-03 — Main sweep (Kaggle T4, `Kaggle_Paper1_AllInOne.ipynb`, ~11h)

Full test set, n=12,869, seed 42:

| Run | Trainable | CER | CER CI95 | AER | WAcc |
| --- | ---: | ---: | --- | ---: | ---: |
| base_prep (zero-shot) | — | 0.1695 | [0.1643, 0.1745] | 0.2213 | 0.6461 |
| full_ft | 184.08M | **0.0961** | [0.0921, 0.1005] | 0.1192 | **0.7980** |
| lora_r16_legacy | 4.62M | 0.1013 | [0.0972, 0.1054] | 0.1307 | 0.7786 |
| lora_r16_attn_ffn | 4.62M | 0.1022 | [0.0981, 0.1062] | 0.1315 | 0.7774 |
| lora_r32_attn | 4.72M | 0.1030 | [0.0988, 0.1075] | 0.1326 | 0.7760 |
| lora_r16_attn | 2.36M | 0.1065 | [0.1022, 0.1108] | 0.1368 | 0.7691 |
| lora_r8_attn | 1.18M | 0.1095 | [0.1053, 0.1139] | 0.1411 | 0.7614 |
| lora_r4_attn | 0.59M | 0.1141 | [0.1099, 0.1185] | 0.1466 | 0.7534 |

- **Paired bootstrap** (2000 resamples, per-sample edit distances, NFC): full_ft vs lora_r16_legacy = **+0.41pp CER, CI95 [0.08, 0.76], p≈0.009**. Full FT statistically better; LoRA recovers **93%** of zero-shot→full-FT gain with 2.5% of params.
- **Placement > rank**: at matched ~4.6M budget, FFN placement (attn_ffn 0.1022, legacy 0.1013) beats attention-only r32 (0.1030).
- **Error taxonomy** (full_ft / lora_r16_legacy, akshara-level): base subst 39.1/40.1%, conjunct subst 31.2/28.6%, deletion 10.3/10.5%, matra 9.9/10.6%, insertion 4.5/5.1%, nasalization sign 3.4/3.6%. Conjunct akshara error rate 0.170/0.190 vs simple 0.108/0.115 (~1.6×). Full FT cuts matra (−13%) and base (−9%) errors vs LoRA; conjunct subs unchanged (+2%) → **conjuncts = encoder-level open problem**, motivates Paper 2 synthetic data.
- Raw artifacts: `paper1/experiments/results/*.json` (per-sample predictions included), adapters `paper1/runs/*/`, full-FT weights `paper1/full_ft_model.zip` (651MB, unextracted).

### 2026-07-03 — Paper assembly

- `numbers.tex` auto-injected (`python -m paper1.experiments.inject_numbers`); tables via `make_tables`.
- `main.tex`: results prose §5.1 (rank curve, placement, paired-bootstrap gap), taxonomy table + analysis §5.3, qualitative figure (6 test crops), parity section quantified, empty synthetic-augmentation section **removed** (deferred to Paper 2), setup corrected to Kaggle T4.
- Figures: `paper/figures/rank_curve.pdf` (CER vs params, CIs); `qual_{00019,00084,00063,00026,00138,00296}.png` — test-parquet row index == results-JSON sample index (verified against references).
  - 19 प्रोसैसर ✓✓ · 84 पदोन्नति ✓✓ · 63 अत्यधिक→अर्थशक (conjunct, both fail) · 26 ह्रास→द्वास (conjunct, both fail) · 138 अपनाता→अपनाना (base subst, both) · 296 दुश्मन (full FT ✓, LoRA दुःशम ✗).
- Static checks pass: no missing macros/refs/bib keys/figures, no TODOs.
- **No LaTeX on Mac** — compile on Overleaf: XeLaTeX, `llncs.cls` (Springer), Noto Sans Devanagari.

### 2026-07-03 — Live demo verification

- BE `uvicorn backend.main:app` :8000 with `TROCR_ADAPTER_PATH=paper1/runs/lora_r16_legacy`; FE Vite :5173. POST `/api/v1/recognize` with real handwriting → works, ~2.1s/word on MPS, `using_adapter: true`.
- Fixed `backend/main.py:580` `str | None` → `Optional[str]` (venv is py3.9).

### 2026-07-04 — Seed robustness runs (Kaggle account 2: sameerwagle, `kaggle-paper1-seed-repeat`)

Seeds 43, 44 for both headline configs (full test, n=12,869):

| Run | CER | AER | WAcc |
| --- | ---: | ---: | ---: |
| lora_r16_legacy_s43 | 0.1017 | 0.1315 | 0.7774 |
| lora_r16_legacy_s44 | 0.1031 | 0.1326 | 0.7766 |
| full_ft_s43 | 0.0954 | 0.1192 | 0.7991 |
| full_ft_s44 | 0.0945 | 0.1172 | 0.8009 |

**3-seed aggregates (42/43/44):** lora_r16_legacy CER **0.1020 ± 0.0010**, WAcc 0.7775 ± 0.0010 · full_ft CER **0.0953 ± 0.0008**, WAcc 0.7993 ± 0.0015. Seed variance ~10× smaller than all reported effects; gap full-FT↔LoRA (~0.6pp) stable across seeds. Robustness sentence added to main.tex §5.1. Artifacts: `paper1/experiments/results/*_s4[34].json`, adapters `paper1/runs/*_s4[34]/`.

## Pending / TODO before submission
- [ ] One line comparing CRNN baseline from dataset paper (Gongidi & Jawahar, ICDAR 2021) vs our WAcc 0.7980.
- [ ] Fill co-author names/affiliations (main.tex line ~30), real GitHub URL in conclusion.
- [x] Compile check — DONE 2026-07-04 locally via `tectonic` (brew): llncs.cls/splncs04.bst from CTAN + Noto font staged in `paper/`; `cd paper1/paper && tectonic main.tex` → 7-page PDF, Devanagari/figures/bib all render. Draft: `paper1/Paper1_Draft_TrOCR_LoRA_Devanagari.pdf`.
- [ ] Pre-submission literature re-check: any new Devanagari/Indic PEFT-OCR paper; DLoRA-TrOCR differentiation intact.
- [ ] Venue: ICDAR / ICFHR / DAS (Springer LNCS).

## Known asymmetries / honest caveats

- Full FT ran 1 epoch (batch 8) vs LoRA 2 epochs (batch 16) — stated in setup; conservative for the LoRA claim (more full-FT training could only widen the gap it already wins).
- Single dataset/language (Hindi words). Depth-over-breadth defense; multi-script = journal extension.
- Bootstrap CIs are over test samples; seed variance addressed by the pending seed runs.

## Nepali adaptation extension (2026-07-04, kernel sameerwagle/paper1-nepali-adaptation v3)

- **Data**: NHD (sweekardahal/nepali-handwritten-images-for-text-detection, CC0) — 1,000 phone-captured pages, detection boxes only (NO transcriptions). Cropped: 61,825 train / 13,942 test word images.
- **Pseudo-labeling**: full_ft + lora_r16_legacy both predict every crop; keep only exact agreement + min-confidence ≥0.9 + nuqta-normalized (ज़→ज etc.) + pure-Devanagari → **16,109 train labels** (26% yield). Rationale: raw confidence unreliable cross-lingually (observed आमा→आकाश @0.973; systematic Hindi nuqta hallucinations).
- **Training**: fresh LoRA r16 legacy, 2 epochs on IIIT-Hindi 70k + 16,109 Nepali (via --synthetic-dir mixing), seed 42, T4.
- **Results**: Hindi test CER **0.1065** [0.1024, 0.1110], WAcc 0.7737 (vs Hindi-only 0.1013/0.7786 → mild forgetting +0.5pp). Nepali heldout agreement with pseudo-consensus: 3120/3589 = **86.9%** (proxy only — trained toward consensus, inflated by construction).
- **Pending human eval**: `verify_sheet/` (500 random NHD-test crops + sheet.csv with both models' predictions + empty human_truth column). Samir fills → real Nepali accuracy for base/full_ft/adapted → decides Paper-3 viability (Hindi→Nepali transfer via agreement-filtered self-training; NHD age stratification enables kids/youth/adults analysis).
- Adapter: `paper1/runs/lora_r16_legacy_nepali/` — serve via `TROCR_ADAPTER_PATH=paper1/runs/lora_r16_legacy_nepali`.
- Ops note: two failed attempts first — (v1) NHD XML boxes exceed image bounds → clamp before crop; (v2) Kaggle assigned P100 where "latest" container torch has no sm_60 kernels → GPU-guard cell (subprocess CUDA test + torch 2.5.1 fallback) now standard in all notebooks; push via raw HTTP with machineShape=NvidiaTeslaT4.
