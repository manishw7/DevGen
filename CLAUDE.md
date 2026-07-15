# DevGen — project context

DevGen = B.Sc. CSIT end-sem project (Devanagari handwriting OCR + generation), feeding **two research papers**. Samir (native Devanagari speaker, Nepal) = first author; target = ICDAR/ICFHR/DAS (Springer LNCS).

## The two papers

- **paper1** = TrOCR-LoRA Devanagari word recognition study. All in `paper1/`: LaTeX (`paper1/paper/main.tex`), experiments, results, trained adapters. Full record: `paper1/EXPERIMENT_LOG.md` — READ before touching paper1. Status: experiments COMPLETE (main sweep + 3-seed robustness), paper written; remaining = admin (co-authors, Overleaf compile, CRNN baseline citation, lit re-check).
- **paper2** = first-ever Devanagari handwriting latent diffusion generator (from scratch, akshara-vs-codepoint conditioning ablation = core novelty). All in `paper2/`: `paper2/README.md` (roadmap), `paper2/UNDERSTANDING.md` (explainer), `paper2/ldm/` (tokenizer/train/sample), `paper2/experiments/` (FID, content fidelity, eval vocab). Status: training in progress on Kaggle (parallel two-GPU runs, target 800 epochs/model; @400 checkpoints in `paper2/runs/`, legibility emerging — flat MSE loss is normal, judge quality late).

## Key facts

- Dataset (both papers): HF `c3rl/IIIT-INDIC-HW-WORDS-Hindi`, local parquets `data/iiit_hindi_parquet/`, pinned rev `2a27244ff5f5f5eaaf86aa4b9411beb356921f51`. ~70k train / 12,869 test words. No writer IDs.
- Training on **Kaggle** (two accounts, one per paper — tokens + CLI recipe in Claude memory `kaggle-accounts.md`). Mac (M5 Pro, MPS) = dev/eval/demo only — no long training locally.
- Notebooks: `paper1/Kaggle_Paper1_AllInOne.ipynb`, `paper1/Kaggle_Paper1_SeedRepeats.ipynb`, `paper2/Kaggle_Paper2_LDM_Scratch.ipynb` — one-shot, resumable, self-contained.
- Python env: `.venv` at repo root, **python 3.9** — no `str | None` runtime annotations. Pin `transformers==4.57.6` (has double-shift loss bug — custom compute_loss in `backend/train_trocr.py` = fix; NEVER train TrOCR via plain `labels=` path).
- Serving (split per paper, 2026-07-04): paper1 recognition API `.venv/bin/python -m uvicorn paper1.backend.app:app --port 8001` (Swagger /docs; all 10 model variants switchable via `?model=` — POST /recognize, /evaluate, GET /models). paper2 generation API `.venv/bin/python -m uvicorn paper2.backend.app:app --port 8002` (Swagger /docs; POST /generate → PNG, /generate/b64). Old monolithic server + college FE backend moved to `legacy/backend_app/` (frontend/ still targets it — run from legacy if demo needed). `backend/` keeps ONLY shared training modules (preprocessing.py, train_trocr.py).
- `report/` = old college end-sem materials (done, don't touch). `notebooks/` = archived original training notebooks.
- LaTeX: `tectonic` installed via brew — compile with `cd paper1/paper && tectonic main.tex` (llncs.cls + splncs04.bst + Noto font staged in that dir; fontspec uses Path=./). Draft PDF: `paper1/Paper1_Draft_TrOCR_LoRA_Devanagari.pdf`.

## Headline numbers (paper1, full test n=12,869)

Zero-shot base CER 16.95% → best LoRA (legacy r16, 4.6M params) 10.13% → full FT 9.61%. LoRA = 93% gain at 2.5% params. 3-seed: full_ft 9.53±0.08, LoRA 10.20±0.10. Conjunct aksharas err 1.6× simple — motivates paper2.