# Legacy

`backend_app/` — the original monolithic DevGen FastAPI server (college end-sem demo):
recognition + NER + dataset browser + GAN/LDM generation, consumed by `frontend/`.

Retired 2026-07-04 when the project split into per-paper backends:

- `paper1/backend/app.py` — TrOCR-LoRA recognition API (Swagger at /docs)
- `paper2/backend/app.py` — LDM handwriting generation API (Swagger at /docs)

`backend/` at the repo root now contains ONLY the shared training/preprocessing
modules (`preprocessing.py`, `train_trocr.py`) that the paper experiments and
Kaggle notebooks import. The old frontend still expects this legacy server; run
it from here if the college demo is ever needed again.
