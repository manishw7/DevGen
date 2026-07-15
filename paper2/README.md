# Paper 2 — Devanagari Handwritten Word Generation with Latent Diffusion

Everything for Paper 2 lives here. **Confirmed (re-verified 2026-07-03): no
latent-diffusion handwriting generator exists for Devanagari or any Indic
script** — this is first-in-script work. One caveat: read Word-Diffusion
(ICPR 2024, PHOS-conditioned) eval section before submission to confirm it is
Latin-only; claim stays hedged as "to our knowledge".

## Layout

```
paper2/
  paper/            main.tex (full LNCS draft, results pending) + references.bib
  experiments/
    build_eval_vocab.py     conjunct-stratified vocabularies (already generated)
    vocab/                  iv.txt / oov_seen.txt / oov_unseen_conjunct.txt (200 each)
    content_fidelity.py     TrOCR-judge: is a generated image readable as the right word
    compute_fid.py          visual realism vs real test images (clean-fid)
  ldm/              (planned) the proper rebuild — training code for the from-scratch model
```

Related but shared with Paper 1 (stays outside):
- `backend/ldm_engine.py` + `ldm/controlnet_devanagari_v4/` — the v1 ControlNet generator
- `paper1/experiments/generate_synthetic.py` — generation driver (used for eval sets)
- `paper1/experiments/run_phase2.sh` — runs v1 evaluation + augmentation end-to-end

## Two-stage plan

### v1 — evaluate the existing ControlNet (ready to run)

The trained checkpoint (`ldm/controlnet_devanagari_v4/checkpoint-28000`) gets the
full protocol via `run_phase2.sh` (needs best Paper-1 adapter as judge, GPU ~20h):
FID, content fidelity per stratum (IV / OOV-seen / OOV-unseen-conjunct),
HTR-augmentation utility. Whatever the numbers, they are the paper's baseline.

### v2 — the proper rebuild (the actual contribution)

Current ControlNet is architecturally wrong for the task: SD1.5 photo prior is
dead weight, CLIP cannot read Devanagari (transliteration hack), 256×256
square wrong for words. Rebuild, WordStylist-recipe but script-aware:

1. **Compact LDM from scratch** in VAE latent space at 64×256 (word aspect).
   Reuse SD VAE (frozen) for encode/decode; train only the UNet.
2. **Akshara-level conditioning** — tokenize target text into orthographic
   syllables (`paper1/experiments/akshara.py`), learned embeddings, UNet
   cross-attention. No CLIP anywhere.
   **Research question nobody has answered: does akshara-level conditioning
   beat codepoint-level for unseen-conjunct generalization?** Both tokenizers
   trained as an ablation — this comparison is the paper.
3. **Style**: writer embeddings if writer IDs recoverable from the original
   IIIT-INDIC-HW-WORDS release; otherwise content-only v2 and style as future work.
4. **Eval**: same protocol as v1 → direct comparison table (v1 ControlNet vs
   v2-codepoint vs v2-akshara), plus qualitative grids.

Compute: WordStylist-scale ≈ a few GPU-days. Kaggle T4 (30h/week) over 2-3
weeks, or M5 Pro overnight sessions ~1 week. Training code will be
Kaggle-one-shot like `Kaggle_Paper1_AllInOne.ipynb`.

## Status

- [x] Literature gap confirmed (2026-07-03 search)
- [x] Full paper draft (`paper/main.tex`) — results section pending
- [x] Eval vocabularies + fidelity/FID harness
- [ ] v1 evaluation (`run_phase2.sh` — after Paper 1 sweep frees the GPU)
- [ ] v2 training code (`ldm/`)
- [ ] v2 training runs (codepoint vs akshara ablation)
- [ ] Results + qualitative figures into paper
- [ ] Check Word-Diffusion (ICPR 2024) eval scripts before submitting
