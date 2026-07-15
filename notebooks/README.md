# Model-Training Notebooks (Kaggle)

| Notebook | Trains | Output used by |
|---|---|---|
| `DevGen_LDM_ControlNet.ipynb` | ControlNet LDM (glyph-conditioned, SD1.5) | `ldm/` checkpoints, backend generation, paper2 v1 |
| `DevGen_StyleGAN_Attn.ipynb` | StyleGAN + attention (64×256 words) | report results, paper2 baseline |
| `Kaggle_TrOCR_Training.ipynb` | original TrOCR-LoRA (pre-paper protocol) | `trocr-devanagari-lora-hf/` (superseded by `paper1/Kaggle_Paper1_AllInOne.ipynb`) |

`NotoSansDevanagari-Regular.ttf` — font used by the LDM notebook to render glyph conditioning images.
