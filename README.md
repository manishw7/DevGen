# DevGen — Devanagari Handwritten Text Recognition & Generation

A hybrid OCR pipeline and GAN-based data augmentation framework for Devanagari (Nepali/Hindi) handwritten text.

## Overview

DevGen combines two core capabilities:

1. **OCR Pipeline** — A hybrid CNN + TrOCR-LoRA system that recognizes handwritten Devanagari characters and words from images.
2. **GAN Generator** — A ResNet GAN that produces synthetic handwritten word images for training data augmentation.

### Architecture

```
Input Image
    │
    ├─ ImageRouter (aspect ratio + ink analysis)
    │       │
    │       ├── Character detected → CNN Classifier (46 classes)
    │       │                         32×32 grayscale, ~500K params
    │       │
    │       └── Word detected ────→ TrOCR + LoRA Adapter
    │                                 microsoft/trocr-base-handwritten
    │                                 Fine-tuned on IIIT-INDIC-HW-WORDS-Hindi
    │
    └─ Output: recognized text + confidence scores + NER entities
```

## Project Structure

```
DevGen/
├── backend/
│   ├── main.py               # FastAPI server
│   ├── trocr_engine.py        # TrOCR + LoRA inference engine
│   ├── cnn_model.py           # CNN character classifier (46 classes)
│   ├── image_router.py        # Smart routing: character vs word
│   ├── preprocessing.py       # Image preprocessing pipeline
│   ├── dataset_loader.py      # HuggingFace dataset utilities
│   ├── ner_extractor.py       # Named Entity Recognition
│   ├── train_trocr.py         # TrOCR LoRA fine-tuning script
│   ├── train_cnn.py           # CNN training script
│   └── train_nepali_gan.py    # GAN training script (local)
├── frontend/                  # React/Vite web interface
├── docs/
│   └── research_paper.md      # Reference: Chhatkuli et al. (2021)
├── test_images/               # Sample handwritten images for testing
├── DevGen_GAN_Notebook.ipynb  # Kaggle notebook for GAN training
├── Kaggle_TrOCR_Training.ipynb # Kaggle notebook for TrOCR fine-tuning
├── devanagari-cnn-classifier.pt # Trained CNN weights
├── trocr-devanagari-lora/     # Trained LoRA adapter weights
├── trocr-devanagari-lora-hf/  # HuggingFace Hub format adapter
├── requirements.txt
└── docker-compose.yml
```

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend)

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify PyTorch device (Apple Silicon):
```bash
python -c "import torch; print(torch.backends.mps.is_available())"
# Should print: True
```

## Usage

### Run the App

**Backend:**
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m uvicorn backend.main:app --reload
```

**Frontend:**
```bash
cd frontend && npm install && npm run dev
```

Open `http://localhost:5173`

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check + model status |
| `GET` | `/api/v1/model/info` | Model configuration details |
| `POST` | `/api/v1/recognize` | OCR: image → recognized text |
| `POST` | `/api/v1/recognize/full` | OCR + NER in one request |
| `POST` | `/api/v1/evaluate` | Calculate CER between strings |
| `POST` | `/api/v1/ner` | Extract entities from text |
| `POST` | `/api/v1/preprocess` | Preprocess a document image |

### Train Models

**TrOCR LoRA (local):**
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m backend.train_trocr \
    --epochs 3 --batch-size 4 --eval-batch-size 4 --grad-accum 2
```

**CNN Classifier:**
```bash
python -m backend.train_cnn
```

**GAN (local):**
```bash
python -m backend.train_nepali_gan \
    --data-dir data/Images/Images --steps 100000 --batch-size 64
```

**GAN (Kaggle — recommended):**
Upload `DevGen_GAN_Notebook.ipynb` to Kaggle with GPU T4 x2 enabled.

## GAN Architecture

The GAN generates 64×64 grayscale handwritten Devanagari word images. Inspired by the methodology in Chhatkuli et al. (2021), enhanced with modern techniques:

| Component | Technique |
|-----------|-----------|
| Generator | ResNet blocks with bilinear upsampling |
| Discriminator | ResNet blocks with spectral normalization + avg-pool |
| Loss | Hinge loss |
| Stabilization | DiffAugment (color, translation, cutout) |
| Inference | Exponential Moving Average (EMA, decay=0.999) |
| Optimizer | Adam with TTUR (G: 1e-4, D: 4e-4, β=(0, 0.9)) |
| Training | 100k steps, n_critic=2 |

## OCR Performance

| Metric | Score |
|--------|-------|
| Exact Match Accuracy | 80% |
| Character Error Rate (CER) | 0.079 |
| Model | TrOCR-base + LoRA (r=16) |
| Dataset | IIIT-INDIC-HW-WORDS-Hindi |

## Models

| Model | Purpose | Size |
|-------|---------|------|
| `devanagari-cnn-classifier.pt` | Character classification (46 classes) | ~2.7 MB |
| `trocr-devanagari-lora/` | Word-level OCR (LoRA adapter) | ~18.5 MB |
| `generator_ema.pth` | GAN generator (trained on Kaggle) | ~10 MB |

## References

- Chhatkuli, R.K., Baral, H.P., & KC, S. (2021). *Generating Nepali Handwritten Letters and Words Using Generative Adversarial Networks.*
- Microsoft TrOCR: `microsoft/trocr-base-handwritten`
- Dataset: `c3rl/IIIT-INDIC-HW-WORDS-Hindi`

## License

This project is for academic and research purposes.
