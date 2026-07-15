# Parameter-Efficient Fine-Tuning of TrOCR for Devanagari Handwritten Word Recognition on Apple Silicon

Manish, DevGen Project

## Abstract

Handwritten Devanagari recognition remains difficult because character shapes are visually dense, writers vary strongly, and word images often contain inconsistent baselines, strokes, and spacing. This work presents a practical word-level handwritten Devanagari OCR system built by fine-tuning a Transformer-based OCR model with Low-Rank Adaptation (LoRA). The system uses the IIIT-INDIC-HW-WORDS Hindi dataset, a Devanagari-specialized TrOCR checkpoint, and a local Apple Silicon training workflow using PyTorch MPS. The best observed validation Character Error Rate (CER) was 0.1380 at checkpoint 14000. The final application exposes the trained model through a FastAPI backend and a React frontend for upload, preprocessing, inference, confidence visualization, and dataset inspection. The project demonstrates that parameter-efficient adaptation is feasible on consumer Apple Silicon hardware, while also revealing a key deployment lesson: inference preprocessing must match the word-crop distribution used during training.

## Keywords

Devanagari OCR, handwritten word recognition, TrOCR, LoRA, Apple Silicon, MPS, character error rate, IIIT-INDIC-HW-WORDS

## 1. Introduction

Optical Character Recognition (OCR) for handwritten Indic scripts is still substantially harder than printed OCR. Devanagari contains many visually similar glyphs, upper headline structures, conjuncts, vowel marks, and writer-specific stroke patterns. Traditional OCR pipelines often split the problem into detection, segmentation, handcrafted preprocessing, classification, and post-processing. These stages can be fragile when handwriting is irregular.

Recent encoder-decoder OCR models reduce the need for explicit character segmentation. TrOCR frames OCR as image-to-text sequence generation using Transformer components: an image encoder extracts visual representations and a text decoder generates recognized tokens. This makes it suitable for word-level handwritten recognition, especially when the training data consists of paired word images and target transcriptions.

This project explores a practical question: can a Devanagari handwritten word recognizer be trained and deployed locally on a MacBook Pro with Apple Silicon, without Docker and without a dedicated CUDA GPU? The resulting system uses a Devanagari TrOCR base model, LoRA fine-tuning, a local extracted dataset, and a FastAPI inference service.

## 2. Scope

This paper covers only the OCR and model-training portion of DevGen:

- Dataset extraction and preparation
- Devanagari TrOCR model selection
- LoRA fine-tuning on Apple Silicon/MPS
- Inference integration into the backend
- OCR preprocessing and confidence reporting
- Observed metrics, limitations, and next steps

Downstream text-analysis modules are intentionally excluded.

## 3. Related Work

TrOCR introduced an end-to-end Transformer-based OCR architecture using pre-trained image and text Transformer models. Unlike older CNN-RNN OCR systems, TrOCR directly maps document or word images to token sequences and was shown to perform strongly on printed and handwritten text recognition tasks.

LoRA is a parameter-efficient fine-tuning method that freezes the base model and injects trainable low-rank matrices into selected layers. This reduces trainable parameters and memory requirements, which is useful for adapting large pretrained models on constrained hardware.

The dataset used in this work is the Hugging Face conversion of IIIT-INDIC-HW-WORDS-Hindi, originally developed for Indic handwritten text recognition. The selected base checkpoint, `paudelanil/trocr-devanagari-2`, is a TrOCR-style Devanagari handwriting model hosted on Hugging Face and trained for Devanagari script recognition.

## 4. Dataset

The project uses `c3rl/IIIT-INDIC-HW-WORDS-Hindi`, a dataset of handwritten Devanagari/Hindi word images paired with text labels. The dataset was extracted into a local directory:

```text
data/
  train/
    images/
    labels.csv
  validation/
    images/
    labels.csv
  test/
    images/
    labels.csv
```

Observed local split sizes were:

| Split | Label rows excluding header |
|---|---:|
| Train | 69,853 |
| Validation | 12,708 |
| Test | 12,869 |
| Total | 95,430 |

The training script supports two dataset modes:

- Direct Hugging Face loading through `datasets.load_dataset`
- Local extracted loading from `./data`

The local extracted workflow was preferred because it avoids repeated dataset downloads and makes debugging image paths, labels, and preprocessing easier.

## 5. Methodology

### 5.1 Model Selection

The project initially considered generic handwritten TrOCR models such as `microsoft/trocr-base-handwritten`. However, generic English/IAM-oriented checkpoints are not ideal for Devanagari script recognition because the tokenizer, decoder prior, and visual writing distribution differ strongly.

The final training run used:

```text
Base model: paudelanil/trocr-devanagari-2
Adapter: LoRA
Best adapter path: trocr-devanagari-lora/best-checkpoint-14000
```

This choice gave the model a better starting point for Devanagari handwriting than an English handwritten OCR checkpoint.

### 5.2 Processor Fallback

The Devanagari checkpoint did not load cleanly through `TrOCRProcessor.from_pretrained` because its image processor metadata was not recognized by the installed Transformers version. To handle this, the project added a fallback processor path:

- Load tokenizer from the selected Devanagari model
- Load a ViT image processor
- Construct a `TrOCRProcessor` manually
- During inference, prefer the local checkpoint `preprocessor_config.json` when available

This made both training and offline inference more robust.

### 5.3 LoRA Fine-Tuning

LoRA was used instead of full fine-tuning to reduce memory pressure and make training practical on a MacBook Pro using PyTorch MPS. The configuration was:

```text
r = 16
lora_alpha = 32
lora_dropout = 0.05
target_modules = ["query", "value", "key", "dense"]
```

The observed trainable-parameter ratio was approximately:

```text
Trainable parameters: 4,620,288
Total parameters:     188,697,376
Trainable ratio:      2.4485%
```

### 5.4 Training Configuration

The main training command used:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m backend.train_trocr \
  --base-model paudelanil/trocr-devanagari-2 \
  --data-dir ./data \
  --output-dir ./trocr-devanagari-lora \
  --epochs 2 \
  --batch-size 4 \
  --eval-batch-size 4 \
  --grad-accum 2 \
  --learning-rate 1e-5 \
  --eval-limit 1000 \
  --eval-steps 1000 \
  --save-steps 1000 \
  --logging-steps 50
```

The effective batch size was:

```text
batch-size x gradient-accumulation = 4 x 2 = 8
```

Validation was limited to 1,000 examples per evaluation to keep MPS evaluation time practical. This means the reported CER is a useful development metric, but a full validation/test evaluation should still be run before claiming final benchmark performance.

### 5.5 Metric

The primary metric is Character Error Rate (CER):

```text
CER = edit_distance(prediction, reference) / number_of_reference_characters
```

Lower CER is better. CER is appropriate here because Devanagari words may differ by small character-level changes that are important for OCR quality.

## 6. System Implementation

### 6.1 Training Script

The training implementation is contained in `backend/train_trocr.py`. It provides:

- MPS/CUDA/CPU device selection
- Hugging Face or local dataset loading
- TrOCR processor fallback
- LoRA attachment
- CER computation
- MPS-compatible training arguments
- Best-model tracking using validation CER

The trainer was customized to avoid PEFT save issues with `VisionEncoderDecoderConfig` by saving adapter weights with `save_embedding_layers=False`.

### 6.2 Inference Engine

The inference implementation is contained in `backend/trocr_engine.py`. It provides:

- Runtime model artifact inspection
- Automatic checkpoint discovery
- Explicit adapter path support through `TROCR_ADAPTER_PATH`
- MPS selection with CPU fallback
- LoRA adapter loading and optional merge
- Beam-search generation
- Token-level confidence extraction

The application is run with:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 \
TROCR_DEVICE=mps \
TROCR_ADAPTER_PATH=./trocr-devanagari-lora/best-checkpoint-14000 \
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### 6.3 OCR Preprocessing

An important deployment issue was discovered during testing. The model was trained on tight word crops, but the initial backend inference pipeline normalized uploaded images into a square 384 x 384 document-like canvas. This added large white padding and changed the input distribution, causing poor predictions on some uploads.

The inference path was updated to crop around foreground handwriting before recognition. This better matches the word-level dataset distribution. The document-style full preprocessing function remains available separately, but `/api/v1/recognize` and `/api/v1/recognize/full` now use OCR-specific foreground cropping.

## 7. Results

Training was run on the local extracted dataset. The best saved adapter observed during the run was:

```text
trocr-devanagari-lora/best-checkpoint-14000
```

Validation metrics from saved checkpoints:

| Checkpoint | Epoch | Validation CER | Validation Loss |
|---:|---:|---:|---:|
| 9000 | 1.0307 | 0.1409 | 0.2461 |
| 10000 | 1.1452 | 0.1471 | 0.2478 |
| 11000 | 1.2597 | 0.1457 | 0.2352 |
| 12000 | 1.3743 | 0.1428 | 0.2345 |
| 13000 | 1.4888 | 0.1440 | 0.2389 |
| 14000 | 1.6033 | 0.1380 | 0.2339 |

The curve shows useful improvement through checkpoint 9000, temporary degradation at 10000 and 11000, then recovery and the best measured CER at 14000. This supports saving and selecting by CER rather than using the final checkpoint blindly.

### 7.1 Smoke Test

A validation sample was used to confirm that the trained adapter loads and predicts successfully:

```text
Reference:  शरावती
Prediction: शरावती
CER:        0.0
```

This test confirms that the saved LoRA adapter and inference engine function correctly, though it is not a substitute for full validation or real-world testing.

## 8. Discussion

The work shows that a Devanagari-specific base model plus LoRA is a better approach than starting from a generic English handwritten OCR checkpoint. It also shows that MPS-based training is feasible for parameter-efficient fine-tuning, although slower and more sensitive to runtime quirks than CUDA training.

The largest practical issue was not only the model but the mismatch between training and inference image distributions. The dataset consists of cropped handwritten word images. When the deployed app sent padded document-style canvases, recognition quality dropped. After adding foreground cropping, the inference input became closer to the training distribution.

Token confidence should also be interpreted carefully. The generated text is selected by sequence-level beam search. A low-confidence token in the UI does not mean the model deliberately selected a low-confidence alternative; it means the final selected sequence contains at least one uncertain generation step. This is useful for debugging, but the recognized word should be judged by CER or human inspection.

## 9. Limitations

This system is not yet a complete robust handwritten document OCR system. Current limitations include:

- The model is word-level, not full-page text detection plus line/word segmentation.
- Validation CER was computed on a 1,000-sample validation subset during training, not the full validation and test splits.
- The dataset distribution may differ from phone-camera uploads, poor lighting, skewed words, or personal handwriting samples.
- The checkpoint is adapted with LoRA only; full fine-tuning or larger parameter-efficient variants may improve performance.
- The preprocessing pipeline now crops foreground handwriting, but robust detection for multi-word or full-document images is still future work.
- Token-level confidence is approximate and should not be treated as calibrated probability.

## 10. Future Work

The next steps should be:

1. Run full validation and test evaluation for `best-checkpoint-14000`.
2. Add a real word/line detection stage for full-page handwritten documents.
3. Collect project-specific handwriting samples and fine-tune on them.
4. Add augmentation during training: blur, contrast shift, rotation, brightness changes, perspective distortion, and padding variation.
5. Compare three baselines: original Devanagari checkpoint, LoRA checkpoint, and full fine-tuning if hardware allows.
6. Add checkpoint selection and early stopping based on CER.
7. Calibrate or improve confidence reporting using sequence scores and validation-set reliability analysis.

## 11. Conclusion

This project developed a working Devanagari handwritten word OCR pipeline using a TrOCR-based Devanagari model and LoRA fine-tuning on the IIIT-INDIC-HW-WORDS Hindi dataset. The training workflow was adapted for Apple Silicon/MPS and reached a best observed validation CER of 0.1380 on the development validation subset. The work also revealed that deployment quality depends strongly on matching inference preprocessing to the training distribution. The resulting system is a useful foundation for word-level Devanagari handwriting recognition, but robust real-world document OCR will require word detection, stronger augmentation, full test-set evaluation, and fine-tuning on user-specific samples.

## References

1. Minghao Li, Tengchao Lv, Jingye Chen, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, and Furu Wei. "TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models." arXiv:2109.10282, 2021. https://huggingface.co/papers/2109.10282
2. Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09685, 2021. https://huggingface.co/papers/2106.09685
3. `c3rl/IIIT-INDIC-HW-WORDS-Hindi`, Hugging Face dataset card. https://huggingface.co/datasets/c3rl/IIIT-INDIC-HW-WORDS-Hindi
4. Anil Paudel et al. `paudelanil/trocr-devanagari-2`, Hugging Face model card. https://huggingface.co/paudelanil/trocr-devanagari-2












t covers the OCR/model work only: dataset, TrOCR, LoRA fine-tuning, MPS/Apple Silicon setup, checkpoint metrics, preprocessing issue, limitations, and future work. I excluded the NER/downstream extraction part.

I also added references for TrOCR, LoRA, the IIIT-INDIC-HW-WORDS-Hindi dataset, and the Devanagari TrOCR model:

https://huggingface.co/papers/2109.10282
https://huggingface.co/papers/2106.09685
https://huggingface.co/datasets/c3rl/IIIT-INDIC-HW-WORDS-Hindi
https://huggingface.co/paudelanil/trocr-devanagari-2