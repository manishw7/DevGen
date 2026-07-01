---
base_model: paudelanil/trocr-devanagari-2
library_name: peft
license: mit
language:
- ne
- hi
pipeline_tag: image-to-text
tags:
- lora
- peft
- transformers
- trocr
- devanagari
- ocr
datasets:
- c3rl/IIIT-INDIC-HW-WORDS-Hindi
sdk: gradio
sdk_version: 3.50.2
python_version: "3.10"
app_file: app.py
pinned: true
---

# DevGen TrOCR Devanagari LoRA Adapter

This repository contains the **DevGen LoRA adapter** for Devanagari OCR. It is designed to be loaded on top of `paudelanil/trocr-devanagari-2` using the PEFT (Parameter-Efficient Fine-Tuning) library.

## Model Details

- **Developed by:** Manish Wagle / DevGen
- **Base model:** `paudelanil/trocr-devanagari-2`
- **Adapter type:** LoRA
- **Task:** Image-to-text OCR for Devanagari handwritten words and short phrases.
- **Library:** PEFT + Transformers

## Intended Use

Use this adapter for high-precision recognition of Devanagari text from cropped handwritten word images. It is particularly optimized for the **IIIT-INDIC-HW-WORDS** distribution but generalizes well to other handwritten Devanagari styles.

> [!IMPORTANT]
> This model is an OCR engine for word-level recognition. It does not perform page layout analysis, table extraction, or full-page segmentation.

## Loading and Inference

To use this model, you need `transformers`, `peft`, and `torch` installed.

```python
from peft import PeftModel
from transformers import AutoTokenizer, TrOCRProcessor, ViTImageProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image

base_model_id = "paudelanil/trocr-devanagari-2"
adapter_id = "manishwagle/devgen-trocr-devanagari-lora" # Or local path

# Load processors
image_processor = ViTImageProcessor.from_pretrained(base_model_id)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
base_model = VisionEncoderDecoderModel.from_pretrained(base_model_id)
model = PeftModel.from_pretrained(base_model, adapter_id)
model.to(device)

# Inference
image = Image.open("sample_handwritten_word.png").convert("RGB")
pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(f"Recognized Text: {generated_text}")
```

## Demo

A hosted Gradio demo and full DevGen OCR Suite interface is available in the [Hugging Face Space](https://huggingface.co/spaces/manishw10/devgen-devanagari-ocr) and the project repository: [DevGen Project Repository](https://github.com/manishwagle/DevGen).

## Limitations

- **Rotation:** Sensitivity to severe text rotation (beyond 15 degrees).
- **Noise:** Performance may degrade on heavily blurred or low-contrast scans.
- **Script:** Optimized for Devanagari; may not work for other Indic scripts without additional tuning.

## Framework Versions

- PEFT 0.19.1
- Transformers 4.40.0+
- PyTorch 2.1.0+
