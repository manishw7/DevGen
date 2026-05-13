---
base_model: paudelanil/trocr-devanagari-2
library_name: peft
license: mit
language:
- ne
- hi
pipeline_tag: image-to-text
tags:
- base_model:adapter:paudelanil/trocr-devanagari-2
- lora
- peft
- transformers
- trocr
- devanagari
- ocr
datasets:
- imagefolder
---

# DevGen TrOCR Devanagari LoRA Adapter

This repository contains the DevGen LoRA adapter for Devanagari OCR. It is intended to be loaded on top of `paudelanil/trocr-devanagari-2` with PEFT.

## Model Details

- Developed by: Manish Wagle / DevGen
- Base model: `paudelanil/trocr-devanagari-2`
- Adapter type: LoRA
- Task: image-to-text OCR for Devanagari word and short-line images
- Library: PEFT + Transformers

## Intended Use

Use this adapter for recognizing Devanagari text from cropped handwritten or printed word images. The DevGen runtime also supports light preprocessing such as foreground cropping and square padding for uploaded document-like images.

This model is not a general document understanding system. It does not perform page layout analysis, table extraction, translation, or language correction.

## Loading

```python
from peft import PeftModel
from transformers import AutoTokenizer, TrOCRProcessor, ViTImageProcessor, VisionEncoderDecoderModel

base_model_id = "paudelanil/trocr-devanagari-2"
adapter_id = "manishwagle/devgen-trocr-devanagari-lora"

image_processor = ViTImageProcessor.from_pretrained(adapter_id)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)

base_model = VisionEncoderDecoderModel.from_pretrained(base_model_id)
model = PeftModel.from_pretrained(base_model, adapter_id)
```

## Demo

A hosted Gradio demo is available as a Hugging Face Space:

`manishwagle/devgen-devanagari-ocr`

## Limitations

The adapter is most reliable on clear Devanagari word or short-line crops. Accuracy can degrade on very noisy images, multi-column documents, severe blur, extreme rotation, or text outside the training distribution.

## Training And Evaluation

The adapter was trained in the DevGen OCR workspace using a LoRA fine-tuning workflow for TrOCR. The local project includes reproducible evaluation scripts for corpus character error rate, word error rate, exact match, and preprocessing ablations.

## Framework Versions

- PEFT 0.19.1
- Transformers
- PyTorch
