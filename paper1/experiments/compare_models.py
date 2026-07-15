"""
Run one or more word images through EVERY Paper-1 model variant and print
predictions side by side — the whole ablation, live.

Usage:
    python -m paper1.experiments.compare_models                      # built-in demo set
    python -m paper1.experiments.compare_models --image my_word.png --truth "नमस्ते"
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.preprocessing import preprocess_pil_for_ocr  # noqa: E402

BASE_MODEL = "paudelanil/trocr-devanagari-2"

VARIANTS = [
    ("base (zero-shot)", None, None),
    ("lora_r4_attn", "paper1/runs/lora_r4_attn", None),
    ("lora_r8_attn", "paper1/runs/lora_r8_attn", None),
    ("lora_r16_attn", "paper1/runs/lora_r16_attn", None),
    ("lora_r32_attn", "paper1/runs/lora_r32_attn", None),
    ("lora_r16_attn_ffn", "paper1/runs/lora_r16_attn_ffn", None),
    ("lora_r16_legacy", "paper1/runs/lora_r16_legacy", None),
    ("lora_r16_legacy_s43", "paper1/runs/lora_r16_legacy_s43", None),
    ("lora_r16_legacy_s44", "paper1/runs/lora_r16_legacy_s44", None),
    ("full_ft", None, "paper1/runs/full_ft"),
]

# (image path, ground truth) — real test-set crops used in the paper figure
DEMO_SET = [
    ("paper1/paper/figures/qual_00019.png", "प्रोसैसर"),
    ("paper1/paper/figures/qual_00084.png", "पदोन्नति"),
    ("paper1/paper/figures/qual_00063.png", "अत्यधिक"),
    ("paper1/paper/figures/qual_00026.png", "ह्रास"),
    ("paper1/paper/figures/qual_00138.png", "अपनाता"),
    ("paper1/paper/figures/qual_00296.png", "दुश्मन"),
]


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_variant(adapter_path, full_model_path, device):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    processor = TrOCRProcessor.from_pretrained(BASE_MODEL)
    src = full_model_path or BASE_MODEL
    model = VisionEncoderDecoderModel.from_pretrained(src)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    return processor, model.to(device).eval()


@torch.inference_mode()
def predict(processor, model, images, device):
    out = []
    for img in images:
        pv = processor(images=preprocess_pil_for_ocr(img.convert("RGB")), return_tensors="pt").pixel_values.to(device)
        ids = model.generate(pv, max_length=32, num_beams=4, early_stopping=True,
                             decoder_start_token_id=model.config.decoder_start_token_id)
        out.append(processor.batch_decode(ids, skip_special_tokens=True)[0].strip())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", action="append", default=[], help="Word image path (repeatable).")
    parser.add_argument("--truth", action="append", default=[], help="Ground truth for each --image.")
    args = parser.parse_args()

    if args.image:
        cases = list(zip(args.image, args.truth + [None] * (len(args.image) - len(args.truth))))
    else:
        cases = DEMO_SET

    device = get_device()
    images = [Image.open(p) for p, _ in cases]
    truths = [t for _, t in cases]
    print(f"device={device}  images={len(images)}\n")

    results = {}
    for name, adapter, full in VARIANTS:
        if adapter and not Path(adapter).exists():
            print(f"[skip] {name} (missing {adapter})")
            continue
        if full and not (Path(full) / "model.safetensors").exists():
            print(f"[skip] {name} (missing weights)")
            continue
        processor, model = load_variant(adapter, full, device)
        results[name] = predict(processor, model, images, device)
        score = sum(p == t for p, t in zip(results[name], truths) if t is not None)
        print(f"[done] {name}: {score}/{len([t for t in truths if t])} exact")
        del model
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    # table
    col = max(len(n) for n in results) + 2
    header = "model".ljust(col) + "".join(f"{(t or '?'):>14}" for t in truths)
    print("\n" + header)
    print("-" * len(header))
    for name, preds in results.items():
        row = name.ljust(col)
        for p, t in zip(preds, truths):
            mark = "✓" if t and p == t else " "
            row += f"{(p + mark):>14}"
        print(row)


if __name__ == "__main__":
    main()
