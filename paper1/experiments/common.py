"""
Shared utilities for Paper 1 experiments: seeding, model loading, and metrics.

All metrics operate on NFC-normalized Unicode. This matters for Devanagari:
the same visual word can be encoded with different codepoint sequences
(e.g. precomposed vs decomposed nukta forms), which silently inflates CER.
"""

from __future__ import annotations

import json
import os
import random
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import editdistance
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoTokenizer, TrOCRProcessor, ViTImageProcessor, VisionEncoderDecoderModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DATASET_NAME = "c3rl/IIIT-INDIC-HW-WORDS-Hindi"
DEFAULT_BASE_MODEL = "paudelanil/trocr-devanagari-2"
DEFAULT_IMAGE_PROCESSOR = "google/vit-base-patch16-224-in21k"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_best_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


def cer(prediction: str, reference: str) -> float:
    """Character error rate over NFC-normalized codepoints."""
    prediction = normalize_text(prediction)
    reference = normalize_text(reference)
    if not reference:
        return 0.0 if not prediction else 1.0
    return editdistance.eval(prediction, reference) / len(reference)


def word_error(prediction: str, reference: str) -> int:
    """Exact-match word error (0 = correct). Dataset is word-level, so
    aggregate word error rate == 1 - word recognition accuracy (WRA)."""
    return int(normalize_text(prediction) != normalize_text(reference))


def akshara_error_rate(prediction: str, reference: str) -> float:
    """Edit distance over akshara (grapheme cluster) sequences."""
    from paper1.experiments.akshara import split_aksharas

    pred_units = split_aksharas(normalize_text(prediction))
    ref_units = split_aksharas(normalize_text(reference))
    if not ref_units:
        return 0.0 if not pred_units else 1.0
    return editdistance.eval(pred_units, ref_units) / len(ref_units)


def bootstrap_ci(values: list[float], n_resamples: int = 1000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of per-sample values."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        means[i] = rng.choice(arr, size=len(arr), replace=True).mean()
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


@dataclass
class EvalSummary:
    run_name: str
    num_samples: int
    cer: float
    cer_ci95: tuple[float, float]
    aer: float  # akshara error rate
    word_accuracy: float
    trainable_params: Optional[int] = None
    total_params: Optional[int] = None
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_name": self.run_name,
            "num_samples": self.num_samples,
            "cer": round(self.cer, 5),
            "cer_ci95": [round(v, 5) for v in self.cer_ci95],
            "aer": round(self.aer, 5),
            "word_accuracy": round(self.word_accuracy, 5),
            "trainable_params": self.trainable_params,
            "total_params": self.total_params,
            "config": self.config,
        }


def load_processor(model_name: str, processor_source: Optional[str] = None) -> TrOCRProcessor:
    """Mirror backend/trocr_engine.py processor resolution so eval matches serving."""
    if processor_source:
        try:
            image_processor = ViTImageProcessor.from_pretrained(processor_source)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
        except Exception:
            pass
    try:
        return TrOCRProcessor.from_pretrained(model_name)
    except Exception as exc:
        print(f"[common] Processor fallback for '{model_name}': {exc}")
        image_processor = ViTImageProcessor.from_pretrained(DEFAULT_IMAGE_PROCESSOR)
        image_processor.image_mean = [0.5, 0.5, 0.5]
        image_processor.image_std = [0.5, 0.5, 0.5]
        image_processor.rescale_factor = 1.0 / 255.0
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)


def load_model(
    base_model: str = DEFAULT_BASE_MODEL,
    adapter_path: Optional[str] = None,
    full_model_path: Optional[str] = None,
    device: Optional[str] = None,
) -> tuple[torch.nn.Module, TrOCRProcessor, str]:
    """Load one of: base model, base+LoRA adapter, or a fully fine-tuned checkpoint.

    Returns (model, processor, device). Model is in eval mode on the device.
    """
    device = device or os.getenv("TROCR_DEVICE") or get_best_torch_device()

    if full_model_path:
        model = VisionEncoderDecoderModel.from_pretrained(full_model_path)
        processor = load_processor(full_model_path)
    else:
        processor_source = None
        if adapter_path and (Path(adapter_path) / "preprocessor_config.json").exists():
            processor_source = adapter_path
        processor = load_processor(base_model, processor_source)
        model = VisionEncoderDecoderModel.from_pretrained(base_model)
        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path)

    base = model.get_base_model() if isinstance(model, PeftModel) else model
    base.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    base.config.pad_token_id = processor.tokenizer.pad_token_id
    base.config.eos_token_id = processor.tokenizer.sep_token_id
    base.config.vocab_size = base.config.decoder.vocab_size
    base.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
    base.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    base.generation_config.eos_token_id = processor.tokenizer.sep_token_id

    model.to(device)
    model.eval()
    return model, processor, device


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if trainable == 0 and isinstance(model, PeftModel):
        trainable = sum(
            p.numel()
            for name, p in model.named_parameters()
            if "lora_" in name or "modules_to_save" in name
        )
    return trainable, total


def save_results(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"[common] Wrote {output_path}")
