"""
Fine-tune TrOCR for Devanagari handwriting with LoRA.

This script is Mac/Apple Silicon friendly by default:
- uses MPS when available
- avoids CUDA-only fp16 training on MPS
- can train directly from HuggingFace or from extracted ./data folders
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
from typing import Any, Optional

import editdistance
import pandas as pd
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    ViTImageProcessor,
    VisionEncoderDecoderModel,
    default_data_collator,
)

MODEL_NAME = "microsoft/trocr-base-handwritten"
DATASET_NAME = "c3rl/IIIT-INDIC-HW-WORDS-Hindi"
DEFAULT_IMAGE_PROCESSOR = "google/vit-base-patch16-224-in21k"


def get_best_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_trocr_processor(model_name: str) -> TrOCRProcessor:
    try:
        return TrOCRProcessor.from_pretrained(model_name)
    except Exception as exc:
        print(f"Falling back to ViT image processor + checkpoint tokenizer: {exc}")
        image_processor = ViTImageProcessor.from_pretrained(DEFAULT_IMAGE_PROCESSOR)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)


class ExtractedDevanagariDataset(Dataset):
    def __init__(
        self,
        split_dir: Path,
        processor: TrOCRProcessor,
        max_target_length: int,
        limit: Optional[int] = None,
    ):
        self.split_dir = split_dir
        self.processor = processor
        self.max_target_length = max_target_length
        self.df = pd.read_csv(split_dir / "labels.csv")
        if limit:
            self.df = self.df.head(limit)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        image_path = self.split_dir / "images" / row["filename"]
        image = Image.open(image_path).convert("RGB")
        return encode_sample(image, str(row["text"]), self.processor, self.max_target_length)


class HuggingFaceDevanagariDataset(Dataset):
    def __init__(
        self,
        split: str,
        processor: TrOCRProcessor,
        max_target_length: int,
        limit: Optional[int] = None,
    ):
        self.dataset = load_dataset(DATASET_NAME, split=split)
        if limit:
            self.dataset = self.dataset.select(range(min(limit, len(self.dataset))))
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.dataset[idx]
        image = coerce_to_pil(sample["image"])
        return encode_sample(image, str(sample["text"]), self.processor, self.max_target_length)


class TrOCRSeq2SeqTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer with PEFT saving adjusted for VisionEncoderDecoder configs."""

    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[dict] = None) -> None:
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if state_dict is None:
            state_dict = self.model.state_dict()

        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(
                output_dir,
                state_dict=state_dict,
                save_embedding_layers=False,
            )
        else:
            torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def coerce_to_pil(image_data: Any) -> Image.Image:
    if isinstance(image_data, Image.Image):
        return image_data.convert("RGB")
    if isinstance(image_data, dict) and "bytes" in image_data:
        return Image.open(io.BytesIO(image_data["bytes"])).convert("RGB")
    if isinstance(image_data, bytes):
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image_data)!r}")


def encode_sample(
    image: Image.Image,
    text: str,
    processor: TrOCRProcessor,
    max_target_length: int,
) -> dict[str, torch.Tensor]:
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
    labels = processor.tokenizer(
        text,
        padding="max_length",
        max_length=max_target_length,
        truncation=True,
    ).input_ids
    labels = [
        label if label != processor.tokenizer.pad_token_id else -100
        for label in labels
    ]
    return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}


def build_datasets(args: argparse.Namespace, processor: TrOCRProcessor):
    if args.data_dir:
        data_dir = Path(args.data_dir).expanduser().resolve()
        train_dataset = ExtractedDevanagariDataset(
            data_dir / "train",
            processor,
            args.max_target_length,
            limit=args.train_limit,
        )
        eval_dataset = ExtractedDevanagariDataset(
            data_dir / "validation",
            processor,
            args.max_target_length,
            limit=args.eval_limit,
        )
        return train_dataset, eval_dataset

    train_dataset = HuggingFaceDevanagariDataset(
        "train",
        processor,
        args.max_target_length,
        limit=args.train_limit,
    )
    eval_dataset = HuggingFaceDevanagariDataset(
        "validation",
        processor,
        args.max_target_length,
        limit=args.eval_limit,
    )
    return train_dataset, eval_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR LoRA for Devanagari OCR.")
    parser.add_argument("--base-model", default=os.getenv("TROCR_BASE_MODEL", MODEL_NAME))
    parser.add_argument("--output-dir", default=os.getenv("TROCR_ADAPTER_ROOT", "./trocr-devanagari-lora"))
    parser.add_argument("--data-dir", default=None, help="Optional extracted dataset directory containing train/validation.")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--train-limit", type=int, default=None, help="Use a small subset for smoke tests.")
    parser.add_argument("--eval-limit", type=int, default=None, help="Use a small validation subset for smoke tests.")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_best_torch_device() if args.device == "auto" else args.device
    if device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    print(f"Initializing TrOCR LoRA training on {device}")
    processor = load_trocr_processor(args.base_model)
    model = VisionEncoderDecoderModel.from_pretrained(args.base_model)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.max_length = args.max_target_length
    model.generation_config.num_beams = 4

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["query", "value", "key", "dense"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset, eval_dataset = build_datasets(args, processor)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")

    def compute_metrics(pred):
        labels_ids = pred.label_ids.copy()
        pred_ids = pred.predictions
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        labels_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

        cer_sum = 0.0
        valid_samples = 0
        for prediction, label in zip(pred_str, labels_str):
            if label:
                cer_sum += editdistance.eval(prediction, label) / len(label)
                valid_samples += 1

        return {"cer": cer_sum / valid_samples if valid_samples else 0.0}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        predict_with_generate=True,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        remove_unused_columns=False,
        fp16=device == "cuda",
        dataloader_pin_memory=device == "cuda",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        report_to="none",
    )

    trainer = TrOCRSeq2SeqTrainer(
        model=model,
        processing_class=processor.image_processor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Training complete. Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
